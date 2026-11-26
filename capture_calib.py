import os, cv2
from flask import Flask, Response, render_template_string
from picamera2 import Picamera2

SAVE_DIR = os.path.expanduser("~/aruco_test/calib_images")
os.makedirs(SAVE_DIR, exist_ok=True)
BOARD_SIZE = (9, 6)  # 9x6 inner corners (10x7 squares)

picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": (1920, 1080)},
    lores={"size": (1280, 720), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()
picam2.set_controls({"AfMode": 0, "LensPosition": 0.0})

app = Flask(__name__)
latest_frame = None
saved_count = len([f for f in os.listdir(SAVE_DIR) if f.endswith(".png")])

HTML = """
<!doctype html>
<html>
<head><title>PiCam3 Wide Calibration (1280x720)</title></head>
<body style="font-family:sans-serif; background:#111; color:#eee; text-align:center;">
  <h2>PiCam v3 Wide Calibration (1280x720 | Focus=0.0)</h2>
  <img src="/video" style="border:2px solid #444; max-width:95%;"><br><br>
  <button onclick="fetch('/snap').then(r=>r.text()).then(t=>document.getElementById('msg').innerText=t)"
          style="font-size:20px; padding:12px 24px; background:#28a745; color:white; border:none; border-radius:6px; cursor:pointer;">
    Capture Image
  </button>
  <p id="msg" style="font-size:18px; color:#0f0;">Saved images: {{count}}</p>
  <p style="color:#aaa; max-width:650px; margin:auto;">
    Capture 18-25 shots of your 9x6 checkerboard. Move it across the center, all 4 corners,
    and outer edges of the wide-angle frame while tilting pitch/roll/yaw (0.4m - 1.5m away).
  </p>
</body>
</html>
"""

def gen_frames():
    global latest_frame
    while True:
        frame = picam2.capture_array("lores")
        latest_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, BOARD_SIZE,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        )
        vis = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if found:
            cv2.drawChessboardCorners(vis, BOARD_SIZE, corners, found)
        _, buf = cv2.imencode(".jpg", vis)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

@app.route("/")
def index():
    return render_template_string(HTML, count=saved_count)

@app.route("/video")
def video():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/snap")
def snap():
    global saved_count
    if latest_frame is None:
        return "No frame yet"
    gray = cv2.cvtColor(latest_frame, cv2.COLOR_RGB2GRAY)
    found, _ = cv2.findChessboardCorners(gray, BOARD_SIZE)
    if not found:
        return f"Board NOT detected! Adjust angle/lighting. (Saved: {saved_count})"
    fname = os.path.join(SAVE_DIR, f"calib_{saved_count:02d}.png")
    cv2.imwrite(fname, cv2.cvtColor(latest_frame, cv2.COLOR_RGB2BGR))
    saved_count += 1
    return f"Saved {os.path.basename(fname)}! Total: {saved_count}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
