import math
import os
import time
import cv2
import numpy as np
from flask import Flask, Response
from picamera2 import Picamera2

# --- 1. Marker Configuration (Update to your exact printed sizes in meters) ---
OUTER_ID = 0
OUTER_SIZE_M = 0.159
INNER_ID = 1
INNER_SIZE_M = 0.159
HANDOFF_Z_M = 1.50    # Switch to inner marker below 1.5 meters if visible

CALIB_FILE = os.path.expanduser("~/aruco_test/camera_calibration.yaml")

# --- 2. Load Camera Calibration ---
fs = cv2.FileStorage(CALIB_FILE, cv2.FILE_STORAGE_READ)
if not fs.isOpened():
    raise RuntimeError(f"Could not open {CALIB_FILE}. Run calibrate.py first.")
camera_matrix = fs.getNode("camera_matrix").mat()
dist_coeffs = fs.getNode("distortion_coefficients").mat()
fs.release()

def get_marker_obj_points(size_m):
    half = size_m / 2.0
    # OpenCV ArUco corner order: top-left, top-right, bottom-right, bottom-left
    return np.array([
        [-half,  half, 0.0],
        [ half,  half, 0.0],
        [ half, -half, 0.0],
        [-half, -half, 0.0]
    ], dtype=np.float32)

OBJ_PTS = {
    OUTER_ID: get_marker_obj_points(OUTER_SIZE_M),
    INNER_ID: get_marker_obj_points(INNER_SIZE_M),
}

# --- 3. Initialize Camera at 1280x720 (Must match calibration resolution!) ---
picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": (1920, 1080)},
    lores={"size": (1280, 720), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()
picam2.set_controls({"AfMode": 0, "LensPosition": 0.0})

# --- 4. Setup ArUco Detector ---
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

app = Flask(__name__)

def estimate_pose(corners_2d, marker_id):
    obj_pts = OBJ_PTS[marker_id]
    img_pts = corners_2d.reshape((4, 2)).astype(np.float32)
    # IPPE_SQUARE is optimal for planar square markers
    ok, rvec, tvec = cv2.solvePnP(
        obj_pts, img_pts, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    if not ok:
        return None
    x, y, z = tvec.ravel()
    angle_x = math.atan2(x, z)
    angle_y = math.atan2(y, z)
    return x, y, z, angle_x, angle_y, rvec, tvec

def generate_frames():
    prev_time = time.time()
    while True:
        frame_rgb = picam2.capture_array("lores")
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = detector.detectMarkers(gray)

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        poses = {}
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            for i, m_id in enumerate(ids.flatten()):
                m_id = int(m_id)
                if m_id in OBJ_PTS:
                    res = estimate_pose(corners[i], m_id)
                    if res is not None:
                        poses[m_id] = res

        # Nested marker selection logic
        active_id = None
        if OUTER_ID in poses and INNER_ID in poses:
            # Use inner marker when closer than HANDOFF_Z_M
            active_id = INNER_ID if poses[OUTER_ID][2] <= HANDOFF_Z_M else OUTER_ID
        elif INNER_ID in poses:
            active_id = INNER_ID
        elif OUTER_ID in poses:
            active_id = OUTER_ID

        if active_id is not None:
            x, y, z, ax, ay, rvec, tvec = poses[active_id]
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.05)
            status = (
                f"ACTIVE ID:{active_id} | X:{x:+.3f}m Y:{y:+.3f}m Z:{z:.3f}m | "
                f"Ang:({math.degrees(ax):+.1f},{math.degrees(ay):+.1f})deg"
            )
            cv2.putText(frame, status, (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            print(f"[{fps:4.1f} FPS] {status}")
        else:
            cv2.putText(frame, "SEARCHING FOR TARGET...", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)

        cv2.putText(frame, f"FPS: {fps:.1f} | 3D Pose Active (1280x720)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

@app.route("/")
def index():
    return """
    <html><body style="background:#111; color:#eee; font-family:sans-serif; text-align:center;">
      <h3>PiCam v3 Wide NoIR - 3D ArUco Pose Estimation</h3>
      <img src="/video_feed" style="max-width:95%; border:2px solid #444;"/>
    </body></html>
    """

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, threaded=True)
    finally:
        picam2.stop()
