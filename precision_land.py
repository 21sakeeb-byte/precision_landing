import math
import os
import time
import cv2
import numpy as np
from flask import Flask, Response
from picamera2 import Picamera2
from pymavlink import mavutil

# --- 1. Configuration ---
ENABLE_MAVLINK = False          # Set True when wired to Pixhawk TELEM2
SERIAL_PORT = "/dev/ttyAMA0"
BAUD_RATE = 921600

OUTER_ID = 0
# Update to your final outdoor pad size (e.g., 0.25)
OUTER_SIZE_M = 0.25
INNER_ID = 1
# Update to your final indoor/inner pad size (e.g., 0.07)
INNER_SIZE_M = 0.04
HANDOFF_Z_M = 1.50              # Switch to inner marker below 1.5m
MAX_COAST_SEC = 0.50            # Kalman prediction coast limit during brief frame drops

CALIB_FILE = os.path.expanduser("~/aruco_test/camera_calibration.yaml")

# --- 2. 2D Constant-Velocity Kalman Filter [x, y, vx, vy] ---


class TargetKalman2D:
    def __init__(self):
        self.x = np.zeros((4, 1), dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 1.0
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=np.float64)
        # Measurement noise (~4.5 cm std)
        self.R = np.eye(2, dtype=np.float64) * 0.002
        # Process noise (target acceleration)
        self.q_accel = 0.5
        self.initialized = False
        self.last_meas_time = 0.0

    def predict(self, dt):
        if not self.initialized or dt <= 0:
            return
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1,  0],
                      [0, 0, 0,  1]], dtype=np.float64)
        dt2 = dt * dt
        dt3 = dt2 * dt / 2.0
        dt4 = dt2 * dt2 / 4.0
        Q = self.q_accel * np.array([
            [dt4, 0,   dt3, 0],
            [0,   dt4, 0,   dt3],
            [dt3, 0,   dt2, 0],
            [0,   dt3, 0,   dt2]
        ], dtype=np.float64)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, meas_x, meas_y, now):
        z = np.array([[meas_x], [meas_y]], dtype=np.float64)
        if not self.initialized:
            self.x = np.array(
                [[meas_x], [meas_y], [0.0], [0.0]], dtype=np.float64)
            self.P = np.eye(4, dtype=np.float64) * 0.1
            self.initialized = True
            self.last_meas_time = now
            return
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y)
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.last_meas_time = now

    def is_valid(self, now):
        return self.initialized and ((now - self.last_meas_time) <= MAX_COAST_SEC)


# --- 3. Load Camera Calibration ---
fs = cv2.FileStorage(CALIB_FILE, cv2.FILE_STORAGE_READ)
if not fs.isOpened():
    raise RuntimeError(f"Could not open {CALIB_FILE}")
camera_matrix = fs.getNode("camera_matrix").mat()
dist_coeffs = fs.getNode("distortion_coefficients").mat()
fs.release()


def get_marker_obj_points(size_m):
    half = size_m / 2.0
    return np.array([
        [-half,  half, 0.0],
        [half,  half, 0.0],
        [half, -half, 0.0],
        [-half, -half, 0.0]
    ], dtype=np.float32)


OBJ_PTS = {
    OUTER_ID: get_marker_obj_points(OUTER_SIZE_M),
    INNER_ID: get_marker_obj_points(INNER_SIZE_M),
}

# --- 4. MAVLink Connection ---
mav = None
if ENABLE_MAVLINK:
    mav = mavutil.mavlink_connection(SERIAL_PORT, baud=BAUD_RATE)
    print(f"Waiting for Pixhawk heartbeat on {SERIAL_PORT}...")
    mav.wait_heartbeat()
    print(f"Connected to Pixhawk (System {mav.target_system})")

# --- 5. Camera & ArUco Setup ---
picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": (1920, 1080)},
    lores={"size": (1280, 720), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()
picam2.set_controls({"AfMode": 0, "LensPosition": 0.0})

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
kf = TargetKalman2D()

app = Flask(__name__)


def estimate_pose(corners_2d, marker_id):
    ok, rvec, tvec = cv2.solvePnP(
        OBJ_PTS[marker_id],
        corners_2d.reshape((4, 2)).astype(np.float32),
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    if not ok:
        return None
    x, y, z = tvec.ravel()
    return x, y, z, rvec, tvec


def generate_frames():
    prev_time = time.time()
    last_z = 1.0

    while True:
        frame_rgb = picam2.capture_array("lores")
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        now = time.time()
        dt = now - prev_time
        fps = 1.0 / max(dt, 1e-6)
        prev_time = now

        kf.predict(dt)

        corners, ids, _ = detector.detectMarkers(gray)
        poses = {}
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            for i, m_id in enumerate(ids.flatten()):
                m_id = int(m_id)
                if m_id in OBJ_PTS:
                    res = estimate_pose(corners[i], m_id)
                    if res is not None:
                        poses[m_id] = res

        active_id = None
        if OUTER_ID in poses and INNER_ID in poses:
            active_id = INNER_ID if poses[OUTER_ID][2] <= HANDOFF_Z_M else OUTER_ID
        elif INNER_ID in poses:
            active_id = INNER_ID
        elif OUTER_ID in poses:
            active_id = OUTER_ID

        if active_id is not None:
            cam_x, cam_y, cam_z, rvec, tvec = poses[active_id]
            last_z = cam_z
            kf.update(cam_x, cam_y, now)
            cv2.drawFrameAxes(frame, camera_matrix,
                              dist_coeffs, rvec, tvec, 0.05)

        if kf.is_valid(now):
            fx, fy, vx, vy = kf.x.ravel()
            # Map camera frame (top of image = drone nose) to MAV_FRAME_BODY_FRD
            body_x = -fy   # Forward
            body_y = fx   # Right
            body_z = last_z

            angle_x = math.atan2(body_y, body_z)  # Rightward angle (rad)
            angle_y = math.atan2(body_x, body_z)  # Forward angle (rad)

            if ENABLE_MAVLINK and mav is not None:
                print(
                    f"[TX] ID={active_id} X={body_x:.2f}m Y={body_y:.2f}m Z={body_z:.2f}m", flush=True)
                mav.mav.landing_target_send(
                    int(now * 1e6),                         # time_usec
                    0,                                      # target_num
                    mavutil.mavlink.MAV_FRAME_BODY_FRD,     # frame
                    angle_x,                                # angle_x (rad)
                    angle_y,                                # angle_y (rad)
                    body_z,                                 # distance (m)
                    0.0, 0.0,                               # size_x, size_y
                    body_x, body_y, body_z,                 # x, y, z (m)
                    (1.0, 0.0, 0.0, 0.0),                   # q (unused)
                    2,                                      # type: MAV_LANDING_TARGET_TYPE_VISION_FIDUCIAL
                    1                                       # position_valid = 1
                )

            mode_tag = "LOCK" if active_id is not None else "COAST"
            status = (
                f"[{mode_tag}] FRD X:{body_x:+.3f}m Y:{body_y:+.3f}m Z:{body_z:.3f}m | "
                f"Vel:({vx:+.2f},{vy:+.2f})m/s"
            )
            cv2.putText(frame, status, (15, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "NO TARGET LOCK (SAFE HOVER)", (15, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 140, 255), 2)

        cv2.putText(frame, f"FPS:{fps:.1f} | MAVLink:{ENABLE_MAVLINK}",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")


@app.route("/")
def index():
    return """
    <html><body style="background:#111; color:#eee; font-family:sans-serif; text-align:center;">
      <h3>ArduPilot Precision Landing - Kalman + MAVLink</h3>
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
