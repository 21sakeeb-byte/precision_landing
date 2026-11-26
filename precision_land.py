#!/usr/bin/env python3
"""
ArduPilot precision landing (ArUco + Kalman + MAVLink LANDING_TARGET).

Default mode is HEADLESS: no video, no web server, just vision + MAVLink.

  python3 precision_land.py                 # headless, MAVLink ON  (flight)
  python3 precision_land.py --no-mavlink    # headless, MAVLink OFF (bench, no Pixhawk)
  python3 precision_land.py --stream        # also serve video on http://<pi-ip>:5000
"""
import argparse
import math
import sys
import threading
import time
import signal
from pathlib import Path

import cv2
import numpy as np
from pymavlink import mavutil
from picamera2 import Picamera2

# --- 1. Configuration ---------------------------------------------------
SERIAL_PORT = "/dev/ttyAMA0"
BAUD_RATE = 921600

OUTER_ID = 0
OUTER_SIZE_M = 0.25     # MEASURE your printed marker (black square edge to edge)
INNER_ID = 1
INNER_SIZE_M = 0.04     # MEASURE your printed marker
HANDOFF_Z_M = 1.50
MAX_COAST_SEC = 0.50

CALIB_FILE = Path(__file__).resolve().parent / "camera_calibration.yaml"
LOG_EVERY_SEC = 1.0


# --- 2. 2D constant-velocity Kalman filter [x, y, vx, vy] ---------------
class TargetKalman2D:
    def __init__(self):
        self.x = np.zeros((4, 1))
        self.P = np.eye(4)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=np.float64)
        self.R = np.eye(2) * 0.002
        self.q_accel = 0.5
        self.initialized = False
        self.last_meas_time = 0.0

    def predict(self, dt):
        if not self.initialized or dt <= 0:
            return
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=np.float64)
        dt2 = dt * dt
        dt3 = dt2 * dt / 2.0
        dt4 = dt2 * dt2 / 4.0
        Q = self.q_accel * np.array([
            [dt4, 0, dt3, 0],
            [0, dt4, 0, dt3],
            [dt3, 0, dt2, 0],
            [0, dt3, 0, dt2]], dtype=np.float64)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, mx, my, now):
        z = np.array([[mx], [my]], dtype=np.float64)
        if not self.initialized:
            self.x = np.array([[mx], [my], [0.0], [0.0]])
            self.P = np.eye(4) * 0.1
            self.initialized = True
            self.last_meas_time = now
            return
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.last_meas_time = now

    def is_valid(self, now):
        return self.initialized and (now - self.last_meas_time) <= MAX_COAST_SEC


# --- 3. Helpers ---------------------------------------------------------
def load_calibration():
    fs = cv2.FileStorage(str(CALIB_FILE), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        sys.exit(f"ERROR: cannot open {CALIB_FILE}. Put camera_calibration.yaml next to this script.")
    K = fs.getNode("camera_matrix").mat()
    D = fs.getNode("distortion_coefficients").mat()
    fs.release()
    return K, D


def marker_obj_points(size_m):
    h = size_m / 2.0
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float32)


OBJ_PTS = {OUTER_ID: marker_obj_points(OUTER_SIZE_M),
           INNER_ID: marker_obj_points(INNER_SIZE_M)}


def estimate_pose(corners_2d, marker_id, K, D):
    ok, rvec, tvec = cv2.solvePnP(OBJ_PTS[marker_id],
                                  corners_2d.reshape((4, 2)).astype(np.float32),
                                  K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    x, y, z = tvec.ravel()
    return x, y, z, rvec, tvec


# --- 4. Vision + MAVLink loop (runs in its own thread) ------------------
class Shared:
    """Latest JPEG for the optional web stream."""
    def __init__(self):
        self.lock = threading.Lock()
        self.jpeg = None


def vision_loop(args, stop, shared, mav):
    K, D = load_calibration()

    picam2 = Picamera2()
    cfg = picam2.create_video_configuration(
        main={"size": (1920, 1080)},
        lores={"size": (1280, 720), "format": "RGB888"})
    picam2.configure(cfg)
    picam2.start()
    picam2.set_controls({"AfMode": 0, "LensPosition": 0.0})

    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
        cv2.aruco.DetectorParameters())
    kf = TargetKalman2D()

    prev = time.time()
    last_z = 1.0
    last_log = 0.0
    last_hb = 0.0
    print("Vision loop running. Ctrl+C to stop.", flush=True)

    try:
        while not stop.is_set():
            frame_rgb = picam2.capture_array("lores")
            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

            now = time.time()
            dt = now - prev
            fps = 1.0 / max(dt, 1e-6)
            prev = now
            kf.predict(dt)

            corners, ids, _ = detector.detectMarkers(gray)
            poses = {}
            if ids is not None:
                for i, m_id in enumerate(ids.flatten()):
                    m_id = int(m_id)
                    if m_id in OBJ_PTS:
                        res = estimate_pose(corners[i], m_id, K, D)
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

            status = "NO TARGET LOCK (SAFE HOVER)"
            if kf.is_valid(now):
                fx, fy, vx, vy = kf.x.ravel()
                body_x = -fy      # forward  (image top = drone nose)
                body_y = fx       # right
                body_z = last_z   # down
                angle_x = math.atan2(body_y, body_z)
                angle_y = math.atan2(body_x, body_z)
                distance = math.sqrt(body_x**2 + body_y**2 + body_z**2)

                if mav is not None:
                    mav.mav.landing_target_send(
                        int(now * 1e6), 0,
                        mavutil.mavlink.MAV_FRAME_BODY_FRD,
                        angle_x, angle_y, distance,
                        0.0, 0.0,
                        body_x, body_y, body_z,
                        (1.0, 0.0, 0.0, 0.0),
                        2,   # VISION_FIDUCIAL
                        1)   # position_valid

                tag = "LOCK" if active_id is not None else "COAST"
                status = (f"[{tag}] ID={active_id} FRD X:{body_x:+.3f} Y:{body_y:+.3f} "
                          f"Z:{body_z:.3f} m | Vel:({vx:+.2f},{vy:+.2f}) m/s")

            # companion heartbeat, 1 Hz
            if mav is not None and now - last_hb >= 1.0:
                mav.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                                       mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
                last_hb = now

            # throttled console log (once per second, not every frame)
            if now - last_log >= LOG_EVERY_SEC:
                print(f"{fps:4.1f} FPS | {status}", flush=True)
                last_log = now

            # drawing + JPEG encoding ONLY if streaming
            if args.stream:
                frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                if active_id is not None:
                    cv2.drawFrameAxes(frame, K, D, rvec, tvec, 0.05)
                cv2.putText(frame, status, (15, 65), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 255, 0), 2)
                cv2.putText(frame, f"FPS:{fps:.1f} MAVLink:{mav is not None}", (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    with shared.lock:
                        shared.jpeg = buf.tobytes()
    finally:
        picam2.stop()


# --- 5. Optional web stream (viewer only; does NOT run the landing logic)
def run_stream(shared, stop):
    from flask import Flask, Response
    app = Flask(__name__)

    def frames():
        while not stop.is_set():
            with shared.lock:
                jpeg = shared.jpeg
            if jpeg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            time.sleep(0.05)

    @app.route("/")
    def index():
        return ('<html><body style="background:#111;color:#eee;text-align:center">'
                '<h3>Precision Landing</h3><img src="/video_feed" style="max-width:95%">'
                '</body></html>')

    @app.route("/video_feed")
    def video_feed():
        return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    app.run(host="0.0.0.0", port=5000, threaded=True)


# --- 6. Main ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", action="store_true", help="serve video on port 5000")
    ap.add_argument("--no-mavlink", action="store_true", help="do not talk to the Pixhawk")
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # clean stop under systemd

    mav = None
    if not args.no_mavlink:
        mav = mavutil.mavlink_connection(SERIAL_PORT, baud=BAUD_RATE)
        print(f"Waiting for Pixhawk heartbeat on {SERIAL_PORT}...", flush=True)
        if mav.wait_heartbeat(timeout=15) is None:
            sys.exit("ERROR: no heartbeat from Pixhawk (check wiring/baud/SERIAL2_PROTOCOL).")
        print(f"Connected to Pixhawk (system {mav.target_system})", flush=True)

    stop = threading.Event()
    shared = Shared()
    worker = threading.Thread(target=vision_loop, args=(args, stop, shared, mav), daemon=True)
    worker.start()

    try:
        if args.stream:
            run_stream(shared, stop)          # blocks until Ctrl+C
        else:
            while worker.is_alive():          # headless: just wait
                time.sleep(0.5)
            sys.exit("Vision thread stopped unexpectedly.")
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        worker.join(timeout=3)


if __name__ == "__main__":
    main()
