import glob, os, cv2, numpy as np

BOARD_SIZE = (9, 6)
SQUARE_SIZE_M = 0.0275
IMG_DIR = os.path.expanduser("~/aruco_test/calib_images/*.png")
OUT_FILE = os.path.expanduser("~/aruco_test/camera_calibration.yaml")

objp = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE_M

objpoints, imgpoints = [], []
img_shape = None
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

for fname in sorted(glob.glob(IMG_DIR)):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_shape = gray.shape[::-1]
    ret, corners = cv2.findChessboardCorners(gray, BOARD_SIZE, None)
    if ret:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners2)

print(f"Using {len(objpoints)} valid calibration images at resolution {img_shape}...")

# Use CALIB_RATIONAL_MODEL (8 distortion coeffs) for 120-deg wide-angle imx708_wide
flags = cv2.CALIB_RATIONAL_MODEL
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, img_shape, None, None, flags=flags
)

# Compute reprojection error
total_err = 0
for i in range(len(objpoints)):
    proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    total_err += cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / len(proj)
mean_err = total_err / len(objpoints)

print(f"\nMean Reprojection Error: {mean_err:.4f} pixels (Target: < 0.50 px)")
print("Camera Matrix (K):\n", mtx)
print("Distortion Coefficients (k1, k2, p1, p2, k3, k4, k5, k6):\n", dist.ravel())

fs = cv2.FileStorage(OUT_FILE, cv2.FILE_STORAGE_WRITE)
fs.write("image_width", img_shape[0])
fs.write("image_height", img_shape[1])
fs.write("camera_matrix", mtx)
fs.write("distortion_coefficients", dist)
fs.write("reprojection_error", mean_err)
fs.release()
print(f"\nSaved calibration to {OUT_FILE}")
