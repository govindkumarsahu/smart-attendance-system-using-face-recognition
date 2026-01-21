import cv2
import os
import sys
import time

# -------------------------
# STUDENT NAME (TERMINAL + WEB)
# -------------------------
if len(sys.argv) > 1:
    student_name = sys.argv[1]
else:
    student_name = input("Enter student name: ")

dataset_path = "dataset"
student_path = os.path.join(dataset_path, student_name)
os.makedirs(student_path, exist_ok=True)

# -------------------------
# CAMERA SETUP
# -------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not accessible")
    sys.exit()

# Use consistent window name for all camera operations
WINDOW_NAME = "Smart Attendance System"

# Destroy any existing windows first to ensure single window
try:
    # Try to destroy the specific window if it exists (Windows compatible check)
    window_prop = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE)
    if window_prop >= 1:
        cv2.destroyWindow(WINDOW_NAME)
except (cv2.error, Exception):
    # Window doesn't exist or error - continue anyway
    pass

# Destroy all windows to ensure clean state
cv2.destroyAllWindows()

# Small delay to ensure windows are fully closed and camera released
time.sleep(0.3)

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 800, 600)

# -------------------------
# 1 SECOND PREVIEW
# -------------------------
preview_start = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.putText(
        frame,
        "Camera Preview - Look at camera",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2
    )

    cv2.imshow(WINDOW_NAME, frame)
    cv2.waitKey(1)

    if time.time() - preview_start >= 1:
        break

# -------------------------
# CAPTURE MODE (5 SECONDS)
# -------------------------
start_time = time.time()
count = 0
capture_interval = 0.3
last_capture = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.putText(
        frame,
        f"Capturing images... {count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.imshow(WINDOW_NAME, frame)
    cv2.waitKey(1)

    current_time = time.time()

    if current_time - last_capture >= capture_interval:
        img_path = os.path.join(student_path, f"{count}.jpg")
        cv2.imwrite(img_path, frame)
        count += 1
        last_capture = current_time

    if current_time - start_time >= 5:
        break

# -------------------------
# END MESSAGE (1 SECOND)
# -------------------------
end_time = time.time()
while time.time() - end_time < 1:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.putText(
        frame,
        "Registration Complete",
        (200, 300),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    cv2.imshow(WINDOW_NAME, frame)
    cv2.waitKey(1)

# Proper cleanup - release camera first, then destroy windows
cap.release()

# Destroy the specific window
try:
    cv2.destroyWindow(WINDOW_NAME)
except:
    pass

# Destroy all windows
cv2.destroyAllWindows()

# Small delay to ensure camera and windows are fully released
time.sleep(0.2)

print(f"✅ Registration completed for {student_name}")
print(f"📸 Total images captured: {count}")
