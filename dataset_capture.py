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

cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera", 800, 600)

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

    cv2.imshow("Camera", frame)
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

    cv2.imshow("Camera", frame)
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

    cv2.imshow("Camera", frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()

print(f"✅ Registration completed for {student_name}")
print(f"📸 Total images captured: {count}")
