import cv2
import os
import sys
import time
from ultralytics import YOLO

# -------------------------
# STUDENT NAME (TERMINAL + WEB)
# -------------------------
if len(sys.argv) > 1:
    student_name = sys.argv[1]
else:
    student_name = input("Enter student name: ")

dataset_path = "TrainingImage"
student_path = os.path.join(dataset_path, student_name)
os.makedirs(student_path, exist_ok=True)

# Load YOLO face model
try:
    model = YOLO('yolov8n-face.pt')
except Exception as e:
    print(f"Failed to load YOLO model: {e}")
    sys.exit(1)

# -------------------------
# CAMERA SETUP
# -------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Camera not accessible")
    sys.exit(1)

WINDOW_NAME = "Smart Attendance System - Registration"

try:
    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1:
        cv2.destroyWindow(WINDOW_NAME)
except Exception:
    pass

cv2.destroyAllWindows()
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

    current_time = time.time()
    
    # Run YOLOv8 face detection
    results = model(frame, verbose=False)
    face_detected = False
    
    # Process detections
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            face_detected = True
            
            # Crop and save if enough time has passed
            if current_time - last_capture >= capture_interval:
                # Ensure coordinates are within frame
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size > 0:
                    img_path = os.path.join(student_path, f"{count}.jpg")
                    cv2.imwrite(img_path, face_crop)
                    count += 1
                    last_capture = current_time

    status_color = (0, 255, 0) if face_detected else (0, 0, 255)
    status_text = f"Capturing images... {count}" if face_detected else "No Face Detected"
    
    cv2.putText(
        frame,
        status_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        status_color,
        2
    )

    cv2.imshow(WINDOW_NAME, frame)
    cv2.waitKey(1)

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

cap.release()

try:
    cv2.destroyWindow(WINDOW_NAME)
except:
    pass

cv2.destroyAllWindows()
time.sleep(0.2)

print(f"[OK] Registration completed for {student_name}")
print(f"[INFO] Total images captured: {count}")
