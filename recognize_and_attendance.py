import cv2
import numpy as np
import csv
from datetime import datetime
import os
import time
import sys

# Check if model files exist
if not os.path.exists("trainer.yml"):
    print("❌ ERROR: Model not trained yet!")
    print("Please train the model first using 'Train / Update Model' button.")
    time.sleep(3)  # Give user time to read the error
    sys.exit(1)

if not os.path.exists("labels.npy"):
    print("❌ ERROR: Labels file not found!")
    print("Please train the model first using 'Train / Update Model' button.")
    time.sleep(3)  # Give user time to read the error
    sys.exit(1)

# Try to load model with error handling
try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer.yml")
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ ERROR: Failed to load model file!")
    print(f"Error: {str(e)}")
    print("Please retrain the model using 'Train / Update Model' button.")
    time.sleep(3)  # Give user time to read the error
    sys.exit(1)

# Try to load labels with error handling
try:
    label_map = np.load("labels.npy", allow_pickle=True).item()
    print(f"✅ Labels loaded successfully: {len(label_map)} person(s) registered")
except Exception as e:
    print(f"❌ ERROR: Failed to load labels file!")
    print(f"Error: {str(e)}")
    print("Please retrain the model using 'Train / Update Model' button.")
    time.sleep(3)  # Give user time to read the error
    sys.exit(1)

face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

attendance_file = "attendance.csv"

if not os.path.exists(attendance_file):
    with open(attendance_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Date", "Time"])

def mark_attendance(name):
    today = datetime.now().strftime("%Y-%m-%d")
    time_now = datetime.now().strftime("%H:%M:%S")

    with open(attendance_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, today, time_now])

    print(f"Attendance marked for {name}")

def filter_overlapping_faces(faces, overlap_threshold=0.3):
    """Remove overlapping face detections, keep the largest one"""
    if len(faces) == 0:
        return []
    
    # Convert to list of (x, y, w, h) tuples
    faces = [(x, y, w, h) for (x, y, w, h) in faces]
    
    # Sort by area (largest first)
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    
    filtered = []
    for current_face in faces:
        cx, cy, cw, ch = current_face
        current_area = cw * ch
        
        # Check if this face overlaps significantly with any already accepted face
        overlap = False
        for accepted_face in filtered:
            ax, ay, aw, ah = accepted_face
            
            # Calculate overlap area
            overlap_x = max(0, min(cx + cw, ax + aw) - max(cx, ax))
            overlap_y = max(0, min(cy + ch, ay + ah) - max(cy, ay))
            overlap_area = overlap_x * overlap_y
            
            # If overlap is significant, skip this face
            if overlap_area > current_area * overlap_threshold:
                overlap = True
                break
        
        if not overlap:
            filtered.append(current_face)
    
    return filtered

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

cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("❌ ERROR: Camera not accessible!")
    print("Please check if camera is connected and not being used by another application.")
    time.sleep(3)  # Give user time to read the error
    sys.exit(1)

# Create and configure window
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 800, 600)

# Make sure window is visible
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
time.sleep(0.1)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 0)

# Set attendance duration to 10 seconds (minimum)
ATTENDANCE_DURATION = 10  # seconds
start_time = time.time()

# Track which students have been marked to avoid duplicates
marked_students = set()

print("📸 Attendance taking started! Camera open for 10 seconds...")
print("-" * 50)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    remaining_time = max(0, ATTENDANCE_DURATION - elapsed_time)
    
    # Break after at least 10 seconds
    if elapsed_time >= ATTENDANCE_DURATION:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces with parameters optimized for multiple faces
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.2,  # Slightly more sensitive
        minNeighbors=4,   # Slightly less strict
        minSize=(80, 80)  # Smaller minimum size to catch more faces
    )
    
    # Filter out overlapping detections (keep unique faces only)
    faces = filter_overlapping_faces(faces)
    
    # Track students detected in current frame
    current_frame_students = []
    
    # Process ALL detected faces in this frame simultaneously
    for (x, y, w, h) in faces:
        # Crop and resize face to match training size
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (200, 200))
        
        # Predict with recognizer
        label, confidence = recognizer.predict(face_img)
        
        name = "Unknown"
        color = (0, 0, 255)
        is_recognized = False
        
        # Lower confidence means better match (threshold adjusted)
        if confidence < 70:
            name = label_map.get(label, "Unknown")
            if name != "Unknown":
                is_recognized = True
                current_frame_students.append(name)
                
                # Mark attendance if not already marked (process all in this frame)
                if name not in marked_students:
                    mark_attendance(name)
                    marked_students.add(name)
                    print(f"✅ Attendance marked: {name} (Confidence: {confidence:.1f})")
                
                color = (0, 255, 0)  # Green for recognized
            else:
                color = (0, 165, 255)  # Orange for unknown
        else:
            color = (0, 0, 255)  # Red for low confidence

        # Draw rectangle and label for each recognized face
        text = f"{name}"
        if is_recognized and name in marked_students:
            text += " ✓"  # Show checkmark for already marked
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(
            frame,
            text,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2
        )
    
    # Display timer and status information
    timer_text = f"Time Remaining: {remaining_time:.1f}s"
    cv2.putText(
        frame,
        timer_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )
    
    # Display count of marked students
    marked_count = len(marked_students)
    count_text = f"Marked: {marked_count} student(s)"
    cv2.putText(
        frame,
        count_text,
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )
    
    # Display current frame detection status
    if current_frame_students:
        detected_text = f"Detected in frame: {len(current_frame_students)}"
        cv2.putText(
            frame,
            detected_text,
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )
    
    # Display list of marked students
    if marked_students:
        y_offset = 130
        cv2.putText(
            frame,
            "Marked Students:",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        student_list = sorted(list(marked_students))
        for idx, student in enumerate(student_list[:5]):  # Show up to 5 names
            cv2.putText(
                frame,
                f"  {idx+1}. {student}",
                (10, y_offset + 25 * (idx + 1)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit early
        break

print("-" * 50)
print(f"✅ Attendance session complete!")
print(f"📊 Total students marked: {len(marked_students)}")
if marked_students:
    print(f"👥 Students: {', '.join(marked_students)}")
else:
    print("⚠️  No students were recognized during this session.")

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
