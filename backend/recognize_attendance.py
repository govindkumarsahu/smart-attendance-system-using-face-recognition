import cv2
import numpy as np
from datetime import datetime
import os
import time
import sys
from collections import defaultdict
from ultralytics import YOLO
from deepface import DeepFace

# ==========================================
# PARSE COMMAND-LINE ARGUMENTS
# ==========================================
# Usage: python recognize_attendance.py <subject_code> <subject_name> <period> <faculty_name> <session_id>
SUBJECT_CODE = sys.argv[1] if len(sys.argv) > 1 else ""
SUBJECT_NAME = sys.argv[2] if len(sys.argv) > 2 else ""
PERIOD = sys.argv[3] if len(sys.argv) > 3 else ""
FACULTY_NAME = sys.argv[4] if len(sys.argv) > 4 else ""
SESSION_ID = int(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5].isdigit() else None

print(f"\n{'='*60}")
print(f"  LECTURE DETAILS")
print(f"{'='*60}")
print(f"  Subject : {SUBJECT_NAME} ({SUBJECT_CODE})")
print(f"  Period  : {PERIOD}")
print(f"  Faculty : {FACULTY_NAME}")
print(f"  Session : #{SESSION_ID}")
print(f"{'='*60}\n")

# Load YOLO face model
try:
    model = YOLO('yolov8n-face.pt')
    print("✅ YOLOv8 Face Model loaded")
except Exception as e:
    print(f"❌ Failed to load YOLO model: {e}")
    sys.exit(1)

import database
import json

LOG_FILE = "system_logs.jsonl"
DATASET_DIR = "TrainingImage"

def write_log(message, log_type="info"):
    """Write log entry to file to be picked up by the UI toast system"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "type": log_type,
        "message": message
    }
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        pass

def mark_attendance(name):
    """Mark attendance with duplicate prevention via SQLite"""
    result = database.mark_attendance(
        name,
        subject_code=SUBJECT_CODE,
        subject_name=SUBJECT_NAME,
        period=PERIOD,
        faculty_name=FACULTY_NAME,
        session_id=SESSION_ID
    )
    
    if result == 'success':
        print(f"✅ Attendance marked for {name} | {SUBJECT_NAME} | {PERIOD}")
        write_log(f"Attendance recorded for {name} in {SUBJECT_NAME} ({PERIOD})", "success")
        return True
    elif result == 'duplicate':
        print(f"ℹ️  {name} already marked for {SUBJECT_NAME} ({PERIOD}) today")
        write_log(f"Attendance already recorded for {name} in {SUBJECT_NAME}", "warning")
        return False
    else:
        print(f"Failed to find {name} in Student Registry")
        return False

# ==========================================
# MULTI-FRAME RECOGNITION BUFFER
# ==========================================
MIN_CONFIRMATIONS = 3
RECOGNITION_BUFFER_SIZE = 10
recognition_buffer = defaultdict(list)
confirmed_students = set()
marked_students = set()

def update_recognition_buffer(name):
    current_time = time.time()
    recognition_buffer[name].append(current_time)
    recognition_buffer[name] = [ts for ts in recognition_buffer[name] if current_time - ts <= 3.0]
    if len(recognition_buffer[name]) > RECOGNITION_BUFFER_SIZE:
        recognition_buffer[name] = recognition_buffer[name][-RECOGNITION_BUFFER_SIZE:]

def get_confirmation_count(name):
    return len(recognition_buffer.get(name, []))

if SUBJECT_NAME:
    WINDOW_NAME = f"Attendance: {SUBJECT_NAME} | {PERIOD} | Faculty: {FACULTY_NAME}"
else:
    WINDOW_NAME = "Smart Attendance System (DeepFace + YOLOv8)"

try:
    window_prop = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE)
    if window_prop >= 1:
        cv2.destroyWindow(WINDOW_NAME)
except (cv2.error, Exception):
    pass

cv2.destroyAllWindows()
time.sleep(0.3)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Camera not accessible!")
    sys.exit(1)

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 900, 700)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
time.sleep(0.1)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 0)

ATTENDANCE_DURATION = 20  # seconds
start_time = time.time()

print("=" * 60)
print("[INFO] DEEPFACE + YOLOV8 ATTENDANCE SYSTEM STARTED")
print("=" * 60)

# Optional: Run a dummy DeepFace find to build representations if missing
print("Initializing DeepFace embeddings... This may take a moment.")
try:
    dummy_img = np.zeros((200, 200, 3), dtype=np.uint8)
    DeepFace.find(img_path=dummy_img, db_path=DATASET_DIR, model_name="ArcFace", enforce_detection=False, silent=True)
except Exception:
    pass
print("Ready!")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    elapsed_time = time.time() - start_time
    remaining_time = max(0, ATTENDANCE_DURATION - elapsed_time)
    
    if elapsed_time >= ATTENDANCE_DURATION:
        break

    # Run YOLOv8 Face Detection
    results = model(frame, verbose=False)
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Extract face crop
            h, w = frame.shape[:2]
            cx1, cy1 = max(0, x1), max(0, y1)
            cx2, cy2 = min(w, x2), min(h, y2)
            
            face_crop = frame[cy1:cy2, cx1:cx2]
            name = "Unknown"
            color = (0, 0, 255)
            confirmation_count = 0
            is_recognized = False

            if face_crop.size > 0:
                try:
                    # Recognize face using DeepFace
                    dfs = DeepFace.find(
                        img_path=face_crop, 
                        db_path=DATASET_DIR, 
                        model_name="ArcFace", 
                        enforce_detection=False,
                        silent=True
                    )
                    
                    if len(dfs) > 0 and len(dfs[0]) > 0:
                        df = dfs[0]
                        # Sort by distance (smaller is better match)
                        df = df.sort_values(by="distance")
                        best_match_path = df.iloc[0]['identity']
                        
                        # Extract folder name (student name) from path
                        matched_name = os.path.basename(os.path.dirname(best_match_path))
                        
                        update_recognition_buffer(matched_name)
                        confirmation_count = get_confirmation_count(matched_name)
                        
                        if confirmation_count >= MIN_CONFIRMATIONS:
                            name = matched_name
                            is_recognized = True
                            
                            if name not in marked_students:
                                if mark_attendance(name):
                                    marked_students.add(name)
                                    confirmed_students.add(name)
                            color = (0, 255, 0)
                        else:
                            name = f"{matched_name}?"
                            color = (0, 165, 255)
                except Exception as e:
                    pass

            text = f"{name}"
            if is_recognized and name in marked_students:
                text += " (OK)"
            elif confirmation_count > 0:
                text += f" ({confirmation_count}/{MIN_CONFIRMATIONS})"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1-text_height-10), (x1+text_width, y1), color, -1)
            cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Overlay logic
    frame_h, frame_w = frame.shape[:2]
    header_height = 70 if SUBJECT_NAME else 40
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame_w, header_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    if SUBJECT_NAME:
        cv2.putText(frame, f"{SUBJECT_NAME} ({SUBJECT_CODE}) | {PERIOD}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 50), 2)
        cv2.putText(frame, f"Faculty: {FACULTY_NAME}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 255), 1)
        cv2.putText(frame, f"Time: {remaining_time:.1f}s", (frame_w - 160, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(frame, f"Marked: {len(marked_students)}", (frame_w - 160, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    else:
        cv2.putText(frame, f"Time: {remaining_time:.1f}s / {ATTENDANCE_DURATION}s", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Marked: {len(marked_students)}", (frame_w - 200, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    if marked_students:
        student_list = sorted(list(marked_students))
        students_text = "Confirmed: " + ", ".join(student_list[:5])
        bar_y = frame_h - 35
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, bar_y - 5), (frame_w, frame_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay2, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, students_text, (10, bar_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    cv2.putText(frame, "GREEN=Confirmed | ORANGE=Pending | RED=Unknown", (10, frame_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(1) & 0xFF == 27:
        print("\n[WARNING] Attendance session ended early by user")
        break

print("=" * 60)
print(f"[OK] ATTENDANCE SESSION COMPLETE!")
print(f"[STATS] Total students marked: {len(marked_students)}")
print("=" * 60)

if SESSION_ID:
    try:
        database.end_lecture_session(SESSION_ID, total_present=len(marked_students))
        print(f"[INFO] Lecture session #{SESSION_ID} marked as completed.")
    except Exception as e:
        print(f"[WARNING] Failed to update lecture session: {e}")

cap.release()
try:
    cv2.destroyWindow(WINDOW_NAME)
except:
    pass
cv2.destroyAllWindows()
time.sleep(0.2)
