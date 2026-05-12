import cv2
import numpy as np
from datetime import datetime
import os
import re
import time
import sys
import glob
from collections import defaultdict
from ultralytics import YOLO
from deepface import DeepFace

# ==========================================
# PARSE COMMAND-LINE ARGUMENTS
# ==========================================
# Usage: python recognize_attendance.py <subject_code> <subject_name> <period> <faculty_name> <session_id> [rtsp_url]
SUBJECT_CODE = sys.argv[1] if len(sys.argv) > 1 else ""
SUBJECT_NAME = sys.argv[2] if len(sys.argv) > 2 else ""
PERIOD = sys.argv[3] if len(sys.argv) > 3 else ""
FACULTY_NAME = sys.argv[4] if len(sys.argv) > 4 else ""
SESSION_ID = int(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5].isdigit() else None
RTSP_URL = sys.argv[6] if len(sys.argv) > 6 else None  # Optional IP camera stream

print(f"\n{'='*60}")
print(f"  LECTURE DETAILS")
print(f"{'='*60}")
print(f"  Subject : {SUBJECT_NAME} ({SUBJECT_CODE})")
print(f"  Period  : {PERIOD}")
print(f"  Faculty : {FACULTY_NAME}")
print(f"  Session : #{SESSION_ID}")
print(f"{'='*60}\n")

import database
import json

LOG_FILE = "system_logs.jsonl"
DATASET_DIR = "TrainingImage"

# ==========================================
# RECOGNITION TUNING PARAMETERS
# ==========================================
DISTANCE_THRESHOLD = 0.40       # ArcFace cosine distance (STRICT: 0.40 prevents false matches)
                                 # Lower = stricter. 0.55 was too loose (wrong names given)
                                 # 0.40 = face must be 60% similar to stored image
MIN_FACE_SIZE = 80              # Minimum face width/height in pixels (larger = clearer face needed)
MIN_CONFIRMATIONS = 5           # Frames needed to confirm — prevents single-frame false match
RECOGNITION_BUFFER_SIZE = 10    # Max buffer entries per student
ATTENDANCE_DURATION = 30        # Seconds for attendance session

# ==========================================
# CLEAN STALE DEEPFACE CACHE
# ==========================================
def clean_deepface_cache():
    """Delete stale DeepFace representation files so fresh embeddings are built"""
    cache_patterns = [
        os.path.join(DATASET_DIR, "representations_*.pkl"),
        os.path.join(DATASET_DIR, "ds_model_*.pkl"),
    ]
    deleted = 0
    for pattern in cache_patterns:
        for f in glob.glob(pattern):
            try:
                os.remove(f)
                deleted += 1
                print(f"  🗑️  Deleted stale cache: {os.path.basename(f)}")
            except Exception:
                pass
    if deleted > 0:
        print(f"  ✅ Cleaned {deleted} stale cache file(s)")
    else:
        print(f"  ℹ️  No stale cache found")

print("[STEP 1/4] Cleaning stale DeepFace cache...")
clean_deepface_cache()

# ==========================================
# LOAD YOLO MODEL
# ==========================================
print("[STEP 2/4] Loading YOLOv8 Face Model...")
try:
    model = YOLO('yolov8n-face.pt')
    print("  ✅ YOLOv8 Face Model loaded")
except Exception as e:
    print(f"  ❌ Failed to load YOLO model: {e}")
    sys.exit(1)

# ==========================================
# OPEN CAMERA EARLY (warms up while models load)
# ==========================================
print("[STEP 3/4] Opening camera...")
# Use RTSP stream if provided, otherwise use local webcam
if RTSP_URL:
    print(f"  🎥 Connecting to IP camera: {RTSP_URL}")
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        print(f"  ⚠️  RTSP stream unreachable: {RTSP_URL}")
        print("  🔄 Falling back to local webcam (index 0)...")
        if sys.platform == 'win32':
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(0)
else:
    if sys.platform == 'win32':
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Camera not accessible!")
    sys.exit(1)

# Set camera properties for faster capture
if RTSP_URL:
    # IP Camera (WiFi) — reduce lag with minimal buffer
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)       # Only keep 1 frame in buffer (latest frame)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)             # Lower FPS = less WiFi bandwidth = less lag
    print("  📡 IP Camera mode: Anti-lag settings applied")
else:
    # Laptop webcam — normal settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
print("  ✅ Camera ready")

# ==========================================
# WINDOW SETUP & UI HELPERS
# ==========================================
if SUBJECT_NAME:
    WINDOW_NAME = f"Attendance: {SUBJECT_NAME} | {PERIOD} | Faculty: {FACULTY_NAME}"
else:
    WINDOW_NAME = "Smart Attendance System (DeepFace + YOLOv8)"

cv2.destroyAllWindows()
time.sleep(0.1)

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 900, 700)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)

# Professional UI Colors
COL_BG         = (20, 20, 25)
COL_PRIMARY    = (230, 160, 50)    # Gold-amber
COL_SUCCESS    = (80, 220, 100)
COL_ERROR      = (60, 60, 230)
COL_WHITE      = (255, 255, 255)
COL_GRAY       = (160, 160, 160)
COL_PANEL      = (40, 40, 45)

def draw_header(frame, text, subtext=""):
    """Draw professional header bar"""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 65), COL_PANEL, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.line(frame, (0, 65), (w, 65), COL_PRIMARY, 2)
    cv2.putText(frame, text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_PRIMARY, 2, cv2.LINE_AA)
    if subtext:
        cv2.putText(frame, subtext, (15, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_GRAY, 1, cv2.LINE_AA)

def draw_footer(frame, text):
    """Draw footer info bar"""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 35), (w, h), COL_PANEL, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.line(frame, (0, h - 35), (w, h - 35), COL_GRAY, 1)
    cv2.putText(frame, text, (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL_GRAY, 1, cv2.LINE_AA)

# Show Professional Loading Screen
loading_img = np.zeros((700, 900, 3), dtype=np.uint8)
loading_img[:] = COL_BG

draw_header(loading_img, "SMART ATTENDANCE SYSTEM", "System Initialization")
cv2.putText(loading_img, "Building Face Database...", (250, 330), cv2.FONT_HERSHEY_SIMPLEX, 1.0, COL_PRIMARY, 2, cv2.LINE_AA)
cv2.putText(loading_img, "This may take a moment. Please wait...", (270, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL_WHITE, 1, cv2.LINE_AA)
draw_footer(loading_img, "Initializing AI Models...")

cv2.imshow(WINDOW_NAME, loading_img)
cv2.waitKey(100)

# ==========================================
# DEEPFACE WARMUP — Build fresh embeddings
# ==========================================
print("[STEP 4/4] Building DeepFace embeddings (first run may be slow)...")
try:
    dummy_img = np.zeros((160, 160, 3), dtype=np.uint8)
    DeepFace.find(
        img_path=dummy_img, 
        db_path=DATASET_DIR, 
        model_name="ArcFace",
        enforce_detection=False, 
        silent=True
    )
    print("  ✅ DeepFace embeddings ready")
except Exception as e:
    print(f"  ⚠️  DeepFace warmup note: {e}")

cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 0)

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
    except Exception:
        pass

def folder_name_to_display_name(folder_name):
    """Convert folder name like 'Sagar Kumar_21104131014' to display name 'Sagar Kumar'"""
    # Strip trailing _<digits> (roll number)
    display = re.sub(r'_\d+$', '', folder_name).strip()
    return display if display else folder_name

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
    
    display_name = folder_name_to_display_name(name)
    
    if result == 'success':
        print(f"✅ Attendance marked for {display_name} | {SUBJECT_NAME} | {PERIOD}")
        write_log(f"Attendance recorded for {display_name} in {SUBJECT_NAME} ({PERIOD})", "success")
        return True
    elif result == 'duplicate':
        print(f"ℹ️  {display_name} already marked for {SUBJECT_NAME} ({PERIOD}) today")
        write_log(f"Attendance already recorded for {display_name} in {SUBJECT_NAME}", "warning")
        return False
    else:
        print(f"⚠️  Failed to find {name} (display: {display_name}) in Student Registry")
        write_log(f"Student not found in registry: {name}", "error")
        return False

# ==========================================
# MULTI-FRAME RECOGNITION BUFFER
# ==========================================
recognition_buffer = defaultdict(list)
confirmed_students = set()
marked_students = set()

def update_recognition_buffer(name):
    current_time = time.time()
    recognition_buffer[name].append(current_time)
    # Only keep entries from last 3 seconds
    recognition_buffer[name] = [ts for ts in recognition_buffer[name] if current_time - ts <= 3.0]
    if len(recognition_buffer[name]) > RECOGNITION_BUFFER_SIZE:
        recognition_buffer[name] = recognition_buffer[name][-RECOGNITION_BUFFER_SIZE:]

def get_confirmation_count(name):
    return len(recognition_buffer.get(name, []))

# ==========================================
# MAIN ATTENDANCE LOOP
# ==========================================
print("=" * 60)
print("[INFO] DEEPFACE + YOLOV8 ATTENDANCE SYSTEM STARTED")
print(f"[INFO] Distance threshold: {DISTANCE_THRESHOLD}")
print(f"[INFO] Min face size: {MIN_FACE_SIZE}px")
print(f"[INFO] Min confirmations: {MIN_CONFIRMATIONS} frames")
print(f"[INFO] Session duration: {ATTENDANCE_DURATION}s")
print("=" * 60)

start_time = time.time()

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
            
            # Calculate face dimensions
            face_w = x2 - x1
            face_h = y2 - y1
            
            # Skip faces that are too small for reliable recognition
            if face_w < MIN_FACE_SIZE or face_h < MIN_FACE_SIZE:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 2)
                cv2.putText(frame, "Too far", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
                continue
            
            # Extract face crop with bounds checking
            h, w = frame.shape[:2]
            cx1, cy1 = max(0, x1), max(0, y1)
            cx2, cy2 = min(w, x2), min(h, y2)
            
            face_crop = frame[cy1:cy2, cx1:cx2]
            name = "Unknown"
            display_name = "Unknown"
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
                        best_distance = df.iloc[0]['distance']
                        best_match_path = df.iloc[0]['identity']
                        
                        # CRITICAL: Only accept match if distance is below threshold
                        if best_distance < DISTANCE_THRESHOLD:
                            # Extract folder name (student name) from path
                            matched_folder = os.path.basename(os.path.dirname(best_match_path))
                            matched_display = folder_name_to_display_name(matched_folder)
                            
                            update_recognition_buffer(matched_folder)
                            confirmation_count = get_confirmation_count(matched_folder)
                            
                            if confirmation_count >= MIN_CONFIRMATIONS:
                                name = matched_folder
                                display_name = matched_display
                                is_recognized = True
                                
                                if name not in marked_students:
                                    if mark_attendance(name):
                                        marked_students.add(name)
                                        confirmed_students.add(name)
                                color = (0, 255, 0)
                            else:
                                display_name = f"{matched_display}?"
                                color = (0, 165, 255)
                        else:
                            # Distance too high — not a reliable match
                            display_name = "Unknown"
                            color = (0, 0, 255)
                except Exception as e:
                    pass

            # Build display text
            text = display_name
            if is_recognized and name in marked_students:
                text += " (OK)"
            elif confirmation_count > 0:
                text += f" ({confirmation_count}/{MIN_CONFIRMATIONS})"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1-text_height-10), (x1+text_width, y1), color, -1)
            cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # ==========================================
    # PROFESSIONAL UI OVERLAY
    # ==========================================
    fh, fw = frame.shape[:2]
    
    # 1. Header
    if SUBJECT_NAME:
        draw_header(frame, f"ATTENDANCE SCANNER", f"{SUBJECT_NAME} ({SUBJECT_CODE})  |  Period: {PERIOD}  |  Faculty: {FACULTY_NAME}")
    else:
        draw_header(frame, "SMART ATTENDANCE SCANNER", "General Attendance Mode")
    
    # 2. Timer Progress Bar (Top right under header)
    timer_w = 150
    timer_x = fw - timer_w - 20
    timer_y = 75
    progress = remaining_time / ATTENDANCE_DURATION
    color_timer = COL_PRIMARY if progress > 0.3 else COL_ERROR
    
    cv2.putText(frame, f"{remaining_time:.1f}s remaining", (timer_x, timer_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL_WHITE, 1, cv2.LINE_AA)
    
    # Custom progress bar for timer
    cv2.rectangle(frame, (timer_x, timer_y), (timer_x + timer_w, timer_y + 6), (60,60,60), -1)
    if progress > 0:
        cv2.rectangle(frame, (timer_x, timer_y), (timer_x + int(timer_w * progress), timer_y + 6), color_timer, -1)
    
    # 3. Status Panels (Bottom area)
    panel_y = fh - 80
    
    # Left Panel: Marked Students Count
    cv2.rectangle(frame, (15, panel_y), (160, fh - 40), COL_PANEL, -1)
    cv2.rectangle(frame, (15, panel_y), (160, fh - 40), COL_PRIMARY, 1)
    cv2.putText(frame, "MARKED", (25, panel_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL_GRAY, 1, cv2.LINE_AA)
    cv2.putText(frame, str(len(marked_students)), (25, panel_y + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COL_PRIMARY, 2, cv2.LINE_AA)
    
    # Center Panel: Recent Confirmations
    if marked_students:
        display_list = sorted([folder_name_to_display_name(s) for s in marked_students])
        recent = display_list[-3:] if len(display_list) > 3 else display_list
        recent.reverse() # Show newest first
        
        cv2.rectangle(frame, (175, panel_y), (fw - 15, fh - 40), COL_PANEL, -1)
        cv2.putText(frame, "RECENTLY MARKED:", (185, panel_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL_GRAY, 1, cv2.LINE_AA)
        
        # Draw tags for recent students
        tag_x = 185
        for s in recent:
            (tw, th), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (tag_x, panel_y + 22), (tag_x + tw + 20, fh - 45), (60, 100, 60), -1)
            cv2.putText(frame, s, (tag_x + 10, fh - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_WHITE, 1, cv2.LINE_AA)
            tag_x += tw + 30

    # 4. Footer
    draw_footer(frame, "GREEN=Confirmed  |  ORANGE=Pending  |  RED=Unknown  |  GRAY=Too Far  |  Press ESC to exit")

    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(1) & 0xFF == 27:
        print("\n[WARNING] Attendance session ended early by user")
        break

print("=" * 60)
print(f"[OK] ATTENDANCE SESSION COMPLETE!")
print(f"[STATS] Total students marked: {len(marked_students)}")
if marked_students:
    for s in sorted(marked_students):
        print(f"  ✅ {folder_name_to_display_name(s)}")
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
