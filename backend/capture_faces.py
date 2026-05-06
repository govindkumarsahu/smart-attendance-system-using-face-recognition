import cv2
import numpy as np
import os
import sys
import time
import glob
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

# -------------------------
# CLEAN STALE DEEPFACE CACHE
# -------------------------
cache_patterns = [
    os.path.join(dataset_path, "representations_*.pkl"),
    os.path.join(dataset_path, "ds_model_*.pkl"),
]
for pattern in cache_patterns:
    for f in glob.glob(pattern):
        try:
            os.remove(f)
            print(f"  Cleaned stale cache: {os.path.basename(f)}")
        except Exception:
            pass

# -------------------------
# LOAD YOLO MODEL
# -------------------------
print("[1/3] Loading YOLOv8 face model...")
try:
    model = YOLO('yolov8n-face.pt')
    print("  Model loaded")
except Exception as e:
    print(f"  Failed to load YOLO model: {e}")
    sys.exit(1)

# -------------------------
# CAMERA SETUP
# -------------------------
print("[2/3] Opening camera...")
if sys.platform == 'win32':
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
else:
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Camera not accessible")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
print("  Camera ready")

WINDOW_NAME = "Smart Attendance - Face Registration"
cv2.destroyAllWindows()
time.sleep(0.1)
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 900, 700)

# =========================================================
# PROFESSIONAL UI DRAWING HELPERS
# =========================================================

# Color palette (BGR)
COL_BG         = (20, 20, 25)
COL_PRIMARY    = (230, 160, 50)    # Gold-amber
COL_SUCCESS    = (80, 220, 100)
COL_ERROR      = (60, 60, 230)
COL_WHITE      = (255, 255, 255)
COL_GRAY       = (160, 160, 160)
COL_DARK_GRAY  = (80, 80, 80)
COL_PANEL      = (40, 40, 45)
COL_ACCENT     = (255, 120, 50)    # Orange-blue
COL_GUIDE_OK   = (80, 255, 80)
COL_GUIDE_WAIT = (50, 160, 230)

def draw_rounded_rect(img, pt1, pt2, color, radius=15, thickness=-1):
    """Draw a rounded rectangle"""
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, (x2-x1)//2, (y2-y1)//2)
    
    # Fill with regular rectangle (approximation for speed)
    overlay = img.copy()
    cv2.rectangle(overlay, (x1+r, y1), (x2-r, y2), color, thickness)
    cv2.rectangle(overlay, (x1, y1+r), (x2, y2-r), color, thickness)
    cv2.circle(overlay, (x1+r, y1+r), r, color, thickness)
    cv2.circle(overlay, (x2-r, y1+r), r, color, thickness)
    cv2.circle(overlay, (x1+r, y2-r), r, color, thickness)
    cv2.circle(overlay, (x2-r, y2-r), r, color, thickness)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)

def draw_face_guide(frame, cx, cy, size, color, thickness=2):
    """Draw a professional face guide oval with corner brackets"""
    axes = (size, int(size * 1.3))
    cv2.ellipse(frame, (cx, cy), axes, 0, 0, 360, color, thickness, cv2.LINE_AA)
    
    # Corner brackets for a scanner-like look
    bracket_len = 20
    offset_x = size + 15
    offset_y = int(size * 1.3) + 15
    
    corners = [
        (cx - offset_x, cy - offset_y),  # Top-left
        (cx + offset_x, cy - offset_y),  # Top-right
        (cx - offset_x, cy + offset_y),  # Bottom-left
        (cx + offset_x, cy + offset_y),  # Bottom-right
    ]
    
    # Top-left
    cv2.line(frame, corners[0], (corners[0][0] + bracket_len, corners[0][1]), color, 3, cv2.LINE_AA)
    cv2.line(frame, corners[0], (corners[0][0], corners[0][1] + bracket_len), color, 3, cv2.LINE_AA)
    # Top-right
    cv2.line(frame, corners[1], (corners[1][0] - bracket_len, corners[1][1]), color, 3, cv2.LINE_AA)
    cv2.line(frame, corners[1], (corners[1][0], corners[1][1] + bracket_len), color, 3, cv2.LINE_AA)
    # Bottom-left
    cv2.line(frame, corners[2], (corners[2][0] + bracket_len, corners[2][1]), color, 3, cv2.LINE_AA)
    cv2.line(frame, corners[2], (corners[2][0], corners[2][1] - bracket_len), color, 3, cv2.LINE_AA)
    # Bottom-right
    cv2.line(frame, corners[3], (corners[3][0] - bracket_len, corners[3][1]), color, 3, cv2.LINE_AA)
    cv2.line(frame, corners[3], (corners[3][0], corners[3][1] - bracket_len), color, 3, cv2.LINE_AA)

def draw_progress_bar(frame, x, y, width, height, progress, color_fill, color_bg=(60,60,60)):
    """Draw a sleek progress bar"""
    cv2.rectangle(frame, (x, y), (x + width, y + height), color_bg, -1)
    fill_w = int(width * min(1.0, progress))
    if fill_w > 0:
        cv2.rectangle(frame, (x, y), (x + fill_w, y + height), color_fill, -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (100,100,100), 1)

def draw_header(frame, text, subtext=""):
    """Draw professional header bar"""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 65), COL_PANEL, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    # Accent line at bottom of header
    cv2.line(frame, (0, 65), (w, 65), COL_PRIMARY, 2)
    cv2.putText(frame, text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_PRIMARY, 2, cv2.LINE_AA)
    if subtext:
        cv2.putText(frame, subtext, (15, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_GRAY, 1, cv2.LINE_AA)

def draw_footer(frame, text, color=COL_GRAY):
    """Draw footer info bar"""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 35), (w, h), COL_PANEL, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.line(frame, (0, h - 35), (w, h - 35), COL_DARK_GRAY, 1)
    cv2.putText(frame, text, (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

def draw_status_badge(frame, x, y, text, color):
    """Draw a small colored status badge"""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    pad = 8
    cv2.rectangle(frame, (x, y - th - pad), (x + tw + pad*2, y + pad), color, -1)
    cv2.putText(frame, text, (x + pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_WHITE, 1, cv2.LINE_AA)

# =========================================================
# GUIDED POSE CAPTURE SEQUENCE
# =========================================================

POSES = [
    {"name": "Look Straight",       "icon": "[ O ]",  "duration": 3.0, "captures": 5,  "instruction": "Look directly at the camera"},
    {"name": "Turn Slightly Left",   "icon": "[ < ]",  "duration": 2.5, "captures": 4,  "instruction": "Slowly turn your head to the LEFT"},
    {"name": "Turn Slightly Right",  "icon": "[ > ]",  "duration": 2.5, "captures": 4,  "instruction": "Slowly turn your head to the RIGHT"},
    {"name": "Tilt Up Slightly",     "icon": "[ ^ ]",  "duration": 2.0, "captures": 3,  "instruction": "Tilt your chin UP slightly"},
    {"name": "Tilt Down Slightly",   "icon": "[ v ]",  "duration": 2.0, "captures": 3,  "instruction": "Tilt your chin DOWN slightly"},
    {"name": "Smile",                "icon": ":)",      "duration": 2.0, "captures": 3,  "instruction": "Give a natural SMILE"},
]

FACE_PADDING = 0.15
TARGET_SIZE = 160
count = 0
total_poses = len(POSES)

print("[3/3] Starting guided face registration...")

for pose_idx, pose in enumerate(POSES):
    pose_start = time.time()
    pose_captures = 0
    capture_interval = pose["duration"] / pose["captures"]
    last_capture_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip for mirror effect — feels more natural
        frame = cv2.flip(frame, 1)
        
        elapsed = time.time() - pose_start
        remaining = max(0, pose["duration"] - elapsed)
        
        if elapsed >= pose["duration"]:
            break
        
        # Run YOLO face detection
        results = model(frame, verbose=False)
        face_detected = False
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face_detected = True
                
                # Draw face bounding box with glow effect
                guide_color = COL_GUIDE_OK if face_detected else COL_GUIDE_WAIT
                cv2.rectangle(frame, (x1-2, y1-2), (x2+2, y2+2), guide_color, 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x1, y1), (x2, y2), COL_SUCCESS, 2, cv2.LINE_AA)
                
                # Capture at intervals
                current_time = time.time()
                if pose_captures < pose["captures"] and current_time - last_capture_time >= capture_interval:
                    h, w = frame.shape[:2]
                    face_w = x2 - x1
                    face_h = y2 - y1
                    pad_x = int(face_w * FACE_PADDING)
                    pad_y = int(face_h * FACE_PADDING)
                    
                    px1 = max(0, x1 - pad_x)
                    py1 = max(0, y1 - pad_y)
                    px2 = min(w, x2 + pad_x)
                    py2 = min(h, y2 + pad_y)
                    
                    face_crop = frame[py1:py2, px1:px2]
                    if face_crop.size > 0:
                        face_resized = cv2.resize(face_crop, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LANCZOS4)
                        img_path = os.path.join(student_path, f"{count}.jpg")
                        cv2.imwrite(img_path, face_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        count += 1
                        pose_captures += 1
                        last_capture_time = current_time
        
        # ---- DRAW PROFESSIONAL UI ----
        fh, fw = frame.shape[:2]
        
        # Header
        draw_header(
            frame,
            f"FACE REGISTRATION  |  {student_name}",
            f"Step {pose_idx + 1} of {total_poses}  |  {pose['name']}"
        )
        
        # Face guide circle in center
        cx, cy = fw // 2, fh // 2 + 15
        guide_col = COL_GUIDE_OK if face_detected else COL_GUIDE_WAIT
        draw_face_guide(frame, cx, cy, 90, guide_col, 2)
        
        # Instruction panel at bottom-center
        instr_text = pose["instruction"]
        (tw, th), _ = cv2.getTextSize(instr_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        panel_w = tw + 40
        panel_x = (fw - panel_w) // 2
        panel_y = fh - 120
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + 45), COL_PANEL, -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        cv2.putText(frame, instr_text, (panel_x + 20, panel_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL_WHITE, 2, cv2.LINE_AA)
        
        # Pose icon (large centered text above instruction)
        icon_text = pose["icon"]
        (iw, ih), _ = cv2.getTextSize(icon_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
        cv2.putText(frame, icon_text, ((fw - iw)//2, panel_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.5, COL_PRIMARY, 3, cv2.LINE_AA)
        
        # Overall progress bar
        overall_progress = (pose_idx + elapsed / pose["duration"]) / total_poses
        draw_progress_bar(frame, 15, 75, fw - 30, 8, overall_progress, COL_PRIMARY)
        
        # Pose countdown ring — small timer
        timer_x = fw - 60
        timer_y = 90
        angle = int(360 * (1 - remaining / pose["duration"]))
        cv2.ellipse(frame, (timer_x, timer_y), (22, 22), -90, 0, angle, COL_ACCENT, 3, cv2.LINE_AA)
        cv2.ellipse(frame, (timer_x, timer_y), (22, 22), -90, angle, 360, COL_DARK_GRAY, 2, cv2.LINE_AA)
        cv2.putText(frame, f"{remaining:.0f}s", (timer_x - 10, timer_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL_WHITE, 1, cv2.LINE_AA)
        
        # Captures counter
        status_text = f"Captured: {pose_captures}/{pose['captures']}"
        status_color = COL_SUCCESS if pose_captures >= pose['captures'] else COL_ACCENT
        draw_status_badge(frame, 15, fh - 55, status_text, status_color)
        
        # Total images badge
        total_text = f"Total: {count}"
        draw_status_badge(frame, fw - 120, fh - 55, total_text, COL_PRIMARY)
        
        # Face detection status
        if not face_detected:
            no_face_text = "No face detected - Position yourself in the frame"
            (nw, nh), _ = cv2.getTextSize(no_face_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.putText(frame, no_face_text, ((fw-nw)//2, cy + 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_ERROR, 1, cv2.LINE_AA)
        
        # Footer
        draw_footer(frame, "Smart Attendance System  |  AI-Powered Face Registration  |  Press ESC to cancel")
        
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("[WARNING] Registration cancelled by user")
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)
    
    # Brief transition between poses
    if pose_idx < total_poses - 1:
        next_pose = POSES[pose_idx + 1]
        trans_start = time.time()
        while time.time() - trans_start < 1.0:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            fh, fw = frame.shape[:2]
            
            # Dark overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (fw, fh), COL_BG, -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            # Transition text
            cv2.putText(frame, "GET READY", ((fw - 200)//2, fh//2 - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, COL_PRIMARY, 2, cv2.LINE_AA)
            cv2.putText(frame, f"Next: {next_pose['name']}", ((fw - 300)//2, fh//2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_WHITE, 2, cv2.LINE_AA)
            cv2.putText(frame, next_pose['instruction'], ((fw - 400)//2, fh//2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_GRAY, 1, cv2.LINE_AA)
            
            # Step progress dots
            dot_y = fh//2 + 100
            dot_start_x = (fw - (total_poses * 25)) // 2
            for i in range(total_poses):
                dot_x = dot_start_x + i * 25
                if i <= pose_idx:
                    cv2.circle(frame, (dot_x, dot_y), 6, COL_SUCCESS, -1, cv2.LINE_AA)
                elif i == pose_idx + 1:
                    cv2.circle(frame, (dot_x, dot_y), 6, COL_PRIMARY, -1, cv2.LINE_AA)
                else:
                    cv2.circle(frame, (dot_x, dot_y), 6, COL_DARK_GRAY, -1, cv2.LINE_AA)
            
            cv2.imshow(WINDOW_NAME, frame)
            cv2.waitKey(1)

# =========================================================
# COMPLETION SCREEN
# =========================================================
completion_start = time.time()
while time.time() - completion_start < 2.5:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    fh, fw = frame.shape[:2]
    
    # Dark overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (fw, fh), COL_BG, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Success panel
    panel_w, panel_h = 450, 200
    px = (fw - panel_w) // 2
    py = (fh - panel_h) // 2
    
    # Panel background
    cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h), COL_PANEL, -1)
    cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h), COL_SUCCESS, 2)
    
    # Checkmark
    check_cx = fw // 2
    check_cy = py + 50
    cv2.circle(frame, (check_cx, check_cy), 25, COL_SUCCESS, 3, cv2.LINE_AA)
    cv2.line(frame, (check_cx - 12, check_cy), (check_cx - 3, check_cy + 10), COL_SUCCESS, 3, cv2.LINE_AA)
    cv2.line(frame, (check_cx - 3, check_cy + 10), (check_cx + 15, check_cy - 10), COL_SUCCESS, 3, cv2.LINE_AA)
    
    cv2.putText(frame, "Registration Complete!", ((fw - 330)//2, py + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COL_SUCCESS, 2, cv2.LINE_AA)
    cv2.putText(frame, f"{count} images captured for {student_name}", ((fw - 380)//2, py + 145), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_WHITE, 1, cv2.LINE_AA)
    
    # Completed dots
    dot_y = py + 175
    dot_start_x = (fw - (total_poses * 25)) // 2
    for i in range(total_poses):
        cv2.circle(frame, (dot_start_x + i * 25, dot_y), 6, COL_SUCCESS, -1, cv2.LINE_AA)
    
    cv2.imshow(WINDOW_NAME, frame)
    cv2.waitKey(1)

cap.release()
try:
    cv2.destroyWindow(WINDOW_NAME)
except:
    pass
cv2.destroyAllWindows()
time.sleep(0.2)

# Clean cache again after capture
for pattern in cache_patterns:
    for f in glob.glob(pattern):
        try:
            os.remove(f)
        except Exception:
            pass

print(f"[OK] Registration completed for {student_name}")
print(f"[INFO] Total images captured: {count}")
print(f"[INFO] Poses completed: {total_poses}")
