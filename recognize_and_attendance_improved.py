import cv2
import numpy as np
import csv
from datetime import datetime
import os
import time
import sys
from collections import defaultdict

# ==========================================
# IMPROVED CONFIGURATION PARAMETERS
# ==========================================
CONFIDENCE_THRESHOLD_INITIAL = 60  # Initial filter (stricter than before)
CONFIDENCE_THRESHOLD_FINAL = 45    # Final average threshold for confirmation
MIN_CONFIRMATIONS = 5              # Minimum frames to confirm a person
RECOGNITION_BUFFER_SIZE = 10       # How many recent recognitions to track

# Check if model files exist
if not os.path.exists("trainer.yml"):
    print("❌ ERROR: Model not trained yet!")
    print("Please train the model first using 'Train / Update Model' button.")
    time.sleep(3)
    sys.exit(1)

if not os.path.exists("labels.npy"):
    print("❌ ERROR: Labels file not found!")
    print("Please train the model first using 'Train / Update Model' button.")
    time.sleep(3)
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
    time.sleep(3)
    sys.exit(1)

# Try to load labels with error handling
try:
    label_map = np.load("labels.npy", allow_pickle=True).item()
    print(f"✅ Labels loaded successfully: {len(label_map)} person(s) registered")
    print(f"📋 Registered students: {', '.join(label_map.values())}")
except Exception as e:
    print(f"❌ ERROR: Failed to load labels file!")
    print(f"Error: {str(e)}")
    print("Please retrain the model using 'Train / Update Model' button.")
    time.sleep(3)
    sys.exit(1)

face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

import database
import json

LOG_FILE = "system_logs.jsonl"

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
    result = database.mark_attendance(name)
    
    if result == 'success':
        print(f"✅ Attendance marked for {name}")
        write_log(f"Attendance recorded for {name}", "success")
        return True
    elif result == 'duplicate':
        print(f"ℹ️  {name} already marked today - skipping duplicate entry")
        write_log(f"Attendance already recorded today for {name}", "warning")
        return False
    else:
        print(f"Failed to find {name} in Student Registry")
        return False

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

# ==========================================
# MULTI-FRAME RECOGNITION BUFFER
# ==========================================
recognition_buffer = defaultdict(list)  # {name: [(confidence, timestamp), ...]}
confirmed_students = set()  # Students who have been confirmed and marked

def update_recognition_buffer(name, confidence):
    """Update the recognition buffer for multi-frame confirmation"""
    current_time = time.time()
    
    # Add new recognition
    recognition_buffer[name].append((confidence, current_time))
    
    # Keep only recent recognitions (last 2 seconds)
    recognition_buffer[name] = [
        (conf, ts) for conf, ts in recognition_buffer[name]
        if current_time - ts <= 2.0
    ]
    
    # Limit buffer size
    if len(recognition_buffer[name]) > RECOGNITION_BUFFER_SIZE:
        recognition_buffer[name] = recognition_buffer[name][-RECOGNITION_BUFFER_SIZE:]

def is_confirmed_recognition(name):
    """Check if a person has been confirmed through multi-frame detection"""
    if name not in recognition_buffer or name == "Unknown":
        return False, 0
    
    recognitions = recognition_buffer[name]
    
    # Need minimum confirmations
    if len(recognitions) < MIN_CONFIRMATIONS:
        return False, len(recognitions)
    
    # Calculate average confidence
    avg_confidence = sum(conf for conf, _ in recognitions) / len(recognitions)
    
    # Must meet strict average threshold
    if avg_confidence < CONFIDENCE_THRESHOLD_FINAL:
        return True, len(recognitions)
    
    return False, len(recognitions)

# Use consistent window name for all camera operations
WINDOW_NAME = "Smart Attendance System - IMPROVED"

# Destroy any existing windows first to ensure single window
try:
    window_prop = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE)
    if window_prop >= 1:
        cv2.destroyWindow(WINDOW_NAME)
except (cv2.error, Exception):
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
    time.sleep(3)
    sys.exit(1)

# Create and configure window
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 900, 700)

# Make sure window is visible
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
time.sleep(0.1)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 0)

# Set attendance duration to 15 seconds (more time for confirmation)
ATTENDANCE_DURATION = 15  # seconds
start_time = time.time()

# Track which students have been marked to avoid duplicates within session
marked_students = set()

print("=" * 60)
print("📸 IMPROVED ATTENDANCE SYSTEM STARTED")
print("=" * 60)
print(f"⏱️  Duration: {ATTENDANCE_DURATION} seconds")
print(f"🎯 Confidence threshold: {CONFIDENCE_THRESHOLD_FINAL} (average)")
print(f"🔄 Minimum confirmations: {MIN_CONFIRMATIONS} frames")
print("-" * 60)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    remaining_time = max(0, ATTENDANCE_DURATION - elapsed_time)
    
    # Break after duration
    if elapsed_time >= ATTENDANCE_DURATION:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces with stricter parameters to reduce false positives
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.2,
        minNeighbors=5,      # More strict for better quality detections
        minSize=(100, 100)   # Larger minimum size
    )
    
    # Filter out overlapping detections
    faces = filter_overlapping_faces(faces)
    
    # Track students detected in current frame
    current_frame_students = []
    
    # Process ALL detected faces in this frame
    for (x, y, w, h) in faces:
        # Crop and resize face to match training size
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (200, 200))
        
        # Predict with recognizer
        label, confidence = recognizer.predict(face_img)
        
        name = "Unknown"
        color = (0, 0, 255)  # Red for unknown
        is_recognized = False
        confirmation_count = 0
        
        # Initial confidence filter (stricter than before)
        if confidence < CONFIDENCE_THRESHOLD_INITIAL:
            potential_name = label_map.get(label, "Unknown")
            
            if potential_name != "Unknown":
                # Update recognition buffer
                update_recognition_buffer(potential_name, confidence)
                
                # Check if this person is confirmed
                is_confirmed, confirmation_count = is_confirmed_recognition(potential_name)
                
                if is_confirmed:
                    name = potential_name
                    is_recognized = True
                    current_frame_students.append(name)
                    
                    # Mark attendance if not already marked
                    if name not in marked_students and name not in confirmed_students:
                        if mark_attendance(name):
                            marked_students.add(name)
                            confirmed_students.add(name)
                            avg_conf = sum(c for c, _ in recognition_buffer[name]) / len(recognition_buffer[name])
                            print(f"✅ CONFIRMED & MARKED: {name} (Avg Confidence: {avg_conf:.1f}, Frames: {len(recognition_buffer[name])})")
                    
                    color = (0, 255, 0)  # Green for confirmed
                else:
                    # Pending confirmation
                    name = f"{potential_name}?"
                    color = (0, 165, 255)  # Orange for pending
            else:
                color = (0, 0, 255)  # Red for unknown
        else:
            # High confidence = poor match = unknown
            color = (0, 0, 255)  # Red for unknown

        # Draw rectangle and label
        text = f"{name}"
        if is_recognized and name in marked_students:
            text += " ✓"  # Checkmark for marked
        elif confirmation_count > 0:
            text += f" ({confirmation_count}/{MIN_CONFIRMATIONS})"  # Show progress
        
        # Add confidence for debugging
        text += f" [{confidence:.0f}]"
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        
        # Draw name with background for better visibility
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x, y-text_height-10), (x+text_width, y), color, -1)
        cv2.putText(
            frame, text, (x, y-5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
    
    # Display header information
    header_y = 30
    cv2.rectangle(frame, (0, 0), (900, 180), (0, 0, 0), -1)  # Black background for header
    
    # Timer
    timer_text = f"Time: {remaining_time:.1f}s / {ATTENDANCE_DURATION}s"
    cv2.putText(frame, timer_text, (10, header_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Marked count
    marked_count = len(marked_students)
    count_text = f"Marked: {marked_count}"
    cv2.putText(frame, count_text, (10, header_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display marked students
    if marked_students:
        y_offset = header_y + 70
        cv2.putText(frame, "Confirmed Students:", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        student_list = sorted(list(marked_students))
        for idx, student in enumerate(student_list[:3]):  # Show up to 3
            cv2.putText(frame, f"  {idx+1}. {student}", (10, y_offset + 25 * (idx + 1)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Show instructions
    cv2.putText(frame, "Legend: GREEN=Confirmed | ORANGE=Pending | RED=Unknown",
               (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit early
        print("\n⚠️  Attendance session ended early by user")
        break

print("=" * 60)
print(f"✅ ATTENDANCE SESSION COMPLETE!")
print("=" * 60)
print(f"📊 Total students marked: {len(marked_students)}")
if marked_students:
    print(f"👥 Students:")
    for student in sorted(marked_students):
        print(f"   ✓ {student}")
else:
    print("⚠️  No students were recognized during this session.")
print("=" * 60)

# Proper cleanup
cap.release()

try:
    cv2.destroyWindow(WINDOW_NAME)
except:
    pass

cv2.destroyAllWindows()

time.sleep(0.2)
