import cv2
import numpy as np
import time

# Load trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

# Load labels
label_map = np.load("labels.npy", allow_pickle=True).item()

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

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

# Create and configure window
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 800, 600)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces with stricter parameters to reduce false positives
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.3, 
        minNeighbors=5,
        minSize=(100, 100)  # Minimum face size to filter small false positives
    )
    
    # Filter out overlapping detections
    faces = filter_overlapping_faces(faces)
    
    # Track the best match for this frame
    best_match = None
    best_confidence = float('inf')
    best_face_box = None
    
    for (x, y, w, h) in faces:
        # Crop and resize face to match training size
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (200, 200))

        label, confidence = recognizer.predict(face_img)
        
        # Track the best match (lowest confidence = best match)
        if confidence < best_confidence:
            best_confidence = confidence
            best_match = (label, confidence, (x, y, w, h))
    
    # Display only the best match
    if best_match is not None:
        label, confidence, (x, y, w, h) = best_match
        
        name = label_map.get(label, "Unknown")

        # Lower confidence = better match
        if confidence < 70:
            text = f"{name}"
            color = (0, 255, 0)
        else:
            text = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(
            frame, text,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9, color, 2
        )
        
        # Show confidence in corner for debugging (optional)
        cv2.putText(
            frame,
            f"Confidence: {confidence:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(1) == 27:  # ESC
        break

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
