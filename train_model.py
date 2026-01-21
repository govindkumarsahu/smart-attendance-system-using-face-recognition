import cv2
import os
import numpy as np
from PIL import Image
import sys

# Fix encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

dataset_path = "dataset"

# Load face cascade for detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
label_map = {}
current_label = 0

print("Starting model training...")
print("-" * 50)

# Read dataset
for person_name in sorted(os.listdir(dataset_path)):
    person_path = os.path.join(dataset_path, person_name)

    if not os.path.isdir(person_path):
        continue

    label_map[current_label] = person_name
    print(f"\nProcessing: {person_name} (Label: {current_label})")

    image_count = 0
    skipped_count = 0

    for image_name in os.listdir(person_path):
        # Skip non-image files
        if not image_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue

        image_path = os.path.join(person_path, image_name)

        # Read image using OpenCV (better for face detection)
        img = cv2.imread(image_path)
        
        if img is None:
            print(f"  [WARNING] Could not read: {image_name}")
            skipped_count += 1
            continue

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        detected_faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.3, 
            minNeighbors=5,
            minSize=(50, 50)
        )

        # If face detected, use cropped face; otherwise use full image
        if len(detected_faces) > 0:
            # Use the largest detected face
            x, y, w, h = detected_faces[0]
            for (fx, fy, fw, fh) in detected_faces:
                if fw * fh > w * h:
                    x, y, w, h = fx, fy, fw, fh
            
            # Crop face region
            face_img = gray[y:y+h, x:x+w]
            
            # Resize to consistent size (recommended for LBPH)
            face_img = cv2.resize(face_img, (200, 200))
        else:
            # If no face detected, use the full image
            print(f"  [WARNING] No face detected in: {image_name}, using full image")
            face_img = cv2.resize(gray, (200, 200))

        faces.append(face_img)
        labels.append(current_label)
        image_count += 1

    print(f"  [OK] Processed {image_count} images for {person_name}")
    if skipped_count > 0:
        print(f"  [WARNING] Skipped {skipped_count} images")
    
    if image_count == 0:
        print(f"  [ERROR] No valid images found for {person_name}!")
    else:
        current_label += 1

print("-" * 50)
print(f"\nTotal persons registered: {len(label_map)}")
print(f"Total images processed: {len(faces)}")

if len(faces) == 0:
    print("\n[ERROR] No faces found in dataset! Cannot train model.")
    sys.exit(1)

# Train model
print("\nTraining model...")
recognizer.train(faces, np.array(labels))

# Save trained model
recognizer.save("trainer.yml")

# Save labels
np.save("labels.npy", label_map)

print("\n[SUCCESS] MODEL TRAINING COMPLETE")
print(f"[SUCCESS] Model saved as: trainer.yml")
print(f"[SUCCESS] Labels saved as: labels.npy")
print(f"\nRegistered persons:")
for label, name in label_map.items():
    print(f"  Label {label}: {name}")

# Verify files were created
if os.path.exists("trainer.yml") and os.path.exists("labels.npy"):
    print("\n[SUCCESS] Model files verified and ready to use!")
else:
    print("\n[ERROR] Model files were not created successfully!")
    sys.exit(1)
