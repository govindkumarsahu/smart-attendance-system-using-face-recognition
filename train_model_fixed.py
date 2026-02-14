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

# ==========================================
# IMPROVED LBPH PARAMETERS (Memory Optimized)
# ==========================================
recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=2,        # Better texture features
    neighbors=12,    # Balanced accuracy (not too high to avoid memory issues)
    grid_x=8,        # Fine feature extraction
    grid_y=8
)

print("=" * 60)
print("OPTIMIZED MODEL TRAINING (Strict Recognition)")
print("=" * 60)
print("Enhanced LBPH parameters for better accuracy:")
print(f"  - Radius: 2 (more texture)")
print(f"  - Neighbors: 12 (better than default)")
print(f"  - Grid: 8x8 (fine features)")
print(f"  - Minimal augmentation (memory optimized)")
print("=" * 60)

faces = []
labels = []
label_map = {}
current_label = 0

print("\nStarting model training...")
print("-" * 60)

# Read dataset
for person_name in sorted(os.listdir(dataset_path)):
    person_path = os.path.join(dataset_path, person_name)

    if not os.path.isdir(person_path):
        continue

    label_map[current_label] = person_name
    print(f"\n📁 Processing: {person_name} (Label: {current_label})")

    image_count = 0
    skipped_count = 0

    for image_name in os.listdir(person_path):
        # Skip non-image files
        if not image_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue

        image_path = os.path.join(person_path, image_name)

        # Read image using OpenCV
        img = cv2.imread(image_path)
        
        if img is None:
            print(f"  ⚠️  Could not read: {image_name}")
            skipped_count += 1
            continue

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces with STRICTER parameters
        detected_faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.2,   # More sensitive
            minNeighbors=5,    # Standard
            minSize=(60, 60)   # Larger minimum
        )

        # If face detected, use cropped face
        if len(detected_faces) > 0:
            # Use the largest detected face
            x, y, w, h = detected_faces[0]
            for (fx, fy, fw, fh) in detected_faces:
                if fw * fh > w * h:
                    x, y, w, h = fx, fy, fw, fh
            
            # Crop face region with some padding
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(gray.shape[1] - x, w + 2*padding)
            h = min(gray.shape[0] - y, h + 2*padding)
            
            face_img = gray[y:y+h, x:x+w]
            
            # Resize to consistent size
            face_img = cv2.resize(face_img, (200, 200))
            
            # Apply histogram equalization for better contrast
            face_img = cv2.equalizeHist(face_img)
            
            # Add original
            faces.append(face_img)
            labels.append(current_label)
            image_count += 1
            
        else:
            # If no face detected, skip
            print(f"  ⚠️  No face detected: {image_name} - SKIPPED")
            skipped_count += 1

    print(f"  ✅ Processed: {image_count} images")
    
    if skipped_count > 0:
        print(f"  ⚠️  Skipped: {skipped_count} images")
    
    if image_count == 0:
        print(f"  ❌ ERROR: No valid images for {person_name}!")
    else:
        current_label += 1

print("=" * 60)
print(f"\n📊 TRAINING SUMMARY:")
print(f"   Persons registered: {len(label_map)}")
print(f"   Total training images: {len(faces)}")
print("=" * 60)

if len(faces) == 0:
    print("\n❌ ERROR: No faces found! Cannot train model.")
    sys.exit(1)

# Train model
print("\n🧠 Training model with enhanced parameters...")

try:
    recognizer.train(faces, np.array(labels))
    print("✅ Training completed successfully!")
except Exception as e:
    print(f"❌ ERROR during training: {str(e)}")
    sys.exit(1)

# Save trained model
try:
    recognizer.save("trainer.yml")
    print("✅ Model saved: trainer.yml")
except Exception as e:
    print(f"❌ ERROR saving model: {str(e)}")
    sys.exit(1)

# Save labels
try:
    np.save("labels.npy", label_map)
    print("✅ Labels saved: labels.npy")
except Exception as e:
    print(f"❌ ERROR saving labels: {str(e)}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ MODEL TRAINING COMPLETE")
print("=" * 60)
print(f"\n👥 Registered students:")
for label, name in sorted(label_map.items()):
    print(f"     {label}: {name}")

# Verify files
print("\n🔍 Verifying files...")
if os.path.exists("trainer.yml") and os.path.exists("labels.npy"):
    size = os.path.getsize("trainer.yml") / (1024 * 1024)
    print(f"✅ trainer.yml: {size:.2f} MB")
    print(f"✅ labels.npy: OK")
    print("\n" + "=" * 60)
    print("🎉 MODEL READY! Use improved attendance script.")
    print("=" * 60)
    print("\nℹ️  Important:")
    print("   • Enhanced LBPH parameters for better accuracy")
    print("   • Stricter face detection")
    print("   • Histogram equalization applied")
    print("   • Use with: python recognize_and_attendance_improved.py")
    print("\n⚠️  For best results:")
    print("   • Delete duplicate student folders (pravin/pravin kumar)")
    print("   • Retrain after cleanup")
    print("=" * 60)
else:
    print("\n❌ ERROR: Files not created!")
    sys.exit(1)
