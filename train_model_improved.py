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
# IMPROVED LBPH PARAMETERS
# ==========================================
# Better parameters for more accurate recognition
recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=2,        # Increased from default 1 - captures more texture
    neighbors=16,    # Increased from default 8 - more neighbors for better accuracy
    grid_x=8,        # Grid size for better feature extraction
    grid_y=8         # Grid size for better feature extraction
)

print("=" * 60)
print("IMPROVED MODEL TRAINING")
print("=" * 60)
print("Using enhanced LBPH parameters:")
print(f"  - Radius: 2 (more texture features)")
print(f"  - Neighbors: 16 (better accuracy)")
print(f"  - Grid: 8x8 (finer feature extraction)")
print("=" * 60)

faces = []
labels = []
label_map = {}
current_label = 0

def augment_image(img):
    """Create variations of images for better training"""
    augmented = [img]
    
    # Brightness variations (subtle)
    bright = cv2.convertScaleAbs(img, alpha=1.15, beta=10)
    dark = cv2.convertScaleAbs(img, alpha=0.85, beta=-10)
    augmented.extend([bright, dark])
    
    # Histogram equalization for better contrast
    equalized = cv2.equalizeHist(img)
    augmented.append(equalized)
    
    # Slight rotations (-3° and +3°)
    rows, cols = img.shape
    for angle in [-3, 3]:
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        rotated = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
        augmented.append(rotated)
    
    return augmented

print("\nStarting model training with image augmentation...")
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
    augmented_count = 0

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

        # Detect faces in the image with stricter parameters
        detected_faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.2,   # More sensitive
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
            
            # Resize to consistent size
            face_img = cv2.resize(face_img, (200, 200))
            
            # Original image
            faces.append(face_img)
            labels.append(current_label)
            image_count += 1
            
            # Create augmented versions for better training
            augmented_images = augment_image(face_img)
            
            # Add augmented images (skip first one as it's the original)
            for aug_img in augmented_images[1:]:
                faces.append(aug_img)
                labels.append(current_label)
                augmented_count += 1
            
        else:
            # If no face detected, skip this image with warning
            print(f"  ⚠️  No face detected in: {image_name} - SKIPPED")
            skipped_count += 1

    total_images = image_count + augmented_count
    print(f"  ✅ Original images: {image_count}")
    print(f"  🔄 Augmented images: {augmented_count}")
    print(f"  📊 Total images for training: {total_images}")
    
    if skipped_count > 0:
        print(f"  ⚠️  Skipped: {skipped_count} images")
    
    if image_count == 0:
        print(f"  ❌ ERROR: No valid images found for {person_name}!")
    else:
        current_label += 1

print("=" * 60)
print(f"\n📊 TRAINING SUMMARY:")
print(f"   Total persons registered: {len(label_map)}")
print(f"   Total images for training: {len(faces)}")
print("=" * 60)

if len(faces) == 0:
    print("\n❌ ERROR: No faces found in dataset! Cannot train model.")
    print("Please register students first using the registration feature.")
    sys.exit(1)

# Train model
print("\n🧠 Training model with improved parameters...")
print("This may take longer due to enhanced parameters and augmentation...")

try:
    recognizer.train(faces, np.array(labels))
    print("✅ Training completed successfully!")
except Exception as e:
    print(f"❌ ERROR during training: {str(e)}")
    sys.exit(1)

# Save trained model
try:
    recognizer.save("trainer.yml")
    print("✅ Model saved as: trainer.yml")
except Exception as e:
    print(f"❌ ERROR saving model: {str(e)}")
    sys.exit(1)

# Save labels
try:
    np.save("labels.npy", label_map)
    print("✅ Labels saved as: labels.npy")
except Exception as e:
    print(f"❌ ERROR saving labels: {str(e)}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ MODEL TRAINING COMPLETE WITH IMPROVEMENTS")
print("=" * 60)
print(f"\n👥 Registered persons:")
for label, name in sorted(label_map.items()):
    print(f"     Label {label}: {name}")

# Verify files were created
print("\n🔍 Verifying model files...")
if os.path.exists("trainer.yml") and os.path.exists("labels.npy"):
    trainer_size = os.path.getsize("trainer.yml") / (1024 * 1024)  # MB
    print(f"✅ trainer.yml: {trainer_size:.2f} MB")
    print(f"✅ labels.npy: OK")
    print("\n" + "=" * 60)
    print("🎉 MODEL IS READY TO USE!")
    print("=" * 60)
    print("\nℹ️  Improvements applied:")
    print("   • Enhanced LBPH parameters for better accuracy")
    print("   • Image augmentation (brightness, rotation, equalization)")
    print("   • Stricter face detection")
    print("   • More training samples per person")
    print("\n💡 Next: Use the improved attendance recognition script")
    print("=" * 60)
else:
    print("\n❌ ERROR: Model files were not created successfully!")
    sys.exit(1)
