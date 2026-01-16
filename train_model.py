import cv2
import os
import numpy as np
from PIL import Image

dataset_path = "dataset"

# LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
label_map = {}
current_label = 0

# Read dataset
for person_name in sorted(os.listdir(dataset_path)):

    person_path = os.path.join(dataset_path, person_name)

    if not os.path.isdir(person_path):
        continue

    label_map[current_label] = person_name

    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)

        img = Image.open(image_path).convert("L")
        img_np = np.array(img, "uint8")

        faces.append(img_np)
        labels.append(current_label)

    current_label += 1

# Train model
recognizer.train(faces, np.array(labels))

# Save trained model
recognizer.save("trainer.yml")

# Save labels
np.save("labels.npy", label_map)

print("MODEL TRAINING COMPLETE")
