import cv2
import numpy as np

# Load trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

# Load labels
label_map = np.load("labels.npy", allow_pickle=True).item()

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]

        label, confidence = recognizer.predict(face_img)

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

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
