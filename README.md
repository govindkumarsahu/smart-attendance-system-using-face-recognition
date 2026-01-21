# Smart Attendance System Using Face Recognition

The **Smart Attendance System** is a Python-based application that automates student attendance using **face recognition**.  
The project is developed using **OpenCV and Flask** and supports both **terminal-based execution** and a **web-based interface** for ease of use.

This system is designed for **academic and college project purposes** and works on a local machine where camera access is available.

---

## 🔧 Technologies Used

- Python 3
- OpenCV
- NumPy
- Pandas
- Flask
- HTML

---

## 📁 Project Structure

Smart_Attendance_System/
│
├── app.py # Flask web application
├── dataset_capture.py # Student face registration (terminal)
├── train_model.py # Train / update face recognition model
├── recognize_and_attendance.py# Attendance recognition
│
├── dataset/ # Stored student face images
│ └── Student_Name/
│ ├── 0.jpg
│ ├── 1.jpg
│ └── ...
│
├── attendance.csv # Attendance records
├── trainer.yml # Trained model file (auto-generated)
├── labels.npy # Label-name mapping (auto-generated)
│
├── templates/
│ └── index.html # Web interface
│
└── README.md

yaml
Copy code

---

## ⚙️ Installation

Install the required Python libraries using:

```bash
pip install opencv-python numpy pandas flask pillow
🧑‍💻 Running the Project (Terminal Mode)
1️⃣ Student Registration
Capture student face images using the camera:

bash
Copy code
python dataset_capture.py
Camera opens on the local system

Multiple face images are captured

Images are stored in the dataset/ folder

2️⃣ Train / Update Model
Train the face recognition model using registered images:

bash
Copy code
python train_model.py
This generates:

trainer.yml

labels.npy

3️⃣ Take Attendance
Recognize faces and mark attendance:

bash
Copy code
python recognize_and_attendance.py
Camera opens

Face is recognized

Attendance is recorded in attendance.csv

🌐 Running the Project (Web Interface)
1️⃣ Start Flask Application
bash
Copy code
python app.py
You will see:

nginx
Copy code
Running on http://127.0.0.1:8000
2️⃣ Open in Browser
Open your browser and visit:

cpp
Copy code
http://127.0.0.1:8000
Web Interface Features
Register Student – Start face registration

Train / Update Model – Train model using dataset

Take Attendance – Recognize face and mark attendance

Reset Model – Remove trained model files

📌 Note:
The camera opens in a separate OpenCV window.
If the window is not visible, check the taskbar or use ALT + TAB.

📊 Attendance Output
Attendance records are saved in:

Copy code
attendance.csv
The file contains:

Student Name

Date

Time

It can be opened using Excel or any spreadsheet software.

🧠 Working Concept
Face registration is performed in a controlled environment

Attendance is marked automatically using face recognition

The system runs locally where camera access is available

This approach follows real-world biometric attendance systems.

⚠️ Important Notes
Camera access works only on the local machine

When shared using tools like ngrok, the web interface can be accessed remotely

Camera functionality remains local to the system