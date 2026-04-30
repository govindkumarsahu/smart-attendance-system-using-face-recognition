# 🎓 Smart Attendance System using Face Recognition

An advanced, automated attendance management system powered by Face Recognition technology. This system replaces traditional manual roll calls with a fast, secure, and accurate facial biometric system. It features a robust Python/Flask backend for image processing and a modern, responsive React + TailwindCSS frontend for an excellent user experience.

---

## ✨ Key Features
- **Face Recognition-based Attendance**: Real-time face detection and recognition using OpenCV and Local Binary Patterns Histograms (LBPH).
- **Dual Portal System**: Separate dashboards and access levels for Faculty (Admins) and Students.
- **Faculty Management**: Complete CRUD operations for faculty members with employee ID tracking.
- **Manual Faculty Check-in/Check-out**: Dedicated module to track faculty working hours and attendance status (Present, Half-Day, On Leave).
- **Comprehensive Reports & Analytics**: View today's stats, detailed history, and download reports as `.csv` files.
- **Interactive Dashboard**: Data visualization with modern UI, stat cards, and real-time status updates.

---

## 🛠️ Tech Stack
- **Frontend**: React.js (v19), Vite, Tailwind CSS (v4), React Router DOM, Chart.js.
- **Backend**: Python, Flask, SQLite3.
- **Computer Vision**: OpenCV (cv2), Haar Cascades, LBPH Face Recognizer.

---

## 🚀 Step-by-Step Installation Guide

Follow these steps to clone and run the project on your local machine.

### Prerequisites
Make sure you have the following installed:
- **Python** (v3.8 or higher)
- **Node.js** (v18 or higher) & **npm**
- A working webcam for capturing faces.

### 📌 Step 1: Clone the Repository
Open your terminal and run this command to download the project:
```bash
git clone <your-repository-url>
cd smart-attendance-system-using-face-recognition
```

---

### 🐍 Step 2: Backend Setup (Python)
You need to install the required Python libraries for Face Recognition and the Flask API.

**1. Create & Activate Virtual Environment (Highly Recommended)**

**For Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**For Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**2. Install All Required Libraries**
Copy and paste the exact command below to install all dependencies at once:
```bash
pip install opencv-python opencv-contrib-python numpy Pillow psutil Flask
```
*(Alternatively, you can also run: `pip install -r requirements.txt`)*

**3. Run the Backend Server**
Once installed, start the Flask server:
```bash
python app.py
```
*(Leave this terminal running! The backend is now live at http://127.0.0.1:5000)*

---

### ⚛️ Step 3: Frontend Setup (React/Vite)
Open a **new terminal window** to set up the frontend UI.

**1. Go to the frontend folder:**
```bash
cd frontend
```

**2. Install Node.js Dependencies:**
Copy and paste this command to install React, TailwindCSS, and other UI packages:
```bash
npm install
```

**3. Start the Frontend Application:**
Run this command to start the React UI:
```bash
npm run dev
```
*(Your frontend is now live at http://localhost:5173 - open this in your browser!)*
   *The React frontend will start running at `http://localhost:5173`.*

---

## 📖 How to Use the System

1. **Access the App**: Open your browser and go to `http://localhost:5173`.
2. **Login as Faculty**: Use default credentials if you haven't created one (e.g., Username: `sharma`, Password: `1234`).
3. **Register Students**: Go to "Register Student", enter details, and the camera will open to capture face samples.
4. **Train Model**: The system automatically trains the LBPH model after capturing samples. You can also view model metrics in "Model Data".
5. **Start Attendance**: Go to "Start Attendance", select Subject/Period, and the camera will recognize faces and log them to the database and CSV.
6. **Faculty Management**: Go to "Faculty Management" to register new staff members and manage their details.
7. **Faculty Attendance**: Go to "Faculty Attendance" to manually Check-In or Check-Out staff members.

---

## 📂 Code Structure & File Explanations

### Root Directory (Backend & AI Models)
- **`app.py`**: The main Flask backend server. It handles routing, API endpoints, user authentication, and triggers Python subprocesses for camera scripts.
- **`database.py`**: Handles all SQLite3 database operations. Contains schemas for students, subjects, faculty, and attendance records, along with CRUD functions.
- **`capture_faces.py`**: Script launched to open the webcam, detect a face using Haar Cascades, and capture 50 grayscale image samples for a newly registered student.
- **`train_model.py`**: Reads all captured face images from the `TrainingImage` folder and trains the `LBPHFaceRecognizer` model. Saves the trained model as `TrainingImageLabel/trainner.yml`.
- **`recognize_and_attendance_improved.py`**: The core attendance script. Opens the webcam, detects faces, predicts their identity using the trained model, and logs their attendance in the database and a daily CSV file.
- **`haarcascade_frontalface_default.xml`**: Pre-trained OpenCV XML file used for detecting human faces in a video stream.
- **`requirements.txt`**: List of all Python dependencies required to run the backend.

### Folders (Backend Data)
- **`TrainingImage/`**: Stores captured raw face images of students during registration.
- **`TrainingImageLabel/`**: Stores the compiled/trained model file (`trainner.yml`).
- **`Attendance/`**: Stores the daily student attendance records in `.csv` format.

### Frontend Directory (`/frontend`)
Built with React and Vite. Key folders and files inside `/src`:
- **`App.jsx`**: The main entry point for routing. Handles protected routes for students and faculty.
- **`main.jsx`**: Bootstraps the React application into the DOM.
- **`services/api.js`**: Contains all Fetch/API calls that communicate with the Flask backend.
- **`components/`**: 
  - `Sidebar.jsx`: The main navigation menu.
  - `StatCard.jsx`: Reusable UI component for displaying dashboard statistics.
  - `ToastContainer.jsx`: Global notification system.
- **`pages/`**:
  - `Dashboard.jsx`: Faculty analytics dashboard.
  - `StudentRegistration.jsx`: Form to register new students and trigger the face capture process.
  - `StartAttendance.jsx`: Interface to select a subject/period and launch the recognition camera.
  - `FacultyRegistration.jsx`: CRUD interface for managing faculty members.
  - `FacultyAttendance.jsx`: Manual check-in/out system for faculty tracking.
  - `Reports.jsx`: View and export student attendance history.

---

*Developed with ❤️ as a Major Project.*