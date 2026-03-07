# Running the Smart Attendance System

Follow these steps to launch and use the system for your demo.

## 1. Start the Web Dashboard
Open your terminal in the project directory and run the main application:
```powershell
python app.py
```
*The server will start at [http://127.0.0.1:8000](http://127.0.0.1:8000)*

## 2. Access and Login
1. Open your browser and go to `http://127.0.0.1:8000`.
2. Select the **Faculty** role.
3. Use the following credentials:
   - **Username:** `faculty`
   - **Password:** `1234`

## 3. Record Attendance (Face Recognition)
You have two ways to start the face recognition engine:
- **From Dashboard:** Click the **"Live Attendance"** link in the sidebar, then click **"Start Attendance"**.
- **Manual Command:** Run the recognition script directly in a new terminal:
  ```powershell
  python recognize_and_attendance_improved.py
  ```

## 4. View Real-time Updates
- Once the camera starts and recognizes a student, the **Today's Attendance Feed** on the Dashboard will update **automatically** every 5 seconds.
- You will see the student's **Photo**, **Name**, **Registration Number**, and **Department** in the live feed.

## Student Login (Demo)
Students can also log in to view their own attendance history:
- **Login with:** Their Registration Number (e.g., `12345678`).
- **Access:** They will only see their personal dashboard and attendance records.

> [!TIP]
> Make sure your camera is well-lit for the best recognition accuracy. If a student is not being recognized, ensure they are registered in the **"New Registration"** section first.
