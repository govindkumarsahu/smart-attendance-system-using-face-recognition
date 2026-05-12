from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response, session
from functools import wraps
import subprocess
import sys
import os
import cv2
import time
import json
import csv
import numpy as np
import shutil
import hashlib
import jwt
from datetime import datetime
import database

# Load .env for email credentials
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
except ImportError:
    pass  # dotenv not installed — credentials must be set as system env vars

from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.secret_key = "attendance_secret_key"

# Initialize database
database.init_db()

# File paths
LOG_FILE = "system.log"
DATASET_DIR = "TrainingImage"
TRAINER_FILE = "TrainingImage/representations_arcface.pkl"

# ============================================================
# UTILITY FUNCTIONS & DECORATORS
# ============================================================

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("logged_in"):
            flash("🔒 Please log in to access this page.", "error")
            return redirect(url_for("login", next=request.url))
        return f(*args, **kwargs)
    return decorated_function

def write_log(message, log_type="info"):
    """Write log entry to file and console"""
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
        print(f"Error writing to log: {e}")
    print(f"[{timestamp}] {message}")

def read_logs(max_lines=100):
    """Read recent log entries from file"""
    logs = []
    try:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines[-max_lines:]:
                    try:
                        log_entry = json.loads(line.strip())
                        logs.append(log_entry)
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"Error reading logs: {e}")
    return logs

def init_log():
    """Initialize log file"""
    write_log("Smart Attendance System started", "success")

# ============================================================
# DATA HELPER FUNCTIONS
# ============================================================

def get_registered_students():
    """Get list of registered student folders from dataset/ enriched with SQLite metadata"""
    import re
    students = database.get_all_students()
    
    # Auto-sync: add any student folders from dataset/ that are missing from the database
    db_names_lower = {s['name'].lower() for s in students}
    if os.path.isdir(DATASET_DIR):
        for folder_name in os.listdir(DATASET_DIR):
            folder_path = os.path.join(DATASET_DIR, folder_name)
            if os.path.isdir(folder_path):
                # Try matching folder name directly and after stripping roll number
                base_name = re.sub(r'_\d+$', '', folder_name).strip()
                if (folder_name.lower() not in db_names_lower and 
                    base_name.lower() not in db_names_lower):
                    # This student exists in dataset but not in DB – auto-register with base name
                    result = database.add_student(base_name, "", "", "")
                    if result is not None:
                        write_log(f"Auto-synced student '{base_name}' from dataset folder '{folder_name}' to database.", "info")
        # Re-fetch after sync
        students = database.get_all_students()

    # Add image count dynamically and verify profile picture existence
    for student in students:
        student['images'] = 0
        student['has_profile_pic'] = False
        
        # Try multiple folder name patterns: exact name, name_rollnumber
        possible_folders = [student['name']]
        if student.get('roll_number'):
            possible_folders.append(f"{student['name']}_{student['roll_number']}")
        
        # Also search for any folder that starts with the student name
        if os.path.isdir(DATASET_DIR):
            for folder_name in os.listdir(DATASET_DIR):
                folder_path = os.path.join(DATASET_DIR, folder_name)
                if os.path.isdir(folder_path):
                    base = re.sub(r'_\d+$', '', folder_name).strip()
                    if base.lower() == student['name'].lower() and folder_name not in possible_folders:
                        possible_folders.append(folder_name)
        
        for folder_candidate in possible_folders:
            folder = os.path.join(DATASET_DIR, folder_candidate)
            if os.path.isdir(folder):
                images = [f for f in os.listdir(folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
                student['images'] = len(images)
                if '1.jpg' in images:
                    student['has_profile_pic'] = True
                elif len(images) > 0:
                    student['has_profile_pic'] = True
                break  # Found the folder, stop looking
    return students

def get_attendance_records():
    """Read all attendance records from SQLite"""
    records = database.get_attendance_records()
    results = []
    for r in records:
        results.append({
            "name": r["name"],
            "date": r["date"],
            "time": r["time"],
            "roll_number": r["roll_number"],
            "department": r["department"],
            "academic_year": r["academic_year"],
            "subject_code": r.get("subject_code", ""),
            "subject_name": r.get("subject_name", ""),
            "period": r.get("period", ""),
            "faculty_name": r.get("faculty_name", "")
        })
    return results

def get_today_records():
    """Get attendance records for today only with full student details"""
    records = database.get_today_records()
    return records

def get_model_info():
    """Get model status information"""
    info = {
        "exists": os.path.exists(TRAINER_FILE),
        "status": "Not Trained",
        "status_color": "red",
        "last_trained": "Never",
        "total_faces": 0,
        "labels": {},
        "pending_count": 0
    }
    if info["exists"]:
        info["status"] = "Ready"
        info["status_color"] = "green"
        try:
            mtime = os.path.getmtime(TRAINER_FILE)
            info["last_trained"] = datetime.fromtimestamp(mtime).strftime("%b %d, %Y %I:%M %p")
        except:
            pass

    # Registered students
    registered = get_registered_students()
    registered_names = set(s["name"] for s in registered)
    
    # In DeepFace, if there's a folder in TrainingImage with images, they are ready
    # We will just consider all registered students with images as "trained"
    trained_names = set(s["name"] for s in registered if s["images"] > 0)
    info["total_faces"] = len(trained_names)
    info["labels"] = {i: name for i, name in enumerate(trained_names)}
    info["pending_count"] = len(registered_names - trained_names)
    info["pending_students"] = list(registered_names - trained_names)

    return info

def get_dashboard_stats():
    """Get all stats needed for dashboard"""
    students = get_registered_students()
    today_records = get_today_records()
    model = get_model_info()
    all_records = get_attendance_records()

    # Unique students who attended today
    present_today = len(today_records)
    total_registered = len(students)
    absent_count = max(0, total_registered - present_today)

    # Model version from trainer.yml modification count or simple version
    model_version = f"v{model['total_faces']}.{len(students)}" if model["exists"] else "N/A"

    return {
        "registered_students": total_registered,
        "present_today": present_today,
        "absent_late": absent_count,
        "model_version": model_version,
        "model_active": model["exists"],
        "recent_records": today_records[:10],  # Last 10 today
        "all_today": today_records,
        "total_records": len(all_records)
    }

@app.route("/api/attendance/today")
@login_required
def api_today_attendance():
    """API endpoint for real-time dashboard updates"""
    if session.get("role") != "faculty":
        return jsonify({"success": False, "message": "Unauthorized"}), 403
    stats = get_dashboard_stats()
    return jsonify(stats)

@app.route("/student_image/<name>")
@login_required
def student_image(name):
    """Serve a student's photo from the dataset directory"""
    # Security: prevent path traversal
    name = os.path.basename(name)
    folder = os.path.join(DATASET_DIR, name)
    if not os.path.exists(folder):
        return abort(404)
    
    # Get the first image in the folder
    images = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        return abort(404)
    
    return send_from_directory(folder, images[0])

def get_recent_activity_logs():
    """Get recent activity from system logs for model page"""
    logs = read_logs(20)
    activity = []
    for log in reversed(logs):
        if any(kw in log.get("message", "") for kw in ["Training", "trained", "Enrolled", "Registration", "Model", "reset"]):
            activity.append({
                "time": log.get("timestamp", ""),
                "message": log.get("message", ""),
                "type": log.get("type", "info")
            })
            if len(activity) >= 5:
                break
    return activity

# ============================================================
# PROCESS MANAGEMENT
# ============================================================

def kill_camera_processes():
    """Kill any running camera-related Python processes"""
    try:
        import psutil
        camera_scripts = ["capture_faces.py", "recognize_attendance.py"]
        killed_count = 0
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                proc_name = proc.info.get('name', '')
                if proc_name and 'python' in proc_name.lower():
                    cmdline = proc.info.get('cmdline', [])
                    if cmdline:
                        cmdline_str = ' '.join(str(c) for c in cmdline).lower()
                        for script in camera_scripts:
                            if script.lower() in cmdline_str:
                                proc.terminate()
                                killed_count += 1
                                try:
                                    proc.wait(timeout=1)
                                except psutil.TimeoutExpired:
                                    proc.kill()
                                break
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        if killed_count > 0:
            time.sleep(0.5)
    except ImportError:
        if sys.platform == 'win32':
            try:
                for script in ["capture_faces.py", "recognize_attendance.py"]:
                    subprocess.run(
                        ["taskkill", "/F", "/FI", f"WINDOWTITLE eq *{script}*", "/T"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
                time.sleep(0.5)
            except Exception:
                pass
        time.sleep(0.5)

# ============================================================
# API ROUTES
# ============================================================

@app.route("/api/logs")
def api_logs():
    logs = read_logs(50)
    return jsonify(logs)

@app.route("/api/stats")
def api_stats():
    """API endpoint for live dashboard stats"""
    stats = get_dashboard_stats()
    return jsonify(stats)

@app.route("/api/session")
def api_session():
    """API endpoint to get current user session state"""
    if "logged_in" in session and session["logged_in"]:
        data = {
            "logged_in": True,
            "username": session.get("username"),
            "role": session.get("role"),
            "roll_number": session.get("roll_number"),
        }
        # Include faculty-specific details
        if session.get("role") == "faculty":
            data["faculty_id"] = session.get("faculty_id")
            data["designation"] = session.get("designation", "")
            data["department"] = session.get("department", "")
        return jsonify(data)
    return jsonify({"logged_in": False})

@app.route("/api/students")
@login_required
def api_students():
    """API endpoint to get all registered students"""
    if session.get("role") != "faculty":
        return jsonify({"success": False, "message": "Unauthorized"}), 403
    students = get_registered_students()
    return jsonify(students)

@app.route("/api/model-info")
@login_required
def api_model_info():
    """API endpoint to get model status and training data"""
    model = get_model_info()
    activity = get_recent_activity_logs()
    students = get_registered_students()
    return jsonify({
        "model_info": model,
        "activity": activity,
        "students": students
    })

# ============================================================
# SUBJECT & LECTURE SESSION APIs
# ============================================================

@app.route("/api/subjects")
@login_required
def api_subjects():
    """API endpoint to get all subjects"""
    subjects = database.get_all_subjects()
    return jsonify(subjects)

@app.route("/api/subjects", methods=["POST"])
@login_required
def api_add_subject():
    """API endpoint to add a new subject"""
    if session.get("role") != "faculty":
        return jsonify({"success": False, "message": "Unauthorized"}), 403
    data = request.get_json()
    code = data.get("code", "").strip()
    name = data.get("name", "").strip()
    department = data.get("department", "").strip()
    semester = data.get("semester", "").strip()
    if not code or not name:
        return jsonify({"success": False, "message": "Subject code and name are required."}), 400
    result = database.add_subject(code, name, department, semester)
    if result is None:
        return jsonify({"success": False, "message": f"Subject with code '{code}' already exists."}), 400
    write_log(f"New subject added: {code} - {name}", "success")
    return jsonify({"success": True, "message": f"Subject '{name}' added successfully."})

@app.route("/api/subjects/<int:subject_id>", methods=["DELETE"])
@login_required
def api_delete_subject(subject_id):
    """API endpoint to delete a subject"""
    if session.get("role") != "faculty":
        return jsonify({"success": False, "message": "Unauthorized"}), 403
    database.delete_subject(subject_id)
    write_log(f"Subject ID {subject_id} deleted.", "warning")
    return jsonify({"success": True, "message": "Subject deleted."})

@app.route("/api/lecture-sessions")
@login_required
def api_lecture_sessions():
    """API endpoint to get lecture sessions"""
    sessions = database.get_today_sessions()
    return jsonify(sessions)

# ============================================================
# FACULTY MANAGEMENT & ATTENDANCE APIs
# ============================================================

@app.route("/api/faculty")
@login_required
def api_faculty():
    """API endpoint to get all faculty members"""
    if session.get("role") != "faculty":
        return jsonify({"success": False, "message": "Unauthorized"}), 403
    faculty = database.get_all_faculty()
    return jsonify(faculty)

@app.route("/api/faculty/register", methods=["POST"])
@login_required
def api_register_faculty():
    """Register a new faculty member"""
    if session.get("role") != "faculty":
        return jsonify({"success": False, "message": "Unauthorized"}), 403

    data = request.get_json()
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()
    full_name = data.get("full_name", "").strip()
    employee_id = data.get("employee_id", "").strip()
    department = data.get("department", "").strip()
    designation = data.get("designation", "Assistant Professor").strip()
    email = data.get("email", "").strip()
    phone = data.get("phone", "").strip()

    if not username or not password or not full_name:
        return jsonify({"success": False, "message": "Username, password, and full name are required."}), 400

    result = database.add_faculty(username, password, full_name, employee_id, department, designation, email, phone)
    if result is None:
        return jsonify({"success": False, "message": f"Faculty with username '{username}' or employee ID '{employee_id}' already exists."}), 400

    write_log(f"New faculty registered: {full_name} (EMP: {employee_id})", "success")
    return jsonify({"success": True, "message": f"Faculty '{full_name}' registered successfully."})

@app.route("/api/faculty/<int:faculty_id>", methods=["PUT"])
@login_required
def api_update_faculty(faculty_id):
    """Update faculty details"""
    if session.get("role") != "faculty":
        return jsonify({"success": False, "message": "Unauthorized"}), 403

    data = request.get_json()
    full_name = data.get("full_name", "").strip()
    employee_id = data.get("employee_id", "").strip()
    department = data.get("department", "").strip()
    designation = data.get("designation", "").strip()
    email = data.get("email", "").strip()
    phone = data.get("phone", "").strip()

    if not full_name:
        return jsonify({"success": False, "message": "Full name is required."}), 400

    success = database.update_faculty(faculty_id, full_name, employee_id, department, designation, email, phone)
    if success:
        write_log(f"Faculty ID {faculty_id} ({full_name}) profile updated.", "info")
        return jsonify({"success": True, "message": f"Faculty '{full_name}' updated successfully."})
    else:
        return jsonify({"success": False, "message": "Update failed. Employee ID might already be taken."}), 400

@app.route("/api/faculty/<int:faculty_id>", methods=["DELETE"])
@login_required
def api_delete_faculty(faculty_id):
    """Delete a faculty member"""
    if session.get("role") != "faculty":
        return jsonify({"success": False, "message": "Unauthorized"}), 403

    faculty = database.get_faculty_by_id(faculty_id)
    if not faculty:
        return jsonify({"success": False, "message": "Faculty not found."}), 404

    database.delete_faculty(faculty_id)
    write_log(f"Faculty '{faculty['full_name']}' deleted.", "warning")
    return jsonify({"success": True, "message": f"Faculty '{faculty['full_name']}' deleted successfully."})

@app.route("/api/faculty-attendance/today")
@login_required
def api_faculty_attendance_today():
    """Get today's faculty attendance summary"""
    if session.get("role") != "faculty":
        return jsonify({"success": False, "message": "Unauthorized"}), 403
    summary = database.get_faculty_today_summary()
    return jsonify(summary)

@app.route("/api/faculty-attendance/history")
@login_required
def api_faculty_attendance_history():
    """Get faculty attendance history with optional filters"""
    if session.get("role") != "faculty":
        return jsonify({"success": False, "message": "Unauthorized"}), 403
    date_from = request.args.get("date_from", "")
    date_to = request.args.get("date_to", "")
    fid = request.args.get("faculty_id", "")
    records = database.get_faculty_attendance(
        date_from=date_from or None,
        date_to=date_to or None,
        faculty_id=int(fid) if fid else None
    )
    return jsonify(records)

@app.route("/api/faculty-attendance/mark", methods=["POST"])
@login_required
def api_mark_faculty_attendance():
    """Mark faculty check-in"""
    if session.get("role") != "faculty":
        return jsonify({"success": False, "message": "Unauthorized"}), 403

    data = request.get_json()
    faculty_id = data.get("faculty_id")
    status = data.get("status", "Present").strip()
    remarks = data.get("remarks", "").strip()
    marked_by = session.get("username", "System")

    if not faculty_id:
        return jsonify({"success": False, "message": "Faculty ID is required."}), 400

    result = database.mark_faculty_attendance(int(faculty_id), status, remarks, marked_by)
    if result == 'success':
        faculty = database.get_faculty_by_id(int(faculty_id))
        fname = faculty['full_name'] if faculty else f"ID {faculty_id}"
        write_log(f"Faculty check-in: {fname} ({status})", "success")
        return jsonify({"success": True, "message": f"Check-in marked for {fname}."})
    elif result == 'duplicate':
        return jsonify({"success": False, "message": "Attendance already marked for today."}), 400
    else:
        return jsonify({"success": False, "message": "Faculty not found."}), 404

@app.route("/api/faculty-attendance/checkout", methods=["POST"])
@login_required
def api_checkout_faculty():
    """Mark faculty check-out"""
    if session.get("role") != "faculty":
        return jsonify({"success": False, "message": "Unauthorized"}), 403

    data = request.get_json()
    faculty_id = data.get("faculty_id")

    if not faculty_id:
        return jsonify({"success": False, "message": "Faculty ID is required."}), 400

    result = database.checkout_faculty(int(faculty_id))
    if result == 'success':
        faculty = database.get_faculty_by_id(int(faculty_id))
        fname = faculty['full_name'] if faculty else f"ID {faculty_id}"
        write_log(f"Faculty check-out: {fname}", "info")
        return jsonify({"success": True, "message": f"Check-out recorded for {fname}."})
    elif result == 'no_checkin':
        return jsonify({"success": False, "message": "No check-in found for today. Please check-in first."}), 400
    elif result == 'already_checkout':
        return jsonify({"success": False, "message": "Already checked out for today."}), 400
    else:
        return jsonify({"success": False, "message": "Error processing check-out."}), 500

@app.route("/api/faculty-attendance/export-csv")
@login_required
def api_export_faculty_csv():
    """Export faculty attendance records as CSV"""
    if session.get("role") != "faculty":
        return jsonify({"success": False, "message": "Unauthorized"}), 403

    date_from = request.args.get("date_from", "")
    date_to = request.args.get("date_to", "")

    records = database.get_faculty_attendance(
        date_from=date_from or None,
        date_to=date_to or None
    )

    write_log(f"Exporting {len(records)} faculty attendance records to CSV.", "info")

    import io
    if not records:
        return jsonify({"success": False, "message": "No faculty attendance data to export."}), 400

    output = io.StringIO()
    output.write('\ufeff')  # UTF-8 BOM
    writer = csv.writer(output)
    writer.writerow(["Faculty Name", "Employee ID", "Department", "Designation", "Date", "Check-In", "Check-Out", "Work Hours", "Status", "Remarks"])

    for r in records:
        writer.writerow([
            r.get("full_name", ""),
            r.get("employee_id", "N/A"),
            r.get("department", "N/A"),
            r.get("designation", ""),
            r.get("date", ""),
            r.get("check_in", "—"),
            r.get("check_out", "—"),
            f"{r.get('work_hours', 0):.2f}" if r.get('work_hours') else "—",
            r.get("status", ""),
            r.get("remarks", "")
        ])

    csv_data = output.getvalue()
    output.close()

    filename = f"faculty_attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.route("/api/attendance/start", methods=["POST"])
@login_required
def api_start_attendance():
    """Start attendance camera with subject, period, and faculty context"""
    if session.get("role") != "faculty":
        return jsonify({"success": False, "message": "Unauthorized"}), 403

    data = request.get_json()
    subject_code = data.get("subject_code", "").strip()
    subject_name = data.get("subject_name", "").strip()
    period = data.get("period", "").strip()
    faculty_name = session.get("username", "Faculty")

    if not subject_code or not period:
        return jsonify({"success": False, "message": "Please select a subject and period."}), 400


    # Create a lecture session in the database
    session_id = database.create_lecture_session(subject_code, subject_name, period, faculty_name)
    if not session_id:
        return jsonify({"success": False, "message": "Failed to create lecture session."}), 500

    kill_camera_processes()
    time.sleep(0.3)

    write_log(f"Attendance started: {subject_name} ({subject_code}) | {period} | Faculty: {faculty_name}", "info")

    try:
        cmd_args = [
            sys.executable, "recognize_attendance.py",
            subject_code, subject_name, period, faculty_name, str(session_id)
        ]
        if sys.platform == 'win32':
            process = subprocess.Popen(cmd_args, creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:
            process = subprocess.Popen(cmd_args)
        write_log("Camera window opened for attendance.", "info")
    except Exception as e:
        write_log(f"ERROR: Failed to start attendance: {str(e)}", "error")
        return jsonify({"success": False, "message": f"Error starting camera: {str(e)}"}), 500

    return jsonify({
        "success": True,
        "message": f"Attendance started for {subject_name} - {period}.",
        "session_id": session_id
    })

@app.route("/api/register", methods=["POST"])
@login_required
def api_register():
    if session.get("role") != "faculty":
        return jsonify({"success": False, "message": "Unauthorized"}), 403
    
    data = request.get_json()
    name = data.get("student_name", "").strip()
    roll_number = data.get("roll_number", "").strip()
    department = data.get("department", "").strip()
    academic_year = data.get("academic_year", "").strip()
    
    if not name:
        return jsonify({"success": False, "message": "Please enter a student name."}), 400

    # Attempt to insert into SQLite
    result = database.add_student(name, roll_number, department, academic_year)
    if result is None: # Name was not unique
        write_log(f"Registration failed: Student {name} already exists", "error")
        return jsonify({"success": False, "message": f"A student with the name {name} already exists!"}), 400

    # Build folder name: "Name_RollNumber" if roll number provided, else just "Name"
    folder_name = f"{name}_{roll_number}" if roll_number else name

    # Clean stale DeepFace cache so embeddings are rebuilt with new student
    import glob
    cache_patterns = [
        os.path.join(DATASET_DIR, "representations_*.pkl"),
        os.path.join(DATASET_DIR, "ds_model_*.pkl"),
    ]
    for pattern in cache_patterns:
        for f in glob.glob(pattern):
            try:
                os.remove(f)
            except Exception:
                pass

    kill_camera_processes()
    write_log(f"Registration started for student: {name} (folder: {folder_name})", "info")

    process = subprocess.Popen(
        [sys.executable, "capture_faces.py", folder_name],
        creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
    )

    write_log("Camera window should open now.", "info")
    return jsonify({"success": True, "message": f"Registration started for {name}. Check camera window."})

# ============================================================
# PAGE ROUTES
# ============================================================

@app.route("/")
def landing():
    return jsonify({"status": "Smart Attendance API is running", "version": "2.0", "message": "Backend is active. Please use the React Frontend at localhost:5173."})

@app.route("/old")
def old_home():
    return redirect(url_for("landing"))

# ============================================================
# AUTHENTICATION
# ============================================================



@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        
        # 1. Faculty Login — authenticate from faculty table in database
        faculty = database.authenticate_faculty(username, password)
        if faculty:
            session["logged_in"] = True
            session["username"] = faculty["full_name"]
            session["faculty_id"] = faculty["id"]
            session["faculty_username"] = faculty["username"]
            session["department"] = faculty.get("department", "")
            session["designation"] = faculty.get("designation", "")
            session["role"] = "faculty"
            write_log(f"Faculty '{faculty['full_name']}' ({faculty['designation']}) logged in", "success")
            flash(f"👋 Welcome, {faculty['full_name']}!")
            return redirect(url_for("dashboard"))
            
        # 2. Student Login (Registration Number)
        conn = database.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM students WHERE roll_number = ?", (username,))
        student = cursor.fetchone()
        conn.close()
        
        if student:
            session["logged_in"] = True
            session["username"] = student["name"]
            session["roll_number"] = student["roll_number"]
            session["role"] = "student"
            session["student_id"] = student["id"]
            
            write_log(f"Student '{student['name']}' logged in via Roll Number", "success")
            flash(f"👋 Welcome, {student['name']}!")
            return redirect(url_for("student_dashboard"))
        
        # 3. Handle Failure
        write_log(f"Failed login attempt for identification '{username}'", "warning")
        flash("❌ Invalid credentials. Use your Faculty username & password.", "error")
            
    return render_template("login.html")

@app.route("/student/dashboard")
@login_required
def student_dashboard():
    if session.get("role") != "student":
        return redirect(url_for("dashboard"))
        
    student_id = session.get("student_id")
    # Get student record
    conn = database.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students WHERE id = ?", (student_id,))
    student = cursor.fetchone()
    
    # Get attendance history for this student
    cursor.execute("""
        SELECT date, time, status 
        FROM attendance 
        WHERE student_id = ? 
        ORDER BY date DESC, time DESC
    """, (student_id,))
    records = cursor.fetchall()
    conn.close()
    
    return render_template("student_dashboard.html", student=student, records=records, active='dashboard')

@app.route("/logout")
def logout():
    username = session.get("username", "Unknown user")
    session.clear()
    write_log(f"User '{username}' logged out", "info")
    flash("👋 Successfully logged out.")
    return redirect(url_for("landing"))

@app.route("/dashboard")
@login_required
def dashboard():
    if session.get("role") == "student":
        return redirect(url_for("student_dashboard"))
    write_log("Dashboard accessed", "info")
    stats = get_dashboard_stats()
    return render_template("dashboard.html", stats=stats)

@app.route("/register-page")
@login_required
def register_page():
    if session.get("role") != "faculty":
        flash("🚫 Access denied. Faculty only.", "error")
        return redirect(url_for("student_dashboard"))
    return render_template("register.html", active='register_page')

@app.route("/students")
@login_required
def students_page():
    if session.get("role") != "faculty":
        flash("🚫 Access denied. Faculty only.", "error")
        return redirect(url_for("student_dashboard"))
    students = get_registered_students()
    return render_template("students.html", students=students, active='students_page')

@app.route("/attendance-page")
@login_required
def attendance_page():
    if session.get("role") != "faculty":
        flash("🚫 Access denied. Faculty only.", "error")
        return redirect(url_for("student_dashboard"))
    today_records = get_today_records()
    return render_template("attendance.html", today_records=today_records, total_present=len(today_records))

@app.route("/reports")
@login_required
def reports():
    if session.get("role") != "faculty":
        flash("🚫 Access denied. Faculty only.", "error")
        return redirect(url_for("student_dashboard"))
    # Get filter params
    search_q = request.args.get("q", "").strip()
    status_filter = request.args.get("status", "all").lower()
    date_from = request.args.get("date_from", "")
    date_to = request.args.get("date_to", "")
    page = int(request.args.get("page", 1))
    per_page = 10

    # Get records from SQLite (filtered directly at DB level if applicable)
    filtered = database.get_attendance_records(date_from, date_to, search_q)
    
    # We still need all records for some stats calculations or we can fetch stats directly
    all_records = database.get_attendance_records()

    # Stats for today
    today = datetime.now().strftime("%Y-%m-%d")
    today_all = [r for r in all_records if r["date"] == today]
    today_unique_names = set(r["name"] for r in today_all)
    total_registered = len(get_registered_students())

    stats = {
        "present_today": len(today_unique_names),
        "absent_today": max(0, total_registered - len(today_unique_names)),
        "late_arrivals": 0,
        "avg_attendance": "N/A"
    }

    # Calculate average attendance percentage
    if total_registered > 0:
        # Get unique dates
        all_dates = set(r["date"] for r in all_records)
        if all_dates:
            total_attend_per_day = []
            for d in all_dates:
                day_names = set(r["name"] for r in all_records if r["date"] == d)
                total_attend_per_day.append(len(day_names))
            avg = sum(total_attend_per_day) / len(total_attend_per_day)
            stats["avg_attendance"] = f"{(avg / total_registered * 100):.1f}%" if total_registered > 0 else "N/A"

    # Pagination
    total_records = len(filtered)
    total_pages = max(1, (total_records + per_page - 1) // per_page)
    page = min(page, total_pages)
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    page_records = filtered[start_idx:end_idx]

    return render_template("reports.html",
        records=page_records,
        stats=stats,
        page=page,
        total_pages=total_pages,
        total_records=total_records,
        start_idx=start_idx + 1,
        end_idx=min(end_idx, total_records),
        search_q=search_q,
        status_filter=status_filter,
        date_from=date_from,
        date_to=date_to
    )

@app.route("/export-csv")
@login_required
def export_csv():
    if session.get("role") != "faculty":
        flash("🚫 Access denied. Faculty only.", "error")
        return redirect(url_for("student_dashboard"))
    """Export filtered attendance records as a downloadable CSV file"""
    search_q = request.args.get("q", "").strip()
    date_from = request.args.get("date_from", "")
    date_to = request.args.get("date_to", "")

    all_records = get_attendance_records()
    all_records = list(reversed(all_records))

    filtered = []
    for r in all_records:
        if search_q and search_q.lower() not in r["name"].lower():
            continue
        if date_from and r["date"] < date_from:
            continue
        if date_to and r["date"] > date_to:
            continue
        filtered.append(r)

    write_log(f"Exporting {len(filtered)} attendance records to CSV.", "info")
    
    import io
    import csv
    if not filtered:
        flash("❌ No attendance data available to export.")
        return redirect(url_for("reports"))

    output = io.StringIO()
    output.write('\ufeff')  # UTF-8 BOM for proper Excel compatibility
    writer = csv.writer(output)
    writer.writerow(["Student Name", "Roll Number", "Department", "Date", "Time", "Status"])
    
    for r in filtered:
        writer.writerow([
            r.get("name", "Unknown"),
            (r.get("roll_number") or "N/A"),
            (r.get("department") or "N/A").upper(),
            r.get("date", ""),
            r.get("time", ""),
            "Present"
        ])

    csv_data = output.getvalue()
    output.close()

    filename = f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    flash("✅ Report downloaded successfully!")
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.route("/model")
@login_required
def model_page():
    model = get_model_info()
    students = get_registered_students()
    activity = get_recent_activity_logs()
    return render_template("model.html", model_info=model, students=students, activity=activity, active='model')

# ============================================================
# ACTION ROUTES
# ============================================================

@app.route("/register", methods=["POST"])
@login_required
def register():
    if session.get("role") != "faculty":
        return jsonify({"success": False, "message": "Unauthorized"}), 403
    name = request.form.get("student_name", "").strip()
    roll_number = request.form.get("roll_number", "").strip()
    department = request.form.get("department", "").strip()
    academic_year = request.form.get("academic_year", "").strip()
    
    if not name:
        flash("❌ Please enter a student name.")
        return redirect(url_for("register_page"))

    # Attempt to insert into SQLite
    result = database.add_student(name, roll_number, department, academic_year)
    if result is None: # Name was not unique
        write_log(f"Registration failed: Student {name} already exists", "error")
        flash(f"❌ A student with the name {name} already exists in the registry!")
        return redirect(url_for("register_page"))

    kill_camera_processes()
    write_log(f"Registration started for student: {name}", "info")
    flash(f"📸 Registration started for {name}. Check camera window.")

    process = subprocess.Popen(
        [sys.executable, "capture_faces.py", name],
        creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
    )

    write_log("Camera window should open now.", "info")
    return redirect(url_for("register_page"))

# ----------- CRITICAL NEW ENDPOINTS -------------
@app.route("/delete_student/<int:id>", methods=["POST"])
@login_required
def delete_student(id):
    if session.get("role") != "faculty":
        return jsonify({"success": False, "message": "Unauthorized"}), 403
    # Determine the student name from the database before deletion
    students = database.get_all_students()
    student = next((s for s in students if s['id'] == id), None)
    
    if not student:
        flash("❌ Error: Student not found in the database.")
        write_log(f"Failed to delete student ID {id}: Not found", "error")
        return redirect(url_for("students_page"))
    
    name = student['name']
    
    # Remove from SQLite
    database.delete_student(id)
    write_log(f"Student '{name}' and associated attendance logs purged from DB.", "success")
    
    # Remove image dataset manually
    folder = os.path.join(DATASET_DIR, name)
    if os.path.exists(folder):
        try:
            shutil.rmtree(folder)
            write_log(f"Dataset images for '{name}' deleted.", "info")
        except Exception as e:
            write_log(f"Error purging dataset images for '{name}': {e}", "warning")
            
    # Invalidate trainer.yml to enforce retraining on next attendance session
    if os.path.exists(TRAINER_FILE):
        try:
            os.remove(TRAINER_FILE)
            write_log("trainer.yml explicitly invalidated due to student deletion.", "warning")
        except:
            pass
        
    flash(f"✅ Successfully deleted {name} and their dataset.")
    return redirect(url_for("students_page"))

@app.route("/edit_student/<int:id>", methods=["POST"])
@login_required
def edit_student(id):
    if session.get("role") != "faculty":
        return jsonify({"success": False, "message": "Unauthorized"}), 403
    name = request.form.get("student_name", "").strip()
    roll_number = request.form.get("roll_number", "").strip()
    department = request.form.get("department", "").strip()
    academic_year = request.form.get("academic_year", "").strip()
    
    if not name:
        flash("❌ Student name cannot be empty.")
        return redirect(url_for("students_page"))
        
    success = database.update_student(id, name, roll_number, department, academic_year)
    if success:
        flash(f"📝 Successfully updated details for {name}.")
        write_log(f"Student ID {id} ({name}) profile updated.", "info")
    else:
        flash(f"❌ Could not update student. Name '{name}' might already be taken.")
        write_log(f"Failed to update Student ID {id}: Integrity error", "error")
        
    return redirect(url_for("students_page"))

@app.route("/train")
@login_required
def train():
    if session.get("role") != "faculty":
        flash("🚫 Access denied. Faculty only.", "error")
        return redirect(url_for("student_dashboard"))
    write_log("Model training invoked - DeepFace auto-manages this", "info")
    flash("⚙️ DeepFace ArcFace automatically manages model representations. No explicit training needed!")
    return redirect(url_for("model_page"))

@app.route("/attendance")
@login_required
def attendance():
    if session.get("role") != "faculty":
        flash("🚫 Access denied. Faculty only.", "error")
        return redirect(url_for("student_dashboard"))
    # DeepFace will automatically generate representations if needed.

    kill_camera_processes()
    time.sleep(0.3)

    write_log("Attendance taking started", "info")
    flash("📸 Attendance started. Please look at the camera.")

    try:
        if sys.platform == 'win32':
            process = subprocess.Popen(
                [sys.executable, "recognize_attendance.py"],
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
        else:
            process = subprocess.Popen(
                [sys.executable, "recognize_attendance.py"]
            )
        write_log("Camera window opened for attendance.", "info")
    except Exception as e:
        write_log(f"❌ ERROR: Failed to start attendance: {str(e)}", "error")
        flash(f"❌ Error starting attendance: {str(e)}")

    return redirect(url_for("attendance_page"))

@app.route("/reset")
@login_required
def reset_model():
    if session.get("role") != "faculty":
        flash("🚫 Access denied. Faculty only.", "error")
        return redirect(url_for("student_dashboard"))
    write_log("Model reset requested", "warning")
    kill_camera_processes()
    time.sleep(0.5)

    try:
        if os.path.exists(TRAINER_FILE):
            os.remove(TRAINER_FILE)
            write_log("trainer.yml deleted", "info")
    except Exception as e:
        write_log(f"Error deleting trainer.yml: {str(e)}", "error")

    try:
        if os.path.exists(LABELS_FILE):
            os.remove(LABELS_FILE)
            write_log("labels.npy deleted", "info")
    except Exception as e:
        write_log(f"Error deleting labels.npy: {str(e)}", "error")

    write_log("Model reset completed.", "success")
    flash("🔄 Model reset successfully. Please train the model before taking attendance.")
    return redirect(url_for("model_page"))

# ============================================================
# ADMIN PANEL API ROUTES
# ============================================================

JWT_SECRET = "smartattend_admin_secret_2024"

@app.route('/api/admin/login', methods=['POST'])
def admin_login():
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        conn = database.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM admins WHERE username = ? AND password = ?", 
                       (username, hashlib.sha256(password.encode()).hexdigest()))
        admin = cursor.fetchone()
        conn.close()
        
        if admin:
            token = jwt.encode({
                'user': username,
                'exp': datetime.utcnow() + __import__('datetime').timedelta(hours=8)
            }, JWT_SECRET, algorithm="HS256")
            return jsonify({'success': True, 'token': token})
        else:
            return jsonify({'success': False, 'message': 'Invalid credentials'}), 401
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/faculty/login', methods=['POST'])
def faculty_login_api():
    try:
        data = request.get_json()
        employee_id = data.get('employee_id', '').strip()
        password = data.get('password', '').strip()

        if not employee_id or not password:
            return jsonify({'success': False, 'message': 'Employee ID and password are required'}), 400

        hashed = hashlib.sha256(password.encode()).hexdigest()

        conn = database.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, full_name, employee_id, department, email FROM faculty WHERE (employee_id = ? OR username = ?) AND password = ?",
            (employee_id, employee_id, hashed)
        )
        faculty = cursor.fetchone()
        conn.close()

        if faculty:
            fac = dict(faculty)
            token = jwt.encode({
                'faculty_id': fac['employee_id'],
                'exp': datetime.utcnow() + __import__('datetime').timedelta(hours=12)
            }, JWT_SECRET, algorithm="HS256")
            return jsonify({
                'success': True,
                'token': token,
                'faculty_id': fac['employee_id'],
                'faculty_name': fac['full_name'],
                'department': fac['department'],
                'email': fac['email']
            })
        else:
            return jsonify({'success': False, 'message': 'Invalid Employee ID or Password'}), 401
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/admin/register-student', methods=['POST'])
def admin_register_student():
    try:
        data = request.get_json()
        name = data.get('name')
        roll = data.get('roll')
        branch = data.get('branch')
        semester = data.get('semester')
        dob = data.get('dob')
        
        if not all([name, roll, branch, semester, dob]):
            return jsonify({'success': False, 'message': 'All fields required'}), 400
            
        username = roll.upper()
        password = hashlib.sha256(dob.encode()).hexdigest()
        
        # Save student to DB first
        result = database.add_student(name, roll, branch, semester, dob, username, password)
        if not result:
            return jsonify({'success': False, 'message': 'Roll or Username already exists'}), 409
        
        # Create dataset folder: TrainingImage/<Name_Roll>
        folder_name = f"{name}_{roll}"
        save_path = f"TrainingImage/{folder_name}"
        os.makedirs(save_path, exist_ok=True)
        
        # Launch capture_faces.py in a NEW CONSOLE WINDOW
        # The script opens the laptop camera, captures YOLO-detected faces for ~5 seconds
        write_log(f"Launching face capture for {name} ({roll})", "info")
        try:
            if sys.platform == 'win32':
                process = subprocess.Popen(
                    [sys.executable, "capture_faces.py", folder_name],
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
            else:
                process = subprocess.Popen(
                    [sys.executable, "capture_faces.py", folder_name]
                )
            
            # Wait for capture to complete (camera window runs for ~7 seconds total)
            process.wait(timeout=30)
            
            # Mark face as registered in DB
            conn = database.get_connection()
            cursor = conn.cursor()
            cursor.execute("UPDATE students SET face_registered = 1 WHERE roll_number = ?", (roll,))
            conn.commit()
            conn.close()
            
            write_log(f"Face capture completed for {name} ({roll})", "success")
            return jsonify({
                'success': True, 
                'message': f'Student registered & face captured!',
                'credentials': {'username': username, 'password': dob}
            })
            
        except subprocess.TimeoutExpired:
            process.kill()
            write_log(f"Face capture timed out for {name}", "warning")
            return jsonify({
                'success': True, 
                'message': 'Student registered but face capture timed out. Try again later.',
                'credentials': {'username': username, 'password': dob}
            })
        except Exception as cam_err:
            write_log(f"Camera error for {name}: {str(cam_err)}", "error")
            return jsonify({
                'success': True,
                'message': f'Student registered in DB but camera failed: {str(cam_err)}',
                'credentials': {'username': username, 'password': dob}
            })

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/admin/student/<int:student_id>', methods=['DELETE'])
def delete_student_admin(student_id):
    conn = None
    try:
        conn = database.get_connection()
        cursor = conn.cursor()
        
        # Get student info to delete folder and attendance
        cursor.execute("SELECT roll_number, name FROM students WHERE id = ?", (student_id,))
        row = cursor.fetchone()
        
        if row:
            roll_number = row['roll_number']
            student_name = row['name']
            
            # Possible folder names
            folder_name1 = f"{student_name}_{roll_number}" if roll_number else student_name
            folder_name2 = student_name
            
            import shutil
            for fname in [folder_name1, folder_name2]:
                folder_path = os.path.join(DATASET_DIR, fname)
                if os.path.exists(folder_path):
                    shutil.rmtree(folder_path)
                    write_log(f"Deleted face data folder: {fname}", "info")
            
            # Clean DeepFace cache so the deleted face is removed from embeddings
            import glob
            cache_patterns = [
                os.path.join(DATASET_DIR, "representations_*.pkl"),
                os.path.join(DATASET_DIR, "ds_model_*.pkl"),
            ]
            for pattern in cache_patterns:
                for f in glob.glob(pattern):
                    try:
                        os.remove(f)
                    except Exception:
                        pass
                        
            # Delete attendance records
            cursor.execute("DELETE FROM attendance WHERE student_id = ?", (student_id,))
                
        # Delete from DB
        cursor.execute("DELETE FROM students WHERE id = ?", (student_id,))
        conn.commit()
        
        write_log(f"Deleted student ID {student_id} and their attendance records", "info")
        return jsonify({'success': True, 'message': 'Student deleted successfully'})
    except Exception as e:
        if conn:
            conn.rollback()
        return jsonify({'success': False, 'message': str(e)}), 500
    finally:
        if conn:
            conn.close()

# ============================================================
# TIMETABLE API ROUTES
# ============================================================

@app.route('/api/timetable', methods=['GET'])
def api_get_timetable():
    """Get timetable entries for a specific faculty"""
    try:
        faculty_id = request.args.get('faculty_id')
        if not faculty_id:
            return jsonify({'success': False, 'message': 'faculty_id required'}), 400
        entries = database.get_timetable_for_faculty(faculty_id)
        return jsonify(entries)
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/timetable/current-class', methods=['GET'])
def api_current_class():
    """Get the currently scheduled class for a faculty (checks day & time)"""
    try:
        faculty_id = request.args.get('faculty_id')
        if not faculty_id:
            return jsonify({'success': False, 'message': 'faculty_id required'}), 400
        entry = database.get_current_class(faculty_id)
        if entry:
            return jsonify({'found': True, 'class': entry})
        return jsonify({'found': False, 'class': None})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/timetable/all', methods=['GET'])
def api_all_timetable():
    """Admin: Get full timetable for all faculty"""
    try:
        entries = database.get_all_timetable()
        return jsonify(entries)
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/timetable', methods=['POST'])
def api_add_timetable():
    """Admin: Add a timetable entry"""
    try:
        data = request.get_json()
        faculty_id = data.get('faculty_id')
        faculty_name = data.get('faculty_name')
        subject_code = data.get('subject_code', '')
        subject_name = data.get('subject_name')
        day_of_week = data.get('day_of_week')
        period = data.get('period')
        branch = data.get('branch', '')
        semester = data.get('semester', '')
        room = data.get('room', '')

        if not all([faculty_id, faculty_name, subject_name, day_of_week, period]):
            return jsonify({'success': False, 'message': 'Missing required fields'}), 400

        result = database.add_timetable_entry(
            faculty_id, faculty_name, subject_code, subject_name,
            day_of_week, period, branch, semester, room
        )
        if result:
            return jsonify({'success': True, 'message': 'Timetable entry added', 'id': result})
        return jsonify({'success': False, 'message': 'Slot already taken for this faculty on this day/period'}), 409
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/timetable/<int:entry_id>', methods=['DELETE'])
def api_delete_timetable(entry_id):
    """Admin: Delete a timetable entry"""
    try:
        database.delete_timetable_entry(entry_id)
        return jsonify({'success': True, 'message': 'Timetable entry deleted'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

# ============================================================
# SESSION ATTENDANCE DETAIL & LOGBOOK ROUTES
# ============================================================

@app.route('/api/session-attendance/<int:session_id>', methods=['GET'])
def api_session_attendance(session_id):
    """Get detailed list of students present in a specific lecture session"""
    try:
        records = database.get_session_attendance(session_id)
        total_students = database.get_total_students_count()
        return jsonify({
            'students': records,
            'total_present': len(records),
            'total_students': total_students
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/faculty-logbook', methods=['GET'])
def api_faculty_logbook():
    """Get all past sessions for a faculty member (regular + extra)"""
    try:
        faculty_id = request.args.get('faculty_id')
        if not faculty_id:
            return jsonify({'success': False, 'message': 'faculty_id required'}), 400
        sessions = database.get_faculty_logbook(faculty_id)
        total_students = database.get_total_students_count()
        return jsonify({'sessions': sessions, 'total_students': total_students})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/faculty-extra-stats', methods=['GET'])
def api_faculty_extra_stats():
    """Get extra classes taken and classes missed/substituted counts"""
    try:
        faculty_id = request.args.get('faculty_id')
        if not faculty_id:
            return jsonify({'success': False, 'message': 'faculty_id required'}), 400
        extra_taken = database.get_faculty_extra_classes_count(faculty_id)
        classes_substituted = database.get_faculty_missed_classes_count(faculty_id)
        return jsonify({
            'extra_taken': extra_taken,
            'classes_substituted': classes_substituted
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/admin/extra-classes-summary', methods=['GET'])
def api_admin_extra_classes_summary():
    """Admin: Get summary of all extra/substitute classes"""
    try:
        sessions = database.get_all_extra_classes_summary()
        return jsonify(sessions)
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

# ============================================================
# TAKE ATTENDANCE ROUTES (SCHEDULED + EXTRA)
# ============================================================

@app.route('/api/take-extra-class', methods=['POST'])
def api_take_extra_class():
    """Faculty: Start an extra/substitute class attendance session"""
    try:
        data = request.get_json() or {}
        faculty_id = data.get('faculty_id', 'UNKNOWN')
        faculty_name = data.get('faculty_name', localStorage_fallback(faculty_id))
        subject_name = data.get('subject_name', '')
        subject_code = data.get('subject_code', '')
        period = data.get('period', '')
        original_faculty_id = data.get('original_faculty_id', '')
        original_faculty_name = data.get('original_faculty_name', '')

        if not subject_name or not period:
            return jsonify({'success': False, 'message': 'Subject and period are required'}), 400

        # Create lecture session with type=extra
        session_id = database.create_lecture_session(
            subject_code=subject_code,
            subject_name=subject_name,
            period=period,
            faculty_name=faculty_name,
            session_type='extra',
            faculty_id=faculty_id,
            original_faculty_id=original_faculty_id,
            original_faculty_name=original_faculty_name
        )

        write_log(f"Extra class started: {subject_name} by {faculty_name} (substituting {original_faculty_name or 'N/A'})", "info")

        # Fetch RTSP URL from rooms if room_id provided
        rtsp_url = None
        room_id = data.get('room_id', None)
        if room_id:
            try:
                conn_rt = database.get_connection()
                cursor_rt = conn_rt.cursor()
                cursor_rt.execute("SELECT rtsp_url FROM rooms WHERE id = ?", (room_id,))
                rt_row = cursor_rt.fetchone()
                conn_rt.close()
                if rt_row:
                    rtsp_url = dict(rt_row).get('rtsp_url')
            except Exception:
                pass

        # Launch recognition
        try:
            cmd_args = [
                sys.executable, "recognize_attendance.py",
                subject_code, subject_name, period, faculty_name,
                str(session_id) if session_id else "0"
            ]
            if rtsp_url:
                cmd_args.append(rtsp_url)
                write_log(f"Using IP camera for extra class: {rtsp_url}", "info")
            if sys.platform == 'win32':
                process = subprocess.Popen(cmd_args, creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:
                process = subprocess.Popen(cmd_args)

            process.wait(timeout=60)

            conn = database.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT total_present FROM lecture_sessions WHERE id = ?", (session_id,))
            row = cursor.fetchone()
            total_marked = dict(row)['total_present'] if row else 0
            conn.close()

            return jsonify({
                'success': True,
                'message': f'Extra class attendance completed! {total_marked} students marked.',
                'session_id': session_id,
                'total_marked': total_marked
            })
        except subprocess.TimeoutExpired:
            process.kill()
            return jsonify({'success': True, 'message': 'Scan timed out.', 'session_id': session_id})
        except Exception as cam_err:
            return jsonify({'success': False, 'message': f'Camera error: {str(cam_err)}'}), 500
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/take-attendance', methods=['POST'])
def api_take_attendance():
    """One-click attendance: launches recognize_attendance.py with laptop camera for 20 seconds"""
    try:
        data = request.get_json() or {}
        faculty_id = data.get('faculty_id', 'UNKNOWN')
        faculty_name = data.get('faculty_name', localStorage_fallback(faculty_id))
        
        # Create a lecture session record
        subject_name = data.get('subject_name', 'General')
        subject_code = data.get('subject_code', 'GEN')
        period = data.get('period', 'Demo')
        session_type = data.get('session_type', 'regular')
        timetable_id = data.get('timetable_id', None)
        
        # Create session in DB
        session_id = database.create_lecture_session(
            subject_code=subject_code,
            subject_name=subject_name,
            period=period,
            faculty_name=faculty_name,
            session_type=session_type,
            faculty_id=faculty_id,
            timetable_id=timetable_id
        )
        
        write_log(f"Starting attendance scan (Session #{session_id}) by {faculty_name} for {subject_name}", "info")

        # AUTO-DETECT CAMERA: timetable room → any single room → laptop fallback
        rtsp_url = None

        # Step 1: Try room linked to timetable
        if timetable_id:
            try:
                conn_rt = database.get_connection()
                cursor_rt = conn_rt.cursor()
                cursor_rt.execute(
                    "SELECT r.rtsp_url FROM timetable t JOIN rooms r ON t.room_id = r.id WHERE t.id = ?",
                    (timetable_id,)
                )
                rt_row = cursor_rt.fetchone()
                conn_rt.close()
                if rt_row:
                    rtsp_url = dict(rt_row).get('rtsp_url')
                    write_log(f"Using timetable room camera: {rtsp_url}", "info")
            except Exception:
                pass

        # Step 2: No timetable room → auto-use the only configured room
        if not rtsp_url:
            try:
                conn_rt = database.get_connection()
                cursor_rt = conn_rt.cursor()
                cursor_rt.execute("SELECT rtsp_url, room_name FROM rooms ORDER BY id LIMIT 1")
                rt_row = cursor_rt.fetchone()
                conn_rt.close()
                if rt_row:
                    rtsp_url = dict(rt_row).get('rtsp_url')
                    room_name = dict(rt_row).get('room_name', '')
                    write_log(f"Auto-selected camera from room '{room_name}': {rtsp_url}", "info")
            except Exception:
                pass

        # Step 3: If still no RTSP → laptop camera (handled in recognize_attendance.py)
        if not rtsp_url:
            write_log("No IP camera configured — using laptop camera", "info")

        # Launch recognize_attendance.py in NEW CONSOLE WINDOW
        try:
            cmd_args = [
                sys.executable, "recognize_attendance.py",
                subject_code, subject_name, period, faculty_name,
                str(session_id) if session_id else "0"
            ]
            if rtsp_url:
                cmd_args.append(rtsp_url)
            
            if sys.platform == 'win32':
                process = subprocess.Popen(
                    cmd_args,
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
            else:
                process = subprocess.Popen(cmd_args)
            
            # Wait for the recognition to complete (~25 seconds including init)
            process.wait(timeout=60)
            
            write_log(f"Attendance scan completed (Session #{session_id})", "success")
            
            # Get how many were marked
            conn = database.get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT total_present FROM lecture_sessions WHERE id = ?",
                (session_id,)
            )
            row = cursor.fetchone()
            total_marked = dict(row)['total_present'] if row else 0
            conn.close()
            
            return jsonify({
                'success': True,
                'message': f'Attendance completed! {total_marked} students marked present.',
                'session_id': session_id,
                'total_marked': total_marked
            })
            
        except subprocess.TimeoutExpired:
            process.kill()
            write_log("Attendance scan timed out", "warning")
            return jsonify({
                'success': True,
                'message': 'Scan completed (timed out). Check logbook for results.',
                'session_id': session_id
            })
        except Exception as cam_err:
            write_log(f"Camera error during attendance: {str(cam_err)}", "error")
            return jsonify({
                'success': False,
                'message': f'Camera error: {str(cam_err)}'
            }), 500
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

def localStorage_fallback(faculty_id):
    """Helper to get faculty name from DB by ID"""
    try:
        conn = database.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT full_name FROM faculty WHERE employee_id = ?", (faculty_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row)['full_name'] if row else faculty_id
    except:
        return faculty_id



@app.route('/api/admin/students', methods=['GET'])
def admin_get_students():
    try:
        conn = database.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, roll_number as roll, department as branch, academic_year as semester, face_registered, created_at FROM students ORDER BY created_at DESC")
        students = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return jsonify(students)
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/admin/add-faculty', methods=['POST'])
def admin_add_faculty():
    try:
        data = request.get_json()
        name = data.get('name')
        employee_id = data.get('employee_id')
        department = data.get('department')
        email = data.get('email')
        plain_password = data.get('password', '').strip()
        
        if not all([name, employee_id, department, email]):
            return jsonify({'success': False, 'message': 'All fields required'}), 400
        
        if not plain_password or len(plain_password) < 4:
            return jsonify({'success': False, 'message': 'Password must be at least 4 characters'}), 400
            
        username = employee_id
        password = hashlib.sha256(plain_password.encode()).hexdigest()
        
        result = database.add_faculty(username, password, name, employee_id, department, "Assistant Professor", email, "")
        if result:
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'message': 'Employee ID or email already exists'}), 409
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/admin/faculty/<int:faculty_id>', methods=['DELETE'])
def delete_faculty_admin(faculty_id):
    conn = None
    try:
        conn = database.get_connection()
        cursor = conn.cursor()
        
        # Get employee_id to remove assignments
        cursor.execute("SELECT employee_id FROM faculty WHERE id = ?", (faculty_id,))
        row = cursor.fetchone()
        
        if row:
            emp_id = row['employee_id']
            # Delete subject assignments linked to this faculty
            cursor.execute("DELETE FROM teacher_assignments WHERE teacher_id = ?", (emp_id,))
            cursor.execute("DELETE FROM teacher_assignments WHERE teacher_id = ?", (str(faculty_id),))
            
        # Delete faculty attendance records
        cursor.execute("DELETE FROM faculty_attendance WHERE faculty_id = ?", (faculty_id,))
        
        # Finally delete the faculty
        cursor.execute("DELETE FROM faculty WHERE id = ?", (faculty_id,))
        conn.commit()
        
        write_log(f"Deleted faculty ID {faculty_id} and their records", "info")
        return jsonify({'success': True, 'message': 'Faculty deleted successfully'})
    except Exception as e:
        if conn:
            conn.rollback()
        return jsonify({'success': False, 'message': str(e)}), 500
    finally:
        if conn:
            conn.close()


@app.route('/api/admin/faculty', methods=['GET'])
def admin_get_faculty():
    try:
        conn = database.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, full_name as name, employee_id, department, email, 1 as is_active, created_at FROM faculty")
        faculty = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return jsonify(faculty)
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/admin/add-classroom', methods=['POST'])
def admin_add_classroom():
    try:
        data = request.get_json()
        room_name = data.get('room_name')
        camera_url = data.get('camera_url')
        
        if not all([room_name, camera_url]):
            return jsonify({'success': False, 'message': 'All fields required'}), 400
            
        conn = database.get_connection()
        cursor = conn.cursor()
        cursor.execute('INSERT INTO classrooms (room_name, camera_url) VALUES (?, ?)', (room_name, camera_url))
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': f"Classroom '{room_name}' added"})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/admin/classrooms', methods=['GET'])
def admin_get_classrooms():
    try:
        conn = database.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM classrooms ORDER BY created_at DESC")
        classrooms = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return jsonify(classrooms)
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/admin/stats', methods=['GET'])
def admin_get_stats():
    try:
        conn = database.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM students")
        students = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM faculty")
        faculty = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM classrooms")
        classrooms = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM teacher_assignments")
        subjects_count = cursor.fetchone()[0]
        
        conn.close()
        return jsonify({
            'students': students, 
            'faculty': faculty, 
            'classrooms': classrooms,
            'subjects_count': subjects_count,
            'attendance': "87%" # Static as per prompt requirements
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/assign-subject', methods=['POST'])
def api_assign_subject():
    try:
        data = request.get_json()
        teacher_id = data.get('teacher_id')
        teacher_name = data.get('teacher_name')
        branch = data.get('branch')
        semester = data.get('semester')
        subject_name = data.get('subject_name')
        subject_code = data.get('subject_code', '')
        
        if not all([teacher_id, teacher_name, branch, semester, subject_name]):
            return jsonify({'success': False, 'message': 'Missing required fields'}), 400
            
        conn = database.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO teacher_assignments 
            (teacher_id, teacher_name, branch, semester, subject_name, subject_code) 
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (teacher_id, teacher_name, branch, semester, subject_name, subject_code))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'Subject assigned successfully'})
    except sqlite3.IntegrityError:
        return jsonify({'success': False, 'message': 'This exact assignment already exists.'}), 409
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/assignments', methods=['GET'])
def api_get_assignments():
    try:
        limit = request.args.get('limit')
        conn = database.get_connection()
        cursor = conn.cursor()
        
        query = "SELECT * FROM teacher_assignments ORDER BY created_at DESC"
        if limit and limit.isdigit():
            query += f" LIMIT {limit}"
            
        cursor.execute(query)
        assignments = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return jsonify(assignments)
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/assignment/<int:id>', methods=['DELETE'])
def api_delete_assignment(id):
    try:
        conn = database.get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM teacher_assignments WHERE id = ?", (id,))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'Assignment removed'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/get-faculty-subjects', methods=['GET'])
def api_get_faculty_subjects():
    try:
        teacher_id = request.args.get('faculty_id') or request.args.get('teacher_id')
        if not teacher_id:
            return jsonify({'success': False, 'message': 'Teacher ID required'}), 400
            
        conn = database.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT subject_name, subject_code, branch, semester FROM teacher_assignments WHERE teacher_id = ? ORDER BY subject_name ASC", (teacher_id,))
        subjects = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return jsonify(subjects)
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/faculty-stats', methods=['GET'])
def api_faculty_stats():
    try:
        faculty_id = request.args.get('faculty_id')
        if not faculty_id:
            return jsonify({'success': False, 'message': 'Faculty ID required'}), 400
            
        conn = database.get_connection()
        cursor = conn.cursor()
        
        today = datetime.now().strftime("%Y-%m-%d")
        cursor.execute("SELECT COUNT(*) FROM class_sessions WHERE faculty_id = ? AND date = ?", (faculty_id, today))
        classes_today = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM teacher_assignments WHERE teacher_id = ?", (faculty_id,))
        subjects_assigned = cursor.fetchone()[0]
        
        cursor.execute("SELECT AVG(total_present * 100.0 / NULLIF(total_students, 0)) FROM class_sessions WHERE faculty_id = ?", (faculty_id,))
        avg_att = cursor.fetchone()[0]
        avg_attendance = float(avg_att) if avg_att else 0.0
        
        conn.close()
        return jsonify({
            'classes_today': classes_today,
            'subjects_assigned': subjects_assigned,
            'avg_attendance': avg_attendance
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/start-session', methods=['POST'])
def api_new_start_session():
    try:
        data = request.get_json()
        faculty_id = data.get('faculty_id')
        subject_raw = data.get('subject') # "Subject Name (Code) — Branch Sem"
        period = data.get('period')
        classroom = data.get('classroom')
        date = data.get('date')
        
        if not all([faculty_id, subject_raw, period, classroom, date]):
            return jsonify({'success': False, 'message': 'Missing fields'}), 400
            
        # Parse subject string
        subject_name = subject_raw.split(' (')[0]
        subject_code = ""
        branch = ""
        semester = ""
        try:
            code_part = subject_raw.split('(')[1].split(')')[0]
            subject_code = code_part
            rest = subject_raw.split('— ')[1].split(' ')
            branch = rest[0]
            semester = " ".join(rest[1:])
        except:
            pass
            
        conn = database.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO class_sessions 
            (faculty_id, subject_name, subject_code, branch, semester, period, classroom, date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (faculty_id, subject_name, subject_code, branch, semester, period, classroom, date))
        session_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # In a real scenario, we trigger camera here using subprocess.Popen
        # e.g., subprocess.Popen(["python", "recognize_attendance.py", "--faculty", faculty_id, "--subject", subject_name, "--period", period, "--classroom", classroom])
        write_log(f"Started session {session_id} for {subject_name} by {faculty_id} in {classroom}", "info")
        
        return jsonify({'success': True, 'session_id': session_id})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/save-session', methods=['POST'])
def api_save_session():
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        total_present = data.get('total_present', 0)
        total_students = data.get('total_students', 60)
        
        if not session_id:
            return jsonify({'success': False, 'message': 'Session ID required'}), 400
            
        conn = database.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE class_sessions 
            SET total_present = ?, total_students = ?, saved_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (total_present, total_students, session_id))
        conn.commit()
        conn.close()
        
        write_log(f"Saved session {session_id} with {total_present} present", "success")
        return jsonify({'success': True, 'message': 'Session saved'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/class-sessions', methods=['GET'])
def api_class_sessions():
    try:
        faculty_id = request.args.get('faculty_id')
        filter_type = request.args.get('filter', 'all')
        
        if not faculty_id:
            return jsonify({'success': False, 'message': 'Faculty ID required'}), 400
            
        conn = database.get_connection()
        cursor = conn.cursor()
        
        query = "SELECT * FROM class_sessions WHERE faculty_id = ?"
        params = [faculty_id]
        
        if filter_type == 'today':
            today = datetime.now().strftime("%Y-%m-%d")
            query += " AND date = ?"
            params.append(today)
            
        query += " ORDER BY started_at DESC"
        
        cursor.execute(query, params)
        sessions = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return jsonify(sessions)
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500




# ============================================================
# STUDENT API ROUTES
# ============================================================

STUDENT_DUMMY = {
    "overall_pct": 84.0,
    "attended": 42,
    "total": 50,
    "subjects": [
        {"name": "Machine Learning", "pct": 88.0, "attended": 22, "total": 25},
        {"name": "Deep Learning",    "pct": 80.0, "attended": 20, "total": 25},
        {"name": "Soft Computing",   "pct": 68.0, "attended": 17, "total": 25},
        {"name": "NLP",              "pct": 92.0, "attended": 23, "total": 25},
        {"name": "Cloud Computing",  "pct": 72.0, "attended": 18, "total": 25},
        {"name": "Major Project",    "pct": 96.0, "attended": 24, "total": 25},
    ],
    "history": [
        {"date": "04 May 2026", "subject": "Machine Learning", "period": "Period 3", "faculty": "Dr. R. Kumar",   "status": "Present"},
        {"date": "04 May 2026", "subject": "Deep Learning",    "period": "Period 1", "faculty": "Dr. R. Kumar",   "status": "Present"},
        {"date": "03 May 2026", "subject": "Soft Computing",   "period": "Period 5", "faculty": "Prof. A. Verma", "status": "Absent"},
        {"date": "03 May 2026", "subject": "NLP",              "period": "Period 3", "faculty": "Prof. S. Patel", "status": "Present"},
        {"date": "02 May 2026", "subject": "Cloud Computing",  "period": "Period 6", "faculty": "Dr. M. Singh",   "status": "Absent"},
    ]
}


@app.route('/api/student-stats/<roll_no>', methods=['GET'])
def student_stats(roll_no):
    try:
        roll_no = roll_no.strip().upper()
        conn = database.get_connection()
        cursor = conn.cursor()

        # 1. Overall stats
        cursor.execute('''
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN a.status='Present' THEN 1 ELSE 0 END) as attended,
                ROUND(SUM(CASE WHEN a.status='Present' THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*),0), 1) as overall_pct
            FROM attendance a
            JOIN students s ON a.student_id = s.id
            WHERE s.roll_number = ?
        ''', (roll_no,))
        overall = cursor.fetchone()

        if not overall or not overall['total']:
            return jsonify({
                'overall_pct': 0.0,
                'attended': 0,
                'total': 0,
                'subjects': [],
                'history': []
            })

        # 2. Subject-wise
        cursor.execute('''
            SELECT
                a.subject_name as name,
                COUNT(*) as total,
                SUM(CASE WHEN a.status='Present' THEN 1 ELSE 0 END) as attended,
                ROUND(SUM(CASE WHEN a.status='Present' THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*),0), 1) as pct
            FROM attendance a
            JOIN students s ON a.student_id = s.id
            WHERE s.roll_number = ?
            GROUP BY a.subject_name ORDER BY pct DESC
        ''', (roll_no,))
        subjects = [dict(r) for r in cursor.fetchall()]

        # 3. Recent history
        cursor.execute('''
            SELECT a.date, a.subject_name as subject, a.period,
                   a.faculty_name as faculty, a.status
            FROM attendance a
            JOIN students s ON a.student_id = s.id
            WHERE s.roll_number = ?
            ORDER BY a.date DESC, a.period ASC LIMIT 5
        ''', (roll_no,))
        history = [dict(r) for r in cursor.fetchall()]

        conn.close()
        return jsonify({
            'overall_pct': overall['overall_pct'] or 0.0,
            'attended': overall['attended'] or 0,
            'total': overall['total'] or 0,
            'subjects': subjects,
            'history': history
        })
    except Exception as e:
        return jsonify({
            'overall_pct': 0.0,
            'attended': 0,
            'total': 0,
            'subjects': [],
            'history': []
        })

# ============================================================
# STUDENT AUTH API (Roll Number + DOB login)
# ============================================================

@app.route('/api/student/login', methods=['POST'])
def student_login_api():
    try:
        data = request.get_json()
        roll_number = data.get('roll_number', '').strip().upper()
        dob = data.get('dob', '').strip()

        if not roll_number or not dob:
            return jsonify({'success': False, 'message': 'Registration Number and Date of Birth are required'}), 400

        conn = database.get_connection()
        cursor = conn.cursor()
        # Match roll_number AND dob stored as hash (set during registration) OR plain dob
        hashed_dob = hashlib.sha256(dob.encode()).hexdigest()
        cursor.execute(
            "SELECT id, name, roll_number, department, academic_year, dob, password FROM students WHERE UPPER(roll_number) = ?",
            (roll_number,)
        )
        student = cursor.fetchone()
        conn.close()

        if student:
            s = dict(student)
            # Accept if password matches hashed dob OR dob field matches plain dob
            pw_match = (s.get('password') == hashed_dob)
            dob_match = (s.get('dob') == dob)
            if pw_match or dob_match:
                token = jwt.encode(
                    {'roll': s['roll_number'], 'exp': datetime.utcnow() + __import__('datetime').timedelta(hours=12)},
                    JWT_SECRET, algorithm="HS256"
                )
                return jsonify({
                    'success': True,
                    'token': token,
                    'name': s['name'],
                    'roll_number': s['roll_number'],
                    'branch': s.get('department', 'CSE'),
                    'semester': s.get('academic_year', '1st Sem')
                })
        return jsonify({'success': False, 'message': 'Invalid Registration Number or Date of Birth'}), 401
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

# ============================================================
# ROOMS (IP/RTSP CAMERA) API ROUTES
# ============================================================

@app.route('/api/rooms', methods=['GET'])
def api_get_rooms():
    try:
        conn = database.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM rooms ORDER BY room_name ASC")
        rooms = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return jsonify(rooms)
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/rooms', methods=['POST'])
def api_add_room():
    try:
        data = request.get_json()
        room_name = data.get('room_name', '').strip()
        rtsp_url = data.get('rtsp_url', '').strip()
        if not room_name or not rtsp_url:
            return jsonify({'success': False, 'message': 'Room name and RTSP URL are required'}), 400
        conn = database.get_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO rooms (room_name, rtsp_url) VALUES (?, ?)", (room_name, rtsp_url))
        conn.commit()
        room_id = cursor.lastrowid
        conn.close()
        write_log(f"Room '{room_name}' added with URL: {rtsp_url}", "info")
        return jsonify({'success': True, 'message': f"Room '{room_name}' added", 'id': room_id})
    except Exception as e:
        if 'UNIQUE' in str(e):
            return jsonify({'success': False, 'message': 'A room with this name already exists'}), 409
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/rooms/<int:room_id>', methods=['DELETE'])
def api_delete_room(room_id):
    try:
        conn = database.get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM rooms WHERE id = ?", (room_id,))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'Room deleted'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

# ============================================================
# STUDENT ATTENDANCE OVERVIEW & DEFAULTER ALERTS
# ============================================================

@app.route('/api/admin/attendance-overview', methods=['GET'])
def api_attendance_overview():
    """Returns per-student attendance summary for Admin defaulter tracker"""
    try:
        conn = database.get_connection()
        cursor = conn.cursor()
        # Count total unique lecture sessions as 'total classes'
        cursor.execute("SELECT COUNT(DISTINCT id) FROM lecture_sessions")
        total_sessions = cursor.fetchone()[0] or 0

        cursor.execute('''
            SELECT s.id, s.name, s.roll_number, s.email,
                   COUNT(a.id) as attended,
                   ? as total
            FROM students s
            LEFT JOIN attendance a ON a.student_id = s.id AND a.status = 'Present'
            GROUP BY s.id, s.name, s.roll_number, s.email
            ORDER BY s.name ASC
        ''', (total_sessions,))
        rows = []
        for r in cursor.fetchall():
            d = dict(r)
            attended = d['attended'] or 0
            total = d['total'] or 0
            d['pct'] = round((attended / total * 100), 1) if total > 0 else 0.0
            rows.append(d)
        conn.close()
        return jsonify({'students': rows, 'total_sessions': total_sessions})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/admin/send-defaulter-alerts', methods=['POST'])
def api_send_defaulter_alerts():
    """Send automated email alerts to students with attendance < 75%"""
    import smtplib
    import os
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    MAIL_USERNAME = os.getenv('MAIL_USERNAME', '')
    MAIL_PASSWORD = os.getenv('MAIL_PASSWORD', '')

    if not MAIL_USERNAME or not MAIL_PASSWORD:
        return jsonify({'success': False, 'message': 'Email credentials not configured. Set MAIL_USERNAME and MAIL_PASSWORD in .env'}), 500

    try:
        conn = database.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(DISTINCT id) FROM lecture_sessions")
        total_sessions = cursor.fetchone()[0] or 0

        cursor.execute('''
            SELECT s.name, s.roll_number, s.email,
                   COUNT(a.id) as attended
            FROM students s
            LEFT JOIN attendance a ON a.student_id = s.id AND a.status = 'Present'
            GROUP BY s.id
            HAVING s.email IS NOT NULL AND s.email != ''
        ''')
        students = [dict(r) for r in cursor.fetchall()]
        conn.close()

        sent = 0
        skipped = 0
        errors = []

        for st in students:
            attended = st['attended'] or 0
            pct = round((attended / total_sessions * 100), 1) if total_sessions > 0 else 0.0
            if pct >= 75:
                skipped += 1
                continue

            try:
                msg = MIMEMultipart('alternative')
                msg['Subject'] = f"⚠️ Attendance Warning — {st['name']}"
                msg['From'] = MAIL_USERNAME
                msg['To'] = st['email']

                html_body = f"""
                <div style="font-family:Arial,sans-serif;background:#0f172a;color:#f1f5f9;padding:30px;border-radius:12px;max-width:480px">
                  <h2 style="color:#ef4444">⚠️ Attendance Shortage Alert</h2>
                  <p>Dear <strong>{st['name']}</strong>,</p>
                  <p>Your current attendance is <strong style="color:#ef4444">{pct}%</strong>
                     ({attended} / {total_sessions} classes attended).</p>
                  <p>You must maintain a minimum of <strong>75%</strong> attendance to be eligible for examinations.</p>
                  <p>Please contact your class coordinator immediately.</p>
                  <hr style="border-color:#334155;margin:20px 0"/>
                  <small style="color:#64748b">SmartAttend AI — Auto-generated Alert</small>
                </div>
                """
                msg.attach(MIMEText(html_body, 'html'))

                with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                    server.login(MAIL_USERNAME, MAIL_PASSWORD)
                    server.sendmail(MAIL_USERNAME, st['email'], msg.as_string())
                sent += 1
                write_log(f"Defaulter alert sent to {st['name']} ({st['email']}) — {pct}%", "info")
            except Exception as mail_err:
                errors.append(f"{st['name']}: {str(mail_err)}")

        return jsonify({
            'success': True,
            'sent': sent,
            'skipped': skipped,
            'errors': errors,
            'message': f"Alerts sent to {sent} defaulter(s). {skipped} student(s) above 75% skipped."
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

# ============================================================
# UPDATED admin/register-student — add email field
# ============================================================

@app.route('/api/admin/register-student-v2', methods=['POST'])
def admin_register_student_v2():
    """Updated registration with email field"""
    try:
        data = request.get_json()
        name = data.get('name', '').strip()
        roll = data.get('roll', '').strip().upper()
        branch = data.get('branch', '').strip()
        semester = data.get('semester', '').strip()
        dob = data.get('dob', '').strip()
        email = data.get('email', '').strip()

        if not all([name, roll, branch, semester, dob]):
            return jsonify({'success': False, 'message': 'Name, roll, branch, semester and DOB are required'}), 400

        username = roll.upper()
        password = hashlib.sha256(dob.encode()).hexdigest()

        conn = database.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO students (name, roll_number, department, academic_year, dob, email, username, password, face_registered) VALUES (?,?,?,?,?,?,?,?,0)",
                (name, roll, branch, semester, dob, email, username, password)
            )
            conn.commit()
            student_id = cursor.lastrowid
        except Exception as db_err:
            conn.close()
            return jsonify({'success': False, 'message': 'Roll number or username already exists'}), 409
        conn.close()

        folder_name = f"{name}_{roll}"
        save_path = os.path.join('TrainingImage', folder_name)
        os.makedirs(save_path, exist_ok=True)

        write_log(f"Launching face capture for {name} ({roll})", "info")
        try:
            if sys.platform == 'win32':
                process = subprocess.Popen(
                    [sys.executable, "capture_faces.py", folder_name],
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
            else:
                process = subprocess.Popen([sys.executable, "capture_faces.py", folder_name])
            process.wait(timeout=30)

            conn2 = database.get_connection()
            cursor2 = conn2.cursor()
            cursor2.execute("UPDATE students SET face_registered = 1 WHERE roll_number = ?", (roll,))
            conn2.commit()
            conn2.close()
            write_log(f"Face capture completed for {name} ({roll})", "success")
            return jsonify({'success': True, 'message': f'Student registered & face captured!', 'credentials': {'username': username, 'password': dob}})
        except subprocess.TimeoutExpired:
            process.kill()
            return jsonify({'success': True, 'message': 'Registered but face capture timed out.', 'credentials': {'username': username, 'password': dob}})
        except Exception as cam_err:
            return jsonify({'success': True, 'message': f'Registered but camera failed: {cam_err}', 'credentials': {'username': username, 'password': dob}})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":
    init_log()
    write_log("Flask server starting on port 8000", "success")
    app.run(debug=True, port=8000)

