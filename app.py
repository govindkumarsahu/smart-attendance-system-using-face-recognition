from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response, session
from functools import wraps
import subprocess
import sys
import os
import time
import json
import csv
import numpy as np
import shutil
from datetime import datetime
import database

app = Flask(__name__)
app.secret_key = "attendance_secret_key"

# Initialize database
database.init_db()

# File paths
LOG_FILE = "system.log"
DATASET_DIR = "dataset"
TRAINER_FILE = "trainer.yml"
LABELS_FILE = "labels.npy"

# ============================================================
# UTILITY FUNCTIONS & DECORATORS
# ============================================================

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user" not in session:
            flash("Please login first.", "warning")
            return redirect(url_for("login"))
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
    students = database.get_all_students()
    
    # Auto-sync: add any student folders from dataset/ that are missing from the database
    db_names_lower = {s['name'].lower() for s in students}
    if os.path.isdir(DATASET_DIR):
        for folder_name in os.listdir(DATASET_DIR):
            folder_path = os.path.join(DATASET_DIR, folder_name)
            if os.path.isdir(folder_path) and folder_name.lower() not in db_names_lower:
                # This student exists in dataset but not in DB – auto-register them
                result = database.add_student(folder_name, "", "", "")
                if result is not None:
                    write_log(f"Auto-synced student '{folder_name}' from dataset folder to database.", "info")
        # Re-fetch after sync
        students = database.get_all_students()

    # Add image count dynamically and verify profile picture existence
    for student in students:
        student['images'] = 0
        student['has_profile_pic'] = False
        folder = os.path.join(DATASET_DIR, student['name'])
        if os.path.isdir(folder):
            images = [f for f in os.listdir(folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
            student['images'] = len(images)
            if '1.jpg' in images:
                student['has_profile_pic'] = True
            elif len(images) > 0:
                student['has_profile_pic'] = True
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
            "academic_year": r["academic_year"]
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
        info["status"] = "Trained"
        info["status_color"] = "green"
        # Get last modified time of trainer.yml
        try:
            mtime = os.path.getmtime(TRAINER_FILE)
            info["last_trained"] = datetime.fromtimestamp(mtime).strftime("%b %d, %Y %I:%M %p")
        except:
            pass

    if os.path.exists(LABELS_FILE):
        try:
            label_map = np.load(LABELS_FILE, allow_pickle=True).item()
            info["labels"] = label_map
            info["total_faces"] = len(label_map)
        except:
            pass

    # Calculate pending: students in dataset/ but not in trained model
    registered = get_registered_students()
    trained_names = set(info["labels"].values()) if info["labels"] else set()
    registered_names = set(s["name"] for s in registered)
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
        camera_scripts = ["dataset_capture.py", "recognize_and_attendance.py", "recognize_face.py"]
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
                for script in ["dataset_capture.py", "recognize_and_attendance.py", "recognize_face.py"]:
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

# ============================================================
# PAGE ROUTES
# ============================================================

@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/old")
def old_home():
    return render_template("Index.html")

# ============================================================
# AUTHENTICATION
# ============================================================

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("logged_in"):
            flash("🔒 Please log in to access this page.", "error")
            return redirect(url_for("login", next=request.url))
        return f(*args, **kwargs)
    return decorated_function

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        
        # 1. Faculty Login (Demo Credentials)
        if username.lower() == "faculty" and password == "1234":
            session["logged_in"] = True
            session["username"] = "Faculty"
            session["role"] = "faculty"
            write_log("Faculty logged in successfully (Demo)", "success")
            flash("👋 Welcome back, Faculty member!")
            return redirect(url_for("dashboard"))
            
        # 2. Student Login (Registration Number)
        # For demo: search by roll_number in the database
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
        flash("❌ Invalid credentials or Registration Number. Hint: Use faculty/1234", "error")
            
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
        [sys.executable, "dataset_capture.py", name],
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
    write_log("Model training started", "info")
    flash("⚙️ Model training started... Please wait.")

    process = subprocess.Popen(
        [sys.executable, "train_model.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )

    def log_output():
        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                if line.strip():
                    log_type = "success" if "COMPLETE" in line or "✅" in line else "info"
                    log_type = "warning" if "⚠️" in line or "Skipped" in line else log_type
                    write_log(line.strip(), log_type)
            process.stdout.close()

    import threading
    log_thread = threading.Thread(target=log_output, daemon=True)
    log_thread.start()

    return redirect(url_for("model_page"))

@app.route("/attendance")
@login_required
def attendance():
    if session.get("role") != "faculty":
        flash("🚫 Access denied. Faculty only.", "error")
        return redirect(url_for("student_dashboard"))
    if not os.path.exists(TRAINER_FILE) or not os.path.exists(LABELS_FILE):
        write_log("❌ ERROR: Model not trained yet!", "error")
        flash("❌ Model not trained yet! Please train the model first.")
        return redirect(url_for("dashboard"))

    kill_camera_processes()
    time.sleep(0.3)

    write_log("Attendance taking started", "info")
    flash("📸 Attendance started. Please look at the camera.")

    try:
        if sys.platform == 'win32':
            process = subprocess.Popen(
                [sys.executable, "recognize_and_attendance_improved.py"],
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
        else:
            process = subprocess.Popen(
                [sys.executable, "recognize_and_attendance_improved.py"]
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
# MAIN
# ============================================================

if __name__ == "__main__":
    init_log()
    write_log("Flask server starting on port 8000", "success")
    app.run(debug=True, port=8000)
