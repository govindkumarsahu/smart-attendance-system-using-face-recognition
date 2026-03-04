from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response
import subprocess
import sys
import os
import time
import json
import csv
import numpy as np
from datetime import datetime

app = Flask(__name__)
app.secret_key = "attendance_secret_key"

# File paths
LOG_FILE = "system.log"
ATTENDANCE_FILE = "attendance.csv"
DATASET_DIR = "dataset"
TRAINER_FILE = "trainer.yml"
LABELS_FILE = "labels.npy"

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

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
    """Get list of registered student folders from dataset/"""
    students = []
    if os.path.exists(DATASET_DIR):
        for name in sorted(os.listdir(DATASET_DIR)):
            folder = os.path.join(DATASET_DIR, name)
            if os.path.isdir(folder):
                img_count = len([f for f in os.listdir(folder) if f.endswith(('.jpg', '.png', '.jpeg'))])
                students.append({"name": name, "images": img_count})
    return students

def get_attendance_records():
    """Read all attendance records from CSV"""
    records = []
    if os.path.exists(ATTENDANCE_FILE):
        try:
            with open(ATTENDANCE_FILE, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("Name") and row.get("Date") and row.get("Time"):
                        records.append({
                            "name": row["Name"].strip(),
                            "date": row["Date"].strip(),
                            "time": row["Time"].strip()
                        })
        except Exception as e:
            print(f"Error reading attendance CSV: {e}")
    return records

def get_today_records():
    """Get attendance records for today only (deduplicated — first entry per student)"""
    today = datetime.now().strftime("%Y-%m-%d")
    all_records = get_attendance_records()
    seen = set()
    today_records = []
    for r in all_records:
        if r["date"] == today and r["name"] not in seen:
            seen.add(r["name"])
            today_records.append(r)
    return today_records

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
    return jsonify({"logs": logs})

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

@app.route("/dashboard")
def dashboard():
    write_log("Dashboard accessed", "info")
    stats = get_dashboard_stats()
    return render_template("dashboard.html", stats=stats)

@app.route("/register-page")
def register_page():
    students = get_registered_students()
    return render_template("register.html", students=students)

@app.route("/attendance-page")
def attendance_page():
    today_records = get_today_records()
    return render_template("attendance.html", today_records=today_records, total_present=len(today_records))

@app.route("/reports")
def reports():
    # Get filter params
    search_q = request.args.get("q", "").strip()
    status_filter = request.args.get("status", "all").lower()
    date_from = request.args.get("date_from", "")
    date_to = request.args.get("date_to", "")
    page = int(request.args.get("page", 1))
    per_page = 10

    all_records = get_attendance_records()
    # Reverse so newest first
    all_records = list(reversed(all_records))

    # Apply filters
    filtered = []
    for r in all_records:
        if search_q and search_q.lower() not in r["name"].lower():
            continue
        if date_from and r["date"] < date_from:
            continue
        if date_to and r["date"] > date_to:
            continue
        filtered.append(r)

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
def export_csv():
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

    import io
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Student Name", "Date", "Time", "Status"])
    for r in filtered:
        writer.writerow([r["name"], r["date"], r["time"], "Present"])

    csv_data = output.getvalue()
    output.close()

    filename = f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.route("/model")
def model_page():
    model = get_model_info()
    students = get_registered_students()
    activity = get_recent_activity_logs()
    return render_template("model.html", model_info=model, students=students, activity=activity)

# ============================================================
# ACTION ROUTES
# ============================================================

@app.route("/register", methods=["POST"])
def register():
    name = request.form.get("student_name", "").strip()
    if not name:
        flash("❌ Please enter a student name.")
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

@app.route("/train")
def train():
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
def attendance():
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
                [sys.executable, "recognize_and_attendance.py"],
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
        else:
            process = subprocess.Popen(
                [sys.executable, "recognize_and_attendance.py"]
            )
        write_log("Camera window opened for attendance.", "info")
    except Exception as e:
        write_log(f"❌ ERROR: Failed to start attendance: {str(e)}", "error")
        flash(f"❌ Error starting attendance: {str(e)}")

    return redirect(url_for("attendance_page"))

@app.route("/reset")
def reset_model():
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
