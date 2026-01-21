from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import subprocess
import sys
import os
import time
import json
from datetime import datetime

app = Flask(__name__)
app.secret_key = "attendance_secret_key"

# Log file path
LOG_FILE = "system.log"

def write_log(message, log_type="info"):
    """Write log entry to file and console"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "type": log_type,
        "message": message
    }
    
    # Write to log file
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"Error writing to log: {e}")
    
    # Also print to console
    print(f"[{timestamp}] {message}")

def read_logs(max_lines=100):
    """Read recent log entries from file"""
    logs = []
    try:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                lines = f.readlines()
                # Get last max_lines
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

@app.route("/api/logs")
def api_logs():
    """API endpoint to get recent logs"""
    logs = read_logs(50)  # Get last 50 log entries
    return jsonify({"logs": logs})

def kill_camera_processes():
    """Kill any running camera-related Python processes to prevent multiple windows"""
    try:
        import psutil
        camera_scripts = ["dataset_capture.py", "recognize_and_attendance.py", "recognize_face.py"]
        
        killed_count = 0
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                # Check if it's a Python process
                proc_name = proc.info.get('name', '')
                if proc_name and 'python' in proc_name.lower():
                    cmdline = proc.info.get('cmdline', [])
                    if cmdline:
                        # Check if it's running any camera script
                        cmdline_str = ' '.join(str(c) for c in cmdline).lower()
                        for script in camera_scripts:
                            if script.lower() in cmdline_str:
                                proc.terminate()
                                killed_count += 1
                                # Wait a bit for process to terminate
                                try:
                                    proc.wait(timeout=1)
                                except psutil.TimeoutExpired:
                                    proc.kill()  # Force kill if it doesn't terminate
                                break
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        
        if killed_count > 0:
            time.sleep(0.5)  # Give time for windows to close
            print(f"Closed {killed_count} previous camera process(es)")
    except ImportError:
        # Fallback: Use taskkill on Windows if psutil not available
        if sys.platform == 'win32':
            try:
                scripts = ["dataset_capture.py", "recognize_and_attendance.py", "recognize_face.py"]
                for script in scripts:
                    # Kill processes running these scripts
                    subprocess.run(
                        ["taskkill", "/F", "/FI", f"WINDOWTITLE eq *{script}*", "/T"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                time.sleep(0.5)
            except Exception:
                pass
        # For other platforms or if taskkill fails, just wait a bit
        time.sleep(0.5)

@app.route("/")
def home():
    write_log("Web interface accessed", "info")
    return render_template("Index.html")

@app.route("/register", methods=["POST"])
def register():
    name = request.form["student_name"]
    kill_camera_processes()  # Close any existing camera windows
    write_log(f"Registration started for student: {name}", "info")
    flash(f"📸 Registration started for {name}. Check camera window.")
    
    # Don't capture stdout/stderr - camera needs display access
    process = subprocess.Popen(
        [sys.executable, "dataset_capture.py", name],
        creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
    )
    
    write_log("Camera window should open now. Check for 'Smart Attendance System' window.", "info")
    
    return redirect(url_for("home"))

@app.route("/train")
def train():
    write_log("Model training started", "info")
    flash("⚙️ Model training started...")
    
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
    
    return redirect(url_for("home"))

@app.route("/attendance")
def attendance():
    # Check if model files exist before starting
    if not os.path.exists("trainer.yml") or not os.path.exists("labels.npy"):
        write_log("❌ ERROR: Model not trained yet! Please train the model first.", "error")
        flash("❌ Model not trained yet! Please train the model first using 'Train / Update Model' button.")
        return redirect(url_for("home"))
    
    kill_camera_processes()  # Close any existing camera windows
    time.sleep(0.3)  # Wait for processes to fully close
    
    write_log("Attendance taking started", "info")
    flash("📸 Attendance started. Please look at the camera.")
    
    # Don't capture stdout/stderr - camera needs display access for OpenCV windows
    # Use CREATE_NEW_CONSOLE on Windows to ensure window visibility
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
        
        write_log("Camera window should open now. Look for 'Smart Attendance System' window.", "info")
    except Exception as e:
        write_log(f"❌ ERROR: Failed to start attendance: {str(e)}", "error")
        flash(f"❌ Error starting attendance: {str(e)}")
    
    return redirect(url_for("home"))

@app.route("/reset")
def reset_model():
    write_log("Model reset requested", "warning")
    
    # Kill any running camera processes first
    kill_camera_processes()
    
    # Wait a bit to ensure processes are killed
    time.sleep(0.5)
    
    # Remove model files
    try:
        if os.path.exists("trainer.yml"):
            os.remove("trainer.yml")
            write_log("trainer.yml deleted", "info")
    except Exception as e:
        write_log(f"Error deleting trainer.yml: {str(e)}", "error")
    
    try:
        if os.path.exists("labels.npy"):
            os.remove("labels.npy")
            write_log("labels.npy deleted", "info")
    except Exception as e:
        write_log(f"Error deleting labels.npy: {str(e)}", "error")
    
    write_log("Model reset completed. Please train the model before taking attendance.", "success")
    flash("🔄 Model reset successfully. Please train the model before taking attendance.")
    return redirect(url_for("home"))

if __name__ == "__main__":
    init_log()
    write_log("Flask server starting on port 8000", "success")
    app.run(debug=True, port=8000)
