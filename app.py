from flask import Flask, render_template, request, redirect, url_for, flash
import subprocess
import sys
import os

app = Flask(__name__)
app.secret_key = "attendance_secret_key"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/register", methods=["POST"])
def register():
    name = request.form["student_name"]
    flash(f"📸 Registration started for {name}. Check camera window.")
    subprocess.Popen([sys.executable, "dataset_capture.py", name])
    return redirect(url_for("home"))

@app.route("/train")
def train():
    flash("⚙️ Model training started...")
    subprocess.Popen([sys.executable, "train_model.py"])
    return redirect(url_for("home"))

@app.route("/attendance")
def attendance():
    flash("📸 Attendance started. Please look at the camera.")
    subprocess.Popen([sys.executable, "recognize_and_attendance.py"])
    return redirect(url_for("home"))

@app.route("/reset")
def reset_model():
    if os.path.exists("trainer.yml"):
        os.remove("trainer.yml")
    if os.path.exists("labels.npy"):
        os.remove("labels.npy")
    flash("🔄 Model reset successfully.")
    return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(debug=True, port=8000)
