from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import hashlib
import os
import jwt
import datetime

app = Flask(__name__)
CORS(app)
DB_PATH = "attendance.db"
JWT_SECRET = "smartattend_admin_secret_2024"

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def hash_password(p): 
    return hashlib.sha256(p.encode()).hexdigest()

def init_db():
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        roll TEXT UNIQUE NOT NULL,
        branch TEXT,
        semester INTEGER,
        dob TEXT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        face_registered INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS faculty (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        employee_id TEXT UNIQUE NOT NULL,
        department TEXT,
        email TEXT UNIQUE,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        is_active INTEGER DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS classrooms (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        room_name TEXT NOT NULL,
        camera_url TEXT NOT NULL,
        is_active INTEGER DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS admins (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER,
        period_id TEXT,
        subject TEXT,
        date TEXT,
        status TEXT DEFAULT 'Present',
        marked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(student_id) REFERENCES students(id),
        UNIQUE(student_id, period_id, date)
    )
    ''')
    
    # Insert default admin
    cursor.execute('INSERT OR IGNORE INTO admins (username, password) VALUES (?, ?)', 
                   ('admin', hash_password('admin123')))
    
    conn.commit()
    conn.close()

@app.route('/api/admin/login', methods=['POST'])
def admin_login():
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM admins WHERE username = ? AND password = ?", 
                       (username, hash_password(password)))
        admin = cursor.fetchone()
        conn.close()
        
        if admin:
            token = jwt.encode({
                'user': username,
                'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=8)
            }, JWT_SECRET, algorithm="HS256")
            return jsonify({'success': True, 'token': token})
        else:
            return jsonify({'success': False, 'message': 'Invalid credentials'}), 401
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/register-student', methods=['POST'])
def register_student():
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
        password = hash_password(dob)
        
        conn = get_db()
        cursor = conn.cursor()
        try:
            cursor.execute('''
            INSERT INTO students (name, roll, branch, semester, dob, username, password) 
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (name, roll, branch, int(semester), dob, username, password))
            conn.commit()
            
            os.makedirs(f"TrainingImage/{name}_{roll}", exist_ok=True)
            return jsonify({'success': True, 'credentials': {'username': username, 'password': dob}})
        except sqlite3.IntegrityError:
            return jsonify({'success': False, 'message': 'Roll already exists'}), 409
        finally:
            conn.close()
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/students', methods=['GET'])
def get_students():
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, roll, branch, semester, face_registered, created_at FROM students ORDER BY created_at DESC")
        students = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return jsonify(students)
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/add-faculty', methods=['POST'])
def add_faculty():
    try:
        data = request.get_json()
        name = data.get('name')
        employee_id = data.get('employee_id')
        department = data.get('department')
        email = data.get('email')
        
        if not all([name, employee_id, department, email]):
            return jsonify({'success': False, 'message': 'All fields required'}), 400
            
        username = employee_id
        password = hash_password(f"{employee_id}@123")
        
        conn = get_db()
        cursor = conn.cursor()
        try:
            cursor.execute('''
            INSERT INTO faculty (name, employee_id, department, email, username, password) 
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (name, employee_id, department, email, username, password))
            conn.commit()
            return jsonify({'success': True})
        except sqlite3.IntegrityError:
            return jsonify({'success': False, 'message': 'Employee ID or email already exists'}), 409
        finally:
            conn.close()
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/faculty', methods=['GET'])
def get_faculty():
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, employee_id, department, email, is_active, created_at FROM faculty")
        faculty = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return jsonify(faculty)
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/add-classroom', methods=['POST'])
def add_classroom():
    try:
        data = request.get_json()
        room_name = data.get('room_name')
        camera_url = data.get('camera_url')
        
        if not all([room_name, camera_url]):
            return jsonify({'success': False, 'message': 'All fields required'}), 400
            
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('INSERT INTO classrooms (room_name, camera_url) VALUES (?, ?)', (room_name, camera_url))
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': f"Classroom '{room_name}' added"})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/classrooms', methods=['GET'])
def get_classrooms():
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM classrooms ORDER BY created_at DESC")
        classrooms = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return jsonify(classrooms)
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM students")
        students = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM faculty")
        faculty = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM classrooms")
        classrooms = cursor.fetchone()[0]
        
        conn.close()
        return jsonify({
            'students': students, 
            'faculty': faculty, 
            'classrooms': classrooms,
            'attendance': "87%" # Static as per prompt requirements
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

if __name__ == "__main__":
    init_db()
    app.run(debug=True, host="0.0.0.0", port=5000)
