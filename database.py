import sqlite3
import os
import json
from datetime import datetime

DB_FILE = "attendance.db"

def get_connection():
    """Establish a connection to the SQLite database with dictionary row factory"""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize the database tables if they do not exist"""
    conn = get_connection()
    cursor = conn.cursor()

    # Students table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            roll_number TEXT,
            department TEXT,
            academic_year TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Attendance table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            status TEXT DEFAULT 'Present',
            FOREIGN KEY (student_id) REFERENCES students (id),
            UNIQUE(student_id, date) 
        )
    ''')
    
    conn.commit()
    conn.close()

def add_student(name, roll_number, department, academic_year):
    """Add a new student to the database"""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO students (name, roll_number, department, academic_year) VALUES (?, ?, ?, ?)",
            (name, roll_number, department, academic_year)
        )
        conn.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError:
        return None # Student name already exists
    finally:
        conn.close()

def get_all_students():
    """Retrieve all students"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students ORDER BY name ASC")
    students = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return students

def get_student_by_name(name):
    """Retrieve a single student by name (case-insensitive)"""
    conn = get_connection()
    cursor = conn.cursor()
    # Use LOWER() for more robust matching across different capitalization in recognition model
    cursor.execute("SELECT * FROM students WHERE LOWER(name) = LOWER(?)", (name,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def update_student(student_id, name, roll_number, department, academic_year):
    """Update student details"""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """UPDATE students 
               SET name = ?, roll_number = ?, department = ?, academic_year = ? 
               WHERE id = ?""",
            (name, roll_number, department, academic_year, student_id)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def delete_student(student_id):
    """Delete a student and their attendance records"""
    conn = get_connection()
    cursor = conn.cursor()
    # First delete their attendance records
    cursor.execute("DELETE FROM attendance WHERE student_id = ?", (student_id,))
    # Then delete the student
    cursor.execute("DELETE FROM students WHERE id = ?", (student_id,))
    conn.commit()
    conn.close()

def mark_attendance(name):
    """
    Attempt to mark attendance for a student on the current date.
    Returns:
       'success' if inserted,
       'duplicate' if already marked today,
       'not_found' if student doesnt exist in students table.
    """
    student = get_student_by_name(name)
    if not student:
        return 'not_found'

    today = datetime.now().strftime("%Y-%m-%d")
    time_now = datetime.now().strftime("%H:%M:%S")

    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO attendance (student_id, date, time) VALUES (?, ?, ?)",
            (student['id'], today, time_now)
        )
        conn.commit()
        return 'success'
    except sqlite3.IntegrityError:
        return 'duplicate'
    finally:
        conn.close()

def get_attendance_records(date_from=None, date_to=None, search_q=None):
    """Get all attendance records joined with student details"""
    conn = get_connection()
    cursor = conn.cursor()
    
    query = """
        SELECT a.date, a.time, a.status, 
               s.name, s.roll_number, s.department, s.academic_year
        FROM attendance a
        JOIN students s ON a.student_id = s.id
        WHERE 1=1
    """
    params = []
    
    if date_from:
        query += " AND a.date >= ?"
        params.append(date_from)
    if date_to:
        query += " AND a.date <= ?"
        params.append(date_to)
    if search_q:
        query += " AND s.name LIKE ?"
        params.append(f"%{search_q}%")
        
    query += " ORDER BY a.date DESC, a.time DESC"
    
    cursor.execute(query, params)
    records = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return records

def get_today_records():
    """Get today's attendance records"""
    today = datetime.now().strftime("%Y-%m-%d")
    return get_attendance_records(date_from=today, date_to=today)
