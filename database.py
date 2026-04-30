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

    # Subjects master table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS subjects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            department TEXT,
            semester TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Lecture sessions table — logs every class taken by a faculty
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS lecture_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_code TEXT NOT NULL,
            subject_name TEXT NOT NULL,
            period TEXT NOT NULL,
            faculty_name TEXT NOT NULL,
            date TEXT NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT,
            total_present INTEGER DEFAULT 0,
            status TEXT DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Attendance table — now tracks per-subject, per-period
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            status TEXT DEFAULT 'Present',
            subject_code TEXT DEFAULT '',
            subject_name TEXT DEFAULT '',
            period TEXT DEFAULT '',
            faculty_name TEXT DEFAULT '',
            session_id INTEGER,
            FOREIGN KEY (student_id) REFERENCES students (id),
            FOREIGN KEY (session_id) REFERENCES lecture_sessions (id),
            UNIQUE(student_id, date, subject_code, period) 
        )
    ''')

    # Faculty table — multiple teachers with their own login
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS faculty (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            full_name TEXT NOT NULL,
            employee_id TEXT UNIQUE,
            department TEXT,
            designation TEXT DEFAULT 'Assistant Professor',
            email TEXT,
            phone TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Faculty attendance table — tracks daily check-in / check-out
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS faculty_attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            faculty_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            check_in TEXT,
            check_out TEXT,
            work_hours REAL DEFAULT 0,
            status TEXT DEFAULT 'Present',
            remarks TEXT DEFAULT '',
            marked_by TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (faculty_id) REFERENCES faculty (id),
            UNIQUE(faculty_id, date)
        )
    ''')

    # Auto-migrate: add employee_id column if missing (for existing DBs)
    try:
        cursor.execute("PRAGMA table_info(faculty)")
        columns = [col[1] for col in cursor.fetchall()]
        if 'employee_id' not in columns:
            cursor.execute("ALTER TABLE faculty ADD COLUMN employee_id TEXT UNIQUE")
    except Exception:
        pass

    # Seed default subjects if the table is empty
    cursor.execute("SELECT COUNT(*) FROM subjects")
    if cursor.fetchone()[0] == 0:
        default_subjects = [
            ('CS501', 'Data Structures & Algorithms', 'CSE', '5th'),
            ('CS502', 'Database Management Systems', 'CSE', '5th'),
            ('CS503', 'Operating Systems', 'CSE', '5th'),
            ('CS504', 'Computer Networks', 'CSE', '5th'),
            ('CS505', 'Artificial Intelligence', 'CSE', '5th'),
            ('CS506', 'Software Engineering', 'CSE', '5th'),
            ('CS507', 'Machine Learning', 'CSE', '7th'),
            ('CS508', 'Cloud Computing', 'CSE', '7th'),
        ]
        cursor.executemany(
            "INSERT INTO subjects (code, name, department, semester) VALUES (?, ?, ?, ?)",
            default_subjects
        )

    # Seed default faculty if the table is empty
    cursor.execute("SELECT COUNT(*) FROM faculty")
    if cursor.fetchone()[0] == 0:
        default_faculty = [
            ('sharma', '1234', 'Prof. Rajesh Sharma', 'EMP001', 'CSE', 'HOD', 'sharma@college.edu', '9876543210'),
            ('verma', '1234', 'Prof. Neha Verma', 'EMP002', 'CSE', 'Assistant Professor', 'verma@college.edu', '9876543211'),
            ('gupta', '1234', 'Prof. Amit Gupta', 'EMP003', 'CSE-AI', 'Assistant Professor', 'gupta@college.edu', '9876543212'),
        ]
        cursor.executemany(
            "INSERT INTO faculty (username, password, full_name, employee_id, department, designation, email, phone) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            default_faculty
        )

    conn.commit()
    conn.close()

# ============================================================
# STUDENT OPERATIONS
# ============================================================

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

# ============================================================
# FACULTY OPERATIONS
# ============================================================

def authenticate_faculty(username, password):
    """Authenticate a faculty member, returns faculty dict or None"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM faculty WHERE LOWER(username) = LOWER(?) AND password = ?", (username, password))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_all_faculty():
    """Retrieve all faculty members"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, full_name, employee_id, department, designation, email, phone, created_at FROM faculty ORDER BY full_name ASC")
    faculty = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return faculty

def add_faculty(username, password, full_name, employee_id="", department="", designation="Assistant Professor", email="", phone=""):
    """Add a new faculty member"""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO faculty (username, password, full_name, employee_id, department, designation, email, phone) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (username, password, full_name, employee_id or None, department, designation, email, phone)
        )
        conn.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()

def update_faculty(faculty_id, full_name, employee_id="", department="", designation="", email="", phone=""):
    """Update faculty details"""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """UPDATE faculty 
               SET full_name = ?, employee_id = ?, department = ?, designation = ?, email = ?, phone = ? 
               WHERE id = ?""",
            (full_name, employee_id or None, department, designation, email, phone, faculty_id)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def delete_faculty(faculty_id):
    """Delete a faculty member and their attendance records"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM faculty_attendance WHERE faculty_id = ?", (faculty_id,))
    cursor.execute("DELETE FROM faculty WHERE id = ?", (faculty_id,))
    conn.commit()
    conn.close()

def get_faculty_by_id(faculty_id):
    """Get a single faculty by ID"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM faculty WHERE id = ?", (faculty_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

# ============================================================
# FACULTY ATTENDANCE OPERATIONS
# ============================================================

def mark_faculty_attendance(faculty_id, status="Present", remarks="", marked_by=""):
    """
    Mark faculty check-in for today.
    Returns: 'success', 'duplicate', or 'not_found'
    """
    faculty = get_faculty_by_id(faculty_id)
    if not faculty:
        return 'not_found'

    today = datetime.now().strftime("%Y-%m-%d")
    time_now = datetime.now().strftime("%H:%M:%S")

    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """INSERT INTO faculty_attendance 
               (faculty_id, date, check_in, status, remarks, marked_by) 
               VALUES (?, ?, ?, ?, ?, ?)""",
            (faculty_id, today, time_now, status, remarks, marked_by)
        )
        conn.commit()
        return 'success'
    except sqlite3.IntegrityError:
        return 'duplicate'
    finally:
        conn.close()

def checkout_faculty(faculty_id):
    """
    Mark faculty check-out for today and calculate work hours.
    Returns: 'success', 'no_checkin', or 'already_checkout'
    """
    today = datetime.now().strftime("%Y-%m-%d")
    time_now = datetime.now().strftime("%H:%M:%S")

    conn = get_connection()
    cursor = conn.cursor()
    
    # Find today's check-in record
    cursor.execute(
        "SELECT * FROM faculty_attendance WHERE faculty_id = ? AND date = ?",
        (faculty_id, today)
    )
    record = cursor.fetchone()
    
    if not record:
        conn.close()
        return 'no_checkin'
    
    record = dict(record)
    if record.get('check_out'):
        conn.close()
        return 'already_checkout'
    
    # Calculate work hours
    check_in_time = record.get('check_in', '')
    work_hours = 0.0
    if check_in_time:
        try:
            fmt = "%H:%M:%S"
            t_in = datetime.strptime(check_in_time, fmt)
            t_out = datetime.strptime(time_now, fmt)
            diff = (t_out - t_in).total_seconds() / 3600.0
            work_hours = round(diff, 2)
        except Exception:
            pass
    
    # Determine status based on work hours
    status = record.get('status', 'Present')
    if status == 'Present' and 0 < work_hours < 4:
        status = 'Half-Day'
    
    cursor.execute(
        """UPDATE faculty_attendance 
           SET check_out = ?, work_hours = ?, status = ? 
           WHERE faculty_id = ? AND date = ?""",
        (time_now, work_hours, status, faculty_id, today)
    )
    conn.commit()
    conn.close()
    return 'success'

def get_faculty_attendance(date_from=None, date_to=None, faculty_id=None):
    """Get faculty attendance records joined with faculty details"""
    conn = get_connection()
    cursor = conn.cursor()
    
    query = """
        SELECT fa.id, fa.date, fa.check_in, fa.check_out, fa.work_hours, fa.status, fa.remarks, fa.marked_by,
               f.full_name, f.employee_id, f.department, f.designation
        FROM faculty_attendance fa
        JOIN faculty f ON fa.faculty_id = f.id
        WHERE 1=1
    """
    params = []
    
    if date_from:
        query += " AND fa.date >= ?"
        params.append(date_from)
    if date_to:
        query += " AND fa.date <= ?"
        params.append(date_to)
    if faculty_id:
        query += " AND fa.faculty_id = ?"
        params.append(faculty_id)
        
    query += " ORDER BY fa.date DESC, fa.check_in DESC"
    
    cursor.execute(query, params)
    records = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return records

def get_faculty_today_summary():
    """Get today's faculty attendance summary"""
    today = datetime.now().strftime("%Y-%m-%d")
    all_faculty = get_all_faculty()
    today_records = get_faculty_attendance(date_from=today, date_to=today)
    
    present_ids = set()
    on_leave_count = 0
    total_work_hours = 0.0
    checked_out_count = 0
    
    for r in today_records:
        present_ids.add(r.get('full_name'))
        if r.get('status') == 'On Leave':
            on_leave_count += 1
        if r.get('work_hours', 0) > 0:
            total_work_hours += r.get('work_hours', 0)
            checked_out_count += 1
    
    avg_hours = round(total_work_hours / checked_out_count, 1) if checked_out_count > 0 else 0
    
    return {
        'total_faculty': len(all_faculty),
        'present_today': len(present_ids),
        'on_leave': on_leave_count,
        'avg_work_hours': avg_hours,
        'records': today_records
    }

# ============================================================
# SUBJECT OPERATIONS
# ============================================================

def get_all_subjects():
    """Retrieve all subjects"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM subjects ORDER BY code ASC")
    subjects = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return subjects

def add_subject(code, name, department="", semester=""):
    """Add a new subject"""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO subjects (code, name, department, semester) VALUES (?, ?, ?, ?)",
            (code, name, department, semester)
        )
        conn.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()

def delete_subject(subject_id):
    """Delete a subject"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM subjects WHERE id = ?", (subject_id,))
    conn.commit()
    conn.close()

# ============================================================
# LECTURE SESSION OPERATIONS
# ============================================================

def create_lecture_session(subject_code, subject_name, period, faculty_name):
    """Create a new lecture session and return the session ID"""
    conn = get_connection()
    cursor = conn.cursor()
    today = datetime.now().strftime("%Y-%m-%d")
    time_now = datetime.now().strftime("%H:%M:%S")
    try:
        cursor.execute(
            """INSERT INTO lecture_sessions 
               (subject_code, subject_name, period, faculty_name, date, start_time, status) 
               VALUES (?, ?, ?, ?, ?, ?, 'active')""",
            (subject_code, subject_name, period, faculty_name, today, time_now)
        )
        conn.commit()
        return cursor.lastrowid
    except Exception:
        return None
    finally:
        conn.close()

def end_lecture_session(session_id, total_present=0):
    """Mark a lecture session as completed"""
    conn = get_connection()
    cursor = conn.cursor()
    time_now = datetime.now().strftime("%H:%M:%S")
    cursor.execute(
        """UPDATE lecture_sessions 
           SET end_time = ?, total_present = ?, status = 'completed' 
           WHERE id = ?""",
        (time_now, total_present, session_id)
    )
    conn.commit()
    conn.close()

def get_lecture_sessions(date_from=None, date_to=None):
    """Get lecture sessions, optionally filtered by date range"""
    conn = get_connection()
    cursor = conn.cursor()
    query = "SELECT * FROM lecture_sessions WHERE 1=1"
    params = []
    if date_from:
        query += " AND date >= ?"
        params.append(date_from)
    if date_to:
        query += " AND date <= ?"
        params.append(date_to)
    query += " ORDER BY date DESC, start_time DESC"
    cursor.execute(query, params)
    sessions = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return sessions

def get_today_sessions():
    """Get today's lecture sessions"""
    today = datetime.now().strftime("%Y-%m-%d")
    return get_lecture_sessions(date_from=today, date_to=today)

# ============================================================
# ATTENDANCE OPERATIONS
# ============================================================

def mark_attendance(name, subject_code="", subject_name="", period="", faculty_name="", session_id=None):
    """
    Attempt to mark attendance for a student on the current date for a specific subject/period.
    Returns:
       'success' if inserted,
       'duplicate' if already marked today for that subject/period,
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
            """INSERT INTO attendance 
               (student_id, date, time, subject_code, subject_name, period, faculty_name, session_id) 
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (student['id'], today, time_now, subject_code, subject_name, period, faculty_name, session_id)
        )
        conn.commit()
        return 'success'
    except sqlite3.IntegrityError:
        return 'duplicate'
    finally:
        conn.close()

def get_attendance_records(date_from=None, date_to=None, search_q=None, subject_code=None):
    """Get all attendance records joined with student details"""
    conn = get_connection()
    cursor = conn.cursor()
    
    query = """
        SELECT a.date, a.time, a.status, a.subject_code, a.subject_name, a.period, a.faculty_name,
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
    if subject_code:
        query += " AND a.subject_code = ?"
        params.append(subject_code)
        
    query += " ORDER BY a.date DESC, a.time DESC"
    
    cursor.execute(query, params)
    records = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return records

def get_today_records():
    """Get today's attendance records"""
    today = datetime.now().strftime("%Y-%m-%d")
    return get_attendance_records(date_from=today, date_to=today)
