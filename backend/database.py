import sqlite3
import os
import json
from datetime import datetime

DB_FILE = "attendance.db"

def get_connection():
    """Establish a connection to the SQLite database with dictionary row factory"""
    conn = sqlite3.connect(DB_FILE, timeout=20)
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

    # Admins table for Admin Panel
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Classrooms table for Admin Panel
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS classrooms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            room_name TEXT NOT NULL,
            camera_url TEXT NOT NULL,
            is_active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Teacher Assignments table for subject-to-faculty mapping
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS teacher_assignments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            teacher_id TEXT NOT NULL,
            teacher_name TEXT NOT NULL,
            branch TEXT NOT NULL,
            semester TEXT NOT NULL,
            subject_name TEXT NOT NULL,
            subject_code TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(teacher_id, branch, semester, subject_name)
        )
    ''')

    # Class sessions table for Faculty logbook
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS class_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            faculty_id TEXT NOT NULL,
            subject_name TEXT NOT NULL,
            subject_code TEXT DEFAULT '',
            branch TEXT NOT NULL,
            semester TEXT NOT NULL,
            period TEXT NOT NULL,
            classroom TEXT NOT NULL,
            date TEXT NOT NULL,
            total_students INTEGER DEFAULT 0,
            total_present INTEGER DEFAULT 0,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            saved_at TIMESTAMP
        )
    ''')

    # Timetable table — maps Faculty -> Subject -> Day -> Period
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS timetable (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            faculty_id TEXT NOT NULL,
            faculty_name TEXT NOT NULL,
            subject_code TEXT NOT NULL,
            subject_name TEXT NOT NULL,
            day_of_week TEXT NOT NULL,
            period TEXT NOT NULL,
            start_time TEXT DEFAULT '',
            end_time TEXT DEFAULT '',
            branch TEXT DEFAULT '',
            semester TEXT DEFAULT '',
            room TEXT DEFAULT '',
            is_active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(faculty_id, day_of_week, period)
        )
    ''')

    # Auto-migrate: add employee_id column if missing (for existing DBs)

    try:
        cursor.execute("PRAGMA table_info(faculty)")
        columns = [col[1] for col in cursor.fetchall()]
        if 'employee_id' not in columns:
            cursor.execute("ALTER TABLE faculty ADD COLUMN employee_id TEXT")
    except Exception:
        pass

    # Rooms table — for RTSP/IP camera management
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS rooms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            room_name TEXT NOT NULL UNIQUE,
            rtsp_url TEXT NOT NULL,
            is_active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Auto-migrate students table to support admin panel requirements
    try:
        cursor.execute("PRAGMA table_info(students)")
        student_columns = [col[1] for col in cursor.fetchall()]
        if 'dob' not in student_columns:
            cursor.execute("ALTER TABLE students ADD COLUMN dob TEXT")
        if 'username' not in student_columns:
            cursor.execute("ALTER TABLE students ADD COLUMN username TEXT")
        if 'password' not in student_columns:
            cursor.execute("ALTER TABLE students ADD COLUMN password TEXT")
        if 'face_registered' not in student_columns:
            cursor.execute("ALTER TABLE students ADD COLUMN face_registered INTEGER DEFAULT 0")
        if 'email' not in student_columns:
            cursor.execute("ALTER TABLE students ADD COLUMN email TEXT DEFAULT ''")
    except Exception as e:
        print(f"Error auto-migrating students table: {e}")

    # Auto-migrate timetable table to add room_id
    try:
        cursor.execute("PRAGMA table_info(timetable)")
        tt_columns = [col[1] for col in cursor.fetchall()]
        if 'room_id' not in tt_columns:
            cursor.execute("ALTER TABLE timetable ADD COLUMN room_id INTEGER REFERENCES rooms(id)")
    except Exception as e:
        print(f"Error auto-migrating timetable table: {e}")

    # Auto-migrate lecture_sessions table for substitute/extra class tracking
    try:
        cursor.execute("PRAGMA table_info(lecture_sessions)")
        ls_columns = [col[1] for col in cursor.fetchall()]
        if 'session_type' not in ls_columns:
            cursor.execute("ALTER TABLE lecture_sessions ADD COLUMN session_type TEXT DEFAULT 'regular'")
        if 'original_faculty_id' not in ls_columns:
            cursor.execute("ALTER TABLE lecture_sessions ADD COLUMN original_faculty_id TEXT DEFAULT ''")
        if 'original_faculty_name' not in ls_columns:
            cursor.execute("ALTER TABLE lecture_sessions ADD COLUMN original_faculty_name TEXT DEFAULT ''")
        if 'timetable_id' not in ls_columns:
            cursor.execute("ALTER TABLE lecture_sessions ADD COLUMN timetable_id INTEGER")
        if 'faculty_id' not in ls_columns:
            cursor.execute("ALTER TABLE lecture_sessions ADD COLUMN faculty_id TEXT DEFAULT ''")
    except Exception as e:
        print(f"Error auto-migrating lecture_sessions table: {e}")

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
    import hashlib as _hl
    cursor.execute("SELECT COUNT(*) FROM faculty")
    if cursor.fetchone()[0] == 0:
        def _hp(p): return _hl.sha256(p.encode()).hexdigest()
        default_faculty = [
            ('EMP001', _hp('EMP001@123'), 'Prof. Rajesh Sharma', 'EMP001', 'CSE', 'HOD', 'sharma@college.edu', '9876543210'),
            ('EMP002', _hp('EMP002@123'), 'Prof. Neha Verma',   'EMP002', 'CSE', 'Assistant Professor', 'verma@college.edu', '9876543211'),
            ('EMP003', _hp('EMP003@123'), 'Prof. Amit Gupta',   'EMP003', 'CSE-AI', 'Assistant Professor', 'gupta@college.edu', '9876543212'),
        ]
        cursor.executemany(
            "INSERT INTO faculty (username, password, full_name, employee_id, department, designation, email, phone) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            default_faculty
        )

    # Seed default admin if missing
    import hashlib
    cursor.execute("SELECT COUNT(*) FROM admins")
    if cursor.fetchone()[0] == 0:
        admin_pass = hashlib.sha256("admin123".encode()).hexdigest()
        cursor.execute("INSERT INTO admins (username, password) VALUES (?, ?)", ("admin", admin_pass))

    # Seed sample timetable if empty
    cursor.execute("SELECT COUNT(*) FROM timetable")
    if cursor.fetchone()[0] == 0:
        _PERIOD_TIMES = {
            'Period 1': ('09:00', '09:50'), 'Period 2': ('10:00', '10:50'),
            'Period 3': ('11:00', '11:50'), 'Period 4': ('12:00', '12:50'),
            'Period 5': ('14:00', '14:50'), 'Period 6': ('15:00', '15:50'),
            'Period 7': ('16:00', '16:50'), 'Period 8': ('17:00', '17:50'),
        }
        sample_timetable = [
            # EMP001 — Prof. Rajesh Sharma
            ('EMP001', 'Prof. Rajesh Sharma', 'CS501', 'Data Structures & Algorithms', 'Monday',    'Period 1', 'CSE', '5th Sem', 'Room 101'),
            ('EMP001', 'Prof. Rajesh Sharma', 'CS502', 'Database Management Systems',  'Monday',    'Period 3', 'CSE', '5th Sem', 'Room 101'),
            ('EMP001', 'Prof. Rajesh Sharma', 'CS501', 'Data Structures & Algorithms', 'Wednesday', 'Period 2', 'CSE', '5th Sem', 'Room 101'),
            ('EMP001', 'Prof. Rajesh Sharma', 'CS502', 'Database Management Systems',  'Thursday',  'Period 1', 'CSE', '5th Sem', 'Room 101'),
            ('EMP001', 'Prof. Rajesh Sharma', 'CS501', 'Data Structures & Algorithms', 'Friday',    'Period 4', 'CSE', '5th Sem', 'Room 101'),
            # EMP002 — Prof. Neha Verma
            ('EMP002', 'Prof. Neha Verma', 'CS503', 'Operating Systems',   'Monday',    'Period 2', 'CSE', '5th Sem', 'Room 102'),
            ('EMP002', 'Prof. Neha Verma', 'CS504', 'Computer Networks',   'Tuesday',   'Period 1', 'CSE', '5th Sem', 'Room 102'),
            ('EMP002', 'Prof. Neha Verma', 'CS503', 'Operating Systems',   'Wednesday', 'Period 3', 'CSE', '5th Sem', 'Room 102'),
            ('EMP002', 'Prof. Neha Verma', 'CS504', 'Computer Networks',   'Thursday',  'Period 2', 'CSE', '5th Sem', 'Room 102'),
            ('EMP002', 'Prof. Neha Verma', 'CS503', 'Operating Systems',   'Friday',    'Period 1', 'CSE', '5th Sem', 'Room 102'),
            # EMP003 — Prof. Amit Gupta
            ('EMP003', 'Prof. Amit Gupta', 'CS507', 'Machine Learning',  'Monday',    'Period 5', 'CSE-AI', '7th Sem', 'Lab A1'),
            ('EMP003', 'Prof. Amit Gupta', 'CS508', 'Cloud Computing',   'Tuesday',   'Period 3', 'CSE-AI', '7th Sem', 'Lab A1'),
            ('EMP003', 'Prof. Amit Gupta', 'CS507', 'Machine Learning',  'Wednesday', 'Period 5', 'CSE-AI', '7th Sem', 'Lab A1'),
            ('EMP003', 'Prof. Amit Gupta', 'CS508', 'Cloud Computing',   'Thursday',  'Period 3', 'CSE-AI', '7th Sem', 'Lab A1'),
            ('EMP003', 'Prof. Amit Gupta', 'CS507', 'Machine Learning',  'Friday',    'Period 5', 'CSE-AI', '7th Sem', 'Lab A1'),
        ]
        for entry in sample_timetable:
            fid, fname, scode, sname, day, period, branch, sem, room = entry
            st, et = _PERIOD_TIMES.get(period, ('', ''))
            try:
                cursor.execute(
                    """INSERT INTO timetable
                       (faculty_id, faculty_name, subject_code, subject_name,
                        day_of_week, period, start_time, end_time, branch, semester, room)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (fid, fname, scode, sname, day, period, st, et, branch, sem, room)
                )
            except Exception:
                pass  # skip duplicates on re-run

    conn.commit()
    conn.close()

# ============================================================
# STUDENT OPERATIONS
# ============================================================

def add_student(name, roll_number, department, academic_year, dob=None, username=None, password=None):
    """Add a new student to the database"""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO students (name, roll_number, department, academic_year, dob, username, password) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (name, roll_number, department, academic_year, dob, username, password)
        )
        conn.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError:
        return None # Student name or username already exists
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

def get_student_by_folder_name(folder_name):
    """
    Robust student lookup from a dataset folder name.
    Handles formats like 'Sagar Kumar_21104131014' where the DB stores 'Sagar Kumar'.
    
    Tries in order:
      1. Exact match on name
      2. Strip trailing _RollNumber and match
      3. Partial LIKE match on the base name part
    """
    import re
    
    # Strategy 1: Exact match
    student = get_student_by_name(folder_name)
    if student:
        return student
    
    # Strategy 2: Strip trailing _<digits> (roll number suffix) and match
    # Pattern: "Name_RollNumber" → "Name"
    base_name = re.sub(r'_\d+$', '', folder_name).strip()
    if base_name and base_name != folder_name:
        student = get_student_by_name(base_name)
        if student:
            return student
    
    # Strategy 3: Also try splitting on last underscore for non-numeric suffixes
    if '_' in folder_name:
        parts = folder_name.rsplit('_', 1)
        name_part = parts[0].strip()
        if name_part:
            student = get_student_by_name(name_part)
            if student:
                return student
    
    # Strategy 4: Partial match — folder name LIKE '%name%' or name LIKE '%folder%'
    conn = get_connection()
    cursor = conn.cursor()
    search_term = base_name if base_name != folder_name else folder_name
    cursor.execute(
        "SELECT * FROM students WHERE LOWER(name) LIKE LOWER(?) ORDER BY id ASC LIMIT 1",
        (f"%{search_term}%",)
    )
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    
    return None

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

def create_lecture_session(subject_code, subject_name, period, faculty_name,
                          session_type='regular', faculty_id='',
                          original_faculty_id='', original_faculty_name='',
                          timetable_id=None):
    """Create a new lecture session and return the session ID"""
    conn = get_connection()
    cursor = conn.cursor()
    today = datetime.now().strftime("%Y-%m-%d")
    time_now = datetime.now().strftime("%H:%M:%S")
    try:
        cursor.execute(
            """INSERT INTO lecture_sessions 
               (subject_code, subject_name, period, faculty_name, date, start_time, status,
                session_type, faculty_id, original_faculty_id, original_faculty_name, timetable_id) 
               VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?)""",
            (subject_code, subject_name, period, faculty_name, today, time_now,
             session_type, faculty_id, original_faculty_id, original_faculty_name, timetable_id)
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
    Uses robust folder-name matching to handle 'Name_RollNumber' format.
    Returns:
       'success' if inserted,
       'duplicate' if already marked today for that subject/period,
       'not_found' if student doesnt exist in students table.
    """
    # Use robust lookup that handles folder naming format (Name_RollNumber)
    student = get_student_by_folder_name(name)
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

# ============================================================
# TIMETABLE OPERATIONS
# ============================================================

PERIOD_TIMES = {
    'Period 1': ('09:00', '09:50'), 'Period 2': ('10:00', '10:50'),
    'Period 3': ('11:00', '11:50'), 'Period 4': ('12:00', '12:50'),
    'Period 5': ('14:00', '14:50'), 'Period 6': ('15:00', '15:50'),
    'Period 7': ('16:00', '16:50'), 'Period 8': ('17:00', '17:50'),
}

def get_timetable_for_faculty(faculty_id):
    """Get all timetable entries for a faculty member"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM timetable WHERE faculty_id = ? AND is_active = 1 ORDER BY "
        "CASE day_of_week "
        "  WHEN 'Monday' THEN 1 WHEN 'Tuesday' THEN 2 WHEN 'Wednesday' THEN 3 "
        "  WHEN 'Thursday' THEN 4 WHEN 'Friday' THEN 5 WHEN 'Saturday' THEN 6 "
        "  ELSE 7 END, start_time ASC",
        (faculty_id,)
    )
    entries = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return entries

def get_current_class(faculty_id):
    """
    Check the current day and time against the timetable.
    Returns the matching timetable entry dict or None.
    """
    now = datetime.now()
    day_name = now.strftime("%A")  # 'Monday', 'Tuesday', etc.
    current_time = now.strftime("%H:%M")

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """SELECT * FROM timetable
           WHERE faculty_id = ? AND day_of_week = ? AND is_active = 1
           ORDER BY start_time ASC""",
        (faculty_id, day_name)
    )
    entries = [dict(row) for row in cursor.fetchall()]
    conn.close()

    for entry in entries:
        st = entry.get('start_time', '')
        et = entry.get('end_time', '')
        if st and et and st <= current_time <= et:
            return entry

    # Also check if we are within 10 minutes before the next class
    for entry in entries:
        st = entry.get('start_time', '')
        if st:
            try:
                start_dt = datetime.strptime(st, "%H:%M")
                now_dt = datetime.strptime(current_time, "%H:%M")
                diff_min = (start_dt - now_dt).total_seconds() / 60
                if 0 < diff_min <= 10:
                    entry['upcoming'] = True
                    return entry
            except Exception:
                pass

    return None

def get_all_timetable():
    """Get the full timetable (admin view)"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM timetable WHERE is_active = 1 ORDER BY faculty_name, "
        "CASE day_of_week "
        "  WHEN 'Monday' THEN 1 WHEN 'Tuesday' THEN 2 WHEN 'Wednesday' THEN 3 "
        "  WHEN 'Thursday' THEN 4 WHEN 'Friday' THEN 5 WHEN 'Saturday' THEN 6 "
        "  ELSE 7 END, start_time ASC"
    )
    entries = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return entries

def add_timetable_entry(faculty_id, faculty_name, subject_code, subject_name,
                        day_of_week, period, branch='', semester='', room=''):
    """Add a new timetable entry"""
    st, et = PERIOD_TIMES.get(period, ('', ''))
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """INSERT INTO timetable
               (faculty_id, faculty_name, subject_code, subject_name,
                day_of_week, period, start_time, end_time, branch, semester, room)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (faculty_id, faculty_name, subject_code, subject_name,
             day_of_week, period, st, et, branch, semester, room)
        )
        conn.commit()
        return cursor.lastrowid
    except Exception:
        return None
    finally:
        conn.close()

def delete_timetable_entry(entry_id):
    """Delete a timetable entry"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM timetable WHERE id = ?", (entry_id,))
    conn.commit()
    conn.close()

# ============================================================
# SESSION ATTENDANCE DETAILS
# ============================================================

def get_session_attendance(session_id):
    """Get detailed list of students marked present in a specific lecture session"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT a.id, a.date, a.time, a.status, a.subject_code, a.subject_name,
               a.period, a.faculty_name,
               s.name, s.roll_number, s.department, s.academic_year
        FROM attendance a
        JOIN students s ON a.student_id = s.id
        WHERE a.session_id = ?
        ORDER BY s.name ASC
    """, (session_id,))
    records = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return records

def get_total_students_count():
    """Get total number of registered students"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM students")
    count = cursor.fetchone()[0]
    conn.close()
    return count

# ============================================================
# EXTRA CLASS / SUBSTITUTE TRACKING
# ============================================================

def get_faculty_extra_classes_count(faculty_id):
    """Count how many extra/substitute classes this faculty has taken"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT COUNT(*) FROM lecture_sessions WHERE faculty_id = ? AND session_type = 'extra'",
        (faculty_id,)
    )
    count = cursor.fetchone()[0]
    conn.close()
    return count

def get_faculty_missed_classes_count(faculty_id):
    """Count how many of this faculty's classes were taken by a substitute"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT COUNT(*) FROM lecture_sessions WHERE original_faculty_id = ? AND session_type = 'extra'",
        (faculty_id,)
    )
    count = cursor.fetchone()[0]
    conn.close()
    return count

def get_all_extra_classes_summary():
    """Admin view: get all extra/substitute classes with details"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, subject_code, subject_name, period, faculty_name, faculty_id,
               date, start_time, end_time, total_present, status,
               session_type, original_faculty_id, original_faculty_name
        FROM lecture_sessions
        WHERE session_type = 'extra'
        ORDER BY date DESC, start_time DESC
    """)
    sessions = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return sessions

def get_faculty_logbook(faculty_id):
    """Get all past lecture sessions for a faculty (both regular and extra)"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, subject_code, subject_name, period, faculty_name, faculty_id,
               date, start_time, end_time, total_present, status,
               session_type, original_faculty_id, original_faculty_name
        FROM lecture_sessions
        WHERE faculty_id = ? OR faculty_name = ?
        ORDER BY date DESC, start_time DESC
    """, (faculty_id, faculty_id))
    sessions = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return sessions
