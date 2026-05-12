// services/api.js

export const api = {
    getSession: async () => {
      const res = await fetch('/api/session');
      return res.json();
    },
    
    // Auth
    login: async (formData) => {
      const res = await fetch('/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams(formData).toString(),
      });
      return { success: res.ok, redirected: res.redirected };
    },
    logout: async () => {
      await fetch('/logout');
    },
  
    // Dashboard Stats
    getDashboardStats: async () => {
      const res = await fetch('/api/attendance/today');
      return res.json();
    },
    
    // Students
    getStudents: async () => {
      const res = await fetch('/api/students');
      return res.json();
    },
    registerStudent: async (formData) => {
      const res = await fetch('/api/register', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });
      return res.json();
    },
    deleteStudent: async (id) => {
      const res = await fetch(`/delete_student/${id}`, { method: 'POST' });
      return { success: res.ok };
    },
    editStudent: async (id, formData) => {
      const res = await fetch(`/edit_student/${id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams(formData).toString(),
      });
      return { success: res.ok };
    },
  
    // ============================================================
    // SUBJECTS
    // ============================================================
    getSubjects: async () => {
      const res = await fetch('/api/subjects');
      return res.json();
    },
    addSubject: async (data) => {
      const res = await fetch('/api/subjects', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      });
      return res.json();
    },
    deleteSubject: async (id) => {
      const res = await fetch(`/api/subjects/${id}`, { method: 'DELETE' });
      return res.json();
    },

    // ============================================================
    // LECTURE SESSIONS
    // ============================================================
    getLectureSessions: async () => {
      const res = await fetch('/api/lecture-sessions');
      return res.json();
    },

    // ============================================================
    // ATTENDANCE (with Subject & Period)
    // ============================================================
    startAttendanceWithSubject: async ({ subject_code, subject_name, period }) => {
      const res = await fetch('/api/attendance/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ subject_code, subject_name, period }),
      });
      return res.json();
    },

    // Legacy attendance start (backward compatible)
    startAttendance: async () => {
      await fetch('/attendance');
      return { success: true };
    },

    // Actions
    trainModel: async () => {
      // In Flask `/train` is a GET that starts a subprocess and redirects.
      // We will handle it by just fetching the endpoint.
      await fetch('/train');
      return { success: true };
    },
    resetModel: async () => {
      await fetch('/reset');
      return { success: true };
    },
  
    // Logs & Reports
    getLogs: async () => {
      const res = await fetch('/api/logs');
      return res.json();
    },
    exportCSV: (params) => {
      const query = new URLSearchParams(params).toString();
      window.location.href = `/export-csv?${query}`;
    },
    
    // Additional Data
    getModelInfo: async () => {
      const res = await fetch('/api/model-info');
      return res.json();
    },

    // ============================================================
    // FACULTY MANAGEMENT
    // ============================================================
    getFaculty: async () => {
      const res = await fetch('/api/faculty');
      return res.json();
    },
    registerFaculty: async (data) => {
      const res = await fetch('/api/faculty/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      });
      return res.json();
    },
    updateFaculty: async (id, data) => {
      const res = await fetch(`/api/faculty/${id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      });
      return res.json();
    },
    deleteFaculty: async (id) => {
      const res = await fetch(`/api/faculty/${id}`, { method: 'DELETE' });
      return res.json();
    },

    // ============================================================
    // FACULTY ATTENDANCE
    // ============================================================
    getFacultyAttendanceToday: async () => {
      const res = await fetch('/api/faculty-attendance/today');
      return res.json();
    },
    getFacultyAttendanceHistory: async (params = {}) => {
      const query = new URLSearchParams(params).toString();
      const res = await fetch(`/api/faculty-attendance/history?${query}`);
      return res.json();
    },
    markFacultyAttendance: async (data) => {
      const res = await fetch('/api/faculty-attendance/mark', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      });
      return res.json();
    },
    checkoutFaculty: async (data) => {
      const res = await fetch('/api/faculty-attendance/checkout', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      });
      return res.json();
    },
    exportFacultyCSV: (params = {}) => {
      const query = new URLSearchParams(params).toString();
      window.location.href = `/api/faculty-attendance/export-csv?${query}`;
    },

    // ============================================================
    // TIMETABLE
    // ============================================================
    getTimetable: async (facultyId) => {
      const res = await fetch(`http://localhost:8000/api/timetable?faculty_id=${facultyId}`);
      return res.json();
    },
    getCurrentClass: async (facultyId) => {
      const res = await fetch(`http://localhost:8000/api/timetable/current-class?faculty_id=${facultyId}`);
      return res.json();
    },
    getAllTimetable: async () => {
      const res = await fetch('http://localhost:8000/api/timetable/all');
      return res.json();
    },
    addTimetableEntry: async (data) => {
      const res = await fetch('http://localhost:8000/api/timetable', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      });
      return res.json();
    },
    deleteTimetableEntry: async (id) => {
      const res = await fetch(`http://localhost:8000/api/timetable/${id}`, { method: 'DELETE' });
      return res.json();
    },

    // ============================================================
    // EXTRA CLASS / SUBSTITUTE
    // ============================================================
    takeExtraClass: async (data) => {
      const res = await fetch('http://localhost:8000/api/take-extra-class', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      });
      return res.json();
    },
    getFacultyExtraStats: async (facultyId) => {
      const res = await fetch(`http://localhost:8000/api/faculty-extra-stats?faculty_id=${facultyId}`);
      return res.json();
    },
    getAdminExtraClassesSummary: async () => {
      const res = await fetch('http://localhost:8000/api/admin/extra-classes-summary');
      return res.json();
    },

    // ============================================================
    // SESSION ATTENDANCE DETAIL & LOGBOOK
    // ============================================================
    getSessionAttendance: async (sessionId) => {
      const res = await fetch(`http://localhost:8000/api/session-attendance/${sessionId}`);
      return res.json();
    },
    getFacultyLogbook: async (facultyId) => {
      const res = await fetch(`http://localhost:8000/api/faculty-logbook?faculty_id=${facultyId}`);
      return res.json();
    },
  };
