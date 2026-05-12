import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

// Shared Components
const Toast = ({ message, type, onClose }) => {
  useEffect(() => {
    const timer = setTimeout(onClose, 3500);
    return () => clearTimeout(timer);
  }, [onClose]);

  const isSuccess = type === 'success';
  return (
    <div style={{
      position: 'fixed',
      bottom: '24px',
      right: '24px',
      zIndex: 999,
      backgroundColor: isSuccess ? 'rgba(245,158,11,0.1)' : 'rgba(239,68,68,0.08)',
      border: isSuccess ? '0.5px solid rgba(245,158,11,0.25)' : '1px solid #ef4444',
      color: isSuccess ? '#fbbf24' : '#fca5a5',
      padding: '12px 16px',
      borderRadius: '8px',
      display: 'flex',
      alignItems: 'center',
      gap: '8px',
      fontSize: '13px',
      boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
    }}>
      {isSuccess ? (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="20 6 9 17 4 12"></polyline></svg>
      ) : (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
      )}
      {message}
    </div>
  );
};

export default function AdminDashboard() {
  const [activeView, setActiveView] = useState('Dashboard');
  const [toast, setToast] = useState(null);
  const navigate = useNavigate();

  const handleLogout = () => {
    localStorage.removeItem("adminToken");
    navigate("/admin/login");
  };

  const showToast = (message, type = 'success') => {
    setToast({ message, type });
  };

  // --- Views ---
  const renderView = () => {
    switch (activeView) {
      case 'Dashboard': return <DashboardView setActiveView={setActiveView} />;
      case 'Manage Students': return <ManageStudentsView showToast={showToast} />;
      case 'Manage Faculty': return <ManageFacultyView showToast={showToast} />;
      case 'Classroom Setup': return <ClassroomSetupView showToast={showToast} />;
      case 'Assign Subjects': return <AssignSubjectsView showToast={showToast} />;
      case 'View Mappings': return <ViewMappingsView />;
      case 'Timetable': return <TimetableView showToast={showToast} />;
      case 'Rooms': return <RoomsView showToast={showToast} />;
      case 'Attendance Tracker': return <AttendanceTrackerView showToast={showToast} />;
      case 'Extra Classes': return <ExtraClassesView />;
      default: return <DashboardView setActiveView={setActiveView} />;
    }
  };

  return (
    <div style={{ display: 'flex', height: '100vh', backgroundColor: '#070b14', fontFamily: '"Segoe UI", system-ui, sans-serif' }}>
      {/* SIDEBAR */}
      <div style={{
        position: 'sticky',
        top: 0,
        height: '100vh',
        width: '215px',
        backgroundColor: '#0d1117',
        borderRight: '0.5px solid rgba(239,68,68,0.1)',
        display: 'flex',
        flexDirection: 'column',
        boxSizing: 'border-box'
      }}>
        {/* TOP section */}
        <div style={{ padding: '20px 16px 16px', borderBottom: '0.5px solid rgba(255,255,255,0.05)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div style={{ width: '34px', height: '34px', backgroundColor: '#ef4444', borderRadius: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>
            </div>
            <div>
              <div style={{ fontSize: '13px', fontWeight: '700', color: '#f1f5f9' }}>SmartAttend AI</div>
              <div style={{ fontSize: '10px', color: '#fca5a5', backgroundColor: 'rgba(239,68,68,0.1)', padding: '2px 6px', borderRadius: '4px', display: 'inline-block', marginTop: '2px' }}>Admin Panel</div>
            </div>
          </div>
        </div>

        {/* NAV section */}
        <div style={{ flex: 1, padding: '12px 0' }}>
          <div style={{ fontSize: '9px', letterSpacing: '1px', color: '#334155', padding: '0 16px', marginBottom: '8px', fontWeight: '600' }}>NAVIGATION</div>
          
          {[
            { name: 'Dashboard', icon: <><rect x="3" y="3" width="7" height="7"></rect><rect x="14" y="3" width="7" height="7"></rect><rect x="14" y="14" width="7" height="7"></rect><rect x="3" y="14" width="7" height="7"></rect></> },
            { name: 'Manage Students', icon: <><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle><path d="M23 21v-2a4 4 0 0 0-3-3.87"></path><path d="M16 3.13a4 4 0 0 1 0 7.75"></path></> },
            { name: 'Manage Faculty', icon: <><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"></path><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"></path></> },
            { name: 'Classroom Setup', icon: <><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path><circle cx="12" cy="13" r="4"></circle></> }
          ].map(btn => {
            const isActive = activeView === btn.name;
            return (
              <button key={btn.name} onClick={() => setActiveView(btn.name)} style={{
                display: 'flex', alignItems: 'center', gap: '9px',
                padding: '9px 16px', fontSize: '12px', fontWeight: '500',
                cursor: 'pointer', border: 'none',
                borderLeft: `2.5px solid ${isActive ? '#ef4444' : 'transparent'}`,
                backgroundColor: isActive ? 'rgba(239,68,68,0.08)' : 'transparent',
                color: isActive ? '#fca5a5' : '#64748b',
                width: '100%', textAlign: 'left', transition: 'all 0.15s'
              }}
              onMouseEnter={(e) => !isActive && (e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.03)')}
              onMouseLeave={(e) => !isActive && (e.currentTarget.style.backgroundColor = 'transparent')}
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill={isActive ? '#ef4444' : 'none'} stroke="currentColor" strokeWidth="2">{btn.icon}</svg>
                {btn.name}
              </button>
            )
          })}

          <div style={{ fontSize: '9px', color: '#334155', letterSpacing: '0.8px', fontWeight: '600', padding: '16px 14px 5px', borderTop: '0.5px solid rgba(255,255,255,0.04)', marginTop: '8px' }}>
            SUBJECT MANAGEMENT
          </div>
          
          {[
            { name: 'Assign Subjects', isNew: true, icon: <><line x1="8" y1="6" x2="21" y2="6"></line><line x1="8" y1="12" x2="21" y2="12"></line><line x1="8" y1="18" x2="21" y2="18"></line><line x1="3" y1="6" x2="3.01" y2="6"></line><line x1="3" y1="12" x2="3.01" y2="12"></line><line x1="3" y1="18" x2="3.01" y2="18"></line></> },
            { name: 'View Mappings', icon: <><rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect><line x1="16" y1="2" x2="16" y2="6"></line><line x1="8" y1="2" x2="8" y2="6"></line><line x1="3" y1="10" x2="21" y2="10"></line></> }
          ].map(btn => {
            const isActive = activeView === btn.name;
            return (
              <button key={btn.name} onClick={() => setActiveView(btn.name)} style={{
                display: 'flex', alignItems: 'center', gap: '9px',
                padding: '9px 16px', fontSize: '12px', fontWeight: '500',
                cursor: 'pointer', border: 'none',
                borderLeft: `2.5px solid ${isActive ? '#ef4444' : 'transparent'}`,
                backgroundColor: isActive ? 'rgba(239,68,68,0.08)' : 'transparent',
                color: isActive ? '#fca5a5' : '#64748b',
                width: '100%', textAlign: 'left', transition: 'all 0.15s'
              }}
              onMouseEnter={(e) => !isActive && (e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.03)')}
              onMouseLeave={(e) => !isActive && (e.currentTarget.style.backgroundColor = 'transparent')}
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill={isActive ? '#ef4444' : 'none'} stroke="currentColor" strokeWidth="2">{btn.icon}</svg>
                {btn.name}
                {btn.isNew && (
                  <span style={{
                    marginLeft: 'auto', backgroundColor: 'rgba(245,158,11,0.15)', color: '#fbbf24', fontSize: '9px', padding: '1px 6px', borderRadius: '4px'
                  }}>NEW</span>
                )}
              </button>
            )
          })}

          <div style={{ fontSize: '9px', color: '#334155', letterSpacing: '0.8px', fontWeight: '600', padding: '16px 14px 5px', borderTop: '0.5px solid rgba(255,255,255,0.04)', marginTop: '8px' }}>
            SCHEDULE & AUDIT
          </div>
          {[
            { name: 'Timetable', isNew: true, icon: <><rect x="3" y="4" width="18" height="18" rx="2"></rect><line x1="16" y1="2" x2="16" y2="6"></line><line x1="8" y1="2" x2="8" y2="6"></line><line x1="3" y1="10" x2="21" y2="10"></line></> },
            { name: 'Rooms', isNew: true, icon: <><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path><polyline points="9 22 9 12 15 12 15 22"></polyline></> },
            { name: 'Attendance Tracker', isNew: true, icon: <><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></> },
            { name: 'Extra Classes', isNew: true, icon: <><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="16"></line><line x1="8" y1="12" x2="16" y2="12"></line></> },
          ].map(btn => {
            const isActive = activeView === btn.name;
            return (
              <button key={btn.name} onClick={() => setActiveView(btn.name)} style={{
                display: 'flex', alignItems: 'center', gap: '9px',
                padding: '9px 16px', fontSize: '12px', fontWeight: '500',
                cursor: 'pointer', border: 'none',
                borderLeft: `2.5px solid ${isActive ? '#ef4444' : 'transparent'}`,
                backgroundColor: isActive ? 'rgba(239,68,68,0.08)' : 'transparent',
                color: isActive ? '#fca5a5' : '#64748b',
                width: '100%', textAlign: 'left', transition: 'all 0.15s'
              }}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill={isActive ? '#ef4444' : 'none'} stroke="currentColor" strokeWidth="2">{btn.icon}</svg>
                {btn.name}
                {btn.isNew && <span style={{ marginLeft:'auto', backgroundColor:'rgba(168,85,247,0.15)', color:'#c084fc', fontSize:'9px', padding:'1px 6px', borderRadius:'4px' }}>NEW</span>}
              </button>
            );
          })}
        </div>

        {/* BOTTOM user bar */}
        <div style={{ marginTop: 'auto', borderTop: '0.5px solid rgba(255,255,255,0.05)', padding: '14px 16px', display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{ width: '30px', height: '30px', backgroundColor: 'rgba(239,68,68,0.15)', border: '0.5px solid rgba(239,68,68,0.3)', borderRadius: '50%', color: '#fca5a5', fontSize: '12px', fontWeight: '700', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>A</div>
          <div>
            <div style={{ fontSize: '12px', color: '#94a3b8', fontWeight: '600' }}>Administrator</div>
            <div style={{ fontSize: '10px', color: '#334155' }}>Full Access</div>
          </div>
          <svg onClick={handleLogout} style={{ marginLeft: 'auto', cursor: 'pointer' }} width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#ef4444" strokeWidth="2"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"></path><polyline points="16 17 21 12 16 7"></polyline><line x1="21" y1="12" x2="9" y2="12"></line></svg>
        </div>
      </div>

      {/* RIGHT COLUMN */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        {/* TOPBAR */}
        <div style={{ backgroundColor: '#0d1117', borderBottom: '0.5px solid rgba(255,255,255,0.04)', padding: '14px 28px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', height: '52px', boxSizing: 'border-box' }}>
          <div style={{ fontSize: '13px', fontWeight: '600', color: '#94a3b8' }}>{activeView}</div>
          <div style={{ fontSize: '11px', color: '#86efac', backgroundColor: 'rgba(34,197,94,0.1)', border: '0.5px solid rgba(34,197,94,0.2)', padding: '4px 10px', borderRadius: '12px', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <div style={{ width: '6px', height: '6px', backgroundColor: '#22c55e', borderRadius: '50%' }}></div>
            System Active
          </div>
        </div>

        {/* MAIN CONTENT */}
        <div style={{ flex: 1, padding: '28px 28px', overflowY: 'auto' }}>
          {renderView()}
        </div>
      </div>
      {toast && <Toast message={toast.message} type={toast.type} onClose={() => setToast(null)} />}
    </div>
  );
}

// --- View Components ---

function DashboardView({ setActiveView }) {
  const [stats, setStats] = useState({ students: 0, faculty: 0, classrooms: 0, subjects_count: 0 });
  const [recentAssignments, setRecentAssignments] = useState([]);

  useEffect(() => {
    axios.get('http://localhost:8000/api/admin/stats').then(res => {
      setStats({
        students: res.data.students || 0,
        classrooms: res.data.classrooms || 0,
        faculty: res.data.faculty || 0,
        subjects_count: res.data.subjects_count || 0
      });
    }).catch(console.error);

    axios.get('http://localhost:8000/api/assignments?limit=3').then(res => {
      setRecentAssignments(res.data);
    }).catch(console.error);
  }, []);

  return (
    <div>
      <h1 style={{ fontSize: '19px', fontWeight: '800', color: '#f1f5f9', letterSpacing: '-0.5px', margin: 0 }}>Dashboard Overview</h1>
      <p style={{ fontSize: '12px', color: '#475569', marginTop: '4px', marginBottom: '20px' }}>System metrics and quick actions</p>
      
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '10px', marginBottom: '20px' }}>
        {[
          { label: 'Total Students', value: stats.students, sub: 'Registered faces', color: '#818cf8' },
          { label: 'Classrooms', value: stats.classrooms, sub: 'Camera feeds live', color: '#22c55e' },
          { label: 'Faculty Members', value: stats.faculty, sub: 'Active accounts', color: '#60a5fa' },
          { label: 'Subjects Mapped', value: stats.subjects_count, sub: 'Assignments', color: '#f59e0b' }
        ].map(card => (
          <div key={card.label} style={{ backgroundColor: '#0d1117', border: '0.5px solid rgba(255,255,255,0.05)', borderRadius: '10px', padding: '15px 16px' }}>
            <div style={{ fontSize: '10px', color: '#475569' }}>{card.label}</div>
            <div style={{ fontSize: '24px', fontWeight: '800', color: '#f1f5f9', margin: '4px 0' }}>{card.value}</div>
            <div style={{ fontSize: '10px', color: card.color }}>{card.sub}</div>
          </div>
        ))}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
        <div style={{ backgroundColor: '#0d1117', border: '0.5px solid rgba(255,255,255,0.05)', borderRadius: '10px', padding: '16px 18px' }}>
          <div style={{ fontSize: '12px', fontWeight: '600', color: '#64748b', marginBottom: '12px' }}>Quick Actions</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px' }}>
            <button onClick={() => setActiveView('Manage Students')} style={{ flex: 1, minWidth: '120px', backgroundColor: 'rgba(129,140,248,0.1)', color: '#818cf8', border: '1px solid rgba(129,140,248,0.2)', borderRadius: '8px', padding: '10px', fontSize: '12px', cursor: 'pointer', fontWeight: '600' }}>+ Register Student</button>
            <button onClick={() => setActiveView('Manage Faculty')} style={{ flex: 1, minWidth: '120px', backgroundColor: 'rgba(34,197,94,0.1)', color: '#22c55e', border: '1px solid rgba(34,197,94,0.2)', borderRadius: '8px', padding: '10px', fontSize: '12px', cursor: 'pointer', fontWeight: '600' }}>+ Add Faculty</button>
            <button onClick={() => setActiveView('Classroom Setup')} style={{ flex: 1, minWidth: '120px', backgroundColor: 'rgba(96,165,250,0.1)', color: '#60a5fa', border: '1px solid rgba(96,165,250,0.2)', borderRadius: '8px', padding: '10px', fontSize: '12px', cursor: 'pointer', fontWeight: '600' }}>+ Add Classroom</button>
          </div>
        </div>

        {/* Recent Subject Assignments */}
        <div style={{ backgroundColor: '#0d1117', border: '0.5px solid rgba(255,255,255,0.05)', borderRadius: '10px', padding: '16px 18px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" strokeWidth="2"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path></svg>
            <span style={{ fontSize: '12px', fontWeight: '600', color: '#64748b' }}>Recent Subject Assignments</span>
          </div>
          <div>
            {recentAssignments.length === 0 ? (
               <div style={{ fontSize: '11px', color: '#475569', marginTop: '10px' }}>No assignments yet.</div>
            ) : (
               recentAssignments.map((assignment, i) => (
                <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '10px', padding: '8px 0', borderBottom: i !== recentAssignments.length - 1 ? '0.5px solid rgba(255,255,255,0.03)' : 'none' }}>
                  <div style={{ width: '28px', height: '28px', borderRadius: '50%', backgroundColor: 'rgba(245,158,11,0.08)', color: '#fbbf24', fontSize: '11px', fontWeight: '700', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    {assignment.teacher_name.charAt(0).toUpperCase()}
                  </div>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: '12px', color: '#e2e8f0', fontWeight: '500' }}>{assignment.teacher_name}</div>
                    <div style={{ fontSize: '10px', color: '#475569', marginTop: '1px' }}>{assignment.subject_name} · {assignment.branch} Sem {assignment.semester}</div>
                  </div>
                  <div style={{ backgroundColor: 'rgba(245,158,11,0.08)', color: '#fbbf24', fontSize: '9px', padding: '2px 7px', borderRadius: '8px' }}>
                    Assigned
                  </div>
                </div>
               ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function AssignSubjectsView({ showToast }) {
  const [faculty, setFaculty] = useState([]);
  const [assignments, setAssignments] = useState([]);
  const [loading, setLoading] = useState(false);
  const [form, setForm] = useState({ teacher_id: '', teacher_name: '', branch: '', semester: '', subject_name: '', subject_code: '' });

  const fetchData = () => {
    axios.get('http://localhost:8000/api/assignments').then(res => setAssignments(res.data)).catch(console.error);
    axios.get('http://localhost:8000/api/admin/faculty').then(res => setFaculty(res.data)).catch(console.error);
  };
  useEffect(() => { fetchData(); }, []);

  const handleTeacherChange = (e) => {
    const t_id = e.target.value;
    if(!t_id) {
       setForm({...form, teacher_id: '', teacher_name: ''});
       return;
    }
    const t_name = e.target.options[e.target.selectedIndex].text.split(" (")[0];
    setForm({...form, teacher_id: t_id, teacher_name: t_name});
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const res = await axios.post('http://localhost:8000/api/assign-subject', form);
      if (res.data.success) {
        showToast(res.data.message);
        setForm({ ...form, subject_name: '', subject_code: '' }); // keep teacher/branch selected
        fetchData();
      } else {
        showToast(res.data.message || 'Error assigning subject', 'error');
      }
    } catch (err) {
      showToast(err.response?.data?.message || 'Server error', 'error');
    } finally { setLoading(false); }
  };

  const handleDelete = async (id) => {
    try {
      const res = await axios.delete(`http://localhost:8000/api/assignment/${id}`);
      if (res.data.success) {
        fetchData();
      }
    } catch (err) { console.error(err); }
  };

  const inputStyle = { width: '100%', backgroundColor: '#0d1117', border: '0.5px solid rgba(255,255,255,0.07)', borderRadius: '6px', padding: '8px 10px', color: '#94a3b8', fontSize: '12px', outline: 'none', boxSizing: 'border-box', marginBottom: '12px', cursor: 'pointer' };

  return (
    <div>
      <h1 style={{ fontSize: '18px', fontWeight: '800', color: '#f1f5f9', letterSpacing: '-0.5px', margin: 0 }}>Assign Subjects</h1>
      <p style={{ fontSize: '12px', color: '#475569', marginTop: '4px', marginBottom: '20px' }}>Map subjects to faculty members by branch and semester.</p>
      
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px' }}>
        {/* LEFT CARD */}
        <div style={{ backgroundColor: '#0d1117', border: '0.5px solid rgba(255,255,255,0.05)', borderRadius: '11px', padding: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', paddingBottom: '13px', borderBottom: '0.5px solid rgba(255,255,255,0.05)', marginBottom: '14px' }}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" strokeWidth="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
            <span style={{ fontSize: '12px', fontWeight: '600', color: '#94a3b8' }}>New Subject Assignment</span>
          </div>
          <form onSubmit={handleSubmit}>
            <select style={inputStyle} value={form.teacher_id} onChange={handleTeacherChange} onFocus={e => e.target.style.borderColor = '#f59e0b'} onBlur={e => e.target.style.borderColor = 'rgba(255,255,255,0.07)'} required>
              <option value="" style={{backgroundColor: '#0d1117', color: 'white'}}>-- Select Faculty --</option>
              {faculty.map(f => (
                <option key={f.id} value={f.employee_id || f.id} style={{backgroundColor: '#0d1117', color: 'white'}}>{f.name} ({f.employee_id})</option>
              ))}
            </select>
            
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
              <select style={inputStyle} value={form.branch} onChange={e => setForm({...form, branch: e.target.value})} onFocus={e => e.target.style.borderColor = '#f59e0b'} onBlur={e => e.target.style.borderColor = 'rgba(255,255,255,0.07)'} required>
                <option value="" style={{backgroundColor: '#0d1117', color: 'white'}}>-- Branch --</option>
                {['CSE-AI', 'CSE', 'IT', 'ECE', 'ME', 'EE'].map(b => <option key={b} value={b} style={{backgroundColor: '#0d1117', color: 'white'}}>{b}</option>)}
              </select>
              <select style={inputStyle} value={form.semester} onChange={e => setForm({...form, semester: e.target.value})} onFocus={e => e.target.style.borderColor = '#f59e0b'} onBlur={e => e.target.style.borderColor = 'rgba(255,255,255,0.07)'} required>
                <option value="" style={{backgroundColor: '#0d1117', color: 'white'}}>-- Semester --</option>
                {['1st Sem', '2nd Sem', '3rd Sem', '4th Sem', '5th Sem', '6th Sem', '7th Sem', '8th Sem'].map(s => <option key={s} value={s} style={{backgroundColor: '#0d1117', color: 'white'}}>{s}</option>)}
              </select>
            </div>

            <input style={{...inputStyle, cursor: 'text'}} placeholder="Subject Name (e.g. Machine Learning)" value={form.subject_name} onChange={e => setForm({...form, subject_name: e.target.value})} onFocus={e => e.target.style.borderColor = '#f59e0b'} onBlur={e => e.target.style.borderColor = 'rgba(255,255,255,0.07)'} required />
            <input style={{...inputStyle, cursor: 'text'}} placeholder="Subject Code (optional) e.g. CS801" value={form.subject_code} onChange={e => setForm({...form, subject_code: e.target.value})} onFocus={e => e.target.style.borderColor = '#f59e0b'} onBlur={e => e.target.style.borderColor = 'rgba(255,255,255,0.07)'} />
            
            <div style={{ backgroundColor: 'rgba(245,158,11,0.06)', border: '0.5px solid rgba(245,158,11,0.15)', borderRadius: '6px', padding: '8px 10px', fontSize: '11px', color: '#78716c', display: 'flex', alignItems: 'center', gap: '7px', marginBottom: '16px' }}>
              <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" strokeWidth="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
              Teacher will see this subject in their Faculty Portal dropdown
            </div>

            <button type="submit" disabled={loading} style={{ width: '100%', backgroundColor: '#b45309', color: 'white', border: 'none', borderRadius: '7px', padding: '10px', fontSize: '12px', fontWeight: '700', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '7px', cursor: 'pointer', opacity: loading ? 0.7 : 1 }}>
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2"><polyline points="20 6 9 17 4 12"></polyline></svg>
              {loading ? 'Saving...' : 'Save Assignment'}
            </button>
          </form>
        </div>

        {/* RIGHT CARD */}
        <div style={{ backgroundColor: '#0d1117', border: '0.5px solid rgba(255,255,255,0.05)', borderRadius: '11px', padding: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', paddingBottom: '13px', borderBottom: '0.5px solid rgba(255,255,255,0.05)', marginBottom: '14px' }}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" strokeWidth="2"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect><line x1="16" y1="2" x2="16" y2="6"></line><line x1="8" y1="2" x2="8" y2="6"></line><line x1="3" y1="10" x2="21" y2="10"></line></svg>
            <span style={{ fontSize: '12px', fontWeight: '600', color: '#94a3b8' }}>Current Assignments ({assignments.length})</span>
          </div>
          <div style={{ maxHeight: '380px', overflowY: 'auto' }}>
            {assignments.map((a, i) => (
              <div key={a.id || i} style={{ display: 'flex', alignItems: 'center', gap: '9px', padding: '9px 0', borderBottom: '0.5px solid rgba(255,255,255,0.03)' }}>
                <div style={{ width: '28px', height: '28px', borderRadius: '50%', backgroundColor: 'rgba(245,158,11,0.08)', color: '#fbbf24', fontSize: '11px', fontWeight: '700', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  {a.teacher_name.charAt(0).toUpperCase()}
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: '12px', color: '#e2e8f0', fontWeight: '500' }}>{a.teacher_name}</div>
                  <div style={{ fontSize: '10px', color: '#475569', marginTop: '1px' }}>{a.subject_name} · {a.branch} {a.semester} {a.subject_code ? `· ${a.subject_code}` : ''}</div>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '3px' }}>
                  <div style={{ backgroundColor: 'rgba(245,158,11,0.08)', color: '#fbbf24', border: '0.5px solid rgba(245,158,11,0.2)', fontSize: '9px', padding: '2px 7px', borderRadius: '8px' }}>Active</div>
                  <div onClick={() => handleDelete(a.id)} style={{ fontSize: '9px', color: '#ef4444', cursor: 'pointer' }}>Remove</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function ViewMappingsView() {
  const [assignments, setAssignments] = useState([]);
  const [filter, setFilter] = useState('All Faculty');

  useEffect(() => {
    axios.get('http://localhost:8000/api/assignments').then(res => setAssignments(res.data)).catch(console.error);
  }, []);

  const grouped = assignments.reduce((acc, curr) => {
    if (!acc[curr.teacher_name]) acc[curr.teacher_name] = [];
    acc[curr.teacher_name].push(curr);
    return acc;
  }, {});

  const displayedKeys = filter === 'All Faculty' ? Object.keys(grouped) : (grouped[filter] ? [filter] : []);

  return (
    <div>
      <h1 style={{ fontSize: '18px', fontWeight: '800', color: '#f1f5f9', letterSpacing: '-0.5px', margin: 0 }}>All Subject Mappings</h1>
      <p style={{ fontSize: '12px', color: '#475569', marginTop: '4px', marginBottom: '20px' }}>Complete overview of all teacher-subject assignments.</p>
      
      <div style={{ backgroundColor: '#0d1117', border: '0.5px solid rgba(255,255,255,0.05)', borderRadius: '11px', padding: '20px', width: '100%' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', paddingBottom: '13px', borderBottom: '0.5px solid rgba(255,255,255,0.05)', marginBottom: '14px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" strokeWidth="2"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect><line x1="16" y1="2" x2="16" y2="6"></line><line x1="8" y1="2" x2="8" y2="6"></line><line x1="3" y1="10" x2="21" y2="10"></line></svg>
            <span style={{ fontSize: '12px', fontWeight: '600', color: '#94a3b8' }}>All Assignments</span>
          </div>
          <select value={filter} onChange={e => setFilter(e.target.value)} style={{ backgroundColor: '#0d1117', border: '0.5px solid rgba(255,255,255,0.1)', color: '#f1f5f9', fontSize: '11px', padding: '4px 8px', borderRadius: '4px', outline: 'none', cursor: 'pointer' }}>
            <option value="All Faculty" style={{backgroundColor:'#0d1117', color:'#f1f5f9'}}>All Faculty</option>
            {Object.keys(grouped).map(name => <option key={name} value={name} style={{backgroundColor:'#0d1117', color:'#f1f5f9'}}>{name}</option>)}
          </select>
        </div>

        <div>
          {displayedKeys.length === 0 ? <div style={{ fontSize: '12px', color: '#475569' }}>No assignments found.</div> : null}
          {displayedKeys.map(teacherName => {
            if (!grouped[teacherName]) return null;
            return (
              <div key={teacherName} style={{ backgroundColor: 'rgba(245,158,11,0.03)', border: '0.5px solid rgba(245,158,11,0.1)', borderRadius: '6px', padding: '10px', marginBottom: '6px', display: 'flex', alignItems: 'center' }}>
                <div style={{ width: '200px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <div style={{ width: '32px', height: '32px', borderRadius: '50%', backgroundColor: 'rgba(245,158,11,0.08)', color: '#fbbf24', fontSize: '12px', fontWeight: '700', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    {teacherName.charAt(0).toUpperCase()}
                  </div>
                  <div style={{ fontSize: '12px', color: '#e2e8f0', fontWeight: '500' }}>{teacherName}</div>
                </div>
                
                <div style={{ flex: 1, display: 'flex', flexWrap: 'wrap', gap: '5px', marginTop: '5px' }}>
                  {grouped[teacherName].map(sub => (
                    <div key={sub.id} style={{ display: 'inline-flex', backgroundColor: 'rgba(245,158,11,0.08)', border: '0.5px solid rgba(245,158,11,0.2)', borderRadius: '6px', padding: '4px 9px', fontSize: '11px', color: '#fbbf24' }}>
                      {sub.subject_name} {sub.subject_code && `(${sub.subject_code})`} - {sub.branch} {sub.semester}
                    </div>
                  ))}
                </div>

                <div style={{ width: '100px', textAlign: 'right', fontSize: '11px', color: '#475569' }}>
                  {grouped[teacherName].length} subjects
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  );
}

function ManageStudentsView({ showToast }) {
  const [students, setStudents] = useState([]);
  const [form, setForm] = useState({ name: '', roll: '', branch: 'CSE-AI', semester: '1', dob: '', email: '' });
  const [loading, setLoading] = useState(false);
  const [captureStatus, setCaptureStatus] = useState('');

  const fetchStudents = () => {
    axios.get('http://localhost:8000/api/admin/students').then(res => setStudents(res.data)).catch(console.error);
  };
  useEffect(() => { fetchStudents(); }, []);

  const handleDeleteStudent = async (id, name) => {
    if (!window.confirm(`Are you sure you want to delete student ${name}? This will also delete their attendance records.`)) return;
    try {
      const res = await axios.delete(`http://localhost:8000/api/admin/student/${id}`);
      if (res.data.success) {
        showToast(`✅ Student deleted!`);
        fetchStudents();
      } else {
        showToast(res.data.message || 'Error deleting student', 'error');
      }
    } catch (err) {
      showToast(err.response?.data?.message || 'Server error', 'error');
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setCaptureStatus('Camera Open: Please move your head for different angles...');
    try {
      const res = await axios.post('http://localhost:8000/api/admin/register-student-v2', form, { timeout: 40000 });
      if (res.data.success) {
        setCaptureStatus('');
        showToast(`✅ ${res.data.message || 'Student registered & face captured!'}`);
        setForm({ name: '', roll: '', branch: 'CSE-AI', semester: '1', dob: '', email: '' });
        fetchStudents();
      } else {
        setCaptureStatus('');
        showToast(res.data.message || 'Error registering student', 'error');
      }
    } catch (err) {
      setCaptureStatus('');
      showToast(err.response?.data?.message || 'Server error — is camera connected?', 'error');
    } finally { setLoading(false); setCaptureStatus(''); }
  };

  const inputStyle = { width: '100%', backgroundColor: '#0d1117', border: '0.5px solid rgba(255,255,255,0.07)', borderRadius: '6px', padding: '8px 10px', color: '#f1f5f9', fontSize: '12px', outline: 'none', boxSizing: 'border-box' };

  return (
    <div>
      <h1 style={{ fontSize: '19px', fontWeight: '800', color: '#f1f5f9', letterSpacing: '-0.5px', margin: 0 }}>Manage Students</h1>
      <p style={{ fontSize: '12px', color: '#475569', marginTop: '4px', marginBottom: '20px' }}>Register new faces and manage existing records</p>
      
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
        <div style={{ backgroundColor: '#0d1117', border: '0.5px solid rgba(255,255,255,0.05)', borderRadius: '10px', padding: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', paddingBottom: '13px', borderBottom: '0.5px solid rgba(255,255,255,0.05)', marginBottom: '14px' }}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#818cf8" strokeWidth="2"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle></svg>
            <span style={{ fontSize: '12px', fontWeight: '600', color: '#94a3b8' }}>Register New Student</span>
          </div>
          <form onSubmit={handleSubmit} className="space-y-3">
            <div style={{ marginBottom: '12px' }}>
              <input style={inputStyle} className="w-full bg-gray-800 text-white rounded p-2" placeholder="Full Name" value={form.name} onChange={e => setForm({...form, name: e.target.value})} required disabled={loading} />
            </div>
            <div style={{ marginBottom: '12px' }}>
              <input style={inputStyle} className="w-full bg-gray-800 text-white rounded p-2" placeholder="Registration Number" value={form.roll} onChange={e => setForm({...form, roll: e.target.value.toUpperCase()})} required disabled={loading} />
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', marginBottom: '12px' }}>
              <select style={inputStyle} className="w-full bg-gray-800 text-white rounded p-2" value={form.branch} onChange={e => setForm({...form, branch: e.target.value})} required disabled={loading}>
                <option value="CSE-AI" style={{backgroundColor: '#0d1117', color: 'white'}}>CSE-AI</option>
                <option value="CSE" style={{backgroundColor: '#0d1117', color: 'white'}}>CSE</option>
                <option value="ECE" style={{backgroundColor: '#0d1117', color: 'white'}}>ECE</option>
                <option value="Electrical" style={{backgroundColor: '#0d1117', color: 'white'}}>Electrical</option>
                <option value="Civil" style={{backgroundColor: '#0d1117', color: 'white'}}>Civil</option>
                <option value="Mechanical" style={{backgroundColor: '#0d1117', color: 'white'}}>Mechanical</option>
              </select>
              <select style={inputStyle} className="w-full bg-gray-800 text-white rounded p-2" value={form.semester} onChange={e => setForm({...form, semester: e.target.value})} required disabled={loading}>
                {[1, 2, 3, 4, 5, 6, 7, 8].map(sem => (
                  <option key={sem} value={sem} style={{backgroundColor: '#0d1117', color: 'white'}}>{sem}</option>
                ))}
              </select>
            </div>
            <div style={{ marginBottom: '12px' }}>
              <input style={inputStyle} className="w-full bg-gray-800 text-white rounded p-2" type="date" value={form.dob} onChange={e => setForm({...form, dob: e.target.value})} required disabled={loading} />
            </div>
            <div style={{ marginBottom: '16px' }}>
              <input style={inputStyle} className="w-full bg-gray-800 text-white rounded p-2" type="email" placeholder="Student Email (for defaulter alerts)" value={form.email} onChange={e => setForm({...form, email: e.target.value})} disabled={loading} />
            </div>
            <div style={{ backgroundColor: 'rgba(96,165,250,0.06)', border: '0.5px solid rgba(96,165,250,0.14)', borderRadius: '6px', padding: '8px 10px', fontSize: '11px', color: '#64748b', display: 'flex', alignItems: 'center', gap: '7px', marginBottom: '16px' }} className="bg-blue-900/20 border border-blue-500/30 rounded p-2 text-blue-300 text-xs flex items-center gap-2 mb-4">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#60a5fa" strokeWidth="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
              Student Username = Registration No | Password = DOB
            </div>

            
            {/* Camera status message */}
            {captureStatus && (
              <div style={{
                backgroundColor: 'rgba(245,158,11,0.08)', border: '0.5px solid rgba(245,158,11,0.25)',
                borderRadius: '8px', padding: '12px', fontSize: '12px', color: '#fbbf24',
                display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '14px'
              }} className="bg-yellow-900/20 border border-yellow-500/30 rounded p-3 text-yellow-400 text-xs flex items-center gap-3 mb-4">
                <div style={{ width: '10px', height: '10px', backgroundColor: '#ef4444', borderRadius: '50%', flexShrink: 0, animation: 'pulse 1s infinite' }} className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
                <div>
                  <div style={{ fontWeight: '700' }} className="font-bold">{captureStatus}</div>
                </div>
                <style>{`@keyframes pulse { 0% { opacity: 1; transform: scale(1); } 50% { opacity: 0.4; transform: scale(1.3); } 100% { opacity: 1; transform: scale(1); } }`}</style>
              </div>
            )}
            
            <button type="submit" disabled={loading} style={{
              width: '100%', backgroundColor: loading ? '#b45309' : '#818cf8', color: 'white', border: 'none',
              borderRadius: '7px', padding: loading ? '14px 10px' : '10px', fontSize: '12px', fontWeight: '700',
              display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '7px',
              cursor: loading ? 'not-allowed' : 'pointer', opacity: loading ? 0.9 : 1,
              transition: 'all 0.2s'
            }} className={`w-full text-white rounded p-2 text-sm font-bold flex items-center justify-center gap-2 transition-all ${loading ? 'bg-orange-700 cursor-not-allowed' : 'bg-indigo-500 hover:bg-indigo-600 cursor-pointer'}`}>
              {loading ? (
                <>
                  <div style={{ width: '14px', height: '14px', border: '2px solid rgba(255,255,255,0.3)', borderTopColor: '#fff', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  Capturing Faces...
                  <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
                </>
              ) : (
                <>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path><circle cx="12" cy="13" r="4"></circle></svg>
                  Register &amp; Capture Faces
                </>
              )}
            </button>
          </form>
        </div>

        <div style={{ backgroundColor: '#0d1117', border: '0.5px solid rgba(255,255,255,0.05)', borderRadius: '10px', padding: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', paddingBottom: '13px', borderBottom: '0.5px solid rgba(255,255,255,0.05)', marginBottom: '14px' }}>
            <span style={{ fontSize: '12px', fontWeight: '600', color: '#94a3b8' }}>Registered Students ({students.length})</span>
          </div>
          <div style={{ maxHeight: '420px', overflowY: 'auto' }}>
            {students.map((st, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '9px', padding: '9px 0', borderBottom: '0.5px solid rgba(255,255,255,0.03)' }}>
                <div style={{ width: '30px', height: '30px', borderRadius: '50%', backgroundColor: 'rgba(129,140,248,0.12)', color: '#a5b4fc', fontSize: '12px', fontWeight: '700', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>{st.name.charAt(0).toUpperCase()}</div>
                <div>
                  <div style={{ fontSize: '12px', color: '#e2e8f0', fontWeight: '500' }}>{st.name}</div>
                  <div style={{ fontSize: '10px', color: '#475569', marginTop: '1px' }}>{st.roll} · {st.branch} · Sem {st.semester}</div>
                </div>
                <div style={{
                  marginLeft: 'auto',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '10px'
                }}>
                  <div style={{
                    backgroundColor: st.face_registered ? 'rgba(34,197,94,0.1)' : 'rgba(129,140,248,0.1)',
                    color: st.face_registered ? '#86efac' : '#a5b4fc',
                    border: `0.5px solid ${st.face_registered ? 'rgba(34,197,94,0.2)' : 'rgba(129,140,248,0.2)'}`,
                    fontSize: '9px', padding: '2px 7px', borderRadius: '8px'
                  }}>
                    {st.face_registered ? '✅ Face Captured' : 'Registered'}
                  </div>
                  <button 
                    onClick={() => handleDeleteStudent(st.id, st.name)}
                    title="Delete Student"
                    style={{
                      background: 'none', border: 'none', cursor: 'pointer', padding: '4px',
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      borderRadius: '4px', backgroundColor: 'rgba(239,68,68,0.1)'
                    }}
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#ef4444" strokeWidth="2"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function ManageFacultyView({ showToast }) {
  const [faculty, setFaculty] = useState([]);
  const [form, setForm] = useState({ name: '', employee_id: '', department: '', email: '', password: '' });
  const [loading, setLoading] = useState(false);
  const [showPass, setShowPass] = useState(false);

  const fetchFaculty = () => { axios.get('http://localhost:8000/api/admin/faculty').then(res => setFaculty(res.data)).catch(console.error); };
  useEffect(() => { fetchFaculty(); }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!form.password || form.password.length < 4) {
      showToast('Password must be at least 4 characters', 'error');
      return;
    }
    setLoading(true);
    try {
      const res = await axios.post('http://localhost:8000/api/admin/add-faculty', form);
      if (res.data.success) {
        showToast(`✅ Faculty "${form.name}" added! Login: ${form.employee_id} / ${form.password}`);
        setForm({ name: '', employee_id: '', department: '', email: '', password: '' });
        fetchFaculty();
      } else showToast(res.data.message || 'Error adding faculty', 'error');
    } catch (err) { showToast(err.response?.data?.message || 'Server error', 'error'); } finally { setLoading(false); }
  };

  const handleDeleteFaculty = async (id, name) => {
    if (!window.confirm(`Are you sure you want to delete faculty ${name}? This will remove them from the system.`)) return;
    try {
      const res = await axios.delete(`http://localhost:8000/api/admin/faculty/${id}`);
      if (res.data.success) {
        showToast(`✅ Faculty ${name} deleted!`);
        fetchFaculty();
      } else {
        showToast(res.data.message || 'Error deleting faculty', 'error');
      }
    } catch (err) {
      showToast(err.response?.data?.message || 'Server error', 'error');
    }
  };

  const inputStyle = { width: '100%', backgroundColor: 'rgba(255,255,255,0.03)', border: '0.5px solid rgba(255,255,255,0.07)', borderRadius: '6px', padding: '8px 10px', color: '#f1f5f9', fontSize: '12px', outline: 'none', boxSizing: 'border-box', marginBottom: '12px' };

  return (
    <div>
      <h1 style={{ fontSize: '19px', fontWeight: '800', color: '#f1f5f9', letterSpacing: '-0.5px', margin: 0 }}>Manage Faculty</h1>
      <p style={{ fontSize: '12px', color: '#475569', marginTop: '4px', marginBottom: '20px' }}>Add faculty members and assign departments</p>
      
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
        <div style={{ backgroundColor: '#0d1117', border: '0.5px solid rgba(255,255,255,0.05)', borderRadius: '10px', padding: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', paddingBottom: '13px', borderBottom: '0.5px solid rgba(255,255,255,0.05)', marginBottom: '14px' }}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#22c55e" strokeWidth="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>
            <span style={{ fontSize: '12px', fontWeight: '600', color: '#94a3b8' }}>Add New Faculty</span>
          </div>
          <form onSubmit={handleSubmit}>
            <input style={inputStyle} placeholder="Full Name" value={form.name} onChange={e => setForm({...form, name: e.target.value})} required />
            <input style={inputStyle} placeholder="Employee ID (used as Login ID)" value={form.employee_id} onChange={e => setForm({...form, employee_id: e.target.value})} required />
            <input style={inputStyle} placeholder="Department" value={form.department} onChange={e => setForm({...form, department: e.target.value})} required />
            <input style={inputStyle} placeholder="Email" type="email" value={form.email} onChange={e => setForm({...form, email: e.target.value})} required />
            
            {/* PASSWORD FIELD */}
            <div style={{ position: 'relative', marginBottom: '12px' }}>
              <input
                style={{...inputStyle, marginBottom: 0, paddingRight: '36px'}}
                placeholder="Set Login Password for Faculty"
                type={showPass ? 'text' : 'password'}
                value={form.password}
                onChange={e => setForm({...form, password: e.target.value})}
                required
              />
              <span
                onClick={() => setShowPass(!showPass)}
                style={{ position: 'absolute', right: '10px', top: '50%', transform: 'translateY(-50%)', cursor: 'pointer', color: '#475569', fontSize: '10px' }}
              >
                {showPass ? 'HIDE' : 'SHOW'}
              </span>
            </div>

            {/* INFO BOX */}
            {form.employee_id && form.password && (
              <div style={{ backgroundColor: 'rgba(34,197,94,0.07)', border: '0.5px solid rgba(34,197,94,0.2)', borderRadius: '6px', padding: '10px 12px', fontSize: '11px', color: '#86efac', marginBottom: '14px' }}>
                <div style={{ fontWeight: '700', marginBottom: '4px' }}>✅ Faculty Login Credentials:</div>
                <div>🆔 Employee ID: <strong>{form.employee_id}</strong></div>
                <div>🔑 Password: <strong>{form.password}</strong></div>
                <div style={{ color: '#475569', marginTop: '4px', fontSize: '10px' }}>Share these with the faculty member</div>
              </div>
            )}

            <button type="submit" disabled={loading} style={{ width: '100%', backgroundColor: '#16a34a', color: 'white', border: 'none', borderRadius: '7px', padding: '10px', fontSize: '12px', fontWeight: '700', cursor: 'pointer', opacity: loading ? 0.7 : 1 }}>
              {loading ? 'Adding...' : 'Add Faculty Member'}
            </button>
          </form>
        </div>

        <div style={{ backgroundColor: '#0d1117', border: '0.5px solid rgba(255,255,255,0.05)', borderRadius: '10px', padding: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', paddingBottom: '13px', borderBottom: '0.5px solid rgba(255,255,255,0.05)', marginBottom: '14px' }}>
            <span style={{ fontSize: '12px', fontWeight: '600', color: '#94a3b8' }}>Faculty Directory ({faculty.length})</span>
          </div>
          <div style={{ maxHeight: '420px', overflowY: 'auto' }}>
            {faculty.map((f, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '9px', padding: '9px 0', borderBottom: '0.5px solid rgba(255,255,255,0.03)' }}>
                <div style={{ width: '30px', height: '30px', borderRadius: '50%', backgroundColor: 'rgba(34,197,94,0.1)', color: '#86efac', fontSize: '12px', fontWeight: '700', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>{f.name.charAt(0).toUpperCase()}</div>
                <div>
                  <div style={{ fontSize: '12px', color: '#e2e8f0', fontWeight: '500' }}>{f.name}</div>
                  <div style={{ fontSize: '10px', color: '#475569', marginTop: '1px' }}>{f.employee_id} · {f.department}</div>
                </div>
                <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <div style={{ backgroundColor: 'rgba(34,197,94,0.1)', color: '#86efac', border: '0.5px solid rgba(34,197,94,0.2)', fontSize: '9px', padding: '2px 7px', borderRadius: '8px' }}>Active</div>
                  <button 
                    onClick={() => handleDeleteFaculty(f.id, f.name)}
                    title="Delete Faculty"
                    style={{
                      background: 'none', border: 'none', cursor: 'pointer', padding: '4px',
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      borderRadius: '4px', backgroundColor: 'rgba(239,68,68,0.1)'
                    }}
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#ef4444" strokeWidth="2"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function ClassroomSetupView({ showToast }) {
  const [classrooms, setClassrooms] = useState([]);
  const [form, setForm] = useState({ room_name: '', camera_url: '' });
  const [loading, setLoading] = useState(false);

  const fetchClassrooms = () => { axios.get('http://localhost:8000/api/admin/classrooms').then(res => setClassrooms(res.data)).catch(console.error); };
  useEffect(() => { fetchClassrooms(); }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const res = await axios.post('http://localhost:8000/api/admin/add-classroom', form);
      if (res.data.success) { showToast(res.data.message); setForm({ room_name: '', camera_url: '' }); fetchClassrooms(); }
      else showToast(res.data.message || 'Error adding classroom', 'error');
    } catch (err) { showToast(err.response?.data?.message || 'Server error', 'error'); } finally { setLoading(false); }
  };

  const inputStyle = { width: '100%', backgroundColor: 'rgba(255,255,255,0.03)', border: '0.5px solid rgba(255,255,255,0.07)', borderRadius: '6px', padding: '8px 10px', color: '#f1f5f9', fontSize: '12px', outline: 'none', boxSizing: 'border-box', marginBottom: '12px' };

  return (
    <div>
      <h1 style={{ fontSize: '19px', fontWeight: '800', color: '#f1f5f9', letterSpacing: '-0.5px', margin: 0 }}>Classroom Setup</h1>
      <p style={{ fontSize: '12px', color: '#475569', marginTop: '4px', marginBottom: '20px' }}>Configure IP cameras and link classrooms</p>
      
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
        <div style={{ backgroundColor: '#0d1117', border: '0.5px solid rgba(255,255,255,0.05)', borderRadius: '10px', padding: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', paddingBottom: '13px', borderBottom: '0.5px solid rgba(255,255,255,0.05)', marginBottom: '14px' }}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#60a5fa" strokeWidth="2"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path><circle cx="12" cy="13" r="4"></circle></svg>
            <span style={{ fontSize: '12px', fontWeight: '600', color: '#94a3b8' }}>Add New Classroom</span>
          </div>
          <form onSubmit={handleSubmit}>
            <input style={inputStyle} placeholder="Room Name (e.g. Room 101 / Lab B)" value={form.room_name} onChange={e => setForm({...form, room_name: e.target.value})} required />
            <input style={inputStyle} placeholder='Camera URL (0 or http://192.168.1.5:8080/video)' value={form.camera_url} onChange={e => setForm({...form, camera_url: e.target.value})} required />
            
            <div style={{ backgroundColor: 'rgba(96,165,250,0.06)', border: '0.5px solid rgba(96,165,250,0.14)', borderRadius: '6px', padding: '8px 10px', fontSize: '11px', color: '#64748b', display: 'flex', alignItems: 'center', gap: '7px', marginBottom: '16px' }}>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#60a5fa" strokeWidth="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
              Use "0" for laptop webcam | IP URL for classroom cameras
            </div>
            <button type="submit" disabled={loading} style={{ width: '100%', backgroundColor: '#1d4ed8', color: 'white', border: 'none', borderRadius: '7px', padding: '10px', fontSize: '12px', fontWeight: '700', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '7px', cursor: 'pointer', opacity: loading ? 0.7 : 1 }}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path><circle cx="12" cy="13" r="4"></circle></svg>
              {loading ? 'Adding...' : 'Add Classroom'}
            </button>
          </form>
        </div>

        <div style={{ backgroundColor: '#0d1117', border: '0.5px solid rgba(255,255,255,0.05)', borderRadius: '10px', padding: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', paddingBottom: '13px', borderBottom: '0.5px solid rgba(255,255,255,0.05)', marginBottom: '14px' }}>
            <span style={{ fontSize: '12px', fontWeight: '600', color: '#94a3b8' }}>Configured Classrooms ({classrooms.length})</span>
          </div>
          <div style={{ maxHeight: '420px', overflowY: 'auto' }}>
            {classrooms.map((c, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '9px', padding: '9px 0', borderBottom: '0.5px solid rgba(255,255,255,0.03)' }}>
                <div style={{ width: '30px', height: '30px', borderRadius: '50%', backgroundColor: 'rgba(96,165,250,0.1)', color: '#60a5fa', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path><circle cx="12" cy="13" r="4"></circle></svg>
                </div>
                <div style={{ overflow: 'hidden' }}>
                  <div style={{ fontSize: '12px', color: '#e2e8f0', fontWeight: '500' }}>{c.room_name}</div>
                  <div style={{ fontSize: '10px', color: '#475569', marginTop: '1px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '200px' }}>{c.camera_url}</div>
                </div>
                <div style={{ marginLeft: 'auto', backgroundColor: 'rgba(96,165,250,0.1)', color: '#60a5fa', border: '0.5px solid rgba(96,165,250,0.2)', fontSize: '9px', padding: '2px 7px', borderRadius: '8px' }}>Active</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// ============================================================
// TIMETABLE VIEW
// ============================================================
function TimetableView({ showToast }) {
  const [entries, setEntries] = useState([]);
  const [faculty, setFaculty] = useState([]);
  const [rooms, setRooms] = useState([]);
  const [facultySubjects, setFacultySubjects] = useState([]);
  const [form, setForm] = useState({ faculty_id:'', faculty_name:'', subject_code:'', subject_name:'', day_of_week:'Monday', period:'Period 1', room:'', room_id:'' });
  const [loading, setLoading] = useState(false);

  const fetchAll = () => {
    axios.get('http://localhost:8000/api/timetable/all').then(r => setEntries(Array.isArray(r.data)?r.data:[])).catch(console.error);
    axios.get('http://localhost:8000/api/admin/faculty').then(r => setFaculty(Array.isArray(r.data)?r.data:[])).catch(console.error);
    axios.get('http://localhost:8000/api/rooms').then(r => setRooms(Array.isArray(r.data)?r.data:[])).catch(console.error);
  };
  useEffect(() => { fetchAll(); }, []);

  // When faculty changes, fetch their assigned subjects
  useEffect(() => {
    if (!form.faculty_id) { setFacultySubjects([]); return; }
    axios.get(`http://localhost:8000/api/get-faculty-subjects?faculty_id=${form.faculty_id}`)
      .then(r => setFacultySubjects(Array.isArray(r.data) ? r.data : []))
      .catch(() => setFacultySubjects([]));
  }, [form.faculty_id]);

  const handleAdd = async () => {
    if(!form.faculty_id || !form.subject_name) return showToast('Fill required fields','error');
    setLoading(true);
    try {
      const res = await axios.post('http://localhost:8000/api/timetable', form);
      if(res.data.success) { showToast('Entry added'); fetchAll(); setForm(p=>({...p, subject_name:'', subject_code:'', room:'', room_id:''})); }
      else showToast(res.data.message,'error');
    } catch(e) { showToast(e.response?.data?.message||'Error','error'); }
    setLoading(false);
  };
  const handleDelete = async (id) => { await axios.delete(`http://localhost:8000/api/timetable/${id}`); showToast('Deleted'); fetchAll(); };
  const cs = { backgroundColor:'#0d1117', border:'0.5px solid rgba(255,255,255,0.05)', borderRadius:'10px', padding:'16px', marginBottom:'14px' };
  const inp = { width:'100%', backgroundColor:'#0d1117', border:'0.5px solid rgba(255,255,255,0.12)', borderRadius:'7px', padding:'9px 11px', color:'#f1f5f9', fontSize:'12px', outline:'none', boxSizing:'border-box' };
  return (
    <div>
      <h2 style={{ color:'#f1f5f9', fontSize:'18px', margin:'0 0 16px' }}>📅 Timetable Management</h2>
      <div style={cs}>
        <div style={{ fontSize:'13px', fontWeight:'700', color:'#f1f5f9', marginBottom:'12px' }}>Add Entry</div>
        <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr 1fr', gap:'10px' }}>
          {/* Faculty Dropdown */}
          <select style={inp} value={form.faculty_id} onChange={e => {
            const f = faculty.find(x => x.employee_id === e.target.value);
            setForm(p => ({...p, faculty_id: e.target.value, faculty_name: f?.name||'', subject_name:'', subject_code:''}));
          }}>
            <option value="" style={{backgroundColor:'#0d1117', color:'#f1f5f9'}}>Select Faculty</option>
            {faculty.map(f => <option key={f.employee_id} value={f.employee_id} style={{backgroundColor:'#0d1117', color:'#f1f5f9'}}>{f.name} ({f.employee_id})</option>)}
          </select>

          {/* Dynamic Subject Dropdown */}
          <select style={inp} value={form.subject_name} onChange={e => {
            const sub = facultySubjects.find(s => s.subject_name === e.target.value);
            setForm(p => ({...p, subject_name: e.target.value, subject_code: sub?.subject_code||''}));
          }} disabled={!form.faculty_id}>
            <option value="" style={{backgroundColor:'#0d1117', color:'#f1f5f9'}}>{form.faculty_id ? (facultySubjects.length ? 'Select Subject' : 'No subjects assigned') : 'Select faculty first'}</option>
            {facultySubjects.map((s,i) => <option key={i} value={s.subject_name} style={{backgroundColor:'#0d1117', color:'#f1f5f9'}}>{s.subject_name} {s.subject_code && `(${s.subject_code})`}</option>)}
          </select>

          {/* Room Dropdown */}
          <select style={inp} value={form.room_id} onChange={e => {
            const rm = rooms.find(r => String(r.id) === e.target.value);
            setForm(p => ({...p, room_id: e.target.value, room: rm?.room_name||''}));
          }}>
            <option value="" style={{backgroundColor:'#0d1117', color:'#f1f5f9'}}>Select Room (optional)</option>
            {rooms.map(r => <option key={r.id} value={r.id} style={{backgroundColor:'#0d1117', color:'#f1f5f9'}}>{r.room_name}</option>)}
          </select>

          <select style={inp} value={form.day_of_week} onChange={e=>setForm(p=>({...p, day_of_week:e.target.value}))}>
            {['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'].map(d => <option key={d} style={{backgroundColor:'#0d1117', color:'#f1f5f9'}}>{d}</option>)}
          </select>
          <select style={inp} value={form.period} onChange={e=>setForm(p=>({...p, period:e.target.value}))}>
            {[1,2,3,4,5,6,7,8].map(n => <option key={n} value={`Period ${n}`} style={{backgroundColor:'#0d1117', color:'#f1f5f9'}}>Period {n}</option>)}
          </select>
        </div>
        <button onClick={handleAdd} disabled={loading} style={{ marginTop:'12px', backgroundColor:'#ef4444', color:'#fff', border:'none', borderRadius:'8px', padding:'10px 24px', fontSize:'12px', fontWeight:'700', cursor:'pointer' }}>
          {loading ? 'Adding...' : '+ Add Entry'}
        </button>
      </div>
      <div style={cs}>
        <div style={{ fontSize:'13px', fontWeight:'700', color:'#f1f5f9', marginBottom:'10px' }}>All Entries ({entries.length})</div>
        <table style={{ width:'100%', borderCollapse:'collapse' }}>
          <thead><tr>{['Faculty','Subject','Day','Period','Time','Room',''].map(h => (
            <th key={h} style={{ fontSize:'10px', color:'#475569', padding:'6px 8px', textAlign:'left', borderBottom:'0.5px solid rgba(255,255,255,0.05)' }}>{h}</th>
          ))}</tr></thead>
          <tbody>{entries.map(e => (
            <tr key={e.id}>
              <td style={{ fontSize:'12px', color:'#e2e8f0', padding:'8px' }}>{e.faculty_name}</td>
              <td style={{ fontSize:'12px', color:'#94a3b8', padding:'8px' }}>{e.subject_name}</td>
              <td style={{ fontSize:'12px', color:'#94a3b8', padding:'8px' }}>{e.day_of_week}</td>
              <td style={{ fontSize:'12px', color:'#94a3b8', padding:'8px' }}>{e.period}</td>
              <td style={{ fontSize:'12px', color:'#94a3b8', padding:'8px' }}>{e.start_time}–{e.end_time}</td>
              <td style={{ fontSize:'12px', color:'#94a3b8', padding:'8px' }}>{e.room||'—'}</td>
              <td style={{ padding:'8px' }}><button onClick={()=>handleDelete(e.id)} style={{ fontSize:'10px', color:'#ef4444', backgroundColor:'rgba(239,68,68,0.08)', border:'0.5px solid rgba(239,68,68,0.2)', borderRadius:'5px', padding:'3px 10px', cursor:'pointer' }}>Delete</button></td>
            </tr>
          ))}</tbody>
        </table>
      </div>
    </div>
  );
}


// ============================================================
// EXTRA CLASSES VIEW
// ============================================================
function ExtraClassesView() {
  const [sessions, setSessions] = useState([]);
  useEffect(() => {
    axios.get('http://localhost:8000/api/admin/extra-classes-summary').then(r => setSessions(Array.isArray(r.data)?r.data:[])).catch(console.error);
  }, []);
  return (
    <div>
      <h2 style={{ color:'#f1f5f9', fontSize:'18px', margin:'0 0 16px' }}>⚡ Extra Classes Summary</h2>
      <div style={{ backgroundColor:'#0d1117', border:'0.5px solid rgba(255,255,255,0.05)', borderRadius:'10px', padding:'16px' }}>
        {sessions.length === 0 ? (
          <div style={{ textAlign:'center', padding:'40px', color:'#475569', fontSize:'13px' }}>No extra classes recorded yet</div>
        ) : (
          <table style={{ width:'100%', borderCollapse:'collapse' }}>
            <thead><tr>{['Date','Subject','Period','Taken By','Original Faculty','Present'].map(h => (
              <th key={h} style={{ fontSize:'10px', color:'#475569', padding:'6px 10px', textAlign:'left', borderBottom:'0.5px solid rgba(255,255,255,0.05)' }}>{h}</th>
            ))}</tr></thead>
            <tbody>{sessions.map(s => (
              <tr key={s.id}>
                <td style={{ fontSize:'12px', color:'#e2e8f0', padding:'10px' }}>{s.date}<br/><span style={{ fontSize:'10px', color:'#475569' }}>{s.start_time}</span></td>
                <td style={{ fontSize:'12px', color:'#e2e8f0', padding:'10px' }}>{s.subject_name}</td>
                <td style={{ fontSize:'12px', color:'#94a3b8', padding:'10px' }}>{s.period}</td>
                <td style={{ fontSize:'12px', color:'#c084fc', padding:'10px' }}>{s.faculty_name}</td>
                <td style={{ fontSize:'12px', color:'#fb923c', padding:'10px' }}>{s.original_faculty_name || '—'}</td>
                <td style={{ fontSize:'12px', color:'#22c55e', padding:'10px' }}>{s.total_present}</td>
              </tr>
            ))}</tbody>
          </table>
        )}
      </div>
    </div>
  );
}

// ============================================================
// ROOMS VIEW — IP / RTSP Camera Management
// ============================================================
function RoomsView({ showToast }) {
  const [rooms, setRooms] = useState([]);
  const [form, setForm] = useState({ room_name: '', rtsp_url: '' });
  const [loading, setLoading] = useState(false);
  const inp = { width:'100%', backgroundColor:'rgba(255,255,255,0.04)', border:'0.5px solid rgba(255,255,255,0.08)', borderRadius:'7px', padding:'9px 11px', color:'#f1f5f9', fontSize:'12px', outline:'none', boxSizing:'border-box', marginBottom:'10px' };

  const fetchRooms = () => {
    axios.get('http://localhost:8000/api/rooms')
      .then(r => setRooms(Array.isArray(r.data) ? r.data : []))
      .catch(console.error);
  };
  useEffect(() => { fetchRooms(); }, []);

  const handleAdd = async () => {
    if (!form.room_name || !form.rtsp_url) return showToast('Room name and RTSP URL are required', 'error');
    setLoading(true);
    try {
      const res = await axios.post('http://localhost:8000/api/rooms', form);
      if (res.data.success) { showToast(`✅ Room "${form.room_name}" added`); setForm({ room_name:'', rtsp_url:'' }); fetchRooms(); }
      else showToast(res.data.message, 'error');
    } catch(e) { showToast(e.response?.data?.message||'Error', 'error'); }
    setLoading(false);
  };

  const handleDelete = async (id, name) => {
    if (!window.confirm(`Delete room "${name}"?`)) return;
    await axios.delete(`http://localhost:8000/api/rooms/${id}`);
    showToast('Room deleted'); fetchRooms();
  };

  const cs = { backgroundColor:'#0d1117', border:'0.5px solid rgba(255,255,255,0.05)', borderRadius:'10px', padding:'16px', marginBottom:'14px' };
  return (
    <div>
      <h2 style={{ color:'#f1f5f9', fontSize:'18px', margin:'0 0 4px' }}>🏫 Room & Camera Management</h2>
      <p style={{ fontSize:'12px', color:'#475569', marginBottom:'20px' }}>Map rooms to their RTSP/IP camera streams for automated attendance scanning.</p>
      <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:'16px' }}>
        <div style={cs}>
          <div style={{ fontSize:'13px', fontWeight:'700', color:'#f1f5f9', marginBottom:'14px' }}>Add New Room</div>
          <input style={inp} placeholder="Room Name (e.g. Lab A1, Room 101)" value={form.room_name} onChange={e => setForm({...form, room_name: e.target.value})} />
          <input style={inp} placeholder="RTSP/IP URL (e.g. rtsp://192.168.1.5:554/stream or 0 for webcam)" value={form.rtsp_url} onChange={e => setForm({...form, rtsp_url: e.target.value})} />
          <div style={{ backgroundColor:'rgba(96,165,250,0.06)', border:'0.5px solid rgba(96,165,250,0.14)', borderRadius:'6px', padding:'8px 10px', fontSize:'11px', color:'#64748b', marginBottom:'12px' }}>
            💡 Use <strong style={{color:'#60a5fa'}}>0</strong> for laptop webcam {' '}|{' '} Use full RTSP URL for IP cameras
          </div>
          <button onClick={handleAdd} disabled={loading} style={{ width:'100%', backgroundColor:'#3b82f6', color:'#fff', border:'none', borderRadius:'7px', padding:'10px', fontSize:'12px', fontWeight:'700', cursor:'pointer', opacity: loading ? 0.7 : 1 }}>
            {loading ? 'Adding...' : '+ Add Room'}
          </button>
        </div>
        <div style={cs}>
          <div style={{ fontSize:'13px', fontWeight:'700', color:'#f1f5f9', marginBottom:'10px' }}>Configured Rooms ({rooms.length})</div>
          <div style={{ maxHeight:'320px', overflowY:'auto' }}>
            {rooms.length === 0 ? <div style={{ color:'#475569', fontSize:'12px', textAlign:'center', padding:'30px' }}>No rooms configured yet</div> : rooms.map(r => (
              <div key={r.id} style={{ display:'flex', alignItems:'center', gap:'10px', padding:'10px 0', borderBottom:'0.5px solid rgba(255,255,255,0.04)' }}>
                <div style={{ width:'32px', height:'32px', borderRadius:'8px', backgroundColor:'rgba(59,130,246,0.1)', color:'#60a5fa', display:'flex', alignItems:'center', justifyContent:'center', flexShrink:0 }}>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path><circle cx="12" cy="13" r="4"></circle></svg>
                </div>
                <div style={{ flex:1, overflow:'hidden' }}>
                  <div style={{ fontSize:'12px', color:'#e2e8f0', fontWeight:'600' }}>{r.room_name}</div>
                  <div style={{ fontSize:'10px', color:'#475569', overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>{r.rtsp_url}</div>
                </div>
                <button onClick={() => handleDelete(r.id, r.room_name)} style={{ background:'none', border:'none', cursor:'pointer', padding:'4px', color:'#ef4444' }}>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
                </button>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// ============================================================
// ATTENDANCE TRACKER VIEW — Defaulter Alert System
// ============================================================
function AttendanceTrackerView({ showToast }) {
  const [students, setStudents] = useState([]);
  const [totalSessions, setTotalSessions] = useState(0);
  const [sending, setSending] = useState(false);
  const [loading, setLoading] = useState(true);

  const fetchOverview = () => {
    setLoading(true);
    axios.get('http://localhost:8000/api/admin/attendance-overview')
      .then(res => { setStudents(res.data.students || []); setTotalSessions(res.data.total_sessions || 0); })
      .catch(console.error)
      .finally(() => setLoading(false));
  };
  useEffect(() => { fetchOverview(); }, []);

  const handleSendAlerts = async () => {
    setSending(true);
    try {
      const res = await axios.post('http://localhost:8000/api/admin/send-defaulter-alerts');
      if (res.data.success) showToast(`✅ ${res.data.message}`);
      else showToast(res.data.message, 'error');
    } catch(e) { showToast(e.response?.data?.message || 'Error sending alerts', 'error'); }
    setSending(false);
  };

  const cs = { backgroundColor:'#0d1117', border:'0.5px solid rgba(255,255,255,0.05)', borderRadius:'10px', padding:'16px' };
  const defaulters = students.filter(s => s.pct < 75).length;

  return (
    <div>
      <h2 style={{ color:'#f1f5f9', fontSize:'18px', margin:'0 0 4px' }}>📊 Student Attendance Overview</h2>
      <p style={{ fontSize:'12px', color:'#475569', marginBottom:'16px' }}>Total Classes Held: <strong style={{color:'#94a3b8'}}>{totalSessions}</strong> {' '}|{' '} Defaulters ({'<'}75%): <strong style={{color:'#ef4444'}}>{defaulters}</strong></p>

      {/* Send Alerts Button */}
      <button onClick={handleSendAlerts} disabled={sending || defaulters === 0} style={{
        marginBottom:'16px', backgroundColor: sending ? '#7f1d1d' : '#ef4444', color:'#fff', border:'none',
        borderRadius:'8px', padding:'11px 24px', fontSize:'13px', fontWeight:'700', cursor: (sending||defaulters===0)?'not-allowed':'pointer',
        opacity: defaulters === 0 ? 0.5 : 1, display:'flex', alignItems:'center', gap:'8px'
      }}>
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2"><path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"></path><polyline points="22,6 12,13 2,6"></polyline></svg>
        {sending ? 'Sending Emails...' : `⚠️ Send Alerts to ${defaulters} Defaulter${defaulters !== 1 ? 's' : ''}`}
      </button>

      <div style={cs}>
        {loading ? (
          <div style={{ textAlign:'center', padding:'40px', color:'#475569' }}>Loading...</div>
        ) : students.length === 0 ? (
          <div style={{ textAlign:'center', padding:'40px', color:'#475569', fontSize:'13px' }}>No students registered yet</div>
        ) : (
          <table style={{ width:'100%', borderCollapse:'collapse' }}>
            <thead><tr>{['Name','Reg No','Email','Attended','Total','Attendance %'].map(h => (
              <th key={h} style={{ fontSize:'10px', color:'#475569', padding:'8px 10px', textAlign:'left', borderBottom:'0.5px solid rgba(255,255,255,0.05)', fontWeight:'600', letterSpacing:'0.5px' }}>{h.toUpperCase()}</th>
            ))}</tr></thead>
            <tbody>{students.map(st => {
              const isDefaulter = st.pct < 75;
              return (
                <tr key={st.id} style={{ backgroundColor: isDefaulter ? 'rgba(239,68,68,0.04)' : 'transparent' }}>
                  <td style={{ fontSize:'12px', color:'#e2e8f0', padding:'10px', fontWeight:'500' }}>{st.name}</td>
                  <td style={{ fontSize:'12px', color:'#94a3b8', padding:'10px' }}>{st.roll_number || '—'}</td>
                  <td style={{ fontSize:'11px', color:'#64748b', padding:'10px' }}>{st.email || <em>No email</em>}</td>
                  <td style={{ fontSize:'12px', color:'#94a3b8', padding:'10px', textAlign:'center' }}>{st.attended}</td>
                  <td style={{ fontSize:'12px', color:'#94a3b8', padding:'10px', textAlign:'center' }}>{st.total}</td>
                  <td style={{ padding:'10px' }}>
                    <span style={{
                      display:'inline-block', fontWeight:'700', fontSize:'12px',
                      color: isDefaulter ? '#fca5a5' : '#86efac',
                      backgroundColor: isDefaulter ? 'rgba(239,68,68,0.1)' : 'rgba(34,197,94,0.08)',
                      border: `0.5px solid ${isDefaulter ? 'rgba(239,68,68,0.3)' : 'rgba(34,197,94,0.2)'}`,
                      borderRadius:'6px', padding:'3px 10px'
                    }}>
                      {st.pct}% {isDefaulter ? '⚠️' : '✅'}
                    </span>
                  </td>
                </tr>
              );
            })}</tbody>
          </table>
        )}
      </div>
    </div>
  );
}
