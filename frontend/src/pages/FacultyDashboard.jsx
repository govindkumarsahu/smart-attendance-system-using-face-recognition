import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

const S = {
  card: { backgroundColor:'#0d1117', border:'0.5px solid rgba(255,255,255,0.05)', borderRadius:'10px', padding:'14px 16px' },
  btn2: { backgroundColor:'rgba(168,85,247,0.15)', color:'#c084fc', border:'1px solid rgba(168,85,247,0.3)', borderRadius:'10px', padding:'14px 32px', fontSize:'14px', fontWeight:'700', cursor:'pointer', display:'flex', alignItems:'center', gap:'8px' },
  modal: { position:'fixed', inset:0, zIndex:1000, display:'flex', alignItems:'center', justifyContent:'center' },
  overlay: { position:'absolute', inset:0, backgroundColor:'rgba(0,0,0,0.7)' },
  mbox: { position:'relative', backgroundColor:'#0d1117', border:'1px solid rgba(255,255,255,0.1)', borderRadius:'14px', padding:'28px', maxWidth:'600px', width:'90%', maxHeight:'80vh', overflowY:'auto' },
  inp: { width:'100%', backgroundColor:'#0d1117', border:'0.5px solid rgba(255,255,255,0.12)', borderRadius:'8px', padding:'10px 12px', color:'#f1f5f9', fontSize:'13px', outline:'none', boxSizing:'border-box' },
};

export default function FacultyDashboard() {
  const navigate = useNavigate();
  
  // Faculty info
  const facultyName = localStorage.getItem("facultyName") || "Dr. Rajesh Kumar";
  const facultyId = localStorage.getItem("facultyId") || "FAC2024001";
  
  // States
  const [time, setTime] = useState("");
  const [dateStr, setDateStr] = useState("");
  const [scanning, setScanning] = useState(false);
  const [scanResult, setScanResult] = useState(null);
  const [countdown, setCountdown] = useState(0);
  const [activeTab, setActiveTab] = useState('dashboard');
  const [currentClass, setCurrentClass] = useState(null);
  const [extraStats, setExtraStats] = useState({ extra_taken:0, classes_substituted:0 });
  const [showExtraModal, setShowExtraModal] = useState(false);
  const [showReviewModal, setShowReviewModal] = useState(false);
  const [reviewData, setReviewData] = useState({ students:[], total_present:0, total_students:0 });
  const [logbookData, setLogbookData] = useState([]);
  const [logbookTotal, setLogbookTotal] = useState(60);
  const [sessionDetailId, setSessionDetailId] = useState(null);
  const [sessionDetail, setSessionDetail] = useState(null);
  const [extraForm, setExtraForm] = useState({ subject_name:'', subject_code:'', period:'', original_faculty_name:'', room_id:'' });
  const [subjects, setSubjects] = useState([]);
  const [rooms, setRooms] = useState([]);
  const [extraScanning, setExtraScanning] = useState(false);
  
  const [stats, setStats] = useState({
    classes_today: 0,
    total_students: 60,
    avg_attendance: "0%",
    subjects_assigned: 0
  });

  // Clock
  useEffect(() => {
    const timer = setInterval(() => {
      const now = new Date();
      setTime(now.toLocaleTimeString('en-US', { hour12: true, hour: '2-digit', minute: '2-digit', second: '2-digit' }));
      setDateStr(now.toLocaleDateString('en-GB', { weekday: 'long', day: '2-digit', month: 'short', year: 'numeric' }));
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  // Fetch stats + timetable + extra stats
  useEffect(() => {
    axios.get(`http://localhost:8000/api/faculty-stats?faculty_id=${facultyId}`)
      .then(res => {
        setStats(prev => ({
          ...prev,
          classes_today: res.data.classes_today || 0,
          subjects_assigned: res.data.subjects_assigned || 0,
          avg_attendance: res.data.avg_attendance ? `${parseFloat(res.data.avg_attendance).toFixed(0)}%` : '0%'
        }));
      }).catch(console.error);
    axios.get(`http://localhost:8000/api/timetable/current-class?faculty_id=${facultyId}`)
      .then(res => { if(res.data.found) setCurrentClass(res.data.class); }).catch(console.error);
    axios.get(`http://localhost:8000/api/faculty-extra-stats?faculty_id=${facultyId}`)
      .then(res => setExtraStats(res.data)).catch(console.error);
    axios.get(`http://localhost:8000/api/get-faculty-subjects?faculty_id=${facultyId}`)
      .then(res => { if(Array.isArray(res.data)) setSubjects(res.data); }).catch(console.error);
    axios.get('http://localhost:8000/api/rooms')
      .then(res => { if(Array.isArray(res.data)) setRooms(res.data); }).catch(console.error);
  }, [facultyId]);

  const fetchLogbook = () => {
    axios.get(`http://localhost:8000/api/faculty-logbook?faculty_id=${facultyId}`)
      .then(res => { setLogbookData(res.data.sessions||[]); setLogbookTotal(res.data.total_students||60); }).catch(console.error);
  };
  useEffect(() => { if(activeTab==='logbook') fetchLogbook(); }, [activeTab]);

  // Countdown timer during scan
  useEffect(() => {
    let timer;
    if (scanning && countdown > 0) {
      timer = setInterval(() => {
        setCountdown(c => {
          if (c <= 1) { clearInterval(timer); return 0; }
          return c - 1;
        });
      }, 1000);
    }
    return () => clearInterval(timer);
  }, [scanning, countdown]);

  // ONE-CLICK ATTENDANCE
  const handleTakeAttendance = async () => {
    setScanning(true);
    setScanResult(null);
    setCountdown(35);
    const subName = currentClass ? currentClass.subject_name : 'General';
    const subCode = currentClass ? currentClass.subject_code : 'GEN';
    const per = currentClass ? currentClass.period : 'Demo';
    const ttId = currentClass ? currentClass.id : null;
    try {
      const res = await axios.post('http://localhost:8000/api/take-attendance', {
        faculty_id: facultyId, faculty_name: facultyName,
        subject_name: subName, subject_code: subCode, period: per,
        session_type: 'regular', timetable_id: ttId
      }, { timeout: 70000 });
      setScanning(false); setCountdown(0);
      if (res.data.success) {
        setScanResult({ type:'success', message: res.data.message, total: res.data.total_marked||0, session_id: res.data.session_id });
        setStats(prev => ({ ...prev, classes_today: prev.classes_today + 1 }));
        // Auto-fetch review data
        if(res.data.session_id) {
          axios.get(`http://localhost:8000/api/session-attendance/${res.data.session_id}`)
            .then(r => { setReviewData(r.data); setShowReviewModal(true); }).catch(console.error);
        }
      } else { setScanResult({ type:'error', message: res.data.message }); }
    } catch (err) {
      setScanning(false); setCountdown(0);
      setScanResult({ type:'error', message: err.response?.data?.message || 'Camera error — check if camera is connected' });
    }
  };

  // EXTRA CLASS
  const handleExtraClass = async () => {
    if(!extraForm.subject_name || !extraForm.period) return;
    setExtraScanning(true);
    try {
      const res = await axios.post('http://localhost:8000/api/take-extra-class', {
        faculty_id: facultyId, faculty_name: facultyName, ...extraForm
      }, { timeout: 70000 });
      setExtraScanning(false); setShowExtraModal(false);
      if(res.data.success) {
        setScanResult({ type:'success', message: res.data.message, total: res.data.total_marked||0, session_id: res.data.session_id });
        setStats(prev => ({ ...prev, classes_today: prev.classes_today + 1 }));
        setExtraStats(prev => ({ ...prev, extra_taken: prev.extra_taken + 1 }));
        if(res.data.session_id) {
          axios.get(`http://localhost:8000/api/session-attendance/${res.data.session_id}`)
            .then(r => { setReviewData(r.data); setShowReviewModal(true); }).catch(console.error);
        }
      }
    } catch(err) { setExtraScanning(false); setScanResult({ type:'error', message:'Camera error' }); }
  };

  // VIEW SESSION DETAIL
  const handleViewSession = (sid) => {
    setSessionDetailId(sid);
    axios.get(`http://localhost:8000/api/session-attendance/${sid}`)
      .then(r => setSessionDetail(r.data)).catch(console.error);
  };


  const handleLogout = () => {
    localStorage.removeItem("facultyToken");
    localStorage.removeItem("facultyId");
    localStorage.removeItem("facultyName");
    navigate("/faculty/login");
  };

  return (
    <div style={{ backgroundColor: '#070b14', minHeight: '100vh', fontFamily: '"Segoe UI", system-ui, sans-serif' }}>
      {/* SECTION 1 — TOP NAVBAR */}
      <div style={{ backgroundColor: '#0d1117', borderBottom: '0.5px solid rgba(34,197,94,0.12)', padding: '0 28px', height: '58px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', boxSizing: 'border-box' }}>
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <div style={{ width: '32px', height: '32px', borderRadius: '8px', backgroundColor: '#16a34a', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="3" y1="9" x2="21" y2="9"></line><line x1="9" y1="21" x2="9" y2="9"></line></svg>
          </div>
          <div style={{ marginLeft: '12px' }}>
            <div style={{ fontSize: '14px', fontWeight: '700', color: '#f1f5f9' }}>Welcome, {facultyName}</div>
            <div style={{ fontSize: '10px', color: '#475569' }}>{facultyId} · CSE Department</div>
          </div>
          <div style={{ marginLeft: '12px', fontSize: '10px', color: '#22c55e', backgroundColor: 'rgba(34,197,94,0.1)', border: '0.5px solid rgba(34,197,94,0.2)', padding: '2px 8px', borderRadius: '4px' }}>
            Faculty Portal
          </div>
        </div>
        
        <div style={{ display: 'flex', alignItems: 'center', gap: '24px' }}>
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: '16px', fontWeight: '700', color: '#f1f5f9', fontVariantNumeric: 'tabular-nums' }}>{time || '--:--:--'}</div>
            <div style={{ fontSize: '10px', color: '#475569' }}>{dateStr || '-- --- ----'}</div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', backgroundColor: 'rgba(34,197,94,0.08)', border: '0.5px solid rgba(34,197,94,0.2)', borderRadius: '20px', padding: '4px 11px', fontSize: '11px', color: '#86efac' }}>
            <div style={{ width: '6px', height: '6px', backgroundColor: '#22c55e', borderRadius: '50%' }}></div>
            System Online
          </div>
          <button onClick={handleLogout} style={{ fontSize: '11px', color: '#475569', backgroundColor: 'transparent', border: '0.5px solid rgba(255,255,255,0.08)', borderRadius: '6px', padding: '5px 12px', cursor: 'pointer' }}>
            Logout
          </button>
        </div>
      </div>

      <div style={{ padding: '22px 28px' }}>
        {/* STATS ROW — 6 cards */}
        <div style={{ display:'grid', gridTemplateColumns:'repeat(6, 1fr)', gap:'10px', marginTop:'22px' }}>
          {[
            { label:'Classes Today', value:stats.classes_today, sub:'Sessions taken', color:'#22c55e' },
            { label:'Total Students', value:stats.total_students, sub:'Registered', color:'#818cf8' },
            { label:'Avg Attendance', value:stats.avg_attendance, sub:'This month', color:'#f59e0b' },
            { label:'Subjects', value:stats.subjects_assigned, sub:'Assigned', color:'#60a5fa' },
            { label:'Extra Classes', value:extraStats.extra_taken, sub:'Taken by you', color:'#c084fc' },
            { label:'Substituted', value:extraStats.classes_substituted, sub:'Your classes covered', color:'#fb923c' },
          ].map(s => (
            <div key={s.label} style={S.card}>
              <div style={{ fontSize:'10px', color:'#475569', marginBottom:'5px' }}>{s.label}</div>
              <div style={{ fontSize:'20px', fontWeight:'800', color:'#f1f5f9' }}>{s.value}</div>
              <div style={{ fontSize:'10px', color:s.color, marginTop:'3px' }}>{s.sub}</div>
            </div>
          ))}
        </div>

        {/* TIMETABLE BANNER */}
        {currentClass && (
          <div style={{ marginTop:'14px', padding:'14px 18px', borderRadius:'10px', backgroundColor: currentClass.upcoming ? 'rgba(245,158,11,0.08)' : 'rgba(34,197,94,0.08)', border: currentClass.upcoming ? '1px solid rgba(245,158,11,0.25)' : '1px solid rgba(34,197,94,0.25)', display:'flex', alignItems:'center', justifyContent:'space-between' }}>
            <div style={{ display:'flex', alignItems:'center', gap:'10px' }}>
              <span style={{ fontSize:'20px' }}>{currentClass.upcoming ? '⏰' : '📚'}</span>
              <div>
                <div style={{ fontSize:'13px', fontWeight:'700', color:'#f1f5f9' }}>
                  {currentClass.upcoming ? 'Upcoming' : 'Current'} Class: {currentClass.subject_name}
                </div>
                <div style={{ fontSize:'11px', color:'#94a3b8' }}>
                  {currentClass.period} · {currentClass.start_time}–{currentClass.end_time} · {currentClass.room || 'No room'}
                </div>
              </div>
            </div>
            <div style={{ fontSize:'11px', color: currentClass.upcoming ? '#fbbf24' : '#86efac', fontWeight:'600' }}>
              {currentClass.upcoming ? 'Starts soon' : 'In Progress'}
            </div>
          </div>
        )}

        {/* TAB NAVIGATION */}
        <div style={{ display:'flex', gap:'4px', marginTop:'18px', borderBottom:'1px solid rgba(255,255,255,0.05)', paddingBottom:'0' }}>
          {['dashboard','logbook'].map(tab => (
            <button key={tab} onClick={() => setActiveTab(tab)} style={{
              padding:'10px 20px', fontSize:'12px', fontWeight:'600', cursor:'pointer', border:'none',
              borderBottom: activeTab===tab ? '2px solid #22c55e' : '2px solid transparent',
              color: activeTab===tab ? '#f1f5f9' : '#475569',
              backgroundColor:'transparent', textTransform:'capitalize'
            }}>{tab === 'dashboard' ? '📷 Dashboard' : '📋 Attendance Logbook'}</button>
          ))}
        </div>

        {/* SECTION 3 — ONE-CLICK ATTENDANCE CARD */}
        <div style={{
          backgroundColor: '#0d1117',
          border: scanning ? '1px solid rgba(239,68,68,0.4)' : '0.5px solid rgba(34,197,94,0.15)',
          borderRadius: '12px', padding: '28px 24px', marginTop: '22px',
          transition: 'border-color 0.3s'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', paddingBottom: '16px', borderBottom: '0.5px solid rgba(255,255,255,0.05)', marginBottom: '20px' }}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#22c55e" strokeWidth="2" style={{ marginRight: '10px' }}><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path><circle cx="12" cy="13" r="4"></circle></svg>
            <span style={{ fontSize: '15px', fontWeight: '800', color: '#f1f5f9' }}>AI Attendance Scanner</span>
            <span style={{ fontSize: '11px', color: '#475569', marginLeft: 'auto' }}>YOLOv8 + DeepFace ArcFace Engine</span>
          </div>

          {/* BIG BUTTON */}
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '10px 0' }}>
            
            {scanning ? (
              /* SCANNING STATE */
              <div style={{ textAlign: 'center', width: '100%' }}>
                {/* Pulsing ring animation */}
                <div style={{ position: 'relative', width: '120px', height: '120px', margin: '0 auto 20px' }}>
                  <div style={{
                    position: 'absolute', inset: 0, borderRadius: '50%',
                    border: '3px solid rgba(239,68,68,0.3)',
                    animation: 'scanPulse 1.5s ease-in-out infinite'
                  }} />
                  <div style={{
                    position: 'absolute', inset: '10px', borderRadius: '50%',
                    border: '3px solid rgba(239,68,68,0.5)',
                    animation: 'scanPulse 1.5s ease-in-out infinite 0.3s'
                  }} />
                  <div style={{
                    position: 'absolute', inset: '20px', borderRadius: '50%',
                    backgroundColor: 'rgba(239,68,68,0.1)',
                    border: '2px solid rgba(239,68,68,0.6)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center'
                  }}>
                    <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#ef4444" strokeWidth="1.5">
                      <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path>
                      <circle cx="12" cy="13" r="4"></circle>
                    </svg>
                  </div>
                  <style>{`
                    @keyframes scanPulse {
                      0% { transform: scale(1); opacity: 1; }
                      50% { transform: scale(1.15); opacity: 0.4; }
                      100% { transform: scale(1); opacity: 1; }
                    }
                  `}</style>
                </div>

                <div style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', backgroundColor: 'rgba(239,68,68,0.1)', border: '0.5px solid rgba(239,68,68,0.25)', borderRadius: '20px', padding: '5px 14px', marginBottom: '12px' }}>
                  <div style={{ width: '8px', height: '8px', backgroundColor: '#ef4444', borderRadius: '50%', animation: 'blink 0.8s infinite' }} />
                  <span style={{ fontSize: '12px', fontWeight: '700', color: '#fca5a5' }}>LIVE — CAMERA ACTIVE</span>
                  <style>{`@keyframes blink { 0%,100% { opacity:1; } 50% { opacity:0.2; } }`}</style>
                </div>

                <div style={{ fontSize: '14px', color: '#f59e0b', fontWeight: '600', marginBottom: '6px' }}>
                  📸 Scanning Faces... Please wait {countdown > 0 ? `${countdown}s` : ''}
                </div>
                <div style={{ fontSize: '11px', color: '#475569' }}>
                  A camera window has opened on your desktop. Students should look at the camera.
                </div>
              </div>
            ) : scanResult ? (
              /* RESULT STATE */
              <div style={{ textAlign: 'center', width: '100%' }}>
                <div style={{
                  width: '80px', height: '80px', borderRadius: '50%', margin: '0 auto 16px',
                  backgroundColor: scanResult.type === 'success' ? 'rgba(34,197,94,0.1)' : 'rgba(239,68,68,0.1)',
                  border: `2px solid ${scanResult.type === 'success' ? 'rgba(34,197,94,0.3)' : 'rgba(239,68,68,0.3)'}`,
                  display: 'flex', alignItems: 'center', justifyContent: 'center'
                }}>
                  {scanResult.type === 'success' ? (
                    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#22c55e" strokeWidth="2"><polyline points="20 6 9 17 4 12"></polyline></svg>
                  ) : (
                    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#ef4444" strokeWidth="2"><circle cx="12" cy="12" r="10"></circle><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line></svg>
                  )}
                </div>
                <div style={{ fontSize: '16px', fontWeight: '700', color: scanResult.type === 'success' ? '#86efac' : '#fca5a5', marginBottom: '6px' }}>
                  {scanResult.message}
                </div>
                {scanResult.total > 0 && (
                  <div style={{ fontSize: '26px', fontWeight: '800', color: '#22c55e', marginBottom: '4px' }}>{scanResult.total} Students</div>
                )}
                <div style={{ fontSize: '11px', color: '#475569', marginBottom: '16px' }}>marked present via face recognition</div>
                
                <button onClick={() => setScanResult(null)} style={{
                  backgroundColor: 'rgba(34,197,94,0.1)', border: '0.5px solid rgba(34,197,94,0.25)',
                  color: '#86efac', borderRadius: '8px', padding: '8px 20px', fontSize: '12px',
                  fontWeight: '600', cursor: 'pointer'
                }}>
                  ← Back to Scanner
                </button>
              </div>
            ) : (
              /* IDLE STATE — BIG BUTTON */
              <>
                <div style={{
                  width: '100px', height: '100px', borderRadius: '50%', margin: '0 auto 20px',
                  background: 'linear-gradient(135deg, rgba(34,197,94,0.15), rgba(34,197,94,0.05))',
                  border: '2px solid rgba(34,197,94,0.25)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center'
                }}>
                  <svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="#22c55e" strokeWidth="1.5">
                    <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path>
                    <circle cx="12" cy="13" r="4"></circle>
                  </svg>
                </div>

                <button
                  onClick={handleTakeAttendance}
                  style={{
                    backgroundColor: '#16a34a', color: '#fff', border: 'none',
                    borderRadius: '10px', padding: '16px 48px', fontSize: '16px',
                    fontWeight: '800', cursor: 'pointer', display: 'flex',
                    alignItems: 'center', gap: '10px', marginBottom: '14px',
                    boxShadow: '0 0 30px rgba(34,197,94,0.2)',
                    transition: 'all 0.2s'
                  }}
                  onMouseEnter={e => e.currentTarget.style.transform = 'scale(1.02)'}
                  onMouseLeave={e => e.currentTarget.style.transform = 'scale(1)'}
                >
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2">
                    <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path>
                    <circle cx="12" cy="13" r="4"></circle>
                  </svg>
                  Take Attendance
                </button>

                <button onClick={() => setShowExtraModal(true)} style={S.btn2}>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
                  Take Extra Class
                </button>

                <div style={{ fontSize:'11px', color:'#475569', textAlign:'center' }}>
                  {currentClass ? `Auto-detected: ${currentClass.subject_name} — ${currentClass.period}` : 'No scheduled class — will use General mode'}
                </div>
              </>
            )}
          </div>
        </div>

        {activeTab === 'logbook' && (
        <div style={{ backgroundColor:'#0d1117', border:'0.5px solid rgba(255,255,255,0.05)', borderRadius:'12px', padding:'18px 20px', marginTop:'18px' }}>
          <div style={{ display:'flex', alignItems:'center', gap:'8px', paddingBottom:'13px', borderBottom:'0.5px solid rgba(255,255,255,0.05)', marginBottom:'10px' }}>
            <span style={{ fontSize:'13px', fontWeight:'700', color:'#f1f5f9' }}>📋 Attendance Logbook</span>
            <span style={{ fontSize:'11px', color:'#475569', marginLeft:'auto' }}>{logbookData.length} sessions</span>
          </div>
          {logbookData.length === 0 ? (
            <div style={{ textAlign:'center', padding:'40px 0', color:'#475569', fontSize:'13px' }}>No sessions recorded yet</div>
          ) : (
          <table style={{ width:'100%', borderCollapse:'collapse' }}>
            <thead><tr>
              {['DATE','SUBJECT','PERIOD','TYPE','PRESENT','ACTION'].map(h => (
                <th key={h} style={{ fontSize:'10px', color:'#334155', fontWeight:'600', letterSpacing:'.5px', padding:'8px 10px', textAlign:'left', borderBottom:'0.5px solid rgba(255,255,255,0.05)' }}>{h}</th>
              ))}
            </tr></thead>
            <tbody>
              {logbookData.map(s => {
                const pct = s.total_present > 0 ? Math.round((s.total_present / Math.max(logbookTotal,1)) * 100) : 0;
                return (
                <tr key={s.id} style={{ cursor:'pointer' }} onClick={() => handleViewSession(s.id)}>
                  <td style={{ fontSize:'12px', color:'#e2e8f0', padding:'10px', borderBottom:'0.5px solid rgba(255,255,255,0.03)' }}>
                    <div>{s.date}</div><div style={{ fontSize:'10px', color:'#475569' }}>{s.start_time}</div>
                  </td>
                  <td style={{ fontSize:'12px', color:'#e2e8f0', fontWeight:'500', padding:'10px', borderBottom:'0.5px solid rgba(255,255,255,0.03)' }}>{s.subject_name}</td>
                  <td style={{ fontSize:'12px', color:'#94a3b8', padding:'10px', borderBottom:'0.5px solid rgba(255,255,255,0.03)' }}>{s.period}</td>
                  <td style={{ padding:'10px', borderBottom:'0.5px solid rgba(255,255,255,0.03)' }}>
                    <span style={{ fontSize:'10px', padding:'2px 8px', borderRadius:'5px',
                      backgroundColor: s.session_type==='extra' ? 'rgba(168,85,247,0.1)' : 'rgba(34,197,94,0.08)',
                      border: s.session_type==='extra' ? '0.5px solid rgba(168,85,247,0.3)' : '0.5px solid rgba(34,197,94,0.2)',
                      color: s.session_type==='extra' ? '#c084fc' : '#86efac' }}>
                      {s.session_type==='extra' ? '⚡ Extra' : '📅 Regular'}
                    </span>
                  </td>
                  <td style={{ fontSize:'12px', color: pct>=75?'#22c55e':'#ef4444', padding:'10px', borderBottom:'0.5px solid rgba(255,255,255,0.03)' }}>
                    {s.total_present}/{logbookTotal} ({pct}%)
                  </td>
                  <td style={{ padding:'10px', borderBottom:'0.5px solid rgba(255,255,255,0.03)' }}>
                    <button style={{ fontSize:'10px', color:'#60a5fa', backgroundColor:'rgba(96,165,250,0.08)', border:'0.5px solid rgba(96,165,250,0.2)', borderRadius:'5px', padding:'3px 10px', cursor:'pointer' }}>View</button>
                  </td>
                </tr>);
              })}
            </tbody>
          </table>)}
        </div>)}
      </div>

      {/* === REVIEW ATTENDANCE MODAL === */}
      {showReviewModal && (
        <div style={S.modal}>
          <div style={S.overlay} onClick={() => setShowReviewModal(false)} />
          <div style={{...S.mbox, maxWidth:'700px'}}>
            <h3 style={{ color:'#f1f5f9', margin:'0 0 4px', fontSize:'18px' }}>✅ Attendance Review</h3>
            <p style={{ color:'#94a3b8', fontSize:'12px', margin:'0 0 16px' }}>Session completed — {reviewData.total_present} / {reviewData.total_students} students present</p>
            <div style={{ backgroundColor:'rgba(34,197,94,0.06)', border:'1px solid rgba(34,197,94,0.2)', borderRadius:'10px', padding:'16px', textAlign:'center', marginBottom:'16px' }}>
              <div style={{ fontSize:'36px', fontWeight:'800', color:'#22c55e' }}>{reviewData.total_present}</div>
              <div style={{ fontSize:'12px', color:'#86efac' }}>out of {reviewData.total_students} students present</div>
            </div>
            {reviewData.students?.length > 0 && (
              <table style={{ width:'100%', borderCollapse:'collapse' }}>
                <thead><tr>{['#','Name','Roll No','Time'].map(h => (<th key={h} style={{ fontSize:'10px', color:'#475569', padding:'6px 8px', textAlign:'left', borderBottom:'0.5px solid rgba(255,255,255,0.05)' }}>{h}</th>))}</tr></thead>
                <tbody>{reviewData.students.map((s,i) => (
                  <tr key={i}><td style={{ fontSize:'12px', color:'#475569', padding:'6px 8px', borderBottom:'0.5px solid rgba(255,255,255,0.03)' }}>{i+1}</td>
                  <td style={{ fontSize:'12px', color:'#e2e8f0', padding:'6px 8px', borderBottom:'0.5px solid rgba(255,255,255,0.03)' }}>{s.name}</td>
                  <td style={{ fontSize:'12px', color:'#94a3b8', padding:'6px 8px', borderBottom:'0.5px solid rgba(255,255,255,0.03)' }}>{s.roll_number||'N/A'}</td>
                  <td style={{ fontSize:'12px', color:'#94a3b8', padding:'6px 8px', borderBottom:'0.5px solid rgba(255,255,255,0.03)' }}>{s.time}</td></tr>
                ))}</tbody>
              </table>)}
            <button onClick={() => setShowReviewModal(false)} style={{ marginTop:'16px', width:'100%', backgroundColor:'#16a34a', color:'#fff', border:'none', borderRadius:'8px', padding:'11px', fontSize:'13px', fontWeight:'700', cursor:'pointer' }}>Close Review</button>
          </div>
        </div>)}

      {/* === EXTRA CLASS MODAL === */}
      {showExtraModal && (
        <div style={S.modal}>
          <div style={S.overlay} onClick={() => !extraScanning && setShowExtraModal(false)} />
          <div style={S.mbox}>
            <h3 style={{ color:'#f1f5f9', margin:'0 0 4px', fontSize:'18px' }}>⚡ Take Extra Class</h3>
            <p style={{ color:'#94a3b8', fontSize:'12px', margin:'0 0 20px' }}>Start an extra/substitute class attendance session</p>
            <div style={{ display:'flex', flexDirection:'column', gap:'14px' }}>
              <div><label style={{ fontSize:'11px', color:'#94a3b8', display:'block', marginBottom:'5px' }}>Subject *</label>
                <select style={{...S.inp, cursor:'pointer'}} value={extraForm.subject_name} onChange={e => {
                  const sel = subjects.find(s => s.subject_name === e.target.value);
                  setExtraForm(p => ({...p, subject_name: e.target.value, subject_code: sel?.subject_code||'' }));
                }}>
                  <option value="" style={{backgroundColor:'#0d1117', color:'#f1f5f9'}}>Select subject...</option>
                  {subjects.map(s => <option key={s.subject_name} value={s.subject_name} style={{backgroundColor:'#0d1117', color:'#f1f5f9'}}>{s.subject_name} ({s.subject_code})</option>)}
                </select>
              </div>
              <div><label style={{ fontSize:'11px', color:'#94a3b8', display:'block', marginBottom:'5px' }}>Period *</label>
                <select style={{...S.inp, cursor:'pointer'}} value={extraForm.period} onChange={e => setExtraForm(p => ({...p, period: e.target.value}))}>
                  <option value="" style={{backgroundColor:'#0d1117', color:'#f1f5f9'}}>Select period...</option>
                  {[1,2,3,4,5,6,7,8].map(n => <option key={n} value={`Period ${n}`} style={{backgroundColor:'#0d1117', color:'#f1f5f9'}}>Period {n}</option>)}
                </select>
              </div>
              <div><label style={{ fontSize:'11px', color:'#94a3b8', display:'block', marginBottom:'5px' }}>Original Faculty (optional)</label>
                <input style={S.inp} placeholder="Name of faculty being substituted" value={extraForm.original_faculty_name} onChange={e => setExtraForm(p => ({...p, original_faculty_name: e.target.value}))} />
              </div>
              <div><label style={{ fontSize:'11px', color:'#94a3b8', display:'block', marginBottom:'5px' }}>📷 Camera / Room (optional)</label>
                <select style={{...S.inp, cursor:'pointer'}} value={extraForm.room_id} onChange={e => setExtraForm(p => ({...p, room_id: e.target.value}))}>
                  <option value="" style={{backgroundColor:'#0d1117', color:'#f1f5f9'}}>🖥️ Use Laptop Camera (default)</option>
                  {rooms.map(r => <option key={r.id} value={r.id} style={{backgroundColor:'#0d1117', color:'#f1f5f9'}}>📱 {r.room_name} — {r.rtsp_url}</option>)}
                </select>
              </div>
            </div>
            <div style={{ display:'flex', gap:'10px', marginTop:'20px' }}>
              <button onClick={() => setShowExtraModal(false)} disabled={extraScanning} style={{ flex:1, backgroundColor:'transparent', color:'#94a3b8', border:'1px solid rgba(255,255,255,0.1)', borderRadius:'8px', padding:'11px', fontSize:'13px', cursor:'pointer' }}>Cancel</button>
              <button onClick={handleExtraClass} disabled={extraScanning || !extraForm.subject_name || !extraForm.period} style={{ flex:1, backgroundColor:'#7c3aed', color:'#fff', border:'none', borderRadius:'8px', padding:'11px', fontSize:'13px', fontWeight:'700', cursor:'pointer', opacity: (!extraForm.subject_name || !extraForm.period) ? 0.5 : 1 }}>
                {extraScanning ? 'Scanning...' : '🎯 Start Scan'}
              </button>
            </div>
          </div>
        </div>)}

      {/* === SESSION DETAIL MODAL === */}
      {sessionDetailId && sessionDetail && (
        <div style={S.modal}>
          <div style={S.overlay} onClick={() => { setSessionDetailId(null); setSessionDetail(null); }} />
          <div style={{...S.mbox, maxWidth:'700px'}}>
            <h3 style={{ color:'#f1f5f9', margin:'0 0 4px', fontSize:'18px' }}>📋 Session #{sessionDetailId} — Details</h3>
            <p style={{ color:'#94a3b8', fontSize:'12px', margin:'0 0 16px' }}>{sessionDetail.total_present} / {sessionDetail.total_students} students present</p>
            {sessionDetail.students?.length > 0 ? (
              <table style={{ width:'100%', borderCollapse:'collapse' }}>
                <thead><tr>{['#','Name','Roll No','Time'].map(h => (<th key={h} style={{ fontSize:'10px', color:'#475569', padding:'6px 8px', textAlign:'left', borderBottom:'0.5px solid rgba(255,255,255,0.05)' }}>{h}</th>))}</tr></thead>
                <tbody>{sessionDetail.students.map((s,i) => (
                  <tr key={i}><td style={{ fontSize:'12px', color:'#475569', padding:'6px 8px', borderBottom:'0.5px solid rgba(255,255,255,0.03)' }}>{i+1}</td>
                  <td style={{ fontSize:'12px', color:'#e2e8f0', padding:'6px 8px', borderBottom:'0.5px solid rgba(255,255,255,0.03)' }}>{s.name}</td>
                  <td style={{ fontSize:'12px', color:'#94a3b8', padding:'6px 8px', borderBottom:'0.5px solid rgba(255,255,255,0.03)' }}>{s.roll_number||'N/A'}</td>
                  <td style={{ fontSize:'12px', color:'#94a3b8', padding:'6px 8px', borderBottom:'0.5px solid rgba(255,255,255,0.03)' }}>{s.time}</td></tr>
                ))}</tbody>
              </table>
            ) : <div style={{ textAlign:'center', padding:'30px', color:'#475569', fontSize:'13px' }}>No students recorded</div>}
            <button onClick={() => { setSessionDetailId(null); setSessionDetail(null); }} style={{ marginTop:'16px', width:'100%', backgroundColor:'rgba(255,255,255,0.05)', color:'#94a3b8', border:'1px solid rgba(255,255,255,0.1)', borderRadius:'8px', padding:'11px', fontSize:'13px', cursor:'pointer' }}>Close</button>
          </div>
        </div>)}
    </div>
  );
}
