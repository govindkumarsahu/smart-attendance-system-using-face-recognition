import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

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
  
  const [stats, setStats] = useState({
    classes_today: 0,
    total_students: 60,
    avg_attendance: "0%",
    subjects_assigned: 0
  });

  const [logData, setLogData] = useState([
    { id: 1, dateStr: "04 May 2026", timeStr: "10:00AM", subject: "Machine Learning", present: 47, total: 60, pct: 78, status: "Saved" },
    { id: 2, dateStr: "04 May 2026", timeStr: "8:00AM", subject: "Deep Learning", present: 52, total: 60, pct: 87, status: "Saved" },
    { id: 3, dateStr: "03 May 2026", timeStr: "2:00PM", subject: "Soft Computing", present: 38, total: 60, pct: 63, status: "Low" },
    { id: 4, dateStr: "03 May 2026", timeStr: "10:00AM", subject: "NLP", present: 55, total: 60, pct: 92, status: "Saved" },
    { id: 5, dateStr: "02 May 2026", timeStr: "11:00AM", subject: "Cloud Computing", present: 50, total: 60, pct: 83, status: "Saved" },
  ]);

  // Clock
  useEffect(() => {
    const timer = setInterval(() => {
      const now = new Date();
      setTime(now.toLocaleTimeString('en-US', { hour12: true, hour: '2-digit', minute: '2-digit', second: '2-digit' }));
      setDateStr(now.toLocaleDateString('en-GB', { weekday: 'long', day: '2-digit', month: 'short', year: 'numeric' }));
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  // Fetch stats
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
  }, [facultyId]);

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
    setCountdown(25); // ~20s recognition + ~5s init
    
    try {
      const res = await axios.post('http://localhost:8000/api/take-attendance', {
        faculty_id: facultyId,
        faculty_name: facultyName,
        subject_name: 'General',
        subject_code: 'GEN',
        period: 'Demo'
      }, { timeout: 70000 });
      
      setScanning(false);
      setCountdown(0);
      
      if (res.data.success) {
        setScanResult({
          type: 'success',
          message: res.data.message || 'Attendance completed!',
          total: res.data.total_marked || 0
        });
        
        // Add to logbook
        const now = new Date();
        setLogData(prev => [{
          id: Date.now(),
          dateStr: now.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' }),
          timeStr: now.toLocaleTimeString('en-US', { hour12: true, hour: '2-digit', minute: '2-digit' }),
          subject: "Attendance Scan",
          present: res.data.total_marked || 0,
          total: 60,
          pct: Math.round(((res.data.total_marked || 0) / 60) * 100),
          status: (res.data.total_marked || 0) > 0 ? "Saved" : "Low"
        }, ...prev]);
        
        // Update stats
        setStats(prev => ({ ...prev, classes_today: prev.classes_today + 1 }));
      } else {
        setScanResult({ type: 'error', message: res.data.message });
      }
    } catch (err) {
      setScanning(false);
      setCountdown(0);
      setScanResult({
        type: 'error',
        message: err.response?.data?.message || 'Camera error — check if camera is connected'
      });
    }
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
        {/* SECTION 2 — STATS ROW */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px', marginTop: '22px' }}>
          {[
            { label: 'Classes Today', value: stats.classes_today, sub: 'Sessions taken', color: '#22c55e' },
            { label: 'Total Students', value: stats.total_students, sub: 'Assigned batch', color: '#818cf8' },
            { label: 'Avg Attendance', value: stats.avg_attendance, sub: 'This month', color: '#f59e0b' },
            { label: 'Subjects Assigned', value: stats.subjects_assigned, sub: 'Active subjects', color: '#60a5fa' }
          ].map(stat => (
            <div key={stat.label} style={{ backgroundColor: '#0d1117', border: '0.5px solid rgba(255,255,255,0.05)', borderRadius: '10px', padding: '14px 16px' }}>
              <div style={{ fontSize: '10px', color: '#475569', marginBottom: '5px' }}>{stat.label}</div>
              <div style={{ fontSize: '22px', fontWeight: '800', color: '#f1f5f9' }}>{stat.value}</div>
              <div style={{ fontSize: '10px', color: stat.color, marginTop: '3px' }}>{stat.sub}</div>
            </div>
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
                  onMouseEnter={e => { e.target.style.backgroundColor = '#15803d'; e.target.style.transform = 'scale(1.02)'; }}
                  onMouseLeave={e => { e.target.style.backgroundColor = '#16a34a'; e.target.style.transform = 'scale(1)'; }}
                >
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2">
                    <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path>
                    <circle cx="12" cy="13" r="4"></circle>
                  </svg>
                  Take Attendance (20 Sec Scan)
                </button>

                <div style={{ fontSize: '11px', color: '#475569', textAlign: 'center' }}>
                  Click to open laptop camera · YOLOv8 detects faces · DeepFace matches with registered students · Auto-marks attendance
                </div>
              </>
            )}
          </div>
        </div>

        {/* SECTION 4 — MY CLASS LOGBOOK TABLE */}
        <div style={{ backgroundColor: '#0d1117', border: '0.5px solid rgba(255,255,255,0.05)', borderRadius: '12px', padding: '18px 20px', marginTop: '22px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', paddingBottom: '13px', borderBottom: '0.5px solid rgba(255,255,255,0.05)', marginBottom: '10px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#60a5fa" strokeWidth="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line></svg>
              <span style={{ fontSize: '13px', fontWeight: '700', color: '#f1f5f9' }}>My Class Logbook</span>
            </div>
          </div>

          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                {["DATE & TIME", "SUBJECT", "PRESENT/TOTAL", "STATUS"].map(th => (
                  <th key={th} style={{ fontSize: '10px', color: '#334155', fontWeight: '600', letterSpacing: '.5px', padding: '8px 10px', textAlign: 'left', borderBottom: '0.5px solid rgba(255,255,255,0.05)' }}>{th}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {logData.map(log => {
                const isGood = log.pct >= 75;
                const statusColor = isGood ? '#86efac' : '#fbbf24';
                const statusBg = isGood ? 'rgba(34,197,94,0.08)' : 'rgba(245,158,11,0.08)';
                const statusBorder = isGood ? 'rgba(34,197,94,0.2)' : 'rgba(245,158,11,0.2)';

                return (
                  <tr key={log.id}>
                    <td style={{ fontSize: '12px', color: '#94a3b8', padding: '10px 10px', borderBottom: '0.5px solid rgba(255,255,255,0.03)' }}>
                      <div style={{ fontSize: '12px', color: '#e2e8f0', fontWeight: '500' }}>{log.dateStr}</div>
                      <div style={{ fontSize: '10px', color: '#475569' }}>{log.timeStr}</div>
                    </td>
                    <td style={{ fontSize: '12px', color: '#e2e8f0', fontWeight: '500', padding: '10px 10px', borderBottom: '0.5px solid rgba(255,255,255,0.03)' }}>{log.subject}</td>
                    <td style={{ fontSize: '12px', color: '#94a3b8', padding: '10px 10px', borderBottom: '0.5px solid rgba(255,255,255,0.03)' }}>
                      <div style={{ color: isGood ? '#22c55e' : '#ef4444' }}>{log.present} / {log.total} ({log.pct}%)</div>
                      <div style={{ height: '3px', width: '60px', borderRadius: '2px', backgroundColor: 'rgba(255,255,255,0.06)', marginTop: '4px' }}>
                        <div style={{ height: '100%', width: `${log.pct}%`, backgroundColor: isGood ? '#22c55e' : '#ef4444', borderRadius: '2px' }}></div>
                      </div>
                    </td>
                    <td style={{ fontSize: '12px', color: '#94a3b8', padding: '10px 10px', borderBottom: '0.5px solid rgba(255,255,255,0.03)' }}>
                      <div style={{ display: 'inline-flex', alignItems: 'center', gap: '4px', fontSize: '10px', padding: '2px 8px', borderRadius: '5px', backgroundColor: statusBg, border: `0.5px solid ${statusBorder}`, color: statusColor }}>
                        {isGood ? (
                          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="20 6 9 17 4 12"></polyline></svg>
                        ) : (
                          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
                        )}
                        {log.status}
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
