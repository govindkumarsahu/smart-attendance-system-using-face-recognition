import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

const EMPTY_DATA = {
  overall_pct: 0,
  attended: 0,
  total: 0,
  subjects: [],
  history: []
};

function subjectColor(pct) {
  if (pct >= 75) return '#22c55e';
  if (pct >= 65) return '#f59e0b';
  return '#ef4444';
}

export default function StudentDashboard() {
  const navigate = useNavigate();
  const studentName = localStorage.getItem("studentName") || "Student";
  const studentRoll = localStorage.getItem("studentRoll") || "CS2024001";
  const studentBranch = localStorage.getItem("studentBranch") || "CSE-AI";
  const studentSem = localStorage.getItem("studentSem") || "8th Sem";

  const [stats, setStats] = useState(EMPTY_DATA);
  const [loading, setLoading] = useState(true);
  const [historyFilter, setHistoryFilter] = useState("Last 5 Classes");

  useEffect(() => {
    const roll = localStorage.getItem("studentRoll");
    if (!roll) { navigate("/student/login"); return; }
    axios.get(`http://localhost:8000/api/student-stats/${roll}`)
      .then(res => { if (res.data && res.data.overall_pct !== undefined) setStats(res.data); })
      .catch(() => setStats(EMPTY_DATA))
      .finally(() => setLoading(false));
  }, [navigate]);

  const handleLogout = () => {
    localStorage.removeItem("studentToken");
    localStorage.removeItem("studentName");
    localStorage.removeItem("studentRoll");
    localStorage.removeItem("studentBranch");
    localStorage.removeItem("studentSem");
    navigate("/student/login");
  };

  const isLow = stats.overall_pct < 75;
  const ringColor = isLow ? '#ef4444' : '#22c55e';
  const circumference = 2 * Math.PI * 60;
  const dashOffset = circumference * (1 - stats.overall_pct / 100);
  const classesNeeded = isLow ? Math.ceil((0.75 * stats.total - stats.attended) / 0.25) : 0;
  const avatarLetter = studentName.charAt(0).toUpperCase();

  return (
    <div style={{ backgroundColor: '#070b14', minHeight: '100vh', fontFamily: '"Segoe UI", system-ui, sans-serif' }}>

      {/* ── SECTION 1: TOPBAR ── */}
      <div style={{
        backgroundColor: '#0d1117',
        borderBottom: '0.5px solid rgba(129,140,248,0.15)',
        height: '56px', padding: '0 28px',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between', boxSizing: 'border-box'
      }}>
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <div style={{ width: '30px', height: '30px', borderRadius: '7px', backgroundColor: '#6366f1', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <svg width="15" height="15" viewBox="0 0 24 24" fill="white">
              <path d="M12 12c2.7 0 4.8-2.1 4.8-4.8S14.7 2.4 12 2.4 7.2 4.5 7.2 7.2 9.3 12 12 12zm0 2.4c-3.2 0-9.6 1.6-9.6 4.8v2.4h19.2v-2.4c0-3.2-6.4-4.8-9.6-4.8z"/>
            </svg>
          </div>
          <span style={{ fontSize: '13px', fontWeight: '700', color: '#f1f5f9', marginLeft: '10px' }}>SmartAttend AI</span>
          <span style={{ fontSize: '10px', color: '#818cf8', backgroundColor: 'rgba(129,140,248,0.12)', border: '0.5px solid rgba(129,140,248,0.25)', padding: '2px 8px', borderRadius: '4px', marginLeft: '6px' }}>
            Student Portal
          </span>
        </div>
        <button onClick={handleLogout} style={{ fontSize: '11px', color: '#475569', backgroundColor: 'transparent', border: '0.5px solid rgba(255,255,255,0.08)', borderRadius: '6px', padding: '5px 12px', cursor: 'pointer' }}>
          Logout
        </button>
      </div>

      <div style={{ padding: '20px 28px', display: 'flex', flexDirection: 'column', gap: '16px' }}>

        {/* ── SECTION 2: PROFILE HEADER CARD ── */}
        <div style={{
          backgroundColor: '#0d1117', border: '0.5px solid rgba(129,140,248,0.18)',
          borderRadius: '12px', padding: '20px 24px',
          display: 'flex', alignItems: 'center', gap: '20px',
          position: 'relative', overflow: 'hidden'
        }}>
          {/* Glow */}
          <div style={{ position: 'absolute', top: '-30px', left: '-30px', width: '140px', height: '140px', background: 'radial-gradient(circle, rgba(99,102,241,0.12) 0%, transparent 70%)', pointerEvents: 'none' }} />

          {/* Avatar */}
          <div style={{
            width: '58px', height: '58px', borderRadius: '50%', flexShrink: 0,
            background: 'linear-gradient(135deg, #6366f1, #818cf8)',
            border: '2.5px solid rgba(129,140,248,0.35)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: '22px', fontWeight: '800', color: '#fff'
          }}>
            {avatarLetter}
          </div>

          {/* Info */}
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: '11px', color: '#64748b', marginBottom: '3px' }}>Welcome back 👋</div>
            <div style={{ fontSize: '18px', fontWeight: '800', color: '#f1f5f9', letterSpacing: '-.4px', marginBottom: '8px' }}>{studentName}</div>
            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
              <span style={{ fontSize: '11px', padding: '3px 10px', borderRadius: '6px', backgroundColor: 'rgba(129,140,248,0.12)', color: '#a5b4fc', border: '0.5px solid rgba(129,140,248,0.25)' }}>
                {studentRoll}
              </span>
              <span style={{ fontSize: '11px', padding: '3px 10px', borderRadius: '6px', backgroundColor: 'rgba(34,197,94,0.1)', color: '#86efac', border: '0.5px solid rgba(34,197,94,0.2)' }}>
                {studentBranch}
              </span>
              <span style={{ fontSize: '11px', padding: '3px 10px', borderRadius: '6px', backgroundColor: 'rgba(96,165,250,0.1)', color: '#93c5fd', border: '0.5px solid rgba(96,165,250,0.2)' }}>
                {studentSem}
              </span>
            </div>
          </div>

          {/* Overall % right */}
          <div style={{ textAlign: 'right', flexShrink: 0 }}>
            <div style={{ fontSize: '10px', color: '#475569', marginBottom: '4px' }}>Overall Attendance</div>
            <div style={{ fontSize: '28px', fontWeight: '800', letterSpacing: '-1px', color: isLow ? '#ef4444' : '#22c55e' }}>
              {stats.overall_pct}%
            </div>
            <div style={{ fontSize: '10px', color: '#475569', marginTop: '2px' }}>Across all subjects</div>
          </div>
        </div>

        {/* ── SECTION 3: MAIN GRID ── */}
        <div style={{ display: 'grid', gridTemplateColumns: '260px 1fr', gap: '16px' }}>

          {/* ── LEFT: RING CARD ── */}
          <div style={{ backgroundColor: '#0d1117', border: '0.5px solid rgba(255,255,255,0.05)', borderRadius: '12px', padding: '20px', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <div style={{ alignSelf: 'flex-start', fontSize: '12px', fontWeight: '700', color: '#94a3b8', marginBottom: '16px' }}>Overall Attendance</div>

            {/* SVG Ring */}
            <div style={{ position: 'relative', width: '150px', height: '150px', marginBottom: '16px' }}>
              <svg width="150" height="150" style={{ transform: 'rotate(-90deg)' }}>
                {/* Track */}
                <circle cx="75" cy="75" r="60" fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="12" />
                {/* Progress */}
                <circle
                  cx="75" cy="75" r="60" fill="none"
                  stroke={ringColor} strokeWidth="12" strokeLinecap="round"
                  strokeDasharray={circumference}
                  strokeDashoffset={dashOffset}
                  style={{ transition: 'stroke-dashoffset 0.6s ease, stroke 0.6s ease' }}
                />
              </svg>
              {/* Center Text */}
              <div style={{
                position: 'absolute', top: '50%', left: '50%',
                transform: 'translate(-50%, -50%)', textAlign: 'center'
              }}>
                <div style={{ fontSize: '26px', fontWeight: '800', color: ringColor, lineHeight: 1 }}>
                  {stats.overall_pct}%
                </div>
                <div style={{ fontSize: '10px', color: '#475569', marginTop: '2px' }}>Present</div>
              </div>
            </div>

            {/* Status Box */}
            {isLow ? (
              <div style={{
                width: '100%', backgroundColor: 'rgba(239,68,68,0.08)',
                border: '0.5px solid rgba(239,68,68,0.25)', borderRadius: '8px',
                padding: '10px 12px', display: 'flex', gap: '7px', fontSize: '11px', color: '#fca5a5'
              }}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="#fca5a5" style={{ flexShrink: 0, marginTop: '1px' }}>
                  <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/>
                </svg>
                <span>Warning! Below 75%. Risk of detention.</span>
              </div>
            ) : (
              <div style={{
                width: '100%', backgroundColor: 'rgba(34,197,94,0.07)',
                border: '0.5px solid rgba(34,197,94,0.2)', borderRadius: '8px',
                padding: '10px 12px', display: 'flex', gap: '7px', fontSize: '11px', color: '#86efac'
              }}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="#86efac" style={{ flexShrink: 0, marginTop: '1px' }}>
                  <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41L9 16.17z"/>
                </svg>
                <span>Good standing! Maintain ≥75% to stay eligible.</span>
              </div>
            )}

            {/* Required Classes Calculator */}
            {isLow && (
              <div style={{
                marginTop: '10px', width: '100%',
                backgroundColor: 'rgba(245,158,11,0.07)', border: '0.5px solid rgba(245,158,11,0.2)',
                borderRadius: '8px', padding: '8px 12px', fontSize: '11px', color: '#fbbf24'
              }}>
                <div style={{ fontWeight: '700', marginBottom: '3px' }}>Classes needed to reach 75%</div>
                <div>Attend next <span style={{ fontSize: '16px', fontWeight: '800' }}>{classesNeeded}</span> classes consecutively</div>
              </div>
            )}

            {/* Progress Bar */}
            <div style={{ marginTop: '12px', width: '100%' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', marginBottom: '6px' }}>
                <span style={{ color: '#475569' }}>Classes Attended</span>
                <span style={{ color: '#f1f5f9', fontWeight: '600' }}>{stats.attended} / {stats.total}</span>
              </div>
              <div style={{ height: '3px', backgroundColor: 'rgba(255,255,255,0.06)', borderRadius: '2px' }}>
                <div style={{
                  height: '100%', width: `${stats.overall_pct}%`,
                  backgroundColor: ringColor, borderRadius: '2px',
                  transition: 'width 0.6s ease'
                }} />
              </div>
            </div>
          </div>

          {/* ── RIGHT COLUMN ── */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>

            {/* Subject-wise Analytics Card */}
            <div style={{ backgroundColor: '#0d1117', border: '0.5px solid rgba(255,255,255,0.05)', borderRadius: '12px', padding: '16px 18px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '7px', marginBottom: '13px' }}>
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#818cf8" strokeWidth="2">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                  <polyline points="14 2 14 8 20 8"/>
                  <line x1="16" y1="13" x2="8" y2="13"/>
                  <line x1="16" y1="17" x2="8" y2="17"/>
                </svg>
                <span style={{ fontSize: '12px', fontWeight: '700', color: '#94a3b8' }}>Subject-wise Analytics</span>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                {stats.subjects.length > 0 ? stats.subjects.map((sub, i) => {
                  const col = subjectColor(sub.pct);
                  return (
                    <SubjectCell key={i} sub={sub} color={col} />
                  );
                }) : <div style={{color:'#64748b', fontSize:'12px', padding:'10px', gridColumn:'span 2', textAlign:'center'}}>No subject data recorded yet.</div>}
              </div>
            </div>

            {/* Recent Class History Card */}
            <div style={{ backgroundColor: '#0d1117', border: '0.5px solid rgba(255,255,255,0.05)', borderRadius: '12px', padding: '16px 18px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '7px', marginBottom: '12px' }}>
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#60a5fa" strokeWidth="2">
                  <rect x="3" y="4" width="18" height="18" rx="2" ry="2"/>
                  <line x1="16" y1="2" x2="16" y2="6"/>
                  <line x1="8" y1="2" x2="8" y2="6"/>
                  <line x1="3" y1="10" x2="21" y2="10"/>
                </svg>
                <span style={{ fontSize: '12px', fontWeight: '700', color: '#94a3b8' }}>Recent Class History</span>
                <select
                  value={historyFilter}
                  onChange={e => setHistoryFilter(e.target.value)}
                  style={{ marginLeft: 'auto', backgroundColor: '#0d1117', border: '0.5px solid rgba(255,255,255,0.1)', borderRadius: '6px', padding: '4px 9px', color: '#f1f5f9', fontSize: '11px', outline: 'none', cursor: 'pointer' }}
                >
                  <option style={{backgroundColor:'#0d1117', color:'#f1f5f9'}}>Last 5 Classes</option>
                  <option style={{backgroundColor:'#0d1117', color:'#f1f5f9'}}>This Week</option>
                  <option style={{backgroundColor:'#0d1117', color:'#f1f5f9'}}>This Month</option>
                </select>
              </div>

              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    {["DATE", "SUBJECT", "PERIOD", "FACULTY", "STATUS"].map(th => (
                      <th key={th} style={{ fontSize: '10px', color: '#334155', fontWeight: '600', letterSpacing: '.5px', padding: '7px 10px', textAlign: 'left', borderBottom: '0.5px solid rgba(255,255,255,0.05)' }}>
                        {th}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {stats.history.length > 0 ? stats.history.map((row, i) => (
                    <tr key={i}>
                      <td style={{ fontSize: '12px', color: '#e2e8f0', fontWeight: '500', padding: '9px 10px', borderBottom: '0.5px solid rgba(255,255,255,0.03)' }}>{row.date}</td>
                      <td style={{ fontSize: '12px', color: '#94a3b8', padding: '9px 10px', borderBottom: '0.5px solid rgba(255,255,255,0.03)' }}>{row.subject}</td>
                      <td style={{ fontSize: '12px', color: '#94a3b8', padding: '9px 10px', borderBottom: '0.5px solid rgba(255,255,255,0.03)' }}>{row.period}</td>
                      <td style={{ fontSize: '12px', color: '#94a3b8', padding: '9px 10px', borderBottom: '0.5px solid rgba(255,255,255,0.03)' }}>{row.faculty}</td>
                      <td style={{ padding: '9px 10px', borderBottom: '0.5px solid rgba(255,255,255,0.03)' }}>
                        {row.status === 'Present' ? (
                          <span style={{ display: 'inline-flex', alignItems: 'center', gap: '4px', backgroundColor: 'rgba(34,197,94,0.08)', border: '0.5px solid rgba(34,197,94,0.2)', color: '#86efac', borderRadius: '5px', padding: '2px 8px', fontSize: '10px' }}>
                            <svg width="8" height="8" viewBox="0 0 24 24" fill="#86efac"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/></svg>
                            Present
                          </span>
                        ) : (
                          <span style={{ display: 'inline-flex', alignItems: 'center', gap: '4px', backgroundColor: 'rgba(239,68,68,0.08)', border: '0.5px solid rgba(239,68,68,0.2)', color: '#fca5a5', borderRadius: '5px', padding: '2px 8px', fontSize: '10px' }}>
                            <svg width="8" height="8" viewBox="0 0 24 24" fill="#fca5a5"><path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/></svg>
                            Absent
                          </span>
                        )}
                      </td>
                    </tr>
                  )) : (
                    <tr>
                      <td colSpan="5" style={{ color: '#64748b', fontSize: '12px', padding: '20px', textAlign: 'center', borderBottom: '0.5px solid rgba(255,255,255,0.03)' }}>
                        No recent classes attended
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function SubjectCell({ sub, color }) {
  const [hovered, setHovered] = React.useState(false);
  return (
    <div
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        backgroundColor: '#070b14',
        border: `0.5px solid ${hovered ? 'rgba(129,140,248,0.3)' : 'rgba(255,255,255,0.05)'}`,
        borderRadius: '9px', padding: '12px 13px', cursor: 'pointer',
        transition: 'border-color 0.15s'
      }}
    >
      <div style={{ fontSize: '11px', fontWeight: '600', color: '#e2e8f0', marginBottom: '6px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
        {sub.name}
      </div>
      <div style={{ height: '4px', backgroundColor: 'rgba(255,255,255,0.06)', borderRadius: '2px', marginBottom: '6px' }}>
        <div style={{ height: '100%', width: `${sub.pct}%`, backgroundColor: color, borderRadius: '2px', transition: 'width 0.6s ease' }} />
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontSize: '13px', fontWeight: '800', color }}>{sub.pct}%</span>
        <span style={{ fontSize: '10px', color: '#475569' }}>{sub.attended}/{sub.total} classes</span>
      </div>
    </div>
  );
}
