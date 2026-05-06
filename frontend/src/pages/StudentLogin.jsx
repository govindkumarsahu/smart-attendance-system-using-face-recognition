import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

export default function StudentLogin() {
  const [rollNo, setRollNo] = useState('');
  const [dob, setDob] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      const res = await axios.post('http://localhost:8000/api/student/login', {
        roll_number: rollNo,
        dob: dob
      });
      if (res.data.success) {
        localStorage.setItem("studentToken", res.data.token || "student-token");
        localStorage.setItem("studentName", res.data.name || rollNo);
        localStorage.setItem("studentRoll", res.data.roll_number || rollNo);
        localStorage.setItem("studentBranch", res.data.branch || "CSE-AI");
        localStorage.setItem("studentSem", res.data.semester || "8th Sem");
        navigate("/student/dashboard");
      } else {
        setError(res.data.message || 'Invalid Registration Number or Password');
      }
    } catch (err) {
      setError(err.response?.data?.message || 'Invalid Registration Number or Password (DOB)');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{
      display: 'flex', minHeight: '100vh', backgroundColor: '#070b14',
      backgroundImage: 'linear-gradient(rgba(99,102,241,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(99,102,241,0.03) 1px, transparent 1px)',
      backgroundSize: '40px 40px', fontFamily: '"Segoe UI", system-ui, sans-serif'
    }}>
      {/* LEFT PANEL */}
      <div style={{
        width: '45%', background: 'linear-gradient(160deg, #0f172a, #0d0d2b)',
        borderRight: '0.5px solid rgba(99,102,241,0.12)',
        padding: '48px 44px', display: 'flex', flexDirection: 'column', justifyContent: 'space-between'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{ width: '44px', height: '44px', borderRadius: '10px', backgroundColor: '#6366f1', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="white">
              <path d="M5 13.18v4L12 21l7-3.82v-4L12 17l-7-3.82zM12 3L1 9l11 6 9-4.91V17h2V9L12 3z"/>
            </svg>
          </div>
          <div>
            <div style={{ fontSize: '16px', fontWeight: '700', color: '#f1f5f9' }}>SmartAttend AI</div>
            <div style={{ fontSize: '11px', color: '#475569' }}>Student Portal</div>
          </div>
        </div>

        <div>
          <h2 style={{ fontSize: '38px', fontWeight: '800', color: '#f1f5f9', lineHeight: '1.15', letterSpacing: '-1px', margin: '0 0 16px 0' }}>
            Track Your<br/>Attendance
          </h2>
          <p style={{ fontSize: '14px', color: '#64748b', lineHeight: '1.7', maxWidth: '320px', margin: '0' }}>
            Check your subject-wise attendance, get shortage alerts, and calculate classes needed to stay eligible.
          </p>
        </div>

        <div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginBottom: '28px' }}>
            {[
              { value: '6+', label: 'Subjects Tracked' },
              { value: '75%', label: 'Min. Required' },
              { value: 'Live', label: 'Real-time Data' },
              { value: 'Auto', label: 'Shortage Alert' }
            ].map((s, i) => (
              <div key={i} style={{ backgroundColor: 'rgba(99,102,241,0.06)', border: '0.5px solid rgba(99,102,241,0.12)', borderRadius: '10px', padding: '14px 16px' }}>
                <div style={{ fontSize: '22px', fontWeight: '800', color: '#a5b4fc' }}>{s.value}</div>
                <div style={{ fontSize: '11px', color: '#475569', marginTop: '3px' }}>{s.label}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* RIGHT PANEL */}
      <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '40px 32px' }}>
        <div style={{
          maxWidth: '400px', width: '100%',
          backgroundColor: 'rgba(15,23,42,0.9)',
          border: '0.5px solid rgba(99,102,241,0.18)',
          borderRadius: '16px', padding: '40px 36px'
        }}>
          <div style={{ width: '56px', height: '56px', backgroundColor: 'rgba(99,102,241,0.1)', border: '0.5px solid rgba(99,102,241,0.2)', borderRadius: '14px', marginBottom: '20px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#818cf8" strokeWidth="2">
              <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
              <circle cx="12" cy="7" r="4"/>
            </svg>
          </div>

          <h1 style={{ fontSize: '24px', fontWeight: '800', color: '#f1f5f9', margin: '0' }}>Student Login</h1>
          <p style={{ fontSize: '13px', color: '#64748b', marginTop: '6px', marginBottom: '28px' }}>
            Enter your Registration Number and Password (DOB) to login
          </p>

          <form onSubmit={handleSubmit}>
            <div style={{ marginBottom: '14px' }}>
              <label style={{ display: 'block', fontSize: '12px', color: '#94a3b8', marginBottom: '7px', fontWeight: '500' }}>
                Registration Number
              </label>
              <input
                type="text"
                value={rollNo}
                onChange={e => setRollNo(e.target.value)}
                placeholder="e.g. CS2024001"
                required
                style={{
                  width: '100%', backgroundColor: 'rgba(255,255,255,0.04)',
                  border: '0.5px solid rgba(255,255,255,0.08)', borderRadius: '8px',
                  padding: '11px 14px', color: '#f1f5f9', fontSize: '14px',
                  outline: 'none', boxSizing: 'border-box'
                }}
                onFocus={e => e.target.style.borderColor = '#6366f1'}
                onBlur={e => e.target.style.borderColor = 'rgba(255,255,255,0.08)'}
              />
            </div>

            <div style={{ marginBottom: '18px' }}>
              <label style={{ display: 'block', fontSize: '12px', color: '#94a3b8', marginBottom: '7px', fontWeight: '500' }}>
                Password (DOB)
              </label>
              <input
                type="date"
                value={dob}
                onChange={e => setDob(e.target.value)}
                required
                style={{
                  width: '100%', backgroundColor: 'rgba(255,255,255,0.04)',
                  border: '0.5px solid rgba(255,255,255,0.08)', borderRadius: '8px',
                  padding: '11px 14px', color: '#f1f5f9', fontSize: '14px',
                  outline: 'none', boxSizing: 'border-box', colorScheme: 'dark'
                }}
                onFocus={e => e.target.style.borderColor = '#6366f1'}
                onBlur={e => e.target.style.borderColor = 'rgba(255,255,255,0.08)'}
              />
            </div>

            {error && (
              <div style={{
                display: 'flex', alignItems: 'center', gap: '8px',
                backgroundColor: 'rgba(239,68,68,0.08)', border: '0.5px solid rgba(239,68,68,0.2)',
                borderRadius: '8px', padding: '10px 12px', fontSize: '13px', color: '#fca5a5', marginBottom: '12px'
              }}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
                </svg>
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              style={{
                width: '100%', backgroundColor: '#6366f1', color: '#fff',
                border: 'none', borderRadius: '9px', padding: '13px',
                fontSize: '14px', fontWeight: '700',
                cursor: loading ? 'not-allowed' : 'pointer',
                opacity: loading ? 0.7 : 1, marginTop: '4px'
              }}
              onMouseEnter={e => !loading && (e.target.style.backgroundColor = '#4f46e5')}
              onMouseLeave={e => !loading && (e.target.style.backgroundColor = '#6366f1')}
            >
              {loading ? 'Verifying...' : 'View My Attendance →'}
            </button>
          </form>

          <button
            onClick={() => navigate('/')}
            style={{ width: '100%', backgroundColor: 'transparent', color: '#475569', border: 'none', padding: '12px', fontSize: '13px', marginTop: '12px', cursor: 'pointer' }}
          >
            ← Back to Home
          </button>
        </div>
      </div>
    </div>
  );
}
