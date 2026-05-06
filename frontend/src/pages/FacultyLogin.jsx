import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

export default function FacultyLogin() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      const res = await axios.post('http://localhost:8000/api/faculty/login', {
        employee_id: username,
        password: password
      });
      if (res.data.success) {
        localStorage.setItem("facultyToken", res.data.token);
        localStorage.setItem("facultyId", res.data.faculty_id);
        localStorage.setItem("facultyName", res.data.faculty_name);
        localStorage.setItem("facultyDept", res.data.department);
        navigate("/faculty/dashboard");
      } else {
        setError(res.data.message || 'Login failed');
      }
    } catch (err) {
      setError(err.response?.data?.message || 'Invalid Employee ID or Password');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{
      display: 'flex',
      minHeight: '100vh',
      backgroundColor: '#070b14',
      backgroundImage: 'linear-gradient(rgba(34, 197, 94, 0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(34, 197, 94, 0.03) 1px, transparent 1px)',
      backgroundSize: '40px 40px',
      fontFamily: '"Segoe UI", system-ui, sans-serif'
    }}>
      {/* LEFT PANEL */}
      <div style={{
        width: '45%',
        background: 'linear-gradient(160deg, #0f172a, #0a1510)',
        borderRight: '0.5px solid rgba(34,197,94,0.12)',
        padding: '48px 44px',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'space-between'
      }}>
        {/* TOP - Logo row */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{
            width: '44px',
            height: '44px',
            borderRadius: '10px',
            backgroundColor: '#16a34a',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M22 10v6M2 10l10-5 10 5-10 5z"/>
              <path d="M6 12v5c3 3 9 3 12 0v-5"/>
            </svg>
          </div>
          <div>
            <div style={{ fontSize: '16px', fontWeight: '700', color: '#f1f5f9' }}>SmartAttend AI</div>
            <div style={{ fontSize: '11px', color: '#475569' }}>Faculty Portal</div>
          </div>
        </div>

        {/* MIDDLE - Branding text */}
        <div>
          <h2 style={{
            fontSize: '38px',
            fontWeight: '800',
            color: '#f1f5f9',
            lineHeight: '1.15',
            letterSpacing: '-1px',
            margin: '0 0 16px 0'
          }}>
            Manage Classes<br/>Effortlessly
          </h2>
          <p style={{
            fontSize: '14px',
            color: '#64748b',
            lineHeight: '1.7',
            maxWidth: '320px',
            margin: '0'
          }}>
            Start AI attendance sessions, monitor live face recognition, and manage your daily class logbook.
          </p>
        </div>

        {/* BOTTOM SECTION */}
        <div>
          <div style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: '12px',
            marginBottom: '28px'
          }}>
            {[
              { value: '30s', label: 'Avg Scan Time' },
              { value: 'Live', label: 'Recognition Feed' },
              { value: 'Auto', label: 'CSV Export' },
              { value: '100%', label: 'Secure' }
            ].map((stat, i) => (
              <div key={i} style={{
                backgroundColor: 'rgba(34,197,94,0.06)',
                border: '0.5px solid rgba(34,197,94,0.12)',
                borderRadius: '10px',
                padding: '14px 16px'
              }}>
                <div style={{ fontSize: '22px', fontWeight: '800', color: '#86efac' }}>{stat.value}</div>
                <div style={{ fontSize: '11px', color: '#475569', marginTop: '3px' }}>{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* RIGHT PANEL */}
      <div style={{
        flex: 1,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '40px 32px'
      }}>
        <div style={{
          maxWidth: '400px',
          width: '100%',
          backgroundColor: 'rgba(15,23,42,0.9)',
          border: '0.5px solid rgba(34,197,94,0.18)',
          borderRadius: '16px',
          padding: '40px 36px'
        }}>
          <div style={{
            width: '56px',
            height: '56px',
            backgroundColor: 'rgba(34,197,94,0.1)',
            border: '0.5px solid rgba(34,197,94,0.2)',
            borderRadius: '14px',
            marginBottom: '20px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#22c55e" strokeWidth="2">
              <rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect>
              <path d="M7 11V7a5 5 0 0 1 10 0v4"></path>
            </svg>
          </div>

          <h1 style={{ fontSize: '24px', fontWeight: '800', color: '#f1f5f9', margin: '0' }}>Faculty Login</h1>
          <p style={{ fontSize: '13px', color: '#64748b', marginTop: '6px', marginBottom: '28px' }}>
            Enter your employee credentials to access your dashboard
          </p>

          <form onSubmit={handleSubmit} style={{ marginTop: '28px' }}>
            <div style={{ marginBottom: '18px' }}>
              <label style={{ display: 'block', fontSize: '12px', color: '#94a3b8', marginBottom: '7px', fontWeight: '500' }}>
                Employee ID
              </label>
              <input 
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="e.g. FAC2024001"
                required
                style={{
                  width: '100%',
                  backgroundColor: 'rgba(255,255,255,0.04)',
                  border: '0.5px solid rgba(255,255,255,0.08)',
                  borderRadius: '8px',
                  padding: '11px 14px',
                  color: '#f1f5f9',
                  fontSize: '14px',
                  outline: 'none',
                  boxSizing: 'border-box'
                }}
                onFocus={(e) => e.target.style.borderColor = '#22c55e'}
                onBlur={(e) => e.target.style.borderColor = 'rgba(255,255,255,0.08)'}
              />
            </div>

            <div style={{ marginBottom: '18px' }}>
              <label style={{ display: 'block', fontSize: '12px', color: '#94a3b8', marginBottom: '7px', fontWeight: '500' }}>
                Password
              </label>
              <input 
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                style={{
                  width: '100%',
                  backgroundColor: 'rgba(255,255,255,0.04)',
                  border: '0.5px solid rgba(255,255,255,0.08)',
                  borderRadius: '8px',
                  padding: '11px 14px',
                  color: '#f1f5f9',
                  fontSize: '14px',
                  outline: 'none',
                  boxSizing: 'border-box'
                }}
                onFocus={(e) => e.target.style.borderColor = '#22c55e'}
                onBlur={(e) => e.target.style.borderColor = 'rgba(255,255,255,0.08)'}
              />
            </div>

            {error && (
              <div style={{
                display: 'flex', alignItems: 'center', gap: '8px',
                backgroundColor: 'rgba(239,68,68,0.08)',
                border: '0.5px solid rgba(239,68,68,0.2)',
                borderRadius: '8px', padding: '10px 12px',
                fontSize: '13px', color: '#fca5a5', marginBottom: '12px'
              }}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
                {error}
              </div>
            )}

            <button 
              type="submit" 
              disabled={loading}
              style={{
                width: '100%',
                backgroundColor: '#16a34a',
                color: '#fff',
                border: 'none',
                borderRadius: '9px',
                padding: '13px',
                fontSize: '14px',
                fontWeight: '700',
                cursor: loading ? 'not-allowed' : 'pointer',
                opacity: loading ? 0.7 : 1,
                transition: 'background-color 0.2s',
                marginTop: '10px'
              }}
              onMouseEnter={(e) => !loading && (e.target.style.backgroundColor = '#15803d')}
              onMouseLeave={(e) => !loading && (e.target.style.backgroundColor = '#16a34a')}
            >
              {loading ? 'Authenticating...' : 'Sign In →'}
            </button>
          </form>


          <button 
            onClick={() => navigate('/')}
            style={{
              width: '100%',
              backgroundColor: 'transparent',
              color: '#475569',
              border: 'none',
              padding: '12px',
              fontSize: '13px',
              marginTop: '12px',
              cursor: 'pointer'
            }}
          >
            ← Back to Home
          </button>
        </div>
      </div>
    </div>
  );
}
