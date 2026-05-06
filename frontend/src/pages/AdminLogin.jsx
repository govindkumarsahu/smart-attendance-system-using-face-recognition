import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

export default function AdminLogin() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      const res = await axios.post('http://localhost:8000/api/admin/login', {
        username,
        password
      });
      if (res.data.success) {
        localStorage.setItem("adminToken", res.data.token);
        navigate("/admin/dashboard");
      } else {
        setError(res.data.message || 'Login failed');
      }
    } catch (err) {
      setError(err.response?.data?.message || 'Server error. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{
      display: 'flex',
      minHeight: '100vh',
      backgroundColor: '#070b14',
      backgroundImage: 'linear-gradient(rgba(239, 68, 68, 0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(239, 68, 68, 0.03) 1px, transparent 1px)',
      backgroundSize: '40px 40px',
      fontFamily: '"Segoe UI", system-ui, sans-serif'
    }}>
      {/* LEFT PANEL */}
      <div style={{
        width: '45%',
        background: 'linear-gradient(160deg, #0f172a, #1a0a0a)',
        borderRight: '0.5px solid rgba(239,68,68,0.12)',
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
            backgroundColor: '#ef4444',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
              <circle cx="12" cy="7" r="4"></circle>
            </svg>
          </div>
          <div>
            <div style={{ fontSize: '16px', fontWeight: '700', color: '#f1f5f9' }}>SmartAttend AI</div>
            <div style={{ fontSize: '11px', color: '#475569' }}>Admin Control Center</div>
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
            Full System<br/>Control
          </h2>
          <p style={{
            fontSize: '14px',
            color: '#64748b',
            lineHeight: '1.7',
            maxWidth: '320px',
            margin: '0'
          }}>
            Manage students, faculty, timetables, and monitor real-time attendance across all classrooms.
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
              { value: '3', label: 'User Roles' },
              { value: '15+', label: 'Features' },
              { value: '99.2%', label: 'Accuracy' },
              { value: '0', label: 'Proxy Passes' }
            ].map((stat, i) => (
              <div key={i} style={{
                backgroundColor: 'rgba(239,68,68,0.06)',
                border: '0.5px solid rgba(239,68,68,0.12)',
                borderRadius: '10px',
                padding: '14px 16px'
              }}>
                <div style={{ fontSize: '22px', fontWeight: '800', color: '#fca5a5' }}>{stat.value}</div>
                <div style={{ fontSize: '11px', color: '#475569', marginTop: '3px' }}>{stat.label}</div>
              </div>
            ))}
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
            {['YOLOv8', 'ArcFace', 'Flask', 'SQLite'].map((tech, i) => (
              <div key={i} style={{
                fontSize: '11px',
                color: '#475569',
                backgroundColor: 'rgba(255,255,255,0.04)',
                border: '0.5px solid rgba(255,255,255,0.07)',
                padding: '4px 10px',
                borderRadius: '20px'
              }}>
                {tech}
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
          border: '0.5px solid rgba(239,68,68,0.18)',
          borderRadius: '16px',
          padding: '40px 36px'
        }}>
          <div style={{
            width: '56px',
            height: '56px',
            backgroundColor: 'rgba(239,68,68,0.1)',
            border: '0.5px solid rgba(239,68,68,0.2)',
            borderRadius: '14px',
            marginBottom: '20px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <svg width="28" height="28" viewBox="0 0 24 24" fill="#f87171">
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
            </svg>
          </div>

          <h1 style={{ fontSize: '24px', fontWeight: '800', color: '#f1f5f9', margin: '0' }}>Admin Login</h1>
          <p style={{ fontSize: '13px', color: '#64748b', marginTop: '6px', marginBottom: '28px' }}>
            Restricted access — authorized personnel only
          </p>

          <form onSubmit={handleSubmit} style={{ marginTop: '28px' }}>
            <div style={{ marginBottom: '18px' }}>
              <label style={{ display: 'block', fontSize: '12px', color: '#94a3b8', marginBottom: '7px', fontWeight: '500' }}>
                Username
              </label>
              <input 
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
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
                onFocus={(e) => e.target.style.borderColor = '#ef4444'}
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
                onFocus={(e) => e.target.style.borderColor = '#ef4444'}
                onBlur={(e) => e.target.style.borderColor = 'rgba(255,255,255,0.08)'}
              />
            </div>

            {error && (
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                backgroundColor: 'rgba(239,68,68,0.08)',
                border: '0.5px solid rgba(239,68,68,0.2)',
                borderRadius: '8px',
                padding: '10px 12px',
                fontSize: '13px',
                color: '#fca5a5',
                marginBottom: '18px'
              }}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
                  <line x1="12" y1="9" x2="12" y2="13"></line>
                  <line x1="12" y1="17" x2="12.01" y2="17"></line>
                </svg>
                {error}
              </div>
            )}

            <button 
              type="submit" 
              disabled={loading}
              style={{
                width: '100%',
                backgroundColor: '#ef4444',
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
              onMouseEnter={(e) => !loading && (e.target.style.backgroundColor = '#dc2626')}
              onMouseLeave={(e) => !loading && (e.target.style.backgroundColor = '#ef4444')}
            >
              {loading ? 'Authenticating...' : 'Access Admin Panel →'}
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
