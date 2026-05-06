import React from "react";
import { useNavigate } from "react-router-dom";

const GraduationIcon = () => (
  <svg viewBox="0 0 24 24" fill="#818cf8" width="22" height="22">
    <path d="M5 13.18v4L12 21l7-3.82v-4L12 17l-7-3.82zM12 3L1 9l11 6 9-4.91V17h2V9L12 3z" />
  </svg>
);

const TeacherIcon = () => (
  <svg viewBox="0 0 24 24" fill="#22c55e" width="22" height="22">
    <path d="M20 3H4v10c0 2.21 1.79 4 4 4h6c2.21 0 4-1.79 4-4v-3h2c1.11 0 2-.89 2-2V5c0-1.11-.89-2-2-2zm0 5h-2V5h2v3zM4 19h16v2H4z" />
  </svg>
);

const ShieldIcon = () => (
  <svg viewBox="0 0 24 24" fill="#f87171" width="22" height="22">
    <path d="M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4zm0 4l5 2.18V11c0 3.5-2.33 6.79-5 7.93-2.67-1.14-5-4.43-5-7.93V7.18L12 5z" />
  </svg>
);

const FaceIcon = () => (
  <svg viewBox="0 0 24 24" fill="white" width="18" height="18">
    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 3c1.66 0 3 1.34 3 3s-1.34 3-3 3-3-1.34-3-3 1.34-3 3-3zm0 14.2c-2.5 0-4.71-1.28-6-3.22.03-1.99 4-3.08 6-3.08 1.99 0 5.97 1.09 6 3.08-1.29 1.94-3.5 3.22-6 3.22z" />
  </svg>
);

const cards = [
  {
    role: "student",
    title: "Student Portal",
    description: "Check your attendance, apply for leaves, and track your academic standing.",
    icon: <GraduationIcon />,
    color: "#818cf8",
    iconBg: "rgba(99,102,241,0.15)",
    cardBg: "linear-gradient(145deg, #0f172a, #1e1b4b)",
    borderColor: "rgba(99,102,241,0.25)",
    borderHover: "rgba(99,102,241,0.65)",
    btnBg: "rgba(99,102,241,0.18)",
    btnBorder: "rgba(99,102,241,0.35)",
    btnColor: "#a5b4fc",
    dotColor: "#818cf8",
    dotBg: "rgba(99,102,241,0.2)",
    features: [
      "View attendance per subject",
      "Apply for medical leave online",
      "75% shortage warnings",
      "Required classes calculator",
    ],
    route: "/student/login",
    label: "Student Login",
  },
  {
    role: "faculty",
    title: "Faculty Portal",
    description: "Launch face-scan sessions, monitor live recognition feed, and export reports.",
    icon: <TeacherIcon />,
    color: "#22c55e",
    iconBg: "rgba(34,197,94,0.12)",
    cardBg: "linear-gradient(145deg, #0f172a, #1a2e1a)",
    borderColor: "rgba(34,197,94,0.2)",
    borderHover: "rgba(34,197,94,0.55)",
    btnBg: "rgba(34,197,94,0.13)",
    btnBorder: "rgba(34,197,94,0.3)",
    btnColor: "#86efac",
    dotColor: "#22c55e",
    dotBg: "rgba(34,197,94,0.15)",
    features: [
      "One-click attendance scan",
      "Live real-time recognition feed",
      "Export PDF / Excel reports",
      "Unknown face intruder alerts",
    ],
    route: "/faculty/login",
    label: "Faculty Login",
  },
  {
    role: "admin",
    title: "Admin Panel",
    description: "Full system control — register students, manage timetables, assign substitutes.",
    icon: <ShieldIcon />,
    color: "#f87171",
    iconBg: "rgba(239,68,68,0.12)",
    cardBg: "linear-gradient(145deg, #0f172a, #1e1414)",
    borderColor: "rgba(239,68,68,0.2)",
    borderHover: "rgba(239,68,68,0.55)",
    btnBg: "rgba(239,68,68,0.13)",
    btnBorder: "rgba(239,68,68,0.3)",
    btnColor: "#fca5a5",
    dotColor: "#f87171",
    dotBg: "rgba(239,68,68,0.15)",
    features: [
      "Register & manage students",
      "Assign substitute teachers",
      "Full audit trail / logs",
      "System-wide analytics",
    ],
    route: "/admin/login",
    label: "Admin Login",
  },
];

const stats = [
  { value: "99.2%", label: "Recognition Accuracy" },
  { value: "< 30s", label: "Full Class Scan" },
  { value: "0", label: "Proxy Attempts Passed" },
];

const techPills = ["YOLOv8", "ArcFace", "React", "Flask", "DeepFace", "SQLite"];

export default function LandingPage() {
  const navigate = useNavigate();

  return (
    <div
      style={{
        background: "#0a0f1e",
        minHeight: "100vh",
        fontFamily: "'Inter', 'Segoe UI', system-ui, sans-serif",
        color: "#fff",
      }}
    >
      {/* Navbar */}
      <nav
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "18px 48px",
          borderBottom: "0.5px solid rgba(255,255,255,0.07)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div
            style={{
              width: 36,
              height: 36,
              background: "#3b82f6",
              borderRadius: 9,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <FaceIcon />
          </div>
          <div>
            <div style={{ fontSize: 15, fontWeight: 700, letterSpacing: "-0.3px" }}>
              SmartAttend AI
            </div>
            <div style={{ fontSize: 11, color: "#475569", marginTop: 1 }}>
              YOLOv8 · ArcFace · DeepFace
            </div>
          </div>
        </div>
        <div
          style={{
            background: "rgba(59,130,246,0.12)",
            border: "0.5px solid rgba(59,130,246,0.28)",
            color: "#60a5fa",
            fontSize: 12,
            padding: "5px 14px",
            borderRadius: 20,
            display: "flex",
            alignItems: "center",
            gap: 6,
          }}
        >
          <span
            style={{
              width: 7,
              height: 7,
              background: "#22c55e",
              borderRadius: "50%",
              display: "inline-block",
            }}
          />
          System Active
        </div>
      </nav>

      {/* Hero */}
      <section style={{ textAlign: "center", padding: "52px 24px 36px" }}>
        <div
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: 7,
            background: "rgba(59,130,246,0.1)",
            border: "0.5px solid rgba(59,130,246,0.22)",
            color: "#60a5fa",
            fontSize: 12,
            padding: "5px 14px",
            borderRadius: 20,
            marginBottom: 20,
          }}
        >
          <span
            style={{
              width: 7,
              height: 7,
              background: "#22c55e",
              borderRadius: "50%",
              display: "inline-block",
            }}
          />
          AI-Powered Face Recognition Attendance
        </div>

        <h1
          style={{
            fontSize: "clamp(26px, 4vw, 36px)",
            fontWeight: 800,
            lineHeight: 1.2,
            letterSpacing: "-0.8px",
            marginBottom: 12,
          }}
        >
          Attendance in{" "}
          <span style={{ color: "#3b82f6" }}>30 Seconds</span>
          <br />
          No Proxy. No Paper.
        </h1>

        <p
          style={{
            fontSize: 14,
            color: "#64748b",
            maxWidth: 420,
            margin: "0 auto 36px",
            lineHeight: 1.7,
          }}
        >
          Automated classroom attendance using YOLOv8 face detection and ArcFace
          recognition — built for real university challenges.
        </p>

        {/* Stats */}
        <div
          style={{
            display: "flex",
            justifyContent: "center",
            gap: "clamp(20px, 5vw, 48px)",
            marginBottom: 48,
            flexWrap: "wrap",
          }}
        >
          {stats.map((s) => (
            <div key={s.label} style={{ textAlign: "center" }}>
              <div style={{ fontSize: 22, fontWeight: 800, color: "#f1f5f9" }}>
                {s.value}
              </div>
              <div style={{ fontSize: 12, color: "#475569", marginTop: 3 }}>
                {s.label}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Cards */}
      <section
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
          gap: 16,
          padding: "0 40px 48px",
          maxWidth: 1100,
          margin: "0 auto",
        }}
      >
        {cards.map((card) => (
          <LoginCard key={card.role} card={card} onLogin={() => navigate(card.route)} />
        ))}
      </section>

      {/* Footer */}
      <footer
        style={{
          borderTop: "0.5px solid rgba(255,255,255,0.05)",
          padding: "14px 48px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          flexWrap: "wrap",
          gap: 12,
        }}
      >
        <div style={{ fontSize: 12, color: "#334155" }}>
          B.Tech CSE-AI · Final Year Major Project · Govind Kumar Sahu
        </div>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          {techPills.map((p) => (
            <span
              key={p}
              style={{
                fontSize: 11,
                color: "#475569",
                background: "rgba(255,255,255,0.04)",
                border: "0.5px solid rgba(255,255,255,0.07)",
                padding: "3px 9px",
                borderRadius: 10,
              }}
            >
              {p}
            </span>
          ))}
        </div>
      </footer>
    </div>
  );
}

function LoginCard({ card, onLogin }) {
  const [hovered, setHovered] = React.useState(false);

  return (
    <div
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        background: card.cardBg,
        border: `0.5px solid ${hovered ? card.borderHover : card.borderColor}`,
        borderRadius: 14,
        padding: "26px 22px",
        cursor: "default",
        transition: "transform 0.2s ease, border-color 0.2s ease",
        transform: hovered ? "translateY(-4px)" : "translateY(0)",
      }}
    >
      {/* Icon */}
      <div
        style={{
          width: 46,
          height: 46,
          background: card.iconBg,
          borderRadius: 11,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          marginBottom: 16,
        }}
      >
        {card.icon}
      </div>

      {/* Title & desc */}
      <div style={{ fontSize: 17, fontWeight: 700, color: "#f1f5f9", marginBottom: 6 }}>
        {card.title}
      </div>
      <div style={{ fontSize: 13, color: "#64748b", lineHeight: 1.6, marginBottom: 18 }}>
        {card.description}
      </div>

      {/* Features */}
      <ul style={{ listStyle: "none", marginBottom: 22, padding: 0 }}>
        {card.features.map((f) => (
          <li
            key={f}
            style={{
              fontSize: 12,
              color: "#94a3b8",
              padding: "4px 0",
              display: "flex",
              alignItems: "center",
              gap: 8,
            }}
          >
            <span
              style={{
                width: 13,
                height: 13,
                borderRadius: "50%",
                background: card.dotBg,
                display: "inline-flex",
                alignItems: "center",
                justifyContent: "center",
                flexShrink: 0,
              }}
            >
              <span
                style={{
                  width: 5,
                  height: 5,
                  background: card.dotColor,
                  borderRadius: "50%",
                  display: "inline-block",
                }}
              />
            </span>
            {f}
          </li>
        ))}
      </ul>

      {/* Button */}
      <button
        onClick={onLogin}
        style={{
          width: "100%",
          padding: "10px 0",
          borderRadius: 8,
          fontSize: 13,
          fontWeight: 600,
          border: `0.5px solid ${card.btnBorder}`,
          background: hovered ? card.btnBg : "transparent",
          color: card.btnColor,
          cursor: "pointer",
          transition: "background 0.2s ease",
          letterSpacing: "0.2px",
        }}
      >
        {card.label} →
      </button>
    </div>
  );
}
