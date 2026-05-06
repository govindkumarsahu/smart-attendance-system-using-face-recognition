import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import LandingPage from './pages/LandingPage';
import AdminLogin from './pages/AdminLogin';
import AdminDashboard from './pages/AdminDashboard';
import FacultyLogin from './pages/FacultyLogin';
import FacultyDashboard from './pages/FacultyDashboard';
import StudentLogin from './pages/StudentLogin';
import StudentDashboard from './pages/StudentDashboard';

// Protected route component for Admin
const AdminRoute = ({ children }) => {
  const token = localStorage.getItem("adminToken");
  if (!token) {
    return <Navigate to="/admin/login" replace />;
  }
  return children;
};

// Protected route component for Faculty
const FacultyRoute = ({ children }) => {
  const token = localStorage.getItem("facultyToken");
  if (!token) {
    return <Navigate to="/faculty/login" replace />;
  }
  return children;
};

// Protected route component for Student
const StudentRoute = ({ children }) => {
  const token = localStorage.getItem("studentToken");
  if (!token) {
    return <Navigate to="/student/login" replace />;
  }
  return children;
};

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        
        <Route path="/admin/login" element={<AdminLogin />} />
        <Route 
          path="/admin/dashboard" 
          element={
            <AdminRoute>
              <AdminDashboard />
            </AdminRoute>
          } 
        />
        
        {/* Placeholders for other roles as requested */}
        <Route path="/student/login" element={<StudentLogin />} />
        <Route 
          path="/student/dashboard" 
          element={
            <StudentRoute>
              <StudentDashboard />
            </StudentRoute>
          } 
        />
        
        <Route path="/faculty/login" element={<FacultyLogin />} />
        <Route 
          path="/faculty/dashboard" 
          element={
            <FacultyRoute>
              <FacultyDashboard />
            </FacultyRoute>
          } 
        />
        
        {/* Catch-all redirect */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
