import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider } from './context/ThemeContext';
import { AuthProvider, useAuth } from './context/AuthContext';
import { ToastProvider } from './context/ToastContext';
import ToastContainer from './components/ToastContainer';

// Layout
import DashboardLayout from './components/DashboardLayout';

// Pages
import LandingPage from './pages/LandingPage';
import Login from './pages/Login';
import Dashboard from './pages/Dashboard';
import StudentRegistration from './pages/StudentRegistration';
import StartAttendance from './pages/StartAttendance';
import RegisteredStudents from './pages/RegisteredStudents';
import Reports from './pages/Reports';
import ModelData from './pages/ModelData';
import StudentDashboard from './pages/StudentDashboard';
import FacultyRegistration from './pages/FacultyRegistration';
import FacultyAttendance from './pages/FacultyAttendance';

function PrivateRoute({ children, requiredRole }) {
  const { user, loading } = useAuth();
  
  if (loading) return null; // or a loading spinner
  
  if (!user) return <Navigate to="/login" replace />;
  
  if (requiredRole && user.role !== requiredRole) {
    // If not faculty but requires faculty
    if (user.role === 'student') return <Navigate to="/student-dashboard" replace />;
    return <Navigate to="/" replace />;
  }
  
  return children;
}

export default function App() {
  return (
    <Router>
      <ThemeProvider>
        <ToastProvider>
          <AuthProvider>
            <ToastContainer />
            <Routes>
              {/* Public Routes */}
              <Route path="/" element={<LandingPage />} />
              <Route path="/login" element={<Login />} />
              
              {/* Dashboard Layout Routes */}
              <Route element={<DashboardLayout />}>
                
                {/* Faculty Routes */}
                <Route path="/dashboard" element={
                  <PrivateRoute requiredRole="faculty">
                    <Dashboard />
                  </PrivateRoute>
                }/>
                <Route path="/register" element={
                  <PrivateRoute requiredRole="faculty">
                    <StudentRegistration />
                  </PrivateRoute>
                }/>
                <Route path="/attendance" element={
                  <PrivateRoute requiredRole="faculty">
                    <StartAttendance />
                  </PrivateRoute>
                }/>
                <Route path="/students" element={
                  <PrivateRoute requiredRole="faculty">
                    <RegisteredStudents />
                  </PrivateRoute>
                }/>
                <Route path="/reports" element={
                  <PrivateRoute requiredRole="faculty">
                    <Reports />
                  </PrivateRoute>
                }/>
                <Route path="/model-data" element={
                  <PrivateRoute requiredRole="faculty">
                    <ModelData />
                  </PrivateRoute>
                }/>
                
                {/* Faculty Management Routes */}
                <Route path="/faculty-registration" element={
                  <PrivateRoute requiredRole="faculty">
                    <FacultyRegistration />
                  </PrivateRoute>
                }/>
                <Route path="/faculty-attendance" element={
                  <PrivateRoute requiredRole="faculty">
                    <FacultyAttendance />
                  </PrivateRoute>
                }/>
                
                {/* Student Route */}
                <Route path="/student-dashboard" element={
                  <PrivateRoute requiredRole="student">
                    <StudentDashboard />
                  </PrivateRoute>
                }/>
                
              </Route>
            </Routes>
          </AuthProvider>
        </ToastProvider>
      </ThemeProvider>
    </Router>
  );
}
