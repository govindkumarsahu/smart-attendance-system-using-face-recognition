import { BrowserRouter, Routes, Route } from "react-router-dom";
import LandingPage from "./LandingPage";

// Placeholder login pages — replace with your actual login components
const StudentLogin  = () => <div style={{color:"#fff",padding:40,background:"#0a0f1e",minHeight:"100vh"}}>Student Login Page</div>;
const FacultyLogin  = () => <div style={{color:"#fff",padding:40,background:"#0a0f1e",minHeight:"100vh"}}>Faculty Login Page</div>;
const AdminLogin    = () => <div style={{color:"#fff",padding:40,background:"#0a0f1e",minHeight:"100vh"}}>Admin Login Page</div>;

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/"               element={<LandingPage />} />
        <Route path="/student/login"  element={<StudentLogin />} />
        <Route path="/faculty/login"  element={<FacultyLogin />} />
        <Route path="/admin/login"    element={<AdminLogin />} />
      </Routes>
    </BrowserRouter>
  );
}
