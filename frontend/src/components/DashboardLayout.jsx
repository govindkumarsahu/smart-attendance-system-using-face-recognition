import { useState, useEffect } from 'react';
import { Outlet, Navigate } from 'react-router-dom';
import Sidebar from './Sidebar';
import Header from './Header';
import { useAuth } from '../context/AuthContext';

export default function DashboardLayout() {
  const { user, loading } = useAuth();
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false);

  // Close mobile sidebar on resize to prevent weird states
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth >= 1024) {
        setIsMobileSidebarOpen(false);
      }
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background-light dark:bg-background-dark">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
      </div>
    );
  }

  if (!user) {
    return <Navigate to="/login" replace />;
  }

  const toggleSidebar = () => {
    if (window.innerWidth >= 1024) {
      setIsSidebarCollapsed(!isSidebarCollapsed);
    } else {
      setIsMobileSidebarOpen(!isMobileSidebarOpen);
    }
  };

  return (
    <div className="flex h-screen overflow-hidden bg-background-light dark:bg-background-dark font-sans text-text-primary-light dark:text-text-primary-dark transition-colors duration-300">
      <Sidebar
        isCollapsed={isSidebarCollapsed}
        onMobileClose={() => setIsMobileSidebarOpen(false)}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 relative">
        <Header toggleSidebar={toggleSidebar} />
        
        {/* Page Content */}
        <main className="flex-1 overflow-y-auto overflow-x-hidden p-4 sm:p-6 lg:p-8 bg-background-light dark:bg-background-dark transition-colors relative">
          {/* Subtle background glow effect */}
          <div className="absolute top-0 left-1/4 w-[500px] h-[500px] bg-primary/5 dark:bg-primary/5 rounded-full blur-[100px] pointer-events-none mix-blend-screen"></div>
          <div className="absolute bottom-0 right-1/4 w-[600px] h-[600px] bg-purple-500/5 dark:bg-purple-500/5 rounded-full blur-[120px] pointer-events-none mix-blend-screen"></div>

          <div className="relative z-10">
            <Outlet />
          </div>
        </main>
      </div>

      {/* Mobile Overlay */}
      {isMobileSidebarOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-40 lg:hidden backdrop-blur-sm"
          onClick={() => setIsMobileSidebarOpen(false)}
        ></div>
      )}
    </div>
  );
}
