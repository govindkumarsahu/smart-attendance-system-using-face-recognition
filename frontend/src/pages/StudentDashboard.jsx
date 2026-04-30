import { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import { Navigate } from 'react-router-dom';

export default function StudentDashboard() {
  const { user } = useAuth();

  if (user?.role !== 'student') {
    return <Navigate to="/dashboard" replace />;
  }

  return (
    <div className="flex flex-col gap-8 max-w-[1200px] mx-auto p-4 sm:p-6 lg:p-8">
      {/* Header Section */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-surface-light dark:bg-surface-dark/90 p-8 rounded-[24px] border border-border-light/60 dark:border-border-dark/60 shadow-sm relative overflow-hidden backdrop-blur-md">
          <div className="absolute top-0 right-0 w-64 h-64 bg-primary/5 rounded-full blur-[80px] pointer-events-none transform translate-x-1/3 -translate-y-1/3"></div>
          
          <div className="relative z-10">
              <h1 className="text-3xl md:text-4xl font-black tracking-tight text-text-primary-light dark:text-text-primary-dark">Student Dashboard</h1>
              <p className="text-text-secondary-light dark:text-text-secondary-dark mt-2 font-medium">Welcome back, <span className="text-primary font-bold">{user.username}</span></p>
          </div>
          <div className="relative z-10 flex items-center gap-4 bg-background-light dark:bg-background-dark p-4 rounded-[16px] border border-border-light dark:border-border-dark">
               <div className="flex flex-col items-end">
                  <span className="text-[11px] font-bold uppercase tracking-wider text-text-secondary-light dark:text-text-secondary-dark">Registration Number</span>
                  <span className="text-xl font-black text-text-primary-light dark:text-text-primary-dark">{user.roll_number}</span>
               </div>
               <div className="size-14 rounded-full bg-primary/10 flex items-center justify-center text-primary border border-primary/20 shrink-0">
                  <span className="material-symbols-outlined text-2xl">school</span>
               </div>
          </div>
      </div>

      {/* Info Notice */}
      <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-[20px] border border-blue-100 dark:border-blue-800/50 flex gap-4">
         <span className="material-symbols-outlined text-blue-500 mt-1">info</span>
         <div>
            <h4 className="text-blue-800 dark:text-blue-300 font-bold mb-1">Student Portal Ready</h4>
            <p className="text-sm text-blue-600/80 dark:text-blue-400/80">You have logged in successfully as a student. To see your full attendance records, the backend API would need to expose your specific data. For this React demo setup, you are fully authenticated and role-restricted!</p>
         </div>
      </div>
      
    </div>
  );
}
