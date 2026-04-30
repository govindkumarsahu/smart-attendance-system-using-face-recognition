import { useState, useEffect } from 'react';
import { api } from '../services/api';
import StatCard from '../components/StatCard';
import { useAuth } from '../context/AuthContext';

export default function Dashboard() {
  const { user } = useAuth();
  const [stats, setStats] = useState({
    registered_students: 0,
    present_today: 0,
    absent_late: 0,
    model_version: 'N/A',
    model_active: false,
    recent_records: [],
    all_today: [],
  });

  const fetchStats = async () => {
    try {
      const data = await api.getDashboardStats();
      setStats(data);
    } catch (error) {
      console.error('Failed to fetch dashboard stats', error);
    }
  };

  useEffect(() => {
    fetchStats();
    const interval = setInterval(fetchStats, 5000); // Poll every 5s like the original app
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex-1 flex justify-center py-4 sm:py-8 px-2 sm:px-6 lg:px-8">
      <div className="w-full max-w-[1400px] flex flex-col gap-6 sm:gap-8">
        
        {/* Header Section */}
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
          <div className="flex flex-col gap-2">
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-primary/10 dark:bg-primary/20 border border-primary/20 text-primary dark:text-primary-light text-xs font-bold w-fit mb-1 shadow-sm">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-primary"></span>
              </span>
              Live System Status
            </div>
            <h1 className="text-text-primary-light dark:text-text-primary-dark text-3xl md:text-4xl font-black leading-tight tracking-[-0.033em]">
              Overview Dashboard
            </h1>
            <p className="text-text-secondary-light dark:text-text-secondary-dark text-base font-medium max-w-2xl">
              Real-time monitoring of today's attendance capture, system metrics, and recognition engine status.
            </p>
          </div>
          
          <div className="flex gap-3">
            <button
              onClick={fetchStats}
              className="flex items-center justify-center gap-2 px-4 py-2 bg-surface-light dark:bg-surface-dark border border-border-light dark:border-border-dark rounded-xl text-text-secondary-light dark:text-text-secondary-dark font-bold text-sm hover:bg-background-light dark:hover:bg-background-dark transition-all hover:-translate-y-0.5 shadow-sm hover:shadow-md active:scale-95"
            >
              <span className="material-symbols-outlined text-[20px]">refresh</span>
              Sync
            </button>
            <button className="flex items-center justify-center gap-2 px-4 py-2 bg-slate-900 border border-slate-800 text-white rounded-xl font-bold text-sm hover:bg-slate-800 transition-all shadow-sm dark:bg-slate-700 dark:border-slate-600 dark:hover:bg-slate-600 hover:-translate-y-0.5 active:scale-95">
              <span className="material-symbols-outlined text-[20px]">download</span>
              Export
            </button>
          </div>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-2 md:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-5">
          <StatCard
            title="Total Students"
            value={stats.registered_students}
            icon="groups"
            colorClass="from-blue-500 to-blue-700 text-blue-500"
            badgeText="Enrolled"
            badgeIcon="school"
            shadowColor="bg-blue-500/10"
          />
          <StatCard
            title="Present Today"
            value={stats.present_today}
            icon="how_to_reg"
            colorClass="from-green-500 to-green-600 text-green-500"
            badgeText="+12% vs avg"
            badgeIcon="trending_up"
            shadowColor="bg-green-500/10"
          />
          <StatCard
            title="Absent Today"
            value={stats.absent_late}
            icon="person_off"
            colorClass="from-red-500 to-red-600 text-red-500"
            shadowColor="bg-red-500/10"
          />
          <StatCard
            title="Model Version"
            value={stats.model_version}
            icon="view_in_ar"
            colorClass="from-purple-500 to-purple-600 text-purple-500"
            badgeText={stats.model_active ? "Trained" : "Pending"}
            badgeIcon={stats.model_active ? "check_circle" : "warning"}
            shadowColor="bg-purple-500/10"
            description={stats.model_active ? "Model is active" : "Needs Retraining"}
          />
        </div>

        {/* Real-time Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          
          <div className="lg:col-span-2 flex flex-col rounded-[20px] bg-surface-light border border-border-light/60 shadow-sm backdrop-blur-md dark:bg-surface-dark/90 dark:border-border-dark/60 overflow-hidden transition-shadow hover:shadow-md">
            <div className="p-5 sm:p-6 border-b border-border-light/60 dark:border-border-dark/60 flex flex-wrap gap-4 justify-between items-center bg-background-light/30 dark:bg-background-dark/30">
              <div>
                <h3 className="text-text-primary-light dark:text-text-primary-dark text-[18px] font-extrabold flex items-center gap-2">
                  <span className="relative flex h-2.5 w-2.5 mr-1">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-500 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-green-500"></span>
                  </span>
                  Live Attendance Feed
                </h3>
                <p className="text-xs font-bold text-text-secondary-light dark:text-text-secondary-dark mt-1">
                  Students detected today: {stats.present_today}
                </p>
              </div>
              <div className="flex gap-2">
                <button className="p-2 border border-border-light dark:border-border-dark rounded-lg text-text-secondary-light dark:text-text-secondary-dark hover:bg-background-light dark:hover:bg-background-dark hover:text-text-primary-light dark:hover:text-text-primary-dark transition-colors">
                  <span className="material-symbols-outlined text-[18px]">filter_list</span>
                </button>
              </div>
            </div>
            
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="bg-background-light/50 dark:bg-background-dark/50 border-b border-border-light/60 dark:border-border-dark/60">
                    <th className="p-4 pl-6 text-xs font-bold tracking-wider text-text-secondary-light dark:text-text-secondary-dark uppercase">Student</th>
                    <th className="p-4 text-xs font-bold tracking-wider text-text-secondary-light dark:text-text-secondary-dark uppercase">Check-in Time</th>
                    <th className="p-4 text-xs font-bold tracking-wider text-text-secondary-light dark:text-text-secondary-dark uppercase">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border-light/60 dark:divide-border-dark/60 text-sm text-text-primary-light dark:text-text-primary-dark" id="attendance-tbody">
                  {stats.recent_records.length > 0 ? (
                    stats.recent_records.map((record, index) => (
                      <tr key={index} className="hover:bg-background-light/30 dark:hover:bg-background-dark/30 transition-colors animate-[slideIn_0.3s_ease-out]">
                        <td className="p-4 pl-6">
                            <div className="flex items-center gap-3">
                                <div className="size-10 rounded-full overflow-hidden bg-gradient-to-br from-primary to-primary-dark text-white flex items-center justify-center text-sm font-bold border-2 border-white dark:border-slate-700 shadow-sm relative group">
                                    {record.name.substring(0, 2).toUpperCase()}
                                    <div className="absolute inset-x-0 bottom-0 top-auto h-1/2 bg-black/20 opacity-0 group-hover:opacity-100 transition-opacity"></div>
                                </div>
                                <div className="flex flex-col">
                                    <p className="font-bold text-text-primary-light dark:text-text-primary-dark text-sm leading-tight">{record.name}</p>
                                    <p className="text-[11px] font-bold text-text-secondary-light dark:text-text-secondary-dark mt-0.5">{record.roll_number || 'N/A'}</p>
                                </div>
                            </div>
                        </td>
                        <td className="p-4 font-mono text-sm tracking-tight text-text-secondary-light dark:text-text-secondary-dark font-medium">{record.time}</td>
                        <td className="p-4">
                            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-bold bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-400 border border-green-200/50 dark:border-green-800/30">
                                <span className="size-1.5 rounded-full bg-green-500 shadow-[0_0_5px_rgba(34,197,94,0.6)]"></span>
                                Recognized
                            </span>
                        </td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan="3" className="p-12 text-center">
                        <div className="flex flex-col items-center justify-center text-text-secondary-light dark:text-text-secondary-dark">
                          <div className="size-16 rounded-full bg-background-light dark:bg-background-dark flex items-center justify-center mb-3">
                            <span className="material-symbols-outlined text-3xl opacity-50">face</span>
                          </div>
                          <p className="font-bold text-sm">No attendance records today</p>
                          <p className="text-xs font-medium mt-1 opacity-70">Start the recognition camera to capture attendance.</p>
                        </div>
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
            {stats.recent_records.length > 0 && (
              <div className="p-3 border-t border-border-light/60 dark:border-border-dark/60 bg-background-light/30 dark:bg-background-dark/30 text-center">
                <a href="/reports" className="text-xs font-bold text-primary hover:text-primary-dark transition-colors flex items-center justify-center gap-1">
                  View full reports <span className="material-symbols-outlined text-[16px]">arrow_forward</span>
                </a>
              </div>
            )}
          </div>

          <div className="flex flex-col gap-6">
            <div className="bg-surface-light dark:bg-surface-dark/90 p-6 rounded-[20px] border border-border-light/60 dark:border-border-dark/60 shadow-sm backdrop-blur-md transition-shadow hover:shadow-md relative overflow-hidden group">
              <div className="absolute -right-10 -bottom-10 w-40 h-40 bg-gradient-to-br from-primary/10 to-transparent rounded-full blur-2xl group-hover:bg-primary/20 transition-colors duration-500"></div>
              <h3 className="text-text-primary-light dark:text-text-primary-dark text-lg font-extrabold mb-2 relative z-10">Quick Actions</h3>
              <p className="text-xs font-medium text-text-secondary-light dark:text-text-secondary-dark mb-6 relative z-10">Common tasks for attendance management.</p>
              
              <div className="flex flex-col gap-3 relative z-10">
                <a href="/attendance" className="flex items-center gap-3 p-3 rounded-xl bg-background-light dark:bg-background-dark border border-border-light/50 dark:border-border-dark/50 hover:border-primary/50 dark:hover:border-primary/50 transition-all hover:-translate-y-0.5 group/btn">
                  <div className="size-10 rounded-lg bg-green-50 dark:bg-green-900/20 text-green-600 dark:text-green-400 flex items-center justify-center shrink-0">
                    <span className="material-symbols-outlined text-[20px]">play_circle</span>
                  </div>
                  <div className="flex-1">
                    <p className="text-sm font-bold text-text-primary-light dark:text-text-primary-dark group-hover/btn:text-primary transition-colors">Start Camera</p>
                  </div>
                  <span className="material-symbols-outlined text-text-secondary-light dark:text-text-secondary-dark opacity-0 group-hover/btn:opacity-100 group-hover/btn:-translate-x-1 transition-all text-[20px]">arrow_forward</span>
                </a>

                <a href="/register" className="flex items-center gap-3 p-3 rounded-xl bg-background-light dark:bg-background-dark border border-border-light/50 dark:border-border-dark/50 hover:border-primary/50 dark:hover:border-primary/50 transition-all hover:-translate-y-0.5 group/btn">
                  <div className="size-10 rounded-lg bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400 flex items-center justify-center shrink-0">
                    <span className="material-symbols-outlined text-[20px]">person_add</span>
                  </div>
                  <div className="flex-1">
                    <p className="text-sm font-bold text-text-primary-light dark:text-text-primary-dark group-hover/btn:text-primary transition-colors">Register Student</p>
                  </div>
                  <span className="material-symbols-outlined text-text-secondary-light dark:text-text-secondary-dark opacity-0 group-hover/btn:opacity-100 group-hover/btn:-translate-x-1 transition-all text-[20px]">arrow_forward</span>
                </a>
              </div>
            </div>

            <div className="bg-gradient-to-br from-primary to-primary-dark p-6 rounded-[20px] shadow-glow text-white relative overflow-hidden transition-transform hover:-translate-y-1">
               <div className="absolute top-0 right-0 w-32 h-32 bg-white/10 rounded-full blur-2xl transform translate-x-1/2 -translate-y-1/2"></div>
               <div className="flex items-center gap-2 mb-2 opacity-90">
                   <span className="material-symbols-outlined text-[18px]">verified_user</span>
                   <span className="text-[11px] font-bold uppercase tracking-wider">System Status</span>
               </div>
               <h3 className="text-xl font-black mb-1">Optimal</h3>
               <p className="text-sm font-medium opacity-80 mb-4 text-balance">The recognition engine is running smoothly with high confidence rates.</p>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}
