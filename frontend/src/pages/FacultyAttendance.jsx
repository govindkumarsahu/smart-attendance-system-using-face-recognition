import { useState, useEffect } from 'react';
import { api } from '../services/api';
import StatCard from '../components/StatCard';
import { useToast } from '../context/ToastContext';

export default function FacultyAttendance() {
  const [faculty, setFaculty] = useState([]);
  const [summary, setSummary] = useState({ total_faculty: 0, present_today: 0, on_leave: 0, avg_work_hours: 0, records: [] });
  const [history, setHistory] = useState([]);
  const [selectedFaculty, setSelectedFaculty] = useState('');
  const [status, setStatus] = useState('Present');
  const [remarks, setRemarks] = useState('');
  const [dateFrom, setDateFrom] = useState('');
  const [dateTo, setDateTo] = useState('');
  const [activeTab, setActiveTab] = useState('today');
  const [isLoading, setIsLoading] = useState(true);
  const { addToast } = useToast();

  const fetchData = async () => {
    setIsLoading(true);
    try {
      const [fac, sum] = await Promise.all([api.getFaculty(), api.getFacultyAttendanceToday()]);
      setFaculty(Array.isArray(fac) ? fac : []);
      setSummary(sum || { total_faculty: 0, present_today: 0, on_leave: 0, avg_work_hours: 0, records: [] });
    } catch (e) { addToast('Failed to load data', 'error'); }
    finally { setIsLoading(false); }
  };

  const fetchHistory = async () => {
    try {
      const params = {};
      if (dateFrom) params.date_from = dateFrom;
      if (dateTo) params.date_to = dateTo;
      const data = await api.getFacultyAttendanceHistory(params);
      setHistory(Array.isArray(data) ? data : []);
    } catch (e) { addToast('Failed to load history', 'error'); }
  };

  useEffect(() => { fetchData(); }, []);
  useEffect(() => { if (activeTab === 'history') fetchHistory(); }, [activeTab]);

  const handleCheckIn = async () => {
    if (!selectedFaculty) return addToast('Please select a faculty member', 'error');
    try {
      const res = await api.markFacultyAttendance({ faculty_id: parseInt(selectedFaculty), status, remarks });
      if (res.success) { addToast(res.message, 'success'); setSelectedFaculty(''); setRemarks(''); fetchData(); }
      else addToast(res.message, 'error');
    } catch (e) { addToast('Check-in failed', 'error'); }
  };

  const handleCheckOut = async (facultyId) => {
    try {
      const res = await api.checkoutFaculty({ faculty_id: facultyId });
      if (res.success) { addToast(res.message, 'success'); fetchData(); }
      else addToast(res.message, 'error');
    } catch (e) { addToast('Check-out failed', 'error'); }
  };

  const handleExport = () => api.exportFacultyCSV({ date_from: dateFrom, date_to: dateTo });

  const getStatusBadge = (s) => {
    const map = {
      'Present': 'bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-400 border-green-200/50 dark:border-green-800/30',
      'Half-Day': 'bg-amber-50 dark:bg-amber-900/20 text-amber-700 dark:text-amber-400 border-amber-200/50 dark:border-amber-800/30',
      'On Leave': 'bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-400 border-red-200/50 dark:border-red-800/30',
      'Absent': 'bg-slate-50 dark:bg-slate-900/20 text-slate-700 dark:text-slate-400 border-slate-200/50 dark:border-slate-800/30',
    };
    return map[s] || map['Present'];
  };

  const getDotColor = (s) => {
    const map = { 'Present': 'bg-green-500', 'Half-Day': 'bg-amber-500', 'On Leave': 'bg-red-500', 'Absent': 'bg-slate-500' };
    return map[s] || 'bg-green-500';
  };

  const selectCls = "block w-full pl-10 pr-10 py-3 border border-border-light/80 dark:border-border-dark/80 rounded-xl bg-background-light/50 dark:bg-background-dark/50 text-text-primary-light dark:text-text-primary-dark focus:ring-2 focus:ring-primary/50 focus:border-primary transition-all font-medium shadow-inner text-sm appearance-none cursor-pointer form-select-arrow";

  return (
    <div className="flex-1 flex flex-col items-center py-4 sm:py-8 px-2 sm:px-6 lg:px-8">
      <div className="w-full max-w-[1400px] flex flex-col gap-6 sm:gap-8">

        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
          <div className="flex flex-col gap-2">
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-amber-500/10 dark:bg-amber-500/20 border border-amber-500/20 text-amber-600 dark:text-amber-400 text-xs font-bold w-fit mb-1 shadow-sm">
              <span className="material-symbols-outlined text-[14px]">badge</span>
              Staff Tracking
            </div>
            <h1 className="text-text-primary-light dark:text-text-primary-dark text-3xl md:text-4xl font-black leading-tight tracking-[-0.033em]">Faculty Attendance</h1>
            <p className="text-text-secondary-light dark:text-text-secondary-dark text-base font-medium max-w-2xl">Track daily check-in, check-out, and work hours for all faculty members.</p>
          </div>
          <div className="flex gap-3">
            <button onClick={fetchData} className="flex items-center gap-2 px-4 py-2 bg-surface-light dark:bg-surface-dark border border-border-light dark:border-border-dark rounded-xl text-text-secondary-light dark:text-text-secondary-dark font-bold text-sm hover:bg-background-light dark:hover:bg-background-dark transition-all hover:-translate-y-0.5 shadow-sm">
              <span className="material-symbols-outlined text-[20px]">refresh</span>Sync
            </button>
            <button onClick={handleExport} className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-primary to-primary-dark text-white rounded-xl font-bold text-sm transition-all shadow-glow hover:shadow-[0_0_25px_rgba(99,102,241,0.6)] hover:-translate-y-0.5">
              <span className="material-symbols-outlined text-[20px]">download</span>Export CSV
            </button>
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 sm:gap-5">
          <StatCard title="Total Faculty" value={summary.total_faculty} icon="groups" colorClass="from-blue-500 to-blue-700 text-blue-500" badgeText="Staff" badgeIcon="badge" shadowColor="bg-blue-500/10" />
          <StatCard title="Present Today" value={summary.present_today} icon="how_to_reg" colorClass="from-green-500 to-green-600 text-green-500" shadowColor="bg-green-500/10" />
          <StatCard title="On Leave" value={summary.on_leave} icon="event_busy" colorClass="from-red-500 to-red-600 text-red-500" shadowColor="bg-red-500/10" />
          <StatCard title="Avg Work Hours" value={summary.avg_work_hours ? `${summary.avg_work_hours}h` : '—'} icon="schedule" colorClass="from-amber-500 to-amber-600 text-amber-500" shadowColor="bg-amber-500/10" />
        </div>

        {/* Mark Attendance Card */}
        <div className="bg-surface-light dark:bg-surface-dark/90 rounded-[20px] shadow-sm border border-border-light/60 dark:border-border-dark/60 p-6 backdrop-blur-md relative overflow-hidden transition-shadow hover:shadow-md">
          <div className="absolute -right-10 -bottom-10 w-40 h-40 bg-gradient-to-br from-primary/10 to-transparent rounded-full blur-2xl"></div>
          <h3 className="text-text-primary-light dark:text-text-primary-dark text-lg font-extrabold mb-4 flex items-center gap-2 relative z-10">
            <span className="material-symbols-outlined text-primary">login</span>Mark Attendance
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4 relative z-10">
            <div className="lg:col-span-2 relative">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none"><span className="material-symbols-outlined text-text-secondary-light text-[20px]">person</span></div>
              <select value={selectedFaculty} onChange={e => setSelectedFaculty(e.target.value)} className={selectCls}>
                <option value="" style={{backgroundColor:'#0d1117', color:'#f1f5f9'}}>Select Faculty</option>
                {faculty.map(f => <option key={f.id} value={f.id} style={{backgroundColor:'#0d1117', color:'#f1f5f9'}}>{f.full_name} ({f.employee_id || 'N/A'})</option>)}
              </select>
            </div>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none"><span className="material-symbols-outlined text-text-secondary-light text-[20px]">fact_check</span></div>
              <select value={status} onChange={e => setStatus(e.target.value)} className={selectCls}>
                <option value="Present" style={{backgroundColor:'#0d1117', color:'#f1f5f9'}}>Present</option>
                <option value="On Leave" style={{backgroundColor:'#0d1117', color:'#f1f5f9'}}>On Leave</option>
                <option value="Half-Day" style={{backgroundColor:'#0d1117', color:'#f1f5f9'}}>Half-Day</option>
              </select>
            </div>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none"><span className="material-symbols-outlined text-text-secondary-light text-[20px]">notes</span></div>
              <input type="text" value={remarks} onChange={e => setRemarks(e.target.value)} placeholder="Remarks (optional)" className="block w-full pl-10 pr-3 py-3 border border-border-light/80 dark:border-border-dark/80 rounded-xl bg-background-light/50 dark:bg-background-dark/50 text-text-primary-light dark:text-text-primary-dark placeholder-text-secondary-light/60 focus:ring-2 focus:ring-primary/50 focus:border-primary transition-all font-medium shadow-inner text-sm" />
            </div>
            <button onClick={handleCheckIn} className="flex items-center justify-center gap-2 py-3 px-4 rounded-xl text-white font-bold bg-gradient-to-r from-green-500 to-green-600 hover:from-green-600 hover:to-green-700 shadow-sm hover:shadow-md transition-all hover:-translate-y-0.5 active:scale-[0.98] text-sm">
              <span className="material-symbols-outlined text-[20px]">login</span>Check In
            </button>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex gap-2">
          {['today', 'history'].map(tab => (
            <button key={tab} onClick={() => setActiveTab(tab)} className={`px-5 py-2.5 rounded-xl font-bold text-sm transition-all ${activeTab === tab ? 'bg-primary text-white shadow-glow' : 'bg-surface-light dark:bg-surface-dark border border-border-light dark:border-border-dark text-text-secondary-light dark:text-text-secondary-dark hover:bg-background-light dark:hover:bg-background-dark'}`}>
              {tab === 'today' ? "Today's Attendance" : 'Attendance History'}
            </button>
          ))}
        </div>

        {/* History Filters */}
        {activeTab === 'history' && (
          <div className="flex flex-wrap gap-4 bg-surface-light/80 dark:bg-surface-dark/90 p-5 rounded-[20px] shadow-sm border border-border-light/60 dark:border-border-dark/60 backdrop-blur-md items-center">
            <div className="flex items-center gap-3 flex-1 min-w-[250px]">
              <input type="date" value={dateFrom} onChange={e => setDateFrom(e.target.value)} className="flex-1 px-3 py-3 border border-border-light/60 dark:border-border-dark/60 rounded-xl bg-background-light/50 dark:bg-background-dark/50 text-text-secondary-light dark:text-text-secondary-dark focus:ring-2 focus:ring-primary/50 text-sm shadow-inner" />
              <span className="text-text-secondary-light font-medium">—</span>
              <input type="date" value={dateTo} onChange={e => setDateTo(e.target.value)} className="flex-1 px-3 py-3 border border-border-light/60 dark:border-border-dark/60 rounded-xl bg-background-light/50 dark:bg-background-dark/50 text-text-secondary-light dark:text-text-secondary-dark focus:ring-2 focus:ring-primary/50 text-sm shadow-inner" />
            </div>
            <button onClick={fetchHistory} className="flex items-center gap-2 px-6 py-3 bg-slate-900 text-white rounded-xl font-bold text-sm hover:bg-slate-800 transition-colors shadow-sm dark:bg-slate-700 dark:hover:bg-slate-600">
              <span className="material-symbols-outlined text-[20px]">filter_list</span>Apply
            </button>
          </div>
        )}

        {/* Table */}
        <div className="bg-surface-light/90 dark:bg-surface-dark/90 rounded-[20px] border border-border-light/60 dark:border-border-dark/60 shadow-sm backdrop-blur-md overflow-hidden transition-shadow hover:shadow-md">
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-background-light/50 dark:bg-background-dark/50 border-b border-border-light/60 dark:border-border-dark/60">
                  <th className="p-4 pl-6 text-xs font-bold tracking-wider text-text-secondary-light dark:text-text-secondary-dark uppercase">Faculty</th>
                  <th className="p-4 text-xs font-bold tracking-wider text-text-secondary-light dark:text-text-secondary-dark uppercase">Employee ID</th>
                  <th className="p-4 text-xs font-bold tracking-wider text-text-secondary-light dark:text-text-secondary-dark uppercase">Dept</th>
                  {activeTab === 'history' && <th className="p-4 text-xs font-bold tracking-wider text-text-secondary-light dark:text-text-secondary-dark uppercase">Date</th>}
                  <th className="p-4 text-xs font-bold tracking-wider text-text-secondary-light dark:text-text-secondary-dark uppercase">Check-In</th>
                  <th className="p-4 text-xs font-bold tracking-wider text-text-secondary-light dark:text-text-secondary-dark uppercase">Check-Out</th>
                  <th className="p-4 text-xs font-bold tracking-wider text-text-secondary-light dark:text-text-secondary-dark uppercase">Work Hrs</th>
                  <th className="p-4 text-xs font-bold tracking-wider text-text-secondary-light dark:text-text-secondary-dark uppercase">Status</th>
                  {activeTab === 'today' && <th className="p-4 text-xs font-bold tracking-wider text-text-secondary-light dark:text-text-secondary-dark uppercase">Action</th>}
                </tr>
              </thead>
              <tbody className="divide-y divide-border-light/60 dark:divide-border-dark/60 text-sm text-text-primary-light dark:text-text-primary-dark">
                {isLoading ? (
                  <tr><td colSpan="9" className="p-10 text-center"><span className="animate-spin inline-block w-6 h-6 border-3 border-primary/30 border-t-primary rounded-full"></span></td></tr>
                ) : (activeTab === 'today' ? summary.records : history).length > 0 ? (
                  (activeTab === 'today' ? summary.records : history).map((r, i) => (
                    <tr key={i} className="hover:bg-background-light/30 dark:hover:bg-background-dark/30 transition-colors">
                      <td className="p-4 pl-6">
                        <div className="flex items-center gap-3">
                          <div className="size-9 rounded-full bg-gradient-to-br from-primary to-primary-dark text-white flex items-center justify-center text-xs font-bold border-2 border-white dark:border-slate-700 shadow-sm">
                            {r.full_name?.substring(0, 2).toUpperCase()}
                          </div>
                          <div>
                            <p className="font-bold text-sm leading-tight">{r.full_name}</p>
                            <p className="text-[11px] text-text-secondary-light dark:text-text-secondary-dark">{r.designation}</p>
                          </div>
                        </div>
                      </td>
                      <td className="p-4 font-mono text-xs font-bold text-primary">{r.employee_id || '—'}</td>
                      <td className="p-4 text-text-secondary-light dark:text-text-secondary-dark">{r.department || '—'}</td>
                      {activeTab === 'history' && <td className="p-4 text-text-secondary-light dark:text-text-secondary-dark">{r.date}</td>}
                      <td className="p-4 font-mono text-sm">{r.check_in || '—'}</td>
                      <td className="p-4 font-mono text-sm">{r.check_out || <span className="text-text-secondary-light/50 italic">pending</span>}</td>
                      <td className="p-4 font-bold">{r.work_hours > 0 ? `${r.work_hours}h` : '—'}</td>
                      <td className="p-4">
                        <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-bold border ${getStatusBadge(r.status)}`}>
                          <span className={`size-1.5 rounded-full ${getDotColor(r.status)}`}></span>{r.status}
                        </span>
                      </td>
                      {activeTab === 'today' && (
                        <td className="p-4">
                          {r.check_in && !r.check_out && r.status !== 'On Leave' ? (
                            <button onClick={() => handleCheckOut(faculty.find(f => f.full_name === r.full_name)?.id)} className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 border border-red-200/50 dark:border-red-800/30 hover:bg-red-100 dark:hover:bg-red-900/40 transition-colors">
                              <span className="material-symbols-outlined text-[16px]">logout</span>Check Out
                            </button>
                          ) : r.check_out ? (
                            <span className="text-xs font-bold text-green-600 dark:text-green-400 flex items-center gap-1"><span className="material-symbols-outlined text-[16px]">check_circle</span>Done</span>
                          ) : null}
                        </td>
                      )}
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan="9" className="p-12 text-center">
                      <div className="flex flex-col items-center text-text-secondary-light dark:text-text-secondary-dark">
                        <div className="size-16 rounded-full bg-background-light dark:bg-background-dark flex items-center justify-center mb-3"><span className="material-symbols-outlined text-3xl opacity-50">badge</span></div>
                        <p className="font-bold text-sm">No attendance records found</p>
                        <p className="text-xs mt-1 opacity-70">{activeTab === 'today' ? 'Mark check-in for faculty above.' : 'Try different date filters.'}</p>
                      </div>
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
