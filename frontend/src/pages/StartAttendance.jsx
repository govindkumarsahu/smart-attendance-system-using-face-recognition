import { useState, useEffect } from 'react';
import { api } from '../services/api';
import { useToast } from '../context/ToastContext';

const PERIODS = [
  { value: 'Lecture 1', label: 'Lecture 1', time: '09:00 - 09:50' },
  { value: 'Lecture 2', label: 'Lecture 2', time: '10:00 - 10:50' },
  { value: 'Lecture 3', label: 'Lecture 3', time: '11:00 - 11:50' },
  { value: 'Lecture 4', label: 'Lecture 4', time: '12:00 - 12:50' },
  { value: 'Lecture 5', label: 'Lecture 5', time: '02:00 - 02:50' },
  { value: 'Lecture 6', label: 'Lecture 6', time: '03:00 - 03:50' },
];

export default function StartAttendance() {
  const [subjects, setSubjects] = useState([]);
  const [sessions, setSessions] = useState([]);
  const [selectedSubject, setSelectedSubject] = useState('');
  const [selectedPeriod, setSelectedPeriod] = useState('');
  const [isStarting, setIsStarting] = useState(false);
  const [stats, setStats] = useState({ recent_records: [], present_today: 0 });
  const { addToast } = useToast();

  const fetchData = async () => {
    try {
      const [subjectData, sessionData, statsData] = await Promise.all([
        api.getSubjects(),
        api.getLectureSessions(),
        api.getDashboardStats(),
      ]);
      setSubjects(subjectData);
      setSessions(sessionData);
      setStats(statsData);
    } catch (error) {
      console.error('Failed to fetch data', error);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 8000);
    return () => clearInterval(interval);
  }, []);

  const selectedSubjectObj = subjects.find(s => s.code === selectedSubject);
  const canStart = selectedSubject && selectedPeriod;

  const handleStartCapture = async () => {
    if (!canStart) {
      addToast('Please select a subject and period first.', 'error');
      return;
    }
    setIsStarting(true);
    addToast(`Starting camera for ${selectedSubjectObj?.name} - ${selectedPeriod}...`, 'info');
    try {
      const result = await api.startAttendanceWithSubject({
        subject_code: selectedSubject,
        subject_name: selectedSubjectObj?.name || selectedSubject,
        period: selectedPeriod,
      });
      if (result.success) {
        addToast(result.message, 'success');
        setTimeout(fetchData, 3000);
      } else {
        addToast(result.message || 'Failed to start attendance.', 'error');
      }
    } catch (error) {
      addToast('Error starting attendance camera.', 'error');
    } finally {
      setIsStarting(false);
    }
  };

  return (
    <div className="flex-1 flex justify-center py-4 md:py-8 px-2 sm:px-4 lg:px-8">
      <div className="w-full max-w-[1200px] flex flex-col gap-6 md:gap-8">

        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
          <div className="flex flex-col gap-2">
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-green-500/10 border border-green-500/20 text-green-600 dark:text-green-400 text-xs font-bold w-fit mb-1">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-500 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
              </span>
              Ready to Capture
            </div>
            <h1 className="text-text-primary-light dark:text-text-primary-dark text-3xl md:text-4xl font-black leading-tight tracking-[-0.033em]">
              Live Attendance
            </h1>
            <p className="text-text-secondary-light dark:text-text-secondary-dark text-base font-medium max-w-2xl">
              Select a subject and period, then launch the recognition engine.
            </p>
          </div>
          <div className="flex gap-3">
            <div className="flex items-center gap-2 px-4 py-2 bg-surface-light dark:bg-surface-dark border border-border-light dark:border-border-dark rounded-xl shadow-sm">
              <span className="text-xs font-bold text-text-secondary-light dark:text-text-secondary-dark uppercase tracking-wider">Today's Total:</span>
              <span className="text-lg font-black text-primary">{stats.present_today}</span>
            </div>
          </div>
        </div>

        {/* Main Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 lg:gap-8">

          {/* Left: Subject & Period Selection + Camera Launch */}
          <div className="lg:col-span-8 flex flex-col gap-6">

            {/* Selection Card */}
            <div className="rounded-[20px] bg-surface-light dark:bg-surface-dark/90 border border-border-light/60 dark:border-border-dark/60 shadow-sm backdrop-blur-md overflow-hidden">
              <div className="p-5 border-b border-border-light/60 dark:border-border-dark/60 bg-background-light/30 dark:bg-background-dark/30">
                <h3 className="text-text-primary-light dark:text-text-primary-dark text-[16px] font-extrabold flex items-center gap-2">
                  <span className="material-symbols-outlined text-primary text-[20px]">tune</span>
                  Configure Attendance Session
                </h3>
                <p className="text-xs text-text-secondary-light dark:text-text-secondary-dark mt-1 font-medium">Select the subject and lecture period before starting the camera.</p>
              </div>

              <div className="p-5 md:p-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">

                  {/* Subject Dropdown */}
                  <div className="flex flex-col gap-2">
                    <label className="text-xs font-bold uppercase tracking-wider text-text-secondary-light dark:text-text-secondary-dark flex items-center gap-1.5">
                      <span className="material-symbols-outlined text-[16px] text-primary">menu_book</span>
                      Subject
                    </label>
                    <div className="relative">
                      <select
                        value={selectedSubject}
                        onChange={(e) => setSelectedSubject(e.target.value)}
                        className="form-select-arrow w-full px-4 py-3.5 pr-10 rounded-xl border-2 border-border-light/60 dark:border-border-dark/60 bg-background-light dark:bg-background-dark text-text-primary-light dark:text-text-primary-dark text-sm font-semibold focus:outline-none focus:ring-2 focus:ring-primary/40 focus:border-primary transition-all cursor-pointer hover:border-primary/50"
                      >
                        <option value="">— Select Subject —</option>
                        {subjects.map((subj) => (
                          <option key={subj.id} value={subj.code}>
                            {subj.code} — {subj.name}
                          </option>
                        ))}
                      </select>
                    </div>
                    {selectedSubjectObj && (
                      <div className="flex items-center gap-2 mt-1 animate-[slideIn_0.3s_ease-out]">
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[10px] font-bold bg-primary/10 text-primary border border-primary/20">
                          {selectedSubjectObj.department || 'CSE'}
                        </span>
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[10px] font-bold bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/20">
                          Sem {selectedSubjectObj.semester || '—'}
                        </span>
                      </div>
                    )}
                  </div>

                  {/* Period Dropdown */}
                  <div className="flex flex-col gap-2">
                    <label className="text-xs font-bold uppercase tracking-wider text-text-secondary-light dark:text-text-secondary-dark flex items-center gap-1.5">
                      <span className="material-symbols-outlined text-[16px] text-primary">schedule</span>
                      Period / Lecture
                    </label>
                    <div className="relative">
                      <select
                        value={selectedPeriod}
                        onChange={(e) => setSelectedPeriod(e.target.value)}
                        className="form-select-arrow w-full px-4 py-3.5 pr-10 rounded-xl border-2 border-border-light/60 dark:border-border-dark/60 bg-background-light dark:bg-background-dark text-text-primary-light dark:text-text-primary-dark text-sm font-semibold focus:outline-none focus:ring-2 focus:ring-primary/40 focus:border-primary transition-all cursor-pointer hover:border-primary/50"
                      >
                        <option value="">— Select Period —</option>
                        {PERIODS.map((p) => (
                          <option key={p.value} value={p.value}>
                            {p.label} ({p.time})
                          </option>
                        ))}
                      </select>
                    </div>
                    {selectedPeriod && (
                      <div className="flex items-center gap-1 mt-1 animate-[slideIn_0.3s_ease-out]">
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[10px] font-bold bg-green-500/10 text-green-600 dark:text-green-400 border border-green-500/20">
                          <span className="material-symbols-outlined text-[12px]">check_circle</span>
                          {PERIODS.find(p => p.value === selectedPeriod)?.time}
                        </span>
                      </div>
                    )}
                  </div>
                </div>

                {/* Selected Summary & Launch Button */}
                <div className={`rounded-2xl p-4 border-2 transition-all duration-300 ${canStart ? 'bg-gradient-to-r from-green-500/5 to-emerald-500/5 border-green-500/30 dark:border-green-500/20' : 'bg-background-light/50 dark:bg-background-dark/50 border-border-light/40 dark:border-border-dark/40'}`}>
                  {canStart ? (
                    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                      <div className="flex items-center gap-3">
                        <div className="size-12 rounded-xl bg-gradient-to-br from-green-500 to-emerald-600 flex items-center justify-center shadow-lg shadow-green-500/20">
                          <span className="material-symbols-outlined text-white text-[24px]">school</span>
                        </div>
                        <div>
                          <p className="text-sm font-black text-text-primary-light dark:text-text-primary-dark">
                            {selectedSubjectObj?.name}
                          </p>
                          <p className="text-xs font-semibold text-text-secondary-light dark:text-text-secondary-dark flex items-center gap-1.5 mt-0.5">
                            <span>{selectedSubjectObj?.code}</span>
                            <span className="text-border-light dark:text-border-dark">•</span>
                            <span>{selectedPeriod}</span>
                            <span className="text-border-light dark:text-border-dark">•</span>
                            <span>{PERIODS.find(p => p.value === selectedPeriod)?.time}</span>
                          </p>
                        </div>
                      </div>
                      <button
                        onClick={handleStartCapture}
                        disabled={isStarting}
                        className="group/btn relative px-6 py-3.5 bg-gradient-to-r from-green-500 to-emerald-600 text-white rounded-2xl font-bold text-[14px] shadow-[0_0_30px_rgba(34,197,94,0.25)] hover:shadow-[0_0_40px_rgba(34,197,94,0.45)] transition-all transform hover:-translate-y-0.5 active:scale-[0.98] flex items-center gap-2.5 overflow-hidden shrink-0"
                      >
                        <div className="absolute inset-0 bg-white/20 translate-y-full group-hover/btn:translate-y-0 transition-transform duration-300 pointer-events-none"></div>
                        {isStarting ? (
                          <span className="animate-spin inline-block w-5 h-5 border-2 border-white/30 border-t-white rounded-full"></span>
                        ) : (
                          <span className="material-symbols-outlined text-[22px]">videocam</span>
                        )}
                        <span>{isStarting ? 'Starting...' : 'Start Attendance'}</span>
                      </button>
                    </div>
                  ) : (
                    <div className="flex items-center gap-3 py-2">
                      <div className="size-10 rounded-lg bg-text-secondary-light/10 dark:bg-text-secondary-dark/10 flex items-center justify-center">
                        <span className="material-symbols-outlined text-text-secondary-light dark:text-text-secondary-dark text-[22px]">info</span>
                      </div>
                      <p className="text-sm font-semibold text-text-secondary-light dark:text-text-secondary-dark">
                        Select both a <strong className="text-text-primary-light dark:text-text-primary-dark">Subject</strong> and <strong className="text-text-primary-light dark:text-text-primary-dark">Period</strong> to enable attendance capture.
                      </p>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Today's Lecture Sessions History */}
            <div className="rounded-[20px] bg-surface-light dark:bg-surface-dark/90 border border-border-light/60 dark:border-border-dark/60 shadow-sm backdrop-blur-md overflow-hidden">
              <div className="p-5 border-b border-border-light/60 dark:border-border-dark/60 bg-background-light/30 dark:bg-background-dark/30 flex items-center justify-between">
                <h3 className="text-text-primary-light dark:text-text-primary-dark text-[16px] font-extrabold flex items-center gap-2">
                  <span className="material-symbols-outlined text-primary text-[20px]">history</span>
                  Today's Lecture Sessions
                </h3>
                <span className="bg-primary/10 text-primary px-2.5 py-0.5 rounded-full text-xs font-bold border border-primary/20">
                  {sessions.length} session{sessions.length !== 1 ? 's' : ''}
                </span>
              </div>
              <div className="divide-y divide-border-light/50 dark:divide-border-dark/50">
                {sessions.length > 0 ? sessions.map((sess, i) => (
                  <div key={i} className="flex items-center gap-4 p-4 hover:bg-background-light/30 dark:hover:bg-background-dark/30 transition-colors">
                    <div className="size-11 rounded-xl bg-gradient-to-br from-primary/10 to-primary/5 dark:from-primary/20 dark:to-primary/5 flex items-center justify-center shrink-0 border border-primary/10">
                      <span className="material-symbols-outlined text-primary text-[20px]">class</span>
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-bold text-text-primary-light dark:text-text-primary-dark truncate">
                        {sess.subject_name} <span className="text-text-secondary-light dark:text-text-secondary-dark font-normal">({sess.subject_code})</span>
                      </p>
                      <p className="text-[11px] font-medium text-text-secondary-light dark:text-text-secondary-dark mt-0.5 flex items-center gap-2 flex-wrap">
                        <span className="flex items-center gap-0.5"><span className="material-symbols-outlined text-[12px]">schedule</span>{sess.period}</span>
                        <span className="flex items-center gap-0.5"><span className="material-symbols-outlined text-[12px]">person</span>{sess.faculty_name}</span>
                        <span className="flex items-center gap-0.5"><span className="material-symbols-outlined text-[12px]">login</span>{sess.start_time?.slice(0,5)}</span>
                      </p>
                    </div>
                    <div className="flex flex-col items-end gap-1 shrink-0">
                      <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[10px] font-black uppercase tracking-wider border ${sess.status === 'completed' ? 'bg-green-50 dark:bg-green-900/20 text-green-600 dark:text-green-400 border-green-200/50 dark:border-green-800/30' : 'bg-amber-50 dark:bg-amber-900/20 text-amber-600 dark:text-amber-400 border-amber-200/50 dark:border-amber-800/30'}`}>
                        {sess.status === 'completed' ? 'Done' : 'Active'}
                      </span>
                      <span className="text-[11px] font-bold text-primary">{sess.total_present || 0} present</span>
                    </div>
                  </div>
                )) : (
                  <div className="p-10 text-center">
                    <span className="material-symbols-outlined text-4xl text-text-secondary-light dark:text-text-secondary-dark opacity-30 mb-2 block">event_busy</span>
                    <p className="text-sm font-semibold text-text-secondary-light dark:text-text-secondary-dark">No lectures conducted today yet.</p>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Right: Recognized Today Feed */}
          <div className="lg:col-span-4 flex flex-col rounded-[20px] bg-surface-light dark:bg-surface-dark/90 border border-border-light/60 dark:border-border-dark/60 shadow-sm backdrop-blur-md overflow-hidden relative min-h-[400px]">
            <div className="p-5 border-b border-border-light/60 dark:border-border-dark/60 bg-background-light/30 dark:bg-background-dark/30 flex justify-between items-center relative z-10">
              <h3 className="text-text-primary-light dark:text-text-primary-dark text-[16px] font-extrabold flex items-center gap-2">
                <span className="material-symbols-outlined text-primary text-[20px]">how_to_reg</span>
                Recognized Today
              </h3>
              <span className="bg-primary/10 text-primary px-2.5 py-0.5 rounded-full text-xs font-bold border border-primary/20">{stats.recent_records.length} recent</span>
            </div>

            <div className="flex-1 overflow-y-auto p-4 space-y-3 relative z-10">
              {stats.recent_records.length > 0 ? (
                stats.recent_records.map((record, index) => (
                  <div key={index} className="flex items-center gap-3 p-3 rounded-xl bg-background-light dark:bg-background-dark border border-border-light/50 dark:border-border-dark/50 hover:border-primary/50 dark:hover:border-primary/50 transition-colors group animate-[slideIn_0.3s_ease-out]">
                    <div className="size-11 rounded-full overflow-hidden bg-gradient-to-br from-primary to-primary-dark text-white flex items-center justify-center text-sm font-bold border-2 border-white dark:border-slate-700 shadow-sm shrink-0">
                      {record.name.substring(0, 2).toUpperCase()}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="font-bold text-text-primary-light dark:text-text-primary-dark text-sm truncate leading-tight">{record.name}</p>
                      <p className="text-[11px] font-medium text-text-secondary-light dark:text-text-secondary-dark mt-0.5 flex items-center gap-1">
                        <span className="material-symbols-outlined text-[12px]">schedule</span> {record.time}
                        {record.subject_name && (
                          <> <span className="text-border-light dark:text-border-dark">•</span> {record.subject_name}</>
                        )}
                      </p>
                    </div>
                    <div className="shrink-0">
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[10px] font-black uppercase tracking-wider bg-green-50 dark:bg-green-900/30 text-green-600 dark:text-green-400 border border-green-200/50 dark:border-green-800/50">
                        Pass
                      </span>
                    </div>
                  </div>
                ))
              ) : (
                <div className="h-full flex flex-col items-center justify-center text-text-secondary-light dark:text-text-secondary-dark opacity-50 space-y-3 py-10">
                  <span className="material-symbols-outlined text-4xl">history</span>
                  <p className="text-sm font-medium text-center">No faces recognized yet.<br />Waiting for camera feed...</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
