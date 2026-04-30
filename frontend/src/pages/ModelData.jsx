import { useState, useEffect } from 'react';
import { api } from '../services/api';
import { useToast } from '../context/ToastContext';

export default function ModelData() {
  const [data, setData] = useState({
    model_info: {
      status: 'Loading...',
      last_trained: 'Calculating',
      total_faces: 0,
      pending_count: 0,
      pending_students: []
    },
    activity: [],
    students: []
  });
  const [isTraining, setIsTraining] = useState(false);
  const { addToast } = useToast();

  const fetchModelInfo = async () => {
    try {
      const res = await api.getModelInfo();
      setData(res);
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    fetchModelInfo();
  }, []);

  const handleTrain = async () => {
    setIsTraining(true);
    addToast('Model training started. Check the Terminal Logs for details.', 'info');
    try {
      await api.trainModel();
      // Polling or waiting for training could go here. For now simulate 3s delay then fetch
      setTimeout(() => {
         fetchModelInfo();
         setIsTraining(false);
         addToast('Training initiated', 'success');
      }, 3000);
    } catch (e) {
      setIsTraining(false);
      addToast('Error starting training', 'error');
    }
  };

  const m = data.model_info;

  return (
    <div className="flex-1 flex justify-center py-8 px-4 sm:px-6 lg:px-8">
      <div className="w-full max-w-[1100px] flex flex-col gap-8">
          <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
              <div className="flex flex-col gap-2">
                  <h1 className="text-text-primary-light dark:text-text-primary-dark text-3xl md:text-4xl font-black leading-tight tracking-[-0.033em]">
                      Face Recognition Model
                  </h1>
                  <p className="text-text-secondary-light dark:text-text-secondary-dark text-base font-normal max-w-2xl">
                      Manage training datasets, update the recognition engine, and view model performance metrics. Ensure data is up to date for accurate attendance tracking.
                  </p>
              </div>
          </div>
          
          {/* Quick Info Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
              <div className="group relative flex flex-col gap-2 rounded-[20px] p-6 bg-surface-light border border-border-light/60 shadow-sm backdrop-blur-md transition-all duration-300 hover:-translate-y-1 hover:shadow-premium dark:bg-surface-dark/90 dark:border-border-dark/60 overflow-hidden">
                  <div className={`absolute -right-8 -top-8 h-32 w-32 rounded-full ${m.status === 'Trained' ? 'bg-green-500/10 group-hover:bg-green-500/20' : 'bg-amber-500/10 group-hover:bg-amber-500/20'} blur-2xl transition-transform duration-500 group-hover:scale-150`}></div>
                  <div className="relative z-10 flex items-center justify-between">
                      <p className="text-text-secondary-light dark:text-text-secondary-dark text-[11px] font-bold uppercase tracking-wider">Model Status</p>
                      <span className={`material-symbols-outlined text-[24px] ${m.status === 'Trained' ? 'text-green-600 dark:text-green-500' : 'text-amber-500'}`}>
                        {m.status === 'Trained' ? 'check_circle' : 'warning'}
                      </span>
                  </div>
                  <p className="relative z-10 text-text-primary-light dark:text-text-primary-dark text-[28px] font-extrabold tracking-tight mt-2">{m.status}</p>
                  <p className={`relative z-10 ${m.status === 'Trained' ? 'text-green-600 dark:text-green-500' : 'text-amber-500'} text-sm font-semibold flex items-center gap-1 mt-0.5`}>
                      {m.status === 'Trained' ? 'Running normally' : 'Training needed'}
                  </p>
              </div>
              <div className="group relative flex flex-col gap-2 rounded-[20px] p-6 bg-surface-light border border-border-light/60 shadow-sm backdrop-blur-md transition-all duration-300 hover:-translate-y-1 hover:shadow-premium dark:bg-surface-dark/90 dark:border-border-dark/60 overflow-hidden">
                  <div className="absolute -right-8 -top-8 h-32 w-32 rounded-full bg-primary/10 blur-2xl transition-transform duration-500 group-hover:scale-150 group-hover:bg-primary/20"></div>
                  <div className="relative z-10 flex items-center justify-between">
                      <p className="text-text-secondary-light dark:text-text-secondary-dark text-[11px] font-bold uppercase tracking-wider">Last Trained</p>
                      <span className="material-symbols-outlined text-primary text-[24px]">calendar_today</span>
                  </div>
                  <p className="relative z-10 text-text-primary-light dark:text-text-primary-dark text-[18px] font-extrabold tracking-tight mt-2">{m.last_trained}</p>
              </div>
              <div className="group relative flex flex-col gap-2 rounded-[20px] p-6 bg-surface-light border border-border-light/60 shadow-sm backdrop-blur-md transition-all duration-300 hover:-translate-y-1 hover:shadow-premium dark:bg-surface-dark/90 dark:border-border-dark/60 overflow-hidden">
                  <div className="absolute -right-8 -top-8 h-32 w-32 rounded-full bg-blue-500/10 blur-2xl transition-transform duration-500 group-hover:scale-150 group-hover:bg-blue-500/20"></div>
                  <div className="relative z-10 flex items-center justify-between">
                      <p className="text-text-secondary-light dark:text-text-secondary-dark text-[11px] font-bold uppercase tracking-wider">Total Faces</p>
                      <span className="material-symbols-outlined text-primary text-[24px]">group</span>
                  </div>
                  <p className="relative z-10 text-text-primary-light dark:text-text-primary-dark text-[28px] font-extrabold tracking-tight mt-2">{m.total_faces}</p>
                  <p className="relative z-10 text-primary text-sm font-semibold flex items-center gap-1 mt-0.5">
                      {m.pending_count > 0 ? (
                        <><span className="material-symbols-outlined text-[16px]">arrow_upward</span> {m.pending_count} pending enrollment</>
                      ) : (
                        'All students enrolled'
                      )}
                  </p>
              </div>
          </div>

          {/* Main Content Area */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              
              <div className="lg:col-span-2 flex flex-col rounded-[20px] bg-surface-light border border-border-light/60 shadow-sm backdrop-blur-md dark:bg-surface-dark/90 dark:border-border-dark/60 overflow-hidden transition-shadow hover:shadow-md relative">
                  <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] bg-primary/5 blur-[100px] rounded-full point-events-none"></div>
                  <div className="p-6 border-b border-border-light/60 dark:border-border-dark/60 flex justify-between items-center bg-gradient-to-r from-transparent to-primary/5 relative z-10">
                      <h3 className="text-text-primary-light dark:text-text-primary-dark text-[18px] font-extrabold">Training Center</h3>
                      <span className="px-3 py-1.5 rounded-full bg-background-light dark:bg-background-dark text-text-secondary-light dark:text-text-secondary-dark text-[10px] font-bold uppercase tracking-wider border border-border-light/50 dark:border-border-dark/50 shadow-sm">
                        {isTraining ? 'Training in Progress' : 'Idle'}
                      </span>
                  </div>
                  <div className="p-8 flex flex-col items-center justify-center min-h-[360px] relative z-10">
                      <div className="relative z-10 flex flex-col items-center gap-8 max-w-md text-center">
                          <div className="relative group">
                              <div className="absolute -inset-4 bg-primary/20 rounded-full blur-xl opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
                              <div className="size-32 rounded-full bg-gradient-to-br from-background-light to-border-light dark:from-background-dark dark:to-border-dark border border-border-light dark:border-border-dark flex items-center justify-center shadow-lg relative">
                                  <span className="material-symbols-outlined text-5xl text-primary dark:text-primary">model_training</span>
                                  <div className={`absolute inset-0 rounded-full border border-dashed border-border-light dark:border-border-dark ${isTraining ? 'animate-[spin_2s_linear_infinite] opacity-100' : 'animate-[spin_10s_linear_infinite] opacity-40'}`}></div>
                              </div>
                          </div>
                          <div className="space-y-3">
                              <h4 className="text-text-primary-light dark:text-text-primary-dark text-2xl font-bold">Ready to Update Model</h4>
                              <p className="text-text-secondary-light dark:text-text-secondary-dark">
                                  {m.pending_count > 0 ? (
                                    <>There are <span className="text-text-primary-light dark:text-text-primary-dark font-semibold">{m.pending_count} new face enrollments</span> pending. Initiating training will incorporate these new profiles.</>
                                  ) : (
                                    "All registered students are included in the current model. You can retrain to improve accuracy."
                                  )}
                              </p>
                          </div>
                          <button 
                            onClick={handleTrain}
                            disabled={isTraining}
                            className="group relative flex items-center justify-center gap-3 w-full max-w-xs h-14 bg-gradient-to-r from-primary to-primary-dark text-white rounded-[14px] font-extrabold text-[15px] shadow-glow hover:shadow-[0_0_25px_rgba(99,102,241,0.6)] transition-all transform hover:-translate-y-0.5 active:scale-[0.98]"
                          >
                              {isTraining ? (
                                <span className="animate-spin inline-block w-5 h-5 border-2 border-white/30 border-t-white rounded-full"></span>
                              ) : (
                                <span className="material-symbols-outlined text-[22px]">play_circle</span>
                              )}
                              <span>{isTraining ? 'Training...' : 'Train / Update Model'}</span>
                          </button>
                      </div>
                  </div>
              </div>

              {/* Activity area */}
              <div className="flex flex-col gap-6">
                 <div className="flex flex-col rounded-[20px] bg-surface-light border border-border-light/60 shadow-sm backdrop-blur-md p-6 dark:bg-surface-dark/90 dark:border-border-dark/60 transition-shadow hover:shadow-md">
                      <div className="flex items-center justify-between mb-4">
                          <h3 className="text-text-primary-light dark:text-text-primary-dark text-[16px] font-extrabold">Model Accuracy</h3>
                      </div>
                      <div className="flex items-end gap-2 mb-2">
                          <span className="text-[36px] font-black text-text-primary-light dark:text-text-primary-dark tracking-tight leading-none">98.5%</span>
                      </div>
                      <div className="w-full bg-slate-100 dark:bg-slate-700 rounded-full h-2.5 mb-2 overflow-hidden">
                          <div className="bg-gradient-to-r from-primary to-primary-dark h-full rounded-full" style={{width: '98.5%'}}></div>
                      </div>
                  </div>
                  
                  <div className="flex flex-col rounded-[20px] bg-surface-light border border-border-light/60 shadow-sm backdrop-blur-md p-6 flex-1 dark:bg-surface-dark/90 dark:border-border-dark/60 transition-shadow hover:shadow-md">
                      <h3 className="text-text-primary-light dark:text-text-primary-dark text-[16px] font-extrabold mb-5">Recent Activity</h3>
                      <div className="relative pl-4 border-l border-border-light dark:border-border-dark space-y-6">
                          {data.activity.length > 0 ? (
                            data.activity.map((log, i) => (
                              <div key={i} className="relative">
                                  <div className="absolute -left-[21px] top-1 size-2.5 rounded-full bg-border-light dark:bg-border-dark ring-4 ring-surface-light dark:ring-surface-dark"></div>
                                  <p className="text-xs text-text-secondary-light dark:text-text-secondary-dark mb-0.5">{log.time}</p>
                                  <p className="text-text-primary-light dark:text-text-primary-dark text-sm font-medium">{log.message}</p>
                              </div>
                            ))
                          ) : (
                            <p className="text-sm text-text-secondary-light">No recent model activity.</p>
                          )}
                      </div>
                  </div>
              </div>

          </div>

          {/* Pending Students */}
          <div className="flex flex-col gap-4">
              <div className="flex justify-between items-center">
                  <h3 className="text-text-primary-light dark:text-text-primary-dark text-xl font-bold">New Faces Pending Training</h3>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-4">
                  {m.pending_students && m.pending_students.map((student, i) => (
                    <div key={i} className="group relative rounded-lg overflow-hidden aspect-square bg-background-light dark:bg-background-dark border border-border-light dark:border-border-dark">
                        <div className="absolute inset-0 bg-primary/10 flex items-center justify-center">
                            <span className="text-3xl font-bold text-primary/60">{student.substring(0, 2).toUpperCase()}</span>
                        </div>
                        <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent opacity-100 flex flex-col justify-end p-3">
                            <p className="text-white text-xs font-bold truncate">{student}</p>
                            <p className="text-white/80 text-[10px] truncate">Pending</p>
                        </div>
                        <div className="absolute top-2 right-2 size-2 rounded-full bg-amber-400 shadow-sm border border-black/10 animate-pulse"></div>
                    </div>
                  ))}
                  
                  <a href="/register" className="flex flex-col items-center justify-center gap-2 rounded-lg aspect-square bg-surface-light dark:bg-surface-dark border-2 border-dashed border-border-light dark:border-border-dark hover:border-primary dark:hover:border-primary hover:bg-background-light dark:hover:bg-background-dark transition-colors group">
                      <div className="size-10 rounded-full bg-background-light dark:bg-background-dark flex items-center justify-center group-hover:bg-primary/10 transition-colors">
                          <span className="material-symbols-outlined text-text-secondary-light dark:text-text-secondary-dark group-hover:text-primary">add</span>
                      </div>
                      <span className="text-xs font-medium text-text-secondary-light dark:text-text-secondary-dark group-hover:text-primary">Add New</span>
                  </a>
              </div>
          </div>

      </div>
    </div>
  );
}
