import { useState } from 'react';
import { api } from '../services/api';
import { useToast } from '../context/ToastContext';

export default function StudentRegistration() {
  const [formData, setFormData] = useState({
    student_name: '',
    roll_number: '',
    department: '',
    academic_year: ''
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const { addToast } = useToast();

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);
    
    try {
      const response = await api.registerStudent(formData);
      
      if (response.success) {
        addToast(response.message, 'success');
        // Clear form
        setFormData({
          student_name: '',
          roll_number: '',
          department: '',
          academic_year: ''
        });
      } else {
        addToast(response.message || 'Registration failed', 'error');
      }
    } catch (error) {
      addToast('Error saving data. Make sure backend is running.', 'error');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="flex-1 flex justify-center py-4 sm:py-8 px-2 sm:px-6 lg:px-8">
      <div className="w-full max-w-[1000px] flex flex-col gap-6 sm:gap-8">
        
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
          <div className="flex flex-col gap-2">
            <h1 className="text-text-primary-light dark:text-text-primary-dark text-3xl md:text-4xl font-black leading-tight tracking-[-0.033em]">
              Student Registration
            </h1>
            <p className="text-text-secondary-light dark:text-text-secondary-dark text-base font-medium max-w-2xl">
              Add new students to the system. Fill out the details below to initiate the facial data capture process.
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          
          <div className="lg:col-span-7 flex flex-col gap-6">
            <div className="bg-surface-light dark:bg-surface-dark/90 rounded-[24px] shadow-sm border border-border-light/60 dark:border-border-dark/60 p-6 sm:p-8 backdrop-blur-md relative overflow-hidden transition-shadow hover:shadow-md">
              <div className="absolute top-0 right-0 w-64 h-64 bg-primary/5 rounded-full blur-3xl pointer-events-none transform translate-x-1/3 -translate-y-1/3"></div>
              
              <div className="flex items-center gap-3 mb-8 relative z-10">
                <div className="size-10 bg-primary/10 dark:bg-primary/20 text-primary dark:text-primary-light flex items-center justify-center rounded-[12px] border border-primary/20">
                  <span className="material-symbols-outlined text-[20px]">badge</span>
                </div>
                <div>
                  <h2 className="text-xl font-extrabold text-text-primary-light dark:text-text-primary-dark">Academic Details</h2>
                  <p className="text-xs font-bold text-text-secondary-light dark:text-text-secondary-dark mt-0.5">Enter student information</p>
                </div>
              </div>

              <form onSubmit={handleSubmit} className="space-y-6 relative z-10">
                
                <div className="space-y-4">
                  <div className="relative">
                    <label className="block text-sm font-bold text-text-primary-light dark:text-text-primary-dark mb-1.5 ml-1">
                      Full Name <span className="text-red-500">*</span>
                    </label>
                    <div className="relative group">
                      <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                        <span className="material-symbols-outlined text-text-secondary-light dark:text-text-secondary-dark text-[20px] group-focus-within:text-primary transition-colors">person</span>
                      </div>
                      <input 
                        type="text" 
                        name="student_name" 
                        value={formData.student_name}
                        onChange={handleInputChange}
                        required 
                        className="block w-full pl-10 pr-3 py-3 border border-border-light/80 dark:border-border-dark/80 rounded-xl bg-background-light/50 dark:bg-background-dark/50 text-text-primary-light dark:text-text-primary-dark placeholder-text-secondary-light/60 dark:placeholder-text-secondary-dark/60 focus:ring-2 focus:ring-primary/50 focus:border-primary transition-all font-medium shadow-inner text-sm" 
                        placeholder="e.g. John Doe"
                      />
                    </div>
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div className="relative">
                      <label className="block text-sm font-bold text-text-primary-light dark:text-text-primary-dark mb-1.5 ml-1">
                        Registration Number
                      </label>
                      <div className="relative group">
                        <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                          <span className="material-symbols-outlined text-text-secondary-light dark:text-text-secondary-dark text-[20px] group-focus-within:text-primary transition-colors">tag</span>
                        </div>
                        <input 
                          type="text" 
                          name="roll_number"
                          value={formData.roll_number}
                          onChange={handleInputChange} 
                          className="block w-full pl-10 pr-3 py-3 border border-border-light/80 dark:border-border-dark/80 rounded-xl bg-background-light/50 dark:bg-background-dark/50 text-text-primary-light dark:text-text-primary-dark placeholder-text-secondary-light/60 dark:placeholder-text-secondary-dark/60 focus:ring-2 focus:ring-primary/50 focus:border-primary transition-all font-medium shadow-inner text-sm uppercase" 
                          placeholder="e.g. CS2023001"
                        />
                      </div>
                    </div>

                    <div className="relative">
                      <label className="block text-sm font-bold text-text-primary-light dark:text-text-primary-dark mb-1.5 ml-1">
                        Academic Year
                      </label>
                      <div className="relative group">
                        <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                          <span className="material-symbols-outlined text-text-secondary-light dark:text-text-secondary-dark text-[20px] group-focus-within:text-primary transition-colors">calendar_month</span>
                        </div>
                        <select 
                          name="academic_year"
                          value={formData.academic_year}
                          onChange={handleInputChange} 
                          className="block w-full pl-10 pr-10 py-3 border border-border-light/80 dark:border-border-dark/80 rounded-xl bg-background-light/50 dark:bg-background-dark/50 text-text-primary-light dark:text-text-primary-dark focus:ring-2 focus:ring-primary/50 focus:border-primary transition-all font-medium shadow-inner text-sm appearance-none cursor-pointer form-select-arrow"
                        >
                          <option value="" style={{backgroundColor:'#0d1117', color:'#f1f5f9'}}>Select Year</option>
                          <option value="1st Year" style={{backgroundColor:'#0d1117', color:'#f1f5f9'}}>1st Year</option>
                          <option value="2nd Year" style={{backgroundColor:'#0d1117', color:'#f1f5f9'}}>2nd Year</option>
                          <option value="3rd Year" style={{backgroundColor:'#0d1117', color:'#f1f5f9'}}>3rd Year</option>
                          <option value="4th Year" style={{backgroundColor:'#0d1117', color:'#f1f5f9'}}>4th Year</option>
                        </select>
                      </div>
                    </div>
                  </div>

                  <div className="relative">
                    <label className="block text-sm font-bold text-text-primary-light dark:text-text-primary-dark mb-1.5 ml-1">
                      Department
                    </label>
                    <div className="relative group">
                      <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                        <span className="material-symbols-outlined text-text-secondary-light dark:text-text-secondary-dark text-[20px] group-focus-within:text-primary transition-colors">school</span>
                      </div>
                      <select 
                        name="department"
                        value={formData.department}
                        onChange={handleInputChange} 
                        className="block w-full pl-10 pr-10 py-3 border border-border-light/80 dark:border-border-dark/80 rounded-xl bg-background-light/50 dark:bg-background-dark/50 text-text-primary-light dark:text-text-primary-dark focus:ring-2 focus:ring-primary/50 focus:border-primary transition-all font-medium shadow-inner text-sm appearance-none cursor-pointer form-select-arrow"
                      >
                        <option value="" style={{backgroundColor:'#0d1117', color:'#f1f5f9'}}>Select Department</option>
                        <option value="Computer Science & Engineering" style={{backgroundColor:'#0d1117', color:'#f1f5f9'}}>Computer Science & Engineering (CSE)</option>
                        <option value="Computer Science & Engineering (AI)" style={{backgroundColor:'#0d1117', color:'#f1f5f9'}}>Computer Science & Engineering (AI)</option>
                        <option value="Electronics and Communication Engineering" style={{backgroundColor:'#0d1117', color:'#f1f5f9'}}>Electronics and Communication Engg. (ECE)</option>
                      </select>
                    </div>
                  </div>
                </div>

                <div className="pt-2 border-t border-border-light/50 dark:border-border-dark/50 mt-8">
                  <button 
                    type="submit" 
                    disabled={isSubmitting}
                    className="w-full flex justify-center items-center gap-2 py-3.5 px-4 rounded-xl text-white font-bold bg-gradient-to-r from-primary to-primary-dark hover:from-primary-dark hover:to-[rgba(79,70,229,0.9)] focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary/50 shadow-glow hover:shadow-[0_0_25px_rgba(99,102,241,0.6)] transition-all transform hover:-translate-y-0.5 active:scale-[0.98] disabled:opacity-70 text-[15px]"
                  >
                    {isSubmitting ? (
                      <span className="animate-spin inline-block w-5 h-5 border-2 border-white/30 border-t-white rounded-full"></span>
                    ) : (
                      <>
                        <span className="material-symbols-outlined text-[22px]">photo_camera</span>
                        Open Camera & Capture
                      </>
                    )}
                  </button>
                </div>
              </form>
            </div>
          </div>

          <div className="lg:col-span-5 flex flex-col gap-6">
            <div className="bg-surface-light dark:bg-surface-dark/90 rounded-[24px] shadow-sm border border-border-light/60 dark:border-border-dark/60 p-6 backdrop-blur-md transition-shadow hover:shadow-md">
              <div className="flex items-center gap-2 mb-4">
                <span className="material-symbols-outlined text-amber-500 text-[20px]">lightbulb</span>
                <h3 className="font-extrabold text-text-primary-light dark:text-text-primary-dark text-[15px]">Registration Guide</h3>
              </div>
              
              <ul className="space-y-4 relative before:absolute before:inset-y-2 before:left-[11px] before:w-px before:bg-border-light dark:before:bg-border-dark">
                <li className="relative pl-8">
                  <span className="absolute left-0 top-1 size-[22px] rounded-full bg-blue-50 dark:bg-blue-900/30 border-2 border-surface-light dark:border-surface-dark ring-1 ring-blue-200 dark:ring-blue-800 flex items-center justify-center text-[10px] font-black text-blue-600 dark:text-blue-400 z-10">1</span>
                  <p className="text-sm font-bold text-text-primary-light dark:text-text-primary-dark leading-tight">Fill details</p>
                  <p className="text-xs font-medium text-text-secondary-light dark:text-text-secondary-dark mt-1">Ensure the name matches official records.</p>
                </li>
                <li className="relative pl-8">
                  <span className="absolute left-0 top-1 size-[22px] rounded-full bg-blue-50 dark:bg-blue-900/30 border-2 border-surface-light dark:border-surface-dark ring-1 ring-blue-200 dark:ring-blue-800 flex items-center justify-center text-[10px] font-black text-blue-600 dark:text-blue-400 z-10">2</span>
                  <p className="text-sm font-bold text-text-primary-light dark:text-text-primary-dark leading-tight">Look at camera</p>
                  <p className="text-xs font-medium text-text-secondary-light dark:text-text-secondary-dark mt-1">A window will open. Look directly into the lens.</p>
                </li>
                <li className="relative pl-8">
                  <span className="absolute left-0 top-1 size-[22px] rounded-full bg-blue-50 dark:bg-blue-900/30 border-2 border-surface-light dark:border-surface-dark ring-1 ring-blue-200 dark:ring-blue-800 flex items-center justify-center text-[10px] font-black text-blue-600 dark:text-blue-400 z-10">3</span>
                  <p className="text-sm font-bold text-text-primary-light dark:text-text-primary-dark leading-tight">Move slightly</p>
                  <p className="text-xs font-medium text-text-secondary-light dark:text-text-secondary-dark mt-1">Turn your head slowly to capture different angles.</p>
                </li>
                <li className="relative pl-8">
                  <span className="absolute left-[3px] top-1 size-[16px] rounded-full bg-green-500 border-2 border-surface-light dark:border-surface-dark z-10 shadow-sm"></span>
                  <p className="text-sm font-bold text-text-primary-light dark:text-text-primary-dark leading-tight mt-0.5">Train Model</p>
                  <p className="text-xs font-medium text-text-secondary-light dark:text-text-secondary-dark mt-1">Don't forget to retrain the model after adding new students.</p>
                </li>
              </ul>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}
