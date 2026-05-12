import { useState, useEffect, useMemo } from 'react';
import { api } from '../services/api';
import Modal from '../components/Modal';
import { useToast } from '../context/ToastContext';

export default function FacultyRegistration() {
  const [faculty, setFaculty] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const { addToast } = useToast();

  const [formData, setFormData] = useState({
    full_name: '', username: '', password: '', employee_id: '',
    department: '', designation: 'Assistant Professor', email: '', phone: ''
  });

  const [editModalOpen, setEditModalOpen] = useState(false);
  const [facultyToEdit, setFacultyToEdit] = useState(null);
  const [deleteModalOpen, setDeleteModalOpen] = useState(false);
  const [facultyToDelete, setFacultyToDelete] = useState(null);

  const fetchFaculty = async () => {
    setIsLoading(true);
    try {
      const data = await api.getFaculty();
      setFaculty(Array.isArray(data) ? data : []);
    } catch (e) { addToast('Failed to fetch faculty', 'error'); }
    finally { setIsLoading(false); }
  };

  useEffect(() => { fetchFaculty(); }, []);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);
    try {
      const res = await api.registerFaculty(formData);
      if (res.success) {
        addToast(res.message, 'success');
        setFormData({ full_name: '', username: '', password: '', employee_id: '', department: '', designation: 'Assistant Professor', email: '', phone: '' });
        fetchFaculty();
      } else { addToast(res.message || 'Registration failed', 'error'); }
    } catch (e) { addToast('Error registering faculty.', 'error'); }
    finally { setIsSubmitting(false); }
  };

  const handleEditSubmit = async (e) => {
    e.preventDefault();
    try {
      const res = await api.updateFaculty(facultyToEdit.id, facultyToEdit);
      if (res.success) { addToast(res.message, 'success'); setEditModalOpen(false); fetchFaculty(); }
      else { addToast(res.message || 'Update failed', 'error'); }
    } catch (e) { addToast('Error updating faculty', 'error'); }
  };

  const handleDelete = async () => {
    try {
      const res = await api.deleteFaculty(facultyToDelete.id);
      if (res.success) { addToast(res.message, 'success'); setDeleteModalOpen(false); fetchFaculty(); }
      else { addToast(res.message || 'Delete failed', 'error'); }
    } catch (e) { addToast('Error deleting faculty', 'error'); }
  };

  const filtered = useMemo(() => faculty.filter(f =>
    f.full_name?.toLowerCase().includes(searchQuery.toLowerCase()) ||
    f.employee_id?.toLowerCase().includes(searchQuery.toLowerCase()) ||
    f.username?.toLowerCase().includes(searchQuery.toLowerCase())
  ), [faculty, searchQuery]);

  const inputCls = "block w-full pl-10 pr-3 py-3 border border-border-light/80 dark:border-border-dark/80 rounded-xl bg-background-light/50 dark:bg-background-dark/50 text-text-primary-light dark:text-text-primary-dark placeholder-text-secondary-light/60 dark:placeholder-text-secondary-dark/60 focus:ring-2 focus:ring-primary/50 focus:border-primary transition-all font-medium shadow-inner text-sm";
  const selectCls = inputCls + " appearance-none cursor-pointer form-select-arrow";

  return (
    <div className="flex-1 flex justify-center py-4 sm:py-8 px-2 sm:px-6 lg:px-8">
      <div className="w-full max-w-[1400px] flex flex-col gap-6 sm:gap-8">

        <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
          <div className="flex flex-col gap-2">
            <h1 className="text-text-primary-light dark:text-text-primary-dark text-3xl md:text-4xl font-black leading-tight tracking-[-0.033em]">
              Faculty Management
            </h1>
            <p className="text-text-secondary-light dark:text-text-secondary-dark text-base font-medium max-w-2xl">
              Register new faculty members and manage existing staff profiles with Employee IDs.
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* Registration Form */}
          <div className="lg:col-span-5 flex flex-col gap-6">
            <div className="bg-surface-light dark:bg-surface-dark/90 rounded-[24px] shadow-sm border border-border-light/60 dark:border-border-dark/60 p-6 sm:p-8 backdrop-blur-md relative overflow-hidden transition-shadow hover:shadow-md">
              <div className="absolute top-0 right-0 w-64 h-64 bg-primary/5 rounded-full blur-3xl pointer-events-none transform translate-x-1/3 -translate-y-1/3"></div>

              <div className="flex items-center gap-3 mb-8 relative z-10">
                <div className="size-10 bg-primary/10 dark:bg-primary/20 text-primary dark:text-primary-light flex items-center justify-center rounded-[12px] border border-primary/20">
                  <span className="material-symbols-outlined text-[20px]">person_add</span>
                </div>
                <div>
                  <h2 className="text-xl font-extrabold text-text-primary-light dark:text-text-primary-dark">Register Faculty</h2>
                  <p className="text-xs font-bold text-text-secondary-light dark:text-text-secondary-dark mt-0.5">Add new staff member</p>
                </div>
              </div>

              <form onSubmit={handleSubmit} className="space-y-5 relative z-10">
                {/* Full Name */}
                <div>
                  <label className="block text-sm font-bold text-text-primary-light dark:text-text-primary-dark mb-1.5 ml-1">Full Name <span className="text-red-500">*</span></label>
                  <div className="relative group">
                    <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none"><span className="material-symbols-outlined text-text-secondary-light dark:text-text-secondary-dark text-[20px] group-focus-within:text-primary transition-colors">person</span></div>
                    <input type="text" name="full_name" value={formData.full_name} onChange={handleInputChange} required className={inputCls} placeholder="e.g. Prof. Rajesh Kumar" />
                  </div>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  {/* Username */}
                  <div>
                    <label className="block text-sm font-bold text-text-primary-light dark:text-text-primary-dark mb-1.5 ml-1">Username <span className="text-red-500">*</span></label>
                    <div className="relative group">
                      <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none"><span className="material-symbols-outlined text-text-secondary-light dark:text-text-secondary-dark text-[20px] group-focus-within:text-primary transition-colors">account_circle</span></div>
                      <input type="text" name="username" value={formData.username} onChange={handleInputChange} required className={inputCls} placeholder="e.g. rajesh" />
                    </div>
                  </div>
                  {/* Password */}
                  <div>
                    <label className="block text-sm font-bold text-text-primary-light dark:text-text-primary-dark mb-1.5 ml-1">Password <span className="text-red-500">*</span></label>
                    <div className="relative group">
                      <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none"><span className="material-symbols-outlined text-text-secondary-light dark:text-text-secondary-dark text-[20px] group-focus-within:text-primary transition-colors">lock</span></div>
                      <input type="password" name="password" value={formData.password} onChange={handleInputChange} required className={inputCls} placeholder="••••••••" />
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  {/* Employee ID */}
                  <div>
                    <label className="block text-sm font-bold text-text-primary-light dark:text-text-primary-dark mb-1.5 ml-1">Employee ID</label>
                    <div className="relative group">
                      <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none"><span className="material-symbols-outlined text-text-secondary-light dark:text-text-secondary-dark text-[20px] group-focus-within:text-primary transition-colors">badge</span></div>
                      <input type="text" name="employee_id" value={formData.employee_id} onChange={handleInputChange} className={inputCls + " uppercase"} placeholder="e.g. EMP004" />
                    </div>
                  </div>
                  {/* Department */}
                  <div>
                    <label className="block text-sm font-bold text-text-primary-light dark:text-text-primary-dark mb-1.5 ml-1">Department</label>
                    <div className="relative group">
                      <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none"><span className="material-symbols-outlined text-text-secondary-light dark:text-text-secondary-dark text-[20px] group-focus-within:text-primary transition-colors">school</span></div>
                      <select name="department" value={formData.department} onChange={handleInputChange} className={selectCls}>
                        <option value="" style={{backgroundColor:'#0d1117', color:'#f1f5f9'}}>Select Dept</option>
                        <option value="CSE" style={{backgroundColor:'#0d1117', color:'#f1f5f9'}}>CSE</option>
                        <option value="CSE-AI" style={{backgroundColor:'#0d1117', color:'#f1f5f9'}}>CSE (AI)</option>
                        <option value="ECE" style={{backgroundColor:'#0d1117', color:'#f1f5f9'}}>ECE</option>
                      </select>
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  {/* Designation */}
                  <div>
                    <label className="block text-sm font-bold text-text-primary-light dark:text-text-primary-dark mb-1.5 ml-1">Designation</label>
                    <div className="relative group">
                      <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none"><span className="material-symbols-outlined text-text-secondary-light dark:text-text-secondary-dark text-[20px] group-focus-within:text-primary transition-colors">work</span></div>
                      <select name="designation" value={formData.designation} onChange={handleInputChange} className={selectCls}>
                        <option value="Assistant Professor" style={{backgroundColor:'#0d1117', color:'#f1f5f9'}}>Assistant Professor</option>
                        <option value="Associate Professor" style={{backgroundColor:'#0d1117', color:'#f1f5f9'}}>Associate Professor</option>
                        <option value="Professor" style={{backgroundColor:'#0d1117', color:'#f1f5f9'}}>Professor</option>
                        <option value="HOD" style={{backgroundColor:'#0d1117', color:'#f1f5f9'}}>HOD</option>
                        <option value="Lab Assistant" style={{backgroundColor:'#0d1117', color:'#f1f5f9'}}>Lab Assistant</option>
                      </select>
                    </div>
                  </div>
                  {/* Phone */}
                  <div>
                    <label className="block text-sm font-bold text-text-primary-light dark:text-text-primary-dark mb-1.5 ml-1">Phone</label>
                    <div className="relative group">
                      <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none"><span className="material-symbols-outlined text-text-secondary-light dark:text-text-secondary-dark text-[20px] group-focus-within:text-primary transition-colors">call</span></div>
                      <input type="text" name="phone" value={formData.phone} onChange={handleInputChange} className={inputCls} placeholder="e.g. 9876543210" />
                    </div>
                  </div>
                </div>

                {/* Email */}
                <div>
                  <label className="block text-sm font-bold text-text-primary-light dark:text-text-primary-dark mb-1.5 ml-1">Email</label>
                  <div className="relative group">
                    <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none"><span className="material-symbols-outlined text-text-secondary-light dark:text-text-secondary-dark text-[20px] group-focus-within:text-primary transition-colors">mail</span></div>
                    <input type="email" name="email" value={formData.email} onChange={handleInputChange} className={inputCls} placeholder="e.g. faculty@college.edu" />
                  </div>
                </div>

                <div className="pt-2 border-t border-border-light/50 dark:border-border-dark/50 mt-6">
                  <button type="submit" disabled={isSubmitting}
                    className="w-full flex justify-center items-center gap-2 py-3.5 px-4 rounded-xl text-white font-bold bg-gradient-to-r from-primary to-primary-dark hover:from-primary-dark hover:to-[rgba(79,70,229,0.9)] focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary/50 shadow-glow hover:shadow-[0_0_25px_rgba(99,102,241,0.6)] transition-all transform hover:-translate-y-0.5 active:scale-[0.98] disabled:opacity-70 text-[15px]">
                    {isSubmitting ? <span className="animate-spin inline-block w-5 h-5 border-2 border-white/30 border-t-white rounded-full"></span> : <><span className="material-symbols-outlined text-[22px]">how_to_reg</span>Register Faculty</>}
                  </button>
                </div>
              </form>
            </div>
          </div>

          {/* Faculty List */}
          <div className="lg:col-span-7 flex flex-col gap-4">
            <div className="bg-surface-light dark:bg-surface-dark/90 p-5 rounded-[20px] shadow-sm border border-border-light/60 dark:border-border-dark/60 backdrop-blur-md flex flex-wrap gap-4 items-center">
              <div className="relative flex-1 min-w-[200px]">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none"><span className="material-symbols-outlined text-text-secondary-light">search</span></div>
                <input type="text" placeholder="Search by name, ID, or username..." value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-10 pr-4 py-2.5 bg-background-light dark:bg-background-dark border border-border-light dark:border-border-dark rounded-xl focus:ring-2 focus:ring-primary/50 outline-none text-sm font-medium" />
              </div>
              <div className="text-sm font-bold text-text-secondary-light">{filtered.length} faculty members</div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {isLoading ? (
                <div className="col-span-full py-12 flex justify-center"><span className="animate-spin inline-block w-8 h-8 border-4 border-primary/30 border-t-primary rounded-full"></span></div>
              ) : filtered.length > 0 ? filtered.map(f => (
                <div key={f.id} className="group bg-surface-light dark:bg-surface-dark/90 rounded-[20px] border border-border-light/60 dark:border-border-dark/60 shadow-sm backdrop-blur-md transition-all duration-300 hover:-translate-y-1 hover:shadow-premium overflow-hidden flex flex-col">
                  <div className="h-20 bg-gradient-to-br from-indigo-100 to-indigo-200 dark:from-indigo-900/50 dark:to-indigo-800/30 relative">
                    <div className="absolute inset-0 opacity-10" style={{backgroundImage: 'radial-gradient(#6366f1 1px, transparent 1px)', backgroundSize: '10px 10px'}}></div>
                    <div className="absolute -bottom-7 left-5">
                      <div className="size-16 rounded-2xl bg-gradient-to-br from-primary to-primary-dark text-white border-4 border-surface-light dark:border-surface-dark shadow-md flex items-center justify-center text-xl font-black">
                        {f.full_name?.substring(0, 2).toUpperCase()}
                      </div>
                    </div>
                    <div className="absolute top-2.5 right-2.5 flex gap-1.5 opacity-0 group-hover:opacity-100 transition-opacity">
                      <button onClick={() => { setFacultyToEdit({...f}); setEditModalOpen(true); }} className="size-7 rounded-full bg-white dark:bg-slate-800 text-blue-500 shadow-sm hover:scale-110 transition-transform flex items-center justify-center" title="Edit"><span className="material-symbols-outlined text-[14px]">edit</span></button>
                      <button onClick={() => { setFacultyToDelete(f); setDeleteModalOpen(true); }} className="size-7 rounded-full bg-white dark:bg-slate-800 text-red-500 shadow-sm hover:scale-110 transition-transform flex items-center justify-center" title="Delete"><span className="material-symbols-outlined text-[14px]">delete</span></button>
                    </div>
                  </div>
                  <div className="p-5 pt-9 flex flex-col flex-1">
                    <h3 className="text-base font-bold text-text-primary-light dark:text-text-primary-dark truncate">{f.full_name}</h3>
                    <p className="text-xs font-bold text-primary mb-3">{f.employee_id || 'No EMP ID'} • {f.designation}</p>
                    <div className="space-y-1.5 mt-auto">
                      <div className="flex items-center text-xs text-text-secondary-light dark:text-text-secondary-dark"><span className="material-symbols-outlined text-[14px] mr-1.5">school</span>{f.department || 'N/A'}</div>
                      <div className="flex items-center text-xs text-text-secondary-light dark:text-text-secondary-dark"><span className="material-symbols-outlined text-[14px] mr-1.5">mail</span>{f.email || 'N/A'}</div>
                      <div className="flex items-center text-xs text-text-secondary-light dark:text-text-secondary-dark"><span className="material-symbols-outlined text-[14px] mr-1.5">call</span>{f.phone || 'N/A'}</div>
                    </div>
                  </div>
                </div>
              )) : (
                <div className="col-span-full py-16 text-center">
                  <span className="material-symbols-outlined text-5xl text-text-secondary-light/50 mb-3 block">group_off</span>
                  <h3 className="text-lg font-bold text-text-primary-light dark:text-text-primary-dark">No faculty found</h3>
                  <p className="text-text-secondary-light mt-1">Register a new faculty member using the form.</p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Edit Modal */}
        {facultyToEdit && (
          <Modal isOpen={editModalOpen} onClose={() => setEditModalOpen(false)} title={<div className="flex items-center gap-2"><span className="material-symbols-outlined text-blue-500">edit_square</span><span>Edit Faculty Profile</span></div>}>
            <form onSubmit={handleEditSubmit} className="space-y-4">
              <div><label className="block text-sm font-bold text-slate-700 dark:text-slate-300 mb-1">Full Name</label><input type="text" value={facultyToEdit.full_name || ''} onChange={e => setFacultyToEdit({...facultyToEdit, full_name: e.target.value})} className="w-full px-4 py-2 rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-900 focus:ring-2 focus:ring-primary outline-none" required /></div>
              <div className="grid grid-cols-2 gap-3">
                <div><label className="block text-sm font-bold text-slate-700 dark:text-slate-300 mb-1">Employee ID</label><input type="text" value={facultyToEdit.employee_id || ''} onChange={e => setFacultyToEdit({...facultyToEdit, employee_id: e.target.value})} className="w-full px-4 py-2 rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-900 focus:ring-2 focus:ring-primary outline-none" /></div>
                <div><label className="block text-sm font-bold text-slate-700 dark:text-slate-300 mb-1">Department</label><input type="text" value={facultyToEdit.department || ''} onChange={e => setFacultyToEdit({...facultyToEdit, department: e.target.value})} className="w-full px-4 py-2 rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-900 focus:ring-2 focus:ring-primary outline-none" /></div>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div><label className="block text-sm font-bold text-slate-700 dark:text-slate-300 mb-1">Designation</label><input type="text" value={facultyToEdit.designation || ''} onChange={e => setFacultyToEdit({...facultyToEdit, designation: e.target.value})} className="w-full px-4 py-2 rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-900 focus:ring-2 focus:ring-primary outline-none" /></div>
                <div><label className="block text-sm font-bold text-slate-700 dark:text-slate-300 mb-1">Phone</label><input type="text" value={facultyToEdit.phone || ''} onChange={e => setFacultyToEdit({...facultyToEdit, phone: e.target.value})} className="w-full px-4 py-2 rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-900 focus:ring-2 focus:ring-primary outline-none" /></div>
              </div>
              <div><label className="block text-sm font-bold text-slate-700 dark:text-slate-300 mb-1">Email</label><input type="email" value={facultyToEdit.email || ''} onChange={e => setFacultyToEdit({...facultyToEdit, email: e.target.value})} className="w-full px-4 py-2 rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-900 focus:ring-2 focus:ring-primary outline-none" /></div>
              <div className="pt-4 flex justify-end gap-3 border-t border-slate-100 dark:border-slate-700">
                <button type="button" onClick={() => setEditModalOpen(false)} className="px-4 py-2 rounded-lg font-bold text-slate-600 hover:bg-slate-100 dark:hover:bg-slate-800">Cancel</button>
                <button type="submit" className="px-4 py-2 rounded-lg font-bold bg-primary text-white hover:bg-primary-dark">Save Changes</button>
              </div>
            </form>
          </Modal>
        )}

        {/* Delete Modal */}
        {facultyToDelete && (
          <Modal isOpen={deleteModalOpen} onClose={() => setDeleteModalOpen(false)} title={<div className="flex items-center gap-2"><span className="material-symbols-outlined text-red-500">warning</span><span className="text-red-600 dark:text-red-400">Confirm Deletion</span></div>}>
            <div className="space-y-4">
              <div className="bg-red-50 dark:bg-red-900/20 p-4 rounded-lg flex gap-3 text-red-800 dark:text-red-200">
                <span className="material-symbols-outlined shrink-0 text-red-500">error</span>
                <p className="text-sm font-medium">Delete <strong>{facultyToDelete.full_name}</strong> ({facultyToDelete.employee_id || 'N/A'})? This will also remove all their attendance records.</p>
              </div>
              <div className="pt-4 flex justify-end gap-3 border-t border-slate-100 dark:border-slate-700">
                <button onClick={() => setDeleteModalOpen(false)} className="px-4 py-2 rounded-lg font-bold text-slate-600 hover:bg-slate-100 dark:hover:bg-slate-800">Cancel</button>
                <button onClick={handleDelete} className="px-4 py-2 rounded-lg font-bold bg-red-500 text-white hover:bg-red-600 flex items-center gap-2"><span className="material-symbols-outlined text-[18px]">delete_forever</span>Confirm Delete</button>
              </div>
            </div>
          </Modal>
        )}
      </div>
    </div>
  );
}
