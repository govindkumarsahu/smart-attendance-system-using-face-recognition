import { useState, useEffect, useMemo } from 'react';
import { api } from '../services/api';
import Modal from '../components/Modal';
import { useToast } from '../context/ToastContext';

export default function RegisteredStudents() {
  const [students, setStudents] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [departmentFilter, setDepartmentFilter] = useState('all');
  const [isLoading, setIsLoading] = useState(true);
  const { addToast } = useToast();

  const [editModalOpen, setEditModalOpen] = useState(false);
  const [studentToEdit, setStudentToEdit] = useState(null);
  
  const [deleteModalOpen, setDeleteModalOpen] = useState(false);
  const [studentToDelete, setStudentToDelete] = useState(null);

  const fetchStudents = async () => {
    setIsLoading(true);
    try {
      const data = await api.getStudents();
      setStudents(data || []);
    } catch (error) {
      addToast('Failed to fetch students', 'error');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchStudents();
  }, []);

  const filteredStudents = useMemo(() => {
    return students.filter(student => {
      const matchesSearch = student.name.toLowerCase().includes(searchQuery.toLowerCase()) || 
                            (student.roll_number && student.roll_number.toLowerCase().includes(searchQuery.toLowerCase()));
      const matchesDept = departmentFilter === 'all' || student.department === departmentFilter;
      return matchesSearch && matchesDept;
    });
  }, [students, searchQuery, departmentFilter]);

  const uniqueDepartments = useMemo(() => {
    const depts = new Set(students.map(s => s.department).filter(Boolean));
    return Array.from(depts);
  }, [students]);

  const handleEditSubmit = async (e) => {
    e.preventDefault();
    try {
      const res = await api.editStudent(studentToEdit.id, studentToEdit);
      if (res.success) {
        addToast(`Updated ${studentToEdit.name} successfully`, 'success');
        setEditModalOpen(false);
        fetchStudents();
      } else {
        addToast('Failed to update student', 'error');
      }
    } catch (error) {
      addToast('Error saving changes', 'error');
    }
  };

  const handleDeleteSubmit = async () => {
    try {
      const res = await api.deleteStudent(studentToDelete.id);
      if (res.success) {
        addToast(`Deleted ${studentToDelete.name} and their dataset`, 'success');
        setDeleteModalOpen(false);
        fetchStudents();
      } else {
        addToast('Failed to delete student', 'error');
      }
    } catch (error) {
      addToast('Error executing delete', 'error');
    }
  };

  return (
    <div className="flex-1 flex flex-col items-center py-8 px-4 sm:px-6 lg:px-8">
        <div className="w-full max-w-[1400px] flex flex-col gap-8">
            <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
                <div className="flex flex-col gap-2">
                    <h1 className="text-text-primary-light dark:text-text-primary-dark text-3xl md:text-4xl font-black leading-tight tracking-[-0.033em]">
                        Student Directory
                    </h1>
                    <p className="text-text-secondary-light dark:text-text-secondary-dark text-base font-medium max-w-2xl">
                        Manage registered students, view their datasets, and edit profile information.
                    </p>
                </div>
                <div className="flex gap-3">
                    <a href="/register" className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-primary to-primary-dark text-white rounded-xl font-bold transition-all duration-300 shadow-glow hover:shadow-[0_0_25px_rgba(99,102,241,0.6)] hover:-translate-y-0.5">
                        <span className="material-symbols-outlined text-[20px]">person_add</span>
                        <span>Add Student</span>
                    </a>
                </div>
            </div>

            {/* Grid display for students */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                
                {/* Search/Filter Card */}
                <div className="col-span-full mb-2 bg-surface-light dark:bg-surface-dark/90 p-5 rounded-[20px] shadow-sm border border-border-light/60 dark:border-border-dark/60 backdrop-blur-md flex flex-wrap gap-4 items-center">
                   <div className="relative flex-1 min-w-[200px]">
                       <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                           <span className="material-symbols-outlined text-text-secondary-light">search</span>
                       </div>
                       <input 
                          type="text" 
                          placeholder="Search by name or reg no..." 
                          value={searchQuery}
                          onChange={(e) => setSearchQuery(e.target.value)}
                          className="w-full pl-10 pr-4 py-2.5 bg-background-light dark:bg-background-dark border border-border-light dark:border-border-dark rounded-xl focus:ring-2 focus:ring-primary/50 outline-none text-sm font-medium"
                       />
                   </div>
                   <div className="relative min-w-[200px]">
                       <select 
                          value={departmentFilter}
                          onChange={(e) => setDepartmentFilter(e.target.value)}
                          className="w-full pl-4 pr-10 py-2.5 bg-background-light dark:bg-background-dark border border-border-light dark:border-border-dark rounded-xl focus:ring-2 focus:ring-primary/50 outline-none text-sm font-medium appearance-none form-select-arrow"
                       >
                           <option value="all" style={{backgroundColor:'#0d1117', color:'#f1f5f9'}}>All Departments</option>
                           {uniqueDepartments.map(dept => (
                               <option key={dept} value={dept} style={{backgroundColor:'#0d1117', color:'#f1f5f9'}}>{dept}</option>
                           ))}
                       </select>
                   </div>
                   <div className="text-sm font-bold text-text-secondary-light">
                       Showing {filteredStudents.length} of {students.length} students
                   </div>
                </div>

                {isLoading ? (
                  <div className="col-span-full py-12 flex justify-center">
                    <span className="animate-spin inline-block w-8 h-8 border-4 border-primary/30 border-t-primary rounded-full"></span>
                  </div>
                ) : filteredStudents.length > 0 ? (
                  filteredStudents.map(student => (
                    <div key={student.id} className="group bg-surface-light dark:bg-surface-dark/90 rounded-[20px] border border-border-light/60 dark:border-border-dark/60 shadow-sm backdrop-blur-md transition-all duration-300 hover:-translate-y-1 hover:shadow-premium overflow-hidden flex flex-col">
                        <div className="h-24 bg-gradient-to-br from-slate-100 to-slate-200 dark:from-slate-800 dark:to-slate-900 relative">
                             {/* Abstract pattern bg */}
                             <div className="absolute inset-0 opacity-10" style={{backgroundImage: 'radial-gradient(#6366f1 1px, transparent 1px)', backgroundSize: '10px 10px'}}></div>
                             
                             <div className="absolute -bottom-8 left-6">
                                 {student.has_profile_pic ? (
                                     <img src={`/student_image/${student.name}?t=${Date.now()}`} alt={student.name} className="size-20 rounded-2xl object-cover border-4 border-surface-light dark:border-surface-dark shadow-md bg-white"/>
                                 ) : (
                                     <div className="size-20 rounded-2xl bg-gradient-to-br from-primary to-primary-dark text-white border-4 border-surface-light dark:border-surface-dark shadow-md flex items-center justify-center text-2xl font-black">
                                         {student.name.substring(0,2).toUpperCase()}
                                     </div>
                                 )}
                             </div>
                             
                             <div className="absolute top-3 right-3 flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                                <button 
                                  onClick={() => { setStudentToEdit(student); setEditModalOpen(true); }}
                                  className="size-8 rounded-full bg-white dark:bg-slate-800 text-blue-500 shadow-sm hover:scale-110 transition-transform flex items-center justify-center"
                                  title="Edit Student"
                                >
                                    <span className="material-symbols-outlined text-[16px]">edit</span>
                                </button>
                                <button 
                                  onClick={() => { setStudentToDelete(student); setDeleteModalOpen(true); }}
                                  className="size-8 rounded-full bg-white dark:bg-slate-800 text-red-500 shadow-sm hover:scale-110 transition-transform flex items-center justify-center"
                                  title="Delete Student"
                                >
                                    <span className="material-symbols-outlined text-[16px]">delete</span>
                                </button>
                             </div>
                        </div>
                        
                        <div className="p-6 pt-10 flex flex-col flex-1">
                            <h3 className="text-lg font-bold text-text-primary-light dark:text-text-primary-dark truncate">{student.name}</h3>
                            <p className="text-xs font-bold text-primary mb-4">{student.roll_number || 'No Reg No.'}</p>
                            
                            <div className="space-y-2 mt-auto">
                               <div className="flex items-center text-xs text-text-secondary-light dark:text-text-secondary-dark">
                                   <span className="material-symbols-outlined text-[16px] mr-2">school</span>
                                   <span className="truncate">{student.department || 'N/A'}</span>
                               </div>
                               <div className="flex items-center text-xs text-text-secondary-light dark:text-text-secondary-dark">
                                   <span className="material-symbols-outlined text-[16px] mr-2">calendar_month</span>
                                   <span>{student.academic_year || 'N/A'}</span>
                               </div>
                               <div className="flex items-center text-xs text-text-secondary-light dark:text-text-secondary-dark">
                                   <span className="material-symbols-outlined text-[16px] mr-2">photo_library</span>
                                   <span>{student.images} dataset images</span>
                               </div>
                            </div>
                        </div>
                    </div>
                  ))
                ) : (
                  <div className="col-span-full py-20 text-center">
                      <span className="material-symbols-outlined text-5xl text-text-secondary-light/50 mb-3 block">person_search</span>
                      <h3 className="text-lg font-bold text-text-primary-light">No students found</h3>
                      <p className="text-text-secondary-light mt-1">Try adjusting your search criteria or register a new student.</p>
                  </div>
                )}
            </div>
        </div>

        {/* Edit Modal */}
        {studentToEdit && (
          <Modal 
            isOpen={editModalOpen} 
            onClose={() => setEditModalOpen(false)} 
            title={(
              <div className="flex items-center gap-2">
                <span className="material-symbols-outlined text-blue-500">edit_square</span>
                <span>Edit Student Profile</span>
              </div>
            )}
          >
            <form onSubmit={handleEditSubmit} className="space-y-4">
              <div>
                <label className="block text-sm font-bold text-slate-700 dark:text-slate-300 mb-1">Full Name</label>
                <input 
                  type="text" 
                  value={studentToEdit.name} 
                  onChange={(e) => setStudentToEdit({...studentToEdit, name: e.target.value})}
                  className="w-full px-4 py-2 rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-900 focus:ring-2 focus:ring-primary outline-none" 
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-bold text-slate-700 dark:text-slate-300 mb-1">Registration Number</label>
                <input 
                  type="text" 
                  value={studentToEdit.roll_number || ''} 
                  onChange={(e) => setStudentToEdit({...studentToEdit, roll_number: e.target.value})}
                  className="w-full px-4 py-2 rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-900 focus:ring-2 focus:ring-primary outline-none" 
                />
              </div>
              <div>
                <label className="block text-sm font-bold text-slate-700 dark:text-slate-300 mb-1">Department</label>
                <input 
                  type="text" 
                  value={studentToEdit.department || ''} 
                  onChange={(e) => setStudentToEdit({...studentToEdit, department: e.target.value})}
                  className="w-full px-4 py-2 rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-900 focus:ring-2 focus:ring-primary outline-none" 
                />
              </div>
              <div>
                <label className="block text-sm font-bold text-slate-700 dark:text-slate-300 mb-1">Academic Year</label>
                <input 
                  type="text" 
                  value={studentToEdit.academic_year || ''} 
                  onChange={(e) => setStudentToEdit({...studentToEdit, academic_year: e.target.value})}
                  className="w-full px-4 py-2 rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-900 focus:ring-2 focus:ring-primary outline-none" 
                />
              </div>
              <div className="pt-4 flex justify-end gap-3 border-t border-slate-100 dark:border-slate-700">
                <button type="button" onClick={() => setEditModalOpen(false)} className="px-4 py-2 rounded-lg font-bold text-slate-600 hover:bg-slate-100 dark:hover:bg-slate-800">Cancel</button>
                <button type="submit" className="px-4 py-2 rounded-lg font-bold bg-primary text-white hover:bg-primary-dark">Save Changes</button>
              </div>
            </form>
          </Modal>
        )}

        {/* Delete Confirmation Modal */}
        {studentToDelete && (
          <Modal 
            isOpen={deleteModalOpen} 
            onClose={() => setDeleteModalOpen(false)}
            title={(
              <div className="flex items-center gap-2">
                <span className="material-symbols-outlined text-red-500">warning</span>
                <span className="text-red-600 dark:text-red-400">Confirm Deletion</span>
              </div>
            )}
          >
            <div className="space-y-4">
              <div className="bg-red-50 dark:bg-red-900/20 p-4 rounded-lg flex gap-3 text-red-800 dark:text-red-200">
                <span className="material-symbols-outlined shrink-0 text-red-500">error</span>
                <p className="text-sm font-medium">You are about to permanently delete <strong>{studentToDelete.name}</strong> from the system.</p>
              </div>
              <p className="text-sm text-slate-600 dark:text-slate-400">This action will:
                <ul className="list-disc pl-5 mt-2 space-y-1">
                  <li>Remove the student profile from the database</li>
                  <li>Delete all their attendance history</li>
                  <li>Permanently erase their dataset of {studentToDelete.images} images</li>
                  <li>Invalidate the current training model</li>
                </ul>
              </p>
              <div className="pt-4 flex justify-end gap-3 border-t border-slate-100 dark:border-slate-700">
                <button type="button" onClick={() => setDeleteModalOpen(false)} className="px-4 py-2 rounded-lg font-bold text-slate-600 hover:bg-slate-100 dark:hover:bg-slate-800">Cancel</button>
                <button onClick={handleDeleteSubmit} className="px-4 py-2 rounded-lg font-bold bg-red-500 text-white hover:bg-red-600 flex items-center gap-2">
                  <span className="material-symbols-outlined text-[18px]">delete_forever</span> Confirm Delete
                </button>
              </div>
            </div>
          </Modal>
        )}

    </div>
  );
}
