import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { useToast } from '../context/ToastContext';

export default function Login() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  
  const { login } = useAuth();
  const { addToast } = useToast();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    
    // We send form data, the backend expects x-www-form-urlencoded
    const formData = { username, password };
    const res = await login(formData);
    
    setIsLoading(false);
    
    if (res.success) {
      addToast('Login successful', 'success');
      // The session should be authenticated now, check role
      navigate('/dashboard'); 
    } else {
      addToast(res.message || 'Invalid credentials', 'error');
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-background-light dark:bg-background-dark p-4 relative overflow-hidden font-sans transition-colors duration-300">
      {/* Background decorations */}
      <div className="absolute top-[-10%] sm:top-[-20%] right-[-10%] sm:right-[-10%] w-[300px] h-[300px] sm:w-[600px] sm:h-[600px] bg-primary/20 dark:bg-primary/10 rounded-full blur-[80px] pointer-events-none mix-blend-screen animate-pulse duration-[8s]"></div>
      <div className="hidden sm:block absolute bottom-[-10%] left-[-10%] w-[500px] h-[500px] bg-purple-500/10 dark:bg-purple-500/5 rounded-full blur-[100px] pointer-events-none mix-blend-screen animate-pulse duration-[10s] delay-1000"></div>

      <div className="w-full max-w-[420px] relative z-10 animate-[slideDown_0.6s_ease-out]">
        <div className="bg-surface-light/80 dark:bg-surface-dark/90 backdrop-blur-xl p-8 sm:p-10 rounded-[24px] shadow-2xl border border-border-light/60 dark:border-border-dark/60 transition-colors duration-300">
          
          <div className="text-center mb-8">
            <div className="size-[72px] mx-auto bg-gradient-to-br from-primary to-primary-dark text-white rounded-[20px] flex items-center justify-center shadow-lg shadow-primary/30 mb-6 group hover:scale-105 transition-transform duration-300 border border-white/20">
              <span className="material-symbols-outlined text-[36px] group-hover:rotate-12 transition-transform duration-300">lock_person</span>
            </div>
            <h1 className="text-3xl font-black text-text-primary-light dark:text-text-primary-dark tracking-tight mb-2">Welcome Back</h1>
            <p className="text-text-secondary-light dark:text-text-secondary-dark font-medium px-4">
              Enter your credentials or registration number to access the portal.
            </p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-5">
            <div className="relative group">
              <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                <span className="material-symbols-outlined text-text-secondary-light dark:text-text-secondary-dark text-[20px] group-focus-within:text-primary transition-colors">person</span>
              </div>
              <input 
                id="username" 
                name="username" 
                type="text" 
                required 
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="block w-full pl-12 pr-4 py-4 border border-border-light/60 dark:border-border-dark/60 rounded-xl leading-5 bg-background-light/50 dark:bg-background-dark/50 placeholder-text-secondary-light/70 dark:placeholder-text-secondary-dark/70 text-text-primary-light dark:text-text-primary-dark focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary transition-all shadow-inner font-medium" 
                placeholder="Username or Reg. No." 
              />
            </div>

            <div className="relative group">
              <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                <span className="material-symbols-outlined text-text-secondary-light dark:text-text-secondary-dark text-[20px] group-focus-within:text-primary transition-colors">key</span>
              </div>
              <input 
                id="password" 
                name="password" 
                type="password" 
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="block w-full pl-12 pr-4 py-4 border border-border-light/60 dark:border-border-dark/60 rounded-xl leading-5 bg-background-light/50 dark:bg-background-dark/50 placeholder-text-secondary-light/70 dark:placeholder-text-secondary-dark/70 text-text-primary-light dark:text-text-primary-dark focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary transition-all shadow-inner font-medium" 
                placeholder="Password (for Faculty)" 
              />
            </div>

            <div className="pt-2">
              <button 
                type="submit" 
                disabled={isLoading}
                className="group relative w-full flex justify-center items-center py-4 px-4 border border-transparent text-sm font-bold rounded-xl text-white bg-gradient-to-r from-primary to-primary-dark hover:from-primary-dark hover:to-[rgba(79,70,229,0.9)] focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary shadow-glow hover:shadow-[0_0_25px_rgba(99,102,241,0.6)] transform transition-all hover:-translate-y-0.5 active:scale-95 disabled:opacity-70 disabled:hover:-translate-y-0 disabled:active:scale-100"
              >
                {isLoading ? (
                  <span className="animate-spin inline-block w-5 h-5 border-2 border-white/30 border-t-white rounded-full"></span>
                ) : (
                  <>
                    <span className="absolute left-0 inset-y-0 flex items-center pl-4 opacity-50 group-hover:opacity-100 transition-opacity">
                      <span className="material-symbols-outlined text-[20px]">login</span>
                    </span>
                    Sign In to Portal
                  </>
                )}
              </button>
            </div>
          </form>
          
          <div className="mt-8 pt-6 border-t border-border-light/50 dark:border-border-dark/50">
            <div className="flex flex-col gap-3">
              <div className="bg-blue-50/50 dark:bg-blue-900/10 rounded-lg p-3 border border-blue-100 dark:border-blue-800 flex items-start gap-3">
                <span className="material-symbols-outlined text-academic-blue text-[18px] shrink-0 mt-0.5">info</span>
                <div className="text-xs text-text-secondary-light dark:text-text-secondary-dark">
                  <span className="font-bold text-text-primary-light dark:text-text-primary-dark block mb-1.5">Faculty Login Credentials:</span>
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <code className="bg-background-light dark:bg-background-dark px-1.5 py-0.5 rounded text-primary font-bold">sharma</code>
                      <span>/</span>
                      <code className="bg-background-light dark:bg-background-dark px-1.5 py-0.5 rounded text-primary font-bold">1234</code>
                      <span className="text-text-secondary-light dark:text-text-secondary-dark">— Prof. Rajesh Sharma (HOD)</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <code className="bg-background-light dark:bg-background-dark px-1.5 py-0.5 rounded text-primary font-bold">verma</code>
                      <span>/</span>
                      <code className="bg-background-light dark:bg-background-dark px-1.5 py-0.5 rounded text-primary font-bold">1234</code>
                      <span className="text-text-secondary-light dark:text-text-secondary-dark">— Prof. Neha Verma</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <code className="bg-background-light dark:bg-background-dark px-1.5 py-0.5 rounded text-primary font-bold">gupta</code>
                      <span>/</span>
                      <code className="bg-background-light dark:bg-background-dark px-1.5 py-0.5 rounded text-primary font-bold">1234</code>
                      <span className="text-text-secondary-light dark:text-text-secondary-dark">— Prof. Amit Gupta</span>
                    </div>
                  </div>
                  <div className="mt-2 pt-1.5 border-t border-blue-100 dark:border-blue-800/50">
                    Students: Login using your <span className="font-semibold text-text-primary-light dark:text-text-primary-dark">Registration Number</span> (No password).
                  </div>
                </div>
              </div>
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}
