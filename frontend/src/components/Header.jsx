import { useState } from 'react';
import { useTheme } from '../context/ThemeContext';
import { useAuth } from '../context/AuthContext';

export default function Header({ toggleSidebar }) {
  const { isDark, toggleTheme } = useTheme();
  const { user } = useAuth();
  const [notifOpen, setNotifOpen] = useState(false);

  return (
    <header className="h-[76px] bg-surface-light/80 dark:bg-surface-dark/90 backdrop-blur-md border-b border-border-light dark:border-border-dark flex items-center justify-between px-4 sm:px-6 z-20 sticky top-0 shadow-sm shrink-0">
      <div className="flex items-center">
        <button
          onClick={toggleSidebar}
          className="p-2.5 rounded-xl text-text-secondary-light dark:text-text-secondary-dark hover:bg-background-light dark:hover:bg-background-dark hover:text-primary transition-colors focus:outline-none focus:ring-2 focus:ring-primary/20 mr-2 sm:mr-4"
        >
          <span className="material-symbols-outlined text-[24px]">menu</span>
        </button>
        <div className="hidden sm:flex items-center bg-background-light dark:bg-background-dark px-4 py-2.5 rounded-xl border border-border-light/60 dark:border-border-dark/60 w-[300px] focus-within:ring-2 focus-within:ring-primary/20 focus-within:border-primary transition-all shadow-inner">
          <span className="material-symbols-outlined text-text-secondary-light dark:text-text-secondary-dark text-[20px]">
            search
          </span>
          <input
            className="bg-transparent border-none focus:outline-none ml-2 text-sm w-full text-text-primary-light dark:text-text-primary-dark placeholder-text-secondary-light dark:placeholder-text-secondary-dark"
            placeholder="Search students, logs, reports..."
            type="text"
          />
        </div>
      </div>

      <div className="flex items-center gap-2 sm:gap-4">
        {/* Dark Mode Toggle */}
        <button
          onClick={toggleTheme}
          className="p-2.5 rounded-xl text-text-secondary-light dark:text-text-secondary-dark hover:bg-background-light dark:hover:bg-background-dark hover:text-primary transition-all hover:-translate-y-0.5"
          title={isDark ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
        >
          <span className="material-symbols-outlined text-[24px]">
            {isDark ? 'light_mode' : 'dark_mode'}
          </span>
        </button>

        {/* Notifications */}
        <div className="relative">
          <button
            onClick={() => setNotifOpen(!notifOpen)}
            className="p-2.5 rounded-xl text-text-secondary-light dark:text-text-secondary-dark hover:bg-background-light dark:hover:bg-background-dark hover:text-primary transition-all relative hover:-translate-y-0.5"
          >
            <span className="material-symbols-outlined text-[24px]">notifications</span>
            <span className="absolute top-2 right-2 w-2 h-2 bg-red-500 rounded-full ring-2 ring-surface-light dark:ring-surface-dark"></span>
          </button>

          {/* Notification Dropdown */}
          <div
            className={`absolute right-0 mt-3 w-80 bg-surface-light dark:bg-surface-dark border border-border-light dark:border-border-dark rounded-2xl shadow-xl z-50 overflow-hidden transform transition-all duration-200 origin-top-right
              ${notifOpen ? 'opacity-100 scale-100' : 'opacity-0 scale-95 pointer-events-none'}`}
          >
            <div className="p-4 border-b border-border-light dark:border-border-dark flex justify-between items-center bg-background-light/50 dark:bg-background-dark/50">
              <h3 className="font-bold text-text-primary-light dark:text-text-primary-dark">
                Notifications
              </h3>
              <span className="text-xs bg-primary/10 text-primary px-2 py-1 rounded-md font-bold">
                1 New
              </span>
            </div>
            <div className="max-h-80 overflow-y-auto">
              <div className="flex items-start gap-4 p-4 hover:bg-background-light dark:hover:bg-background-dark cursor-pointer transition-colors border-b border-border-light/50 dark:border-border-dark/50">
                <div className="size-10 rounded-full bg-blue-50 dark:bg-blue-900/20 text-blue-500 flex items-center justify-center shrink-0 border border-blue-100 dark:border-blue-800">
                  <span className="material-symbols-outlined text-[20px]">info</span>
                </div>
                <div>
                  <p className="text-sm font-bold text-text-primary-light dark:text-text-primary-dark mb-0.5">
                    System Reactivated
                  </p>
                  <p className="text-xs text-text-secondary-light dark:text-text-secondary-dark">
                    Welcome to the new React interface!
                  </p>
                  <p className="text-[10px] text-text-secondary-light/70 dark:text-text-secondary-dark/70 mt-1 font-medium">
                    Just now
                  </p>
                </div>
              </div>
            </div>
            <div className="p-3 text-center border-t border-border-light dark:border-border-dark bg-background-light/50 dark:bg-background-dark/50 hover:bg-background-light dark:hover:bg-background-dark cursor-pointer transition-colors">
              <span className="text-sm font-bold text-primary">View all alerts</span>
            </div>
          </div>
        </div>

        <div className="h-8 w-px bg-border-light dark:bg-border-dark mx-1"></div>

        {/* User Profile */}
        <div className="flex items-center gap-3 pl-1">
          <div className="flex flex-col text-right hidden sm:flex">
            <span className="text-sm font-bold text-text-primary-light dark:text-text-primary-dark leading-tight">
              {user?.username || 'User'}
            </span>
            <span className="text-[11px] font-bold text-primary uppercase tracking-wider">
              {user?.role || 'Guest'}
            </span>
          </div>
          <div className="size-10 rounded-xl bg-gradient-to-br from-indigo-100 to-purple-100 dark:from-indigo-900/40 dark:to-purple-900/40 text-primary flex items-center justify-center border-2 border-primary/20 shadow-sm cursor-pointer hover:shadow-md transition-shadow">
            <span className="font-black text-sm">
              {user?.username ? user.username.substring(0, 2).toUpperCase() : 'ME'}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}
