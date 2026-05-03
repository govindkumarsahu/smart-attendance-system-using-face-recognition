import { NavLink } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export default function Sidebar({ isCollapsed, onMobileClose }) {
  const { user, logout } = useAuth();
  const isAdmin = user?.role === 'faculty';

  const menuItems = [
    { name: 'Dashboard', icon: 'dashboard', path: '/dashboard', show: true },
    {
      name: 'Register Student',
      icon: 'person_add',
      path: '/register',
      show: isAdmin,
    },
    {
      name: 'Start Attendance',
      icon: 'document_scanner',
      path: '/attendance',
      show: isAdmin,
    },
    {
      name: 'Registered Students',
      icon: 'group',
      path: '/students',
      show: isAdmin,
    },
    { name: 'Reports', icon: 'analytics', path: '/reports', show: isAdmin },
    { name: 'Model Data', icon: 'model_training', path: '/model-data', show: isAdmin },
    {
      name: 'Faculty Management',
      icon: 'supervised_user_circle',
      path: '/faculty-registration',
      show: isAdmin,
    },
    {
      name: 'Faculty Attendance',
      icon: 'badge',
      path: '/faculty-attendance',
      show: isAdmin,
    },
  ];

  return (
    <aside
      className={`
        bg-surface-light dark:bg-surface-dark border-r border-border-light dark:border-border-dark flex-col transition-all duration-300
        ${isCollapsed ? 'w-[80px] px-2 hidden lg:flex' : 'w-[280px] px-4 flex absolute h-full z-50 lg:relative'}
      `}
    >
      {/* Brand area */}
      <div className="h-[76px] flex items-center mb-6 mt-2 relative">
        <div className="size-12 rounded-xl bg-gradient-to-br from-primary to-primary-dark text-white flex items-center justify-center shrink-0 shadow-lg shadow-primary/30 ml-1">
          <span className="material-symbols-outlined text-[28px] animate-pulse">
            center_focus_weak
          </span>
        </div>
        <div
          className={`flex flex-col ml-3 transition-opacity duration-300 ${
            isCollapsed ? 'hidden' : 'block'
          }`}
        >
          <span className="text-[18px] font-black tracking-tight text-text-primary-light dark:text-text-primary-dark leading-tight mt-1">
            Smart
          </span>
          <span className="text-[14px] font-bold text-text-secondary-light dark:text-text-secondary-dark tracking-wide">
            Attendance
          </span>
        </div>

        {/* Mobile close button */}
        {!isCollapsed && (
          <button
            onClick={onMobileClose}
            className="lg:hidden absolute right-0 top-1/2 -translate-y-1/2 p-2 text-text-secondary-light dark:text-text-secondary-dark"
          >
            <span className="material-symbols-outlined text-2xl">close</span>
          </button>
        )}
      </div>

      <p
        className={`text-xs font-bold uppercase tracking-wider text-text-secondary-light/70 dark:text-text-secondary-dark/70 mb-3 px-3 transition-opacity duration-300 ${
          isCollapsed ? 'hidden' : 'block'
        }`}
      >
        Main Menu
      </p>

      {/* Nav Links */}
      <nav className="flex-1 space-y-2">
        {menuItems
          .filter((item) => item.show)
          .map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) => `
              group relative flex items-center justify-between p-3 rounded-xl transition-all duration-300
              ${
                isActive
                  ? 'bg-primary/10 text-primary dark:bg-primary/20 dark:text-primary-light font-bold shadow-sm'
                  : 'text-text-secondary-light dark:text-text-secondary-dark hover:bg-background-light dark:hover:bg-background-dark hover:text-text-primary-light dark:hover:text-text-primary-dark font-medium'
              }
            `}
            >
              {({ isActive }) => (
                <div className="flex items-center">
                  <div
                    className={`shrink-0 flex items-center justify-center transition-transform duration-300 ${
                      isActive ? 'scale-110' : 'group-hover:scale-110'
                    }`}
                  >
                    <span className="material-symbols-outlined text-[24px]">
                      {item.icon}
                    </span>
                  </div>
                  <span
                    className={`ml-4 text-[15px] tracking-wide transition-opacity duration-300 ${
                      isCollapsed ? 'hidden' : 'block'
                    }`}
                  >
                    {item.name}
                  </span>
                </div>
              )}
            </NavLink>
          ))}
      </nav>

      {/* User Area */}
      <div className="mt-auto mb-6">
        <div
          className={`h-px bg-gradient-to-r from-transparent via-border-light dark:via-border-dark to-transparent mb-4 transition-opacity duration-300 ${
            isCollapsed ? 'hidden' : 'block'
          }`}
        ></div>
        <button
          onClick={logout}
          className={`
            w-full flex items-center p-3 rounded-xl transition-all duration-300 text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20 font-bold
            ${isCollapsed ? 'justify-center' : ''}
          `}
        >
          <span className="material-symbols-outlined text-[24px]">logout</span>
          <span
            className={`ml-4 text-[15px] tracking-wide transition-opacity duration-300 ${
              isCollapsed ? 'hidden' : 'block'
            }`}
          >
            Logout
          </span>
        </button>
      </div>
    </aside>
  );
}
