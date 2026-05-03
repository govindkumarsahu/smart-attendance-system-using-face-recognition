/* ============================================
   Smart Attendance System - Custom Scripts
   ============================================ */

// ============================================
// DARK MODE TOGGLE
// ============================================

/**
 * Apply the saved theme (dark/light) on page load.
 * Uses localStorage key "theme". Falls back to system preference.
 */
(function applyThemeOnLoad() {
   const saved = localStorage.getItem('theme');
   const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
   if (saved === 'dark' || (!saved && prefersDark)) {
      document.documentElement.classList.add('dark');
   } else {
      document.documentElement.classList.remove('dark');
   }
})();

/**
 * Toggle between dark and light mode.
 * Persists choice to localStorage and updates the toggle button icon.
 */
function toggleDarkMode() {
   const html = document.documentElement;
   const isDark = html.classList.toggle('dark');
   localStorage.setItem('theme', isDark ? 'dark' : 'light');
   updateDarkModeIcon(isDark);
}

/**
 * Update the icon on the dark mode button.
 * @param {boolean} isDark - current dark mode state
 */
function updateDarkModeIcon(isDark) {
   const btn = document.getElementById('dark-mode-btn');
   if (!btn) return;
   const icon = btn.querySelector('.material-symbols-outlined');
   if (!icon) return;
   icon.textContent = isDark ? 'light_mode' : 'dark_mode';
   btn.title = isDark ? 'Switch to Light Mode' : 'Switch to Dark Mode';
}

// Wire the button and sync icon once DOM is ready
document.addEventListener('DOMContentLoaded', function () {
   const darkBtn = document.getElementById('dark-mode-btn');
   if (darkBtn) {
      darkBtn.addEventListener('click', toggleDarkMode);
      // Sync icon to current theme
      updateDarkModeIcon(document.documentElement.classList.contains('dark'));
   }

   // Sidebar toggle functionality
   const sidebarBtn = document.getElementById('sidebar-toggle-btn');
   const sidebar = document.getElementById('main-sidebar');
   // Select dynamic components to hide when collapsing
   const sidebarTexts = document.querySelectorAll('.sidebar-text');
   const sidebarHeaders = document.querySelectorAll('.sidebar-section-header');
   const logoText = document.querySelector('.sidebar-logo-text');

   if (sidebarBtn && sidebar) {
      sidebarBtn.addEventListener('click', function () {
         // Desktop Toggle (Collapse width, hide text)
         if (window.innerWidth >= 1024) {
            const isCollapsed = sidebar.classList.contains('w-[80px]');
            
            if (isCollapsed) {
               // Expand
               sidebar.classList.remove('w-[80px]', 'px-2');
               sidebar.classList.add('w-[280px]', 'px-4');
               
               setTimeout(() => {
                  sidebarTexts.forEach(el => el.classList.remove('hidden'));
                  sidebarHeaders.forEach(el => el.classList.remove('hidden'));
                  if(logoText) logoText.classList.remove('hidden');
               }, 150);
            } else {
               // Collapse
               sidebar.classList.remove('w-[280px]', 'px-4');
               sidebar.classList.add('w-[80px]', 'px-2');
               
               sidebarTexts.forEach(el => el.classList.add('hidden'));
               sidebarHeaders.forEach(el => el.classList.add('hidden'));
               if(logoText) logoText.classList.add('hidden');
            }
         } 
         // Mobile Toggle (Slide in/out)
         else {
            if (sidebar.classList.contains('hidden')) {
               sidebar.classList.remove('hidden', 'lg:hidden');
               sidebar.classList.add('flex', 'absolute', 'h-full', 'z-50', 'w-[280px]');
               // Ensure text is visible on mobile open
               sidebarTexts.forEach(el => el.classList.remove('hidden'));
               sidebarHeaders.forEach(el => el.classList.remove('hidden'));
               if(logoText) logoText.classList.remove('hidden');
            } else {
               sidebar.classList.add('hidden');
               sidebar.classList.remove('flex', 'absolute', 'h-full', 'z-50');
            }
         }
      });
   }

   // Notification Dropdown Logic
   const notifBtn = document.getElementById('notif-menu-btn');
   const notifDropdown = document.getElementById('notif-dropdown');

   if (notifBtn && notifDropdown) {
      notifBtn.addEventListener('click', function(e) {
         e.stopPropagation();
         notifDropdown.classList.toggle('opacity-0');
         notifDropdown.classList.toggle('invisible');
         notifDropdown.classList.toggle('translate-y-2');
      });

      // Close if clicking outside
      document.addEventListener('click', function(e) {
         if (!notifDropdown.contains(e.target) && !notifBtn.contains(e.target)) {
            notifDropdown.classList.add('opacity-0', 'invisible', 'translate-y-2');
         }
      });
   }

   // Start Log Polling system for UI Notifications
   startLogPolling();
});

// ============================================
// TOAST NOTIFICATION SYSTEM
// ============================================

let lastLogCount = 0;

function startLogPolling() {
   // Poll the server every 2 seconds for new logs
   setInterval(fetchLogs, 2000);
}

async function fetchLogs() {
   try {
       const response = await fetch('/api/logs');
       if (!response.ok) return;
       
       const logs = await response.json();
       
       if (logs.length < lastLogCount) {
           lastLogCount = 0;
       }

       if (logs.length > lastLogCount) {
           const newLogs = logs.slice(lastLogCount);
           
           newLogs.forEach(log => {
               let icon = 'info';
               let colorClass = 'bg-surface-light border-border-light text-text-main dark:bg-surface-dark dark:border-border-dark dark:text-text-primary-dark';
               
               if (log.type === 'success') {
                   icon = 'check_circle';
                   colorClass = 'bg-green-50 border-green-200 text-green-800 dark:bg-green-900/30 dark:border-green-800 dark:text-green-200';
               } else if (log.type === 'error') {
                   icon = 'error';
                   colorClass = 'bg-red-50 border-red-200 text-red-800 dark:bg-red-900/30 dark:border-red-800 dark:text-red-200';
               } else if (log.type === 'warning') {
                   icon = 'warning';
                   colorClass = 'bg-yellow-50 border-yellow-200 text-yellow-800 dark:bg-yellow-900/30 dark:border-yellow-800 dark:text-yellow-200';
               } else if (log.type === 'info') {
                   icon = 'info';
                   colorClass = 'bg-blue-50 border-blue-200 text-blue-800 dark:bg-blue-900/30 dark:border-blue-800 dark:text-blue-200';
               }
               
               createToast(log.message, icon, colorClass);
           });
           
           lastLogCount = logs.length;
       }
   } catch (error) {
       console.error("Error fetching logs:", error);
   }
}

function createToast(message, icon, colorClass) {
   const container = document.getElementById('toast-container');
   if (!container) return;
   
   const toast = document.createElement('div');
   toast.className = `flex items-center gap-3 p-4 rounded-xl shadow-lg border backdrop-blur-md animate-[slideIn_0.3s_ease-out] pointer-events-auto transition-all ${colorClass}`;
   
   toast.innerHTML = `
       <span class="material-symbols-outlined text-xl shrink-0">${icon}</span>
       <span class="text-sm font-medium flex-1">${message}</span>
       <button onclick="this.parentElement.remove()" class="opacity-60 hover:opacity-100 transition-opacity ml-2 shrink-0">
           <span class="material-symbols-outlined text-sm">close</span>
       </button>
   `;
   
   if (container.firstChild) {
       container.insertBefore(toast, container.firstChild);
   } else {
       container.appendChild(toast);
   }
   
   while (container.children.length > 5) {
       container.removeChild(container.lastChild);
   }
   
   setTimeout(() => {
       if (toast.parentElement) {
           toast.style.opacity = '0';
           toast.style.transform = 'translateY(10px)';
           setTimeout(() => {
               if (toast.parentElement) toast.remove();
           }, 300);
       }
   }, 5000);
}
