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
   const btn = document.getElementById('dark-mode-btn');
   if (btn) {
      btn.addEventListener('click', toggleDarkMode);
      // Sync icon to current theme
      updateDarkModeIcon(document.documentElement.classList.contains('dark'));
   }
});
