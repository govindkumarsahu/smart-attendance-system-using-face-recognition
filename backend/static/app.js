/**
 * ============================================================
 * SMART ATTENDANCE SYSTEM - JAVASCRIPT
 * Main application logic for web interface
 * ============================================================
 */

// ==========================================
// GLOBAL VARIABLES
// ==========================================
let logMessages = new Set();
let autoScrollEnabled = true;

// ==========================================
// LOG MANAGEMENT FUNCTIONS
// ==========================================

/**
 * Add a new log entry to the terminal
 * @param {string} message - Log message
 * @param {string} type - Log type: info, success, warning, error
 */
function addLogEntry(message, type = 'info') {
    const terminal = document.getElementById('terminal-output');
    const messageKey = message + type; // Unique key for deduplication
    
    // Prevent duplicate messages
    if (!logMessages.has(messageKey)) {
        const timestamp = new Date().toLocaleTimeString('en-IN', {
            hour12: true,
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
        
        const entry = document.createElement('div');
        entry.className = `log-entry ${type}`;
        
        // Add appropriate icon based on type
        let icon = '';
        switch(type) {
            case 'success': icon = '✓'; break;
            case 'error': icon = '✗'; break;
            case 'warning': icon = '⚠'; break;
            default: icon = 'ℹ';
        }
        
        entry.textContent = `[${timestamp}] ${icon} ${message}`;
        terminal.appendChild(entry);
        logMessages.add(messageKey);
        
        // Auto-scroll to bottom if enabled
        if (autoScrollEnabled) {
            terminal.scrollTop = terminal.scrollHeight;
        }
        
        // Add entrance animation
        entry.style.animation = 'slideInRight 0.3s ease-out';
    }
}

/**
 * Clear all log entries
 */
function clearLog() {
    const terminal = document.getElementById('terminal-output');
    
    // Fade out effect
    terminal.style.opacity = '0.3';
    
    setTimeout(() => {
        terminal.innerHTML = '';
        logMessages.clear();
        terminal.style.opacity = '1';
        addLogEntry('Log cleared successfully. System ready.', 'success');
    }, 200);
}

/**
 * Fetch latest logs from server
 */
async function fetchLogs() {
    try {
        const response = await fetch('/api/logs');
        const data = await response.json();
        
        if (data.logs && Array.isArray(data.logs) && data.logs.length > 0) {
            data.logs.forEach(log => {
                const messageKey = log.message + (log.type || 'info');
                if (!logMessages.has(messageKey)) {
                    addLogEntry(log.message, log.type || 'info');
                }
            });
        }
    } catch (error) {
        console.error('Failed to fetch logs:', error);
        // Don't show error to user to avoid clutter
    }
}

/**
 * Toggle auto-scroll on terminal
 */
function toggleAutoScroll() {
    autoScrollEnabled = !autoScrollEnabled;
    console.log('Auto-scroll:', autoScrollEnabled ? 'Enabled' : 'Disabled');
}

// ==========================================
// FORM VALIDATION
// ==========================================

/**
 * Initialize form validation on page load
 */
function initializeFormValidation() {
    const registrationForm = document.getElementById('registration-form');
    
    if (registrationForm) {
        registrationForm.addEventListener('submit', function(e) {
            const nameInput = document.getElementById('student_name');
            const name = nameInput.value.trim();
            
            // Validate name length
            if (name.length < 2) {
                e.preventDefault();
                alert('⚠️ Please enter a valid name (at least 2 characters)');
                nameInput.focus();
                return;
            }
            
            // Check for numbers or special characters
            if (!/^[A-Za-z\s]+$/.test(name)) {
                e.preventDefault();
                alert('⚠️ Name should only contain letters and spaces');
                nameInput.focus();
                return;
            }
            
            // Confirm before submission
            const confirmed = confirm(
                `📝 Register new student:\n\n"${name}"\n\n` +
                `Camera will open for face capture.\n` +
                `Make sure:\n` +
                `• Good lighting\n` +
                `• Face clearly visible\n` +
                `• Look directly at camera\n\n` +
                `Proceed?`
            );
            
            if (!confirmed) {
                e.preventDefault();
            }
        });
    }
}

// ==========================================
// TERMINAL SCROLL DETECTION
// ==========================================

/**
 * Detect when user scrolls up to pause auto-scroll
 */
function initializeScrollDetection() {
    const terminal = document.getElementById('terminal-output');
    
    if (terminal) {
        terminal.addEventListener('scroll', function(e) {
            const element = e.target;
            const isAtBottom = element.scrollHeight - element.scrollTop === element.clientHeight;
            
            if (!isAtBottom && autoScrollEnabled) {
                // User scrolled up, temporarily disable auto-scroll
                autoScrollEnabled = false;
                setTimeout(() => {
                    autoScrollEnabled = true;
                }, 5000); // Re-enable after 5 seconds
            }
        });
    }
}

// ==========================================
// KEYBOARD SHORTCUTS
// ==========================================

/**
 * Setup keyboard shortcuts
 */
function initializeKeyboardShortcuts() {
    document.addEventListener('keydown', function(e) {
        // Ctrl+L or Cmd+L to clear logs
        if ((e.ctrlKey || e.metaKey) && e.key === 'l') {
            e.preventDefault();
            clearLog();
        }
    });
}

// ==========================================
// PAGE VISIBILITY API
// ==========================================

/**
 * Handle page visibility changes
 */
function initializeVisibilityHandler() {
    document.addEventListener('visibilitychange', function() {
        if (document.hidden) {
            console.log('Tab hidden - log polling continues in background');
        } else {
            console.log('Tab visible - ensuring log sync');
            fetchLogs();
        }
    });
}

// ==========================================
// INITIALIZATION
// ==========================================

/**
 * Initialize all components when DOM is ready
 */
function initializeApp() {
    console.log('Smart Attendance System - JavaScript loaded');
    
    // Setup form validation
    initializeFormValidation();
    
    // Setup scroll detection
    initializeScrollDetection();
    
    // Setup keyboard shortcuts
    initializeKeyboardShortcuts();
    
    // Setup visibility handler
    initializeVisibilityHandler();
    
    // Start log polling (every 1 second)
    setInterval(fetchLogs, 1000);
    
    // Initial fetch
    fetchLogs();
    
    // Add welcome messages
    setTimeout(() => {
        addLogEntry('Smart Attendance System v2.0 initialized', 'success');
        addLogEntry('All systems operational. Ready to accept commands.', 'info');
    }, 500);
}

// ==========================================
// PAGE LOAD EVENT
// ==========================================

// Wait for DOM to be fully loaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeApp);
} else {
    // DOM already loaded
    initializeApp();
}

// ==========================================
// EXPORT FOR DEBUGGING (Optional)
// ==========================================
window.AttendanceApp = {
    addLogEntry,
    clearLog,
    fetchLogs,
    toggleAutoScroll
};
