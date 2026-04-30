import { useToast } from '../context/ToastContext';

export default function ToastContainer() {
  const { toasts, removeToast } = useToast();

  if (toasts.length === 0) return null;

  return (
    <div
      id="toast-container"
      className="fixed bottom-5 right-5 z-[9999] flex flex-col gap-3 max-w-sm w-full pointer-events-none"
    >
      {toasts.map((toast) => {
        let icon = 'info';
        let colorClass =
          'bg-surface-light border-border-light text-slate-800 dark:bg-surface-dark dark:border-border-dark dark:text-slate-200';

        if (toast.type === 'success') {
          icon = 'check_circle';
          colorClass =
            'bg-green-50 border-green-200 text-green-800 dark:bg-green-900/30 dark:border-green-800 dark:text-green-200';
        } else if (toast.type === 'error') {
          icon = 'error';
          colorClass =
            'bg-red-50 border-red-200 text-red-800 dark:bg-red-900/30 dark:border-red-800 dark:text-red-200';
        } else if (toast.type === 'warning') {
          icon = 'warning';
          colorClass =
            'bg-yellow-50 border-yellow-200 text-yellow-800 dark:bg-yellow-900/30 dark:border-yellow-800 dark:text-yellow-200';
        }

        return (
          <div
            key={toast.id}
            className={`flex items-center gap-3 p-4 rounded-xl shadow-lg border backdrop-blur-md animate-[slideIn_0.3s_ease-out] pointer-events-auto transition-all ${colorClass}`}
          >
            <span className="material-symbols-outlined text-xl shrink-0">{icon}</span>
            <span className="text-sm font-medium flex-1">{toast.message}</span>
            <button
              onClick={() => removeToast(toast.id)}
              className="opacity-60 hover:opacity-100 transition-opacity ml-2 shrink-0"
            >
              <span className="material-symbols-outlined text-sm">close</span>
            </button>
          </div>
        );
      })}
    </div>
  );
}
