export default function StatCard({ 
    title, 
    value, 
    icon, 
    colorClass, 
    badgeText, 
    badgeIcon = 'arrow_upward', 
    description,
    shadowColor 
  }) {
    // Default gradients/colors if none provided
    const defaultColor = colorClass || 'from-primary to-primary-dark text-primary';
    const hasGradient = defaultColor.includes('to-');
    
    return (
      <div className="group bg-surface-light dark:bg-surface-dark/90 p-5 rounded-[20px] border border-border-light/60 dark:border-border-dark/60 shadow-sm backdrop-blur-md flex flex-col justify-between transition-all duration-300 hover:-translate-y-1 hover:shadow-premium relative overflow-hidden h-full">
        <div className={`absolute -right-6 -top-6 w-32 h-32 rounded-full blur-2xl group-hover:scale-150 transition-transform duration-500 ${shadowColor || 'bg-primary/10'}`}></div>
        
        <div className="flex justify-between items-start relative z-10 mb-4">
          <div className={`size-14 rounded-[16px] flex items-center justify-center border shadow-sm relative z-10 transition-transform group-hover:scale-110 duration-300 ${
            hasGradient ? `bg-gradient-to-br ${defaultColor.split(' ')[0]} ${defaultColor.split(' ')[1]} text-white border-white/20` : `bg-${defaultColor}-50 dark:bg-${defaultColor}-900/30 text-${defaultColor}-600 dark:text-${defaultColor}-400 border-${defaultColor}-100 dark:border-${defaultColor}-800/50`
          }`}>
            <span className="material-symbols-outlined text-[28px] drop-shadow-sm">{icon}</span>
          </div>
          {badgeText && (
            <span className="flex items-center gap-1 text-[11px] font-bold text-green-600 dark:text-green-400 bg-green-50 dark:bg-green-900/30 px-2.5 py-1 rounded-full border border-green-100 dark:border-green-800/50">
              <span className="material-symbols-outlined text-[12px]">{badgeIcon}</span> {badgeText}
            </span>
          )}
        </div>
        
        <div className="relative z-10 mt-auto">
          <p className="text-text-secondary-light dark:text-text-secondary-dark text-[12px] font-bold uppercase tracking-wider mb-1">
            {title}
          </p>
          <div className="flex items-baseline gap-2">
            <h3 className="text-text-primary-light dark:text-text-primary-dark text-[32px] font-black tracking-[-0.03em] leading-none">
              {value}
            </h3>
          </div>
          {description && (
            <p className="text-text-secondary-light dark:text-text-secondary-dark text-xs mt-2 font-medium">
              {description}
            </p>
          )}
        </div>
      </div>
    );
  }
