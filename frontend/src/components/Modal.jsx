import React from 'react';

export default function Modal({ isOpen, onClose, title, children, maxWidth = 'max-w-md' }) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      <div 
        className="fixed inset-0 bg-slate-900/40 dark:bg-slate-900/60 backdrop-blur-sm transition-opacity" 
        onClick={onClose}
      ></div>
      
      <div className="flex min-h-full items-center justify-center p-4 text-center sm:p-0">
        <div className={`relative transform overflow-hidden rounded-2xl bg-white dark:bg-slate-800 text-left shadow-xl transition-all sm:my-8 sm:w-full ${maxWidth}`}>
          
          {/* Header */}
          <div className="bg-white dark:bg-slate-800 px-4 pb-4 pt-5 sm:p-6 sm:pb-4 border-b border-slate-100 dark:border-slate-700 font-sans">
            <div className="sm:flex sm:items-start">
              <div className="mt-3 text-center sm:ml-4 sm:mt-0 sm:text-left w-full flex justify-between items-center">
                <h3 className="text-lg font-bold leading-6 text-slate-900 dark:text-white" id="modal-title">
                  {title}
                </h3>
                <button 
                  onClick={onClose}
                  className="text-slate-400 hover:text-slate-500 focus:outline-none"
                >
                  <span className="material-symbols-outlined shrink-0 text-xl">close</span>
                </button>
              </div>
            </div>
          </div>
          
          {/* Body */}
          <div className="px-4 py-5 sm:p-6 bg-white dark:bg-slate-800">
            {children}
          </div>
          
        </div>
      </div>
    </div>
  );
}
