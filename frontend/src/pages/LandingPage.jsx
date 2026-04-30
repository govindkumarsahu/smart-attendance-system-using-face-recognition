import { Link } from 'react-router-dom';

export default function LandingPage() {
  return (
    <div className="relative flex min-h-screen w-full flex-col group/design-root bg-background-light dark:bg-[#101622] text-slate-900 dark:text-slate-100 font-sans antialiased overflow-x-hidden">
        {/* Header */}
        <header className="sticky top-0 z-50 w-full bg-white/90 backdrop-blur-md border-b border-slate-200 dark:bg-slate-900/90 dark:border-slate-800">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex items-center justify-between h-16">
                    {/* Logo */}
                    <div className="flex items-center gap-3">
                        <div className="flex items-center justify-center size-8 rounded-lg bg-[#003366] text-white">
                            <span className="material-symbols-outlined text-[20px]">school</span>
                        </div>
                        <h2 className="text-slate-900 dark:text-white text-lg font-bold tracking-tight">Smart Attendance</h2>
                    </div>
                    {/* Desktop Nav */}
                    <nav className="hidden md:flex flex-1 justify-center gap-8">
                        <a className="text-slate-600 hover:text-[#003366] dark:text-slate-300 dark:hover:text-white text-sm font-medium transition-colors" href="#features">Features</a>
                        <a className="text-slate-600 hover:text-[#003366] dark:text-slate-300 dark:hover:text-white text-sm font-medium transition-colors" href="#how-it-works">How it Works</a>
                        <a className="text-slate-600 hover:text-[#003366] dark:text-slate-300 dark:hover:text-white text-sm font-medium transition-colors" href="#roles">Roles</a>
                    </nav>
                    {/* CTA */}
                    <div className="flex items-center gap-4">
                        <Link className="hidden sm:flex items-center justify-center rounded-lg h-9 px-4 bg-[#003366] hover:bg-[#002852] text-white text-sm font-bold transition-colors shadow-sm" to="/login">
                            <span>Login</span>
                        </Link>
                        {/* Mobile Menu Icon (Visual Only) */}
                        <button className="md:hidden text-slate-600 dark:text-slate-300">
                            <span className="material-symbols-outlined">menu</span>
                        </button>
                    </div>
                </div>
            </div>
        </header>

        <main className="flex-grow">
            {/* Hero Section */}
            <section className="relative bg-white dark:bg-[#101622] overflow-hidden">
                {/* Background Decoration */}
                <div className="absolute inset-0 z-0 opacity-10 dark:opacity-5">
                    <div className="absolute inset-0" aria-label="Subtle geometric dot pattern" style={{backgroundImage: 'radial-gradient(#003366 1px, transparent 1px)', backgroundSize: '32px 32px'}}></div>
                </div>
                <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 lg:py-32">
                    <div className="grid lg:grid-cols-2 gap-12 items-center">
                        <div className="flex flex-col gap-6 text-left">
                            <div className="inline-flex items-center rounded-full border border-blue-200 bg-blue-50 px-3 py-1 text-sm font-medium text-[#003366] dark:bg-blue-900/20 dark:border-blue-800 dark:text-blue-300 w-fit">
                                <span className="flex h-2 w-2 rounded-full bg-[#003366] mr-2"></span>
                                Now available for Faculty
                            </div>
                            <h1 className="text-4xl lg:text-6xl font-black leading-tight tracking-tight text-slate-900 dark:text-white">
                                Smart Attendance <br />
                                <span className="text-transparent bg-clip-text bg-gradient-to-r from-[#003366] to-blue-500">System</span>
                            </h1>
                            <p className="text-lg text-slate-600 dark:text-slate-400 max-w-xl leading-relaxed">
                                Automated Face Recognition-Based Attendance for Smart Campuses. Experience seamless classroom management with high-precision AI technology.
                            </p>
                            <div className="flex flex-wrap gap-4 pt-4">
                                <Link className="flex items-center justify-center rounded-lg h-12 px-8 bg-[#003366] hover:bg-[#002852] text-white text-base font-bold transition-all shadow-lg hover:shadow-xl hover:-translate-y-0.5" to="/login">
                                    Login
                                </Link>
                                <Link className="flex items-center justify-center rounded-lg h-12 px-8 border-2 border-slate-200 bg-transparent hover:border-[#003366] hover:text-[#003366] text-slate-700 dark:border-slate-700 dark:text-slate-300 dark:hover:border-blue-400 dark:hover:text-blue-400 text-base font-bold transition-all" to="/login">
                                    Faculty Sign Up
                                </Link>
                            </div>
                        </div>
                        <div className="relative lg:h-[500px] w-full rounded-2xl overflow-hidden shadow-2xl bg-slate-100 dark:bg-slate-800 group">
                            {/* Abstract Representation of Face Scan */}
                            <div className="absolute inset-0 bg-cover bg-center" aria-label="Modern classroom collaborative students" style={{backgroundImage: "url('https://lh3.googleusercontent.com/aida-public/AB6AXuDXFfgq11a7k4tQRPnVr8RKO-RS2VXi0udwUOPjTLXbNlJoGM5Os5_HhbD5s1tJgotgHJvzdG8Voth3MyYMlNHiQrKnsp8bsFzTNPs3mJApWNq0XlN1nTgKsSInCI2_LJIBJPEXEtmo686A6-MEuhd5S9wucfm0Zrmd9Ji4LG0YGiclc5dJ4QKwaOXzmiW3yLSkRmTplP1jjuYc2Z82YPpri862Syv4ECOuWnRangMAxStKiV2FDEZm1uVkcJeIi3_dEbtvmY3svJQ')"}}></div>
                            <div className="absolute inset-0 bg-gradient-to-t from-[#003366]/90 to-transparent mix-blend-multiply"></div>
                            <div className="absolute bottom-0 left-0 p-8 w-full">
                                <div className="bg-white/10 backdrop-blur-md border border-white/20 rounded-xl p-4 flex items-center gap-4">
                                    <div className="h-12 w-12 rounded-full bg-green-500 flex items-center justify-center text-white shadow-lg">
                                        <span className="material-symbols-outlined">check</span>
                                    </div>
                                    <div>
                                        <p className="text-white font-bold text-lg">Attendance Marked</p>
                                        <p className="text-blue-100 text-sm">Real-time verification complete</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            {/* Key Features Section */}
            <section className="py-20 bg-background-light dark:bg-slate-900 border-y border-slate-200 dark:border-slate-800" id="features">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="text-center max-w-3xl mx-auto mb-16">
                        <h2 className="text-[#003366] dark:text-blue-400 font-semibold tracking-wide uppercase text-sm mb-2">Key Features</h2>
                        <h3 className="text-3xl font-bold text-slate-900 dark:text-white sm:text-4xl">Why Choose Smart Attendance?</h3>
                        <p className="mt-4 text-lg text-slate-600 dark:text-slate-400">Experience the future of classroom management with our cutting-edge features designed for efficiency and security.</p>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
                        <div className="group bg-white dark:bg-slate-800 rounded-xl p-6 shadow-sm hover:shadow-md transition-all border border-slate-200 dark:border-slate-700">
                            <div className="w-12 h-12 rounded-lg bg-blue-50 dark:bg-blue-900/30 flex items-center justify-center text-[#003366] dark:text-blue-400 mb-4 group-hover:scale-110 transition-transform">
                                <span className="material-symbols-outlined">face</span>
                            </div>
                            <h4 className="text-lg font-bold text-slate-900 dark:text-white mb-2">Face Recognition</h4>
                            <p className="text-slate-600 dark:text-slate-400 text-sm leading-relaxed">Instant identification using advanced AI models ensuring high accuracy attendance.</p>
                        </div>
                        <div className="group bg-white dark:bg-slate-800 rounded-xl p-6 shadow-sm hover:shadow-md transition-all border border-slate-200 dark:border-slate-700">
                            <div className="w-12 h-12 rounded-lg bg-blue-50 dark:bg-blue-900/30 flex items-center justify-center text-[#003366] dark:text-blue-400 mb-4 group-hover:scale-110 transition-transform">
                                <span className="material-symbols-outlined">visibility</span>
                            </div>
                            <h4 className="text-lg font-bold text-slate-900 dark:text-white mb-2">Real-Time Monitoring</h4>
                            <p className="text-slate-600 dark:text-slate-400 text-sm leading-relaxed">Track student presence as it happens in real-time with live dashboard updates.</p>
                        </div>
                        <div className="group bg-white dark:bg-slate-800 rounded-xl p-6 shadow-sm hover:shadow-md transition-all border border-slate-200 dark:border-slate-700">
                            <div className="w-12 h-12 rounded-lg bg-blue-50 dark:bg-blue-900/30 flex items-center justify-center text-[#003366] dark:text-blue-400 mb-4 group-hover:scale-110 transition-transform">
                                <span className="material-symbols-outlined">description</span>
                            </div>
                            <h4 className="text-lg font-bold text-slate-900 dark:text-white mb-2">Automated Reports</h4>
                            <p className="text-slate-600 dark:text-slate-400 text-sm leading-relaxed">Generate detailed Excel and PDF reports instantly for monthly or semester reviews.</p>
                        </div>
                        <div className="group bg-white dark:bg-slate-800 rounded-xl p-6 shadow-sm hover:shadow-md transition-all border border-slate-200 dark:border-slate-700">
                            <div className="w-12 h-12 rounded-lg bg-blue-50 dark:bg-blue-900/30 flex items-center justify-center text-[#003366] dark:text-blue-400 mb-4 group-hover:scale-110 transition-transform">
                                <span className="material-symbols-outlined">admin_panel_settings</span>
                            </div>
                            <h4 className="text-lg font-bold text-slate-900 dark:text-white mb-2">Secure Access</h4>
                            <p className="text-slate-600 dark:text-slate-400 text-sm leading-relaxed">Role-based access controls ensuring data privacy and security for all users.</p>
                        </div>
                    </div>
                </div>
            </section>

            {/* How It Works Section */}
            <section className="py-20 bg-white dark:bg-[#101622]" id="how-it-works">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="text-center mb-16">
                        <h2 className="text-3xl font-bold text-slate-900 dark:text-white">How It Works</h2>
                        <p className="mt-4 text-slate-600 dark:text-slate-400">Streamlined process from registration to reporting.</p>
                    </div>
                    <div className="relative">
                        <div className="hidden md:block absolute top-1/2 left-0 w-full h-0.5 bg-slate-200 dark:bg-slate-700 -translate-y-1/2 z-0"></div>
                        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 relative z-10">
                            <div className="bg-white dark:bg-[#101622] p-4 flex flex-col items-center text-center">
                                <div className="w-16 h-16 rounded-full bg-[#003366] text-white flex items-center justify-center text-2xl font-bold shadow-lg mb-4 border-4 border-white dark:border-[#101622]">1</div>
                                <h4 className="font-bold text-lg text-slate-900 dark:text-white mb-2">Register Students</h4>
                                <p className="text-sm text-slate-500 dark:text-slate-400">Input student details and initial photos into the database.</p>
                            </div>
                            <div className="bg-white dark:bg-[#101622] p-4 flex flex-col items-center text-center">
                                <div className="w-16 h-16 rounded-full bg-white border-2 border-[#003366] text-[#003366] flex items-center justify-center text-2xl font-bold shadow-lg mb-4 dark:bg-slate-800 dark:border-blue-500 dark:text-blue-400">2</div>
                                <h4 className="font-bold text-lg text-slate-900 dark:text-white mb-2">Train Model</h4>
                                <p className="text-sm text-slate-500 dark:text-slate-400">System learns face encodings for accurate recognition.</p>
                            </div>
                            <div className="bg-white dark:bg-[#101622] p-4 flex flex-col items-center text-center">
                                <div className="w-16 h-16 rounded-full bg-white border-2 border-[#003366] text-[#003366] flex items-center justify-center text-2xl font-bold shadow-lg mb-4 dark:bg-slate-800 dark:border-blue-500 dark:text-blue-400">3</div>
                                <h4 className="font-bold text-lg text-slate-900 dark:text-white mb-2">Take Attendance</h4>
                                <p className="text-sm text-slate-500 dark:text-slate-400">Camera scans classroom and marks present students.</p>
                            </div>
                            <div className="bg-white dark:bg-[#101622] p-4 flex flex-col items-center text-center">
                                <div className="w-16 h-16 rounded-full bg-white border-2 border-[#003366] text-[#003366] flex items-center justify-center text-2xl font-bold shadow-lg mb-4 dark:bg-slate-800 dark:border-blue-500 dark:text-blue-400">4</div>
                                <h4 className="font-bold text-lg text-slate-900 dark:text-white mb-2">Generate Reports</h4>
                                <p className="text-sm text-slate-500 dark:text-slate-400">Export attendance data for academic records.</p>
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            {/* Roles Section */}
            <section className="py-20 bg-slate-50 dark:bg-slate-900/50" id="roles">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="text-center mb-12">
                        <h2 className="text-3xl font-bold text-slate-900 dark:text-white">User Roles</h2>
                        <p className="mt-4 text-slate-600 dark:text-slate-400">Tailored dashboards for different responsibilities.</p>
                    </div>
                    <div className="grid md:grid-cols-2 gap-8 max-w-4xl mx-auto">
                        <div className="bg-white dark:bg-slate-800 rounded-2xl p-8 shadow-sm border border-slate-100 dark:border-slate-700 flex flex-col items-start hover:border-[#003366] dark:hover:border-blue-500 transition-colors">
                            <div className="p-3 bg-purple-100 text-purple-700 rounded-lg mb-6 dark:bg-purple-900/30 dark:text-purple-300">
                                <span className="material-symbols-outlined text-3xl">person</span>
                            </div>
                            <h3 className="text-2xl font-bold text-slate-900 dark:text-white mb-4">Student</h3>
                            <ul className="space-y-3 mb-8 flex-1">
                                <li className="flex items-start gap-3 text-slate-600 dark:text-slate-400">
                                    <span className="material-symbols-outlined text-green-500 text-xl">check_circle</span>
                                    <span>View personal attendance history</span>
                                </li>
                                <li className="flex items-start gap-3 text-slate-600 dark:text-slate-400">
                                    <span className="material-symbols-outlined text-green-500 text-xl">check_circle</span>
                                    <span>Track subject-wise presence</span>
                                </li>
                                <li className="flex items-start gap-3 text-slate-600 dark:text-slate-400">
                                    <span className="material-symbols-outlined text-green-500 text-xl">check_circle</span>
                                    <span>Download attendance certificates</span>
                                </li>
                            </ul>
                            <Link to="/login" className="w-full py-3 px-4 bg-slate-100 hover:bg-slate-200 text-slate-800 font-semibold rounded-lg transition-colors dark:bg-slate-700 dark:text-white dark:hover:bg-slate-600 text-center block">
                                Student Login
                            </Link>
                        </div>
                        <div className="bg-white dark:bg-slate-800 rounded-2xl p-8 shadow-sm border border-slate-100 dark:border-slate-700 flex flex-col items-start hover:border-[#003366] dark:hover:border-blue-500 transition-colors">
                            <div className="p-3 bg-blue-100 text-[#003366] rounded-lg mb-6 dark:bg-blue-900/30 dark:text-blue-300">
                                <span className="material-symbols-outlined text-3xl">cast_for_education</span>
                            </div>
                            <h3 className="text-2xl font-bold text-slate-900 dark:text-white mb-4">Faculty</h3>
                            <ul className="space-y-3 mb-8 flex-1">
                                <li className="flex items-start gap-3 text-slate-600 dark:text-slate-400">
                                    <span className="material-symbols-outlined text-green-500 text-xl">check_circle</span>
                                    <span>Class-specific attendance management</span>
                                </li>
                                <li className="flex items-start gap-3 text-slate-600 dark:text-slate-400">
                                    <span className="material-symbols-outlined text-green-500 text-xl">check_circle</span>
                                    <span>Download subject-wise reports</span>
                                </li>
                                <li className="flex items-start gap-3 text-slate-600 dark:text-slate-400">
                                    <span className="material-symbols-outlined text-green-500 text-xl">check_circle</span>
                                    <span>View individual student analytics</span>
                                </li>
                            </ul>
                            <Link to="/login" className="w-full py-3 px-4 bg-[#003366] hover:bg-[#002852] text-white font-semibold rounded-lg transition-colors shadow-md text-center block">
                                Faculty Login
                            </Link>
                        </div>
                    </div>
                </div>
            </section>
        </main>
        
        {/* Footer */}
        <footer className="bg-[#001a33] text-slate-300 py-12 border-t border-slate-800">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-8">
                    <div className="col-span-1">
                        <div className="flex items-center gap-2 mb-4 text-white">
                            <span className="material-symbols-outlined">school</span>
                            <span className="text-xl font-bold">Smart Attendance</span>
                        </div>
                        <p className="text-sm text-slate-400 mb-4">
                            College of Engineering &amp; Technology<br />
                            Department of Computer Science
                        </p>
                    </div>
                    <div className="col-span-1">
                        <h4 className="text-white font-bold mb-4">Quick Links</h4>
                        <ul className="space-y-2 text-sm">
                            <li><Link className="hover:text-white transition-colors" to="/">Home</Link></li>
                            <li><Link className="hover:text-white transition-colors" to="/">Privacy Policy</Link></li>
                            <li><Link className="hover:text-white transition-colors" to="/">Terms of Service</Link></li>
                        </ul>
                    </div>
                    <div className="col-span-1">
                        <h4 className="text-white font-bold mb-4">Contact</h4>
                        <ul className="space-y-2 text-sm">
                            <li className="flex items-center gap-2">
                                <span className="material-symbols-outlined text-sm">email</span>
                                <a className="hover:text-white transition-colors" href="mailto:admin@college.edu">admin@college.edu</a>
                            </li>
                            <li className="flex items-center gap-2">
                                <span className="material-symbols-outlined text-sm">code</span>
                                <a className="hover:text-white transition-colors" href="#">View on GitHub</a>
                            </li>
                        </ul>
                    </div>
                </div>
                <div className="border-t border-white/10 pt-8 flex flex-col md:flex-row justify-between items-center text-xs text-slate-500">
                    <p>&copy; 2025 Smart Attendance System. All rights reserved.</p>
                    <p>Designed for Academic Excellence</p>
                </div>
            </div>
        </footer>
    </div>
  );
}
