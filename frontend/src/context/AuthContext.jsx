import { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { api } from '../services/api';

const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  const checkSession = useCallback(async () => {
    try {
      setLoading(true);
      const session = await api.getSession();
      if (session.logged_in) {
        setUser({
          username: session.username,
          role: session.role,
          roll_number: session.roll_number,
          faculty_id: session.faculty_id,
          designation: session.designation,
          department: session.department,
        });
      } else {
        setUser(null);
      }
    } catch (error) {
      console.error('Failed to verify session', error);
      setUser(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    checkSession();
  }, [checkSession]);

  const login = async (formData) => {
    try {
      const response = await api.login(formData);
      if (response.success || response.redirected) {
         await checkSession();
         return { success: true };
      }
      return { success: false, message: 'Invalid credentials' };
    } catch (error) {
      return { success: false, message: 'Login failed' };
    }
  };

  const logout = async () => {
    try {
      await api.logout();
      setUser(null);
    } catch (error) {
      console.error('Logout failed', error);
    }
  };

  return (
    <AuthContext.Provider value={{ user, loading, login, logout, checkSession }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
