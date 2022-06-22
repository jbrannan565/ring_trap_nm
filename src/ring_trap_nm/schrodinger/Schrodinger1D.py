import numpy as np, matplotlib.pyplot as plt
from numpy.fft import fft, ifft
from scipy.integrate import solve_ivp


class Schrodinger1D:
    def __init__(self, N=100, xs=None, V0=0.0):
        self.hbar = 1.0
        self.m = 0.5
        
        if xs is None:
            self.xs = np.linspace(-2*np.pi, 2*np.pi, N)
        else:
            self.xs = xs

        self.dx = None
        self.ks = 2*np.pi*np.fft.fftfreq(N, self.get_dx())
        self.V0 = V0
        self.N = N
        
        
    def get_dx(self):
        """return average difference of `self.xs`"""
        if self.dx is not None: return self.dx
        self.dx = np.diff(self.xs).mean()
        return self.dx

        
    def get_n_dydx(self, y, n=1):
        """return the `n`th derivative of `y' using fourier methods"""
        if n < 1:
            raise ValueError("n to small")

        return ifft(fft(y) * (1j*self.ks)**n)

    def get_V(self, t):
        """return V(t) -- the external potential"""
        return self.V0
    
    
    def get_dydt(self, t, y):
        """return dy/dt by schrodinger"""
        ddy = self.get_n_dydx(y, n=2)
        Vy = self.get_V(t)*y
        return (-self.hbar/self.m/2*ddy + Vy) / 1j


    def get_expectation(self, t, y, operator):
        """return expectation value of `operator`"""
        return np.sum(y.conj()*operator(t,y))

       
    def get_standard_deviation(self, t, y, \
        expectation_operator, square_expectation_operator):
        """
        return standard deviation of an operator using
        - math::
            \sqrt{square_expectation_operator - expectation_operator^2}
        """
        return np.sqrt(square_expectation_operator(t, y) \
            - expectation_operator(t, y)**2)

        
    def get_Ey(self, t, y):
        """
        get 
        - math::
            \hat{E}y = i\hbar dy / dt
        """
        return 1j*self.hbar*self.get_dydt(t, y)
    
      
    def get_E_expectation(self, t, y):
        """return expectation value of Hamiltonian operator"""
        return self.get_expectation(t, y, self.get_Ey)

    
    def get_E_squared(self, t, y):
        """
        Returns expectation of Hamiltonian squared:
        .. math::
            \lang \hat{E}^2 \rang &= - \hbar \partial / \partial t
            &= H^2

            -(-\hbar / m / 2 \cdot d^2/dx^2 + V)^2 
            &= ([\hbar / m / 2]^2 d^4/dx^4 \\
                + 2V -\hbar / 2 / m d^2/dx^2 \\
                + V^2)
        """
        V = self.get_V(t)
        ts = np.zeros(3)
        ts[0] = (self.hbar / self.m / 2)**2 * self.get_n_dydx(y, n=4)
        ts[1]= (-self.hbar/self.m) * V * y
        ts[2]= V**2 * y
        return np.sum(ts)

    
    def get_E_deviation(self, t, y):
        """return standard deviation of Hamiltonian operator"""
        return self.get_standard_deviation(t, y, \
            self.get_E_expectation, self.get_E_squared)
    
    
    def get_p(self, t, y):
        """
        get 
        - math::
            \hat{p}y = - i\hbar dy/ dt
        """
        return -1j*self.hbar*self.get_n_dydx(y, n=1)
    
    
    def get_p_expectation(self, t, y):
        """return expextation value of momentum operator"""
        return self.get_expectation(t, y, self.get_p)
    
    
    def get_p_squared(self, t, y):
        """return expextation value of momentum operator squared"""
        return -self.hbar*self.get_ddyddx(y)
    
    
    def get_p_deviation(self, t, y):
        """return standard deviation of momentum operator `p`"""
        return self.get_standard_deviation(t, y, \
            self.get_p_expectation, self.get_p_squared)

    
    def evolve(self, y0, tf, **kwargs):
        """evolve state `y0` to time `tf`"""
        return solve_ivp(fun=self.get_dydt, t_span=(0,tf), y0=y0, **kwargs)
    

    def plot_state(self, y):
        """plot state 'y' over 'self.xs'"""
        fig, ax = plt.subplots()
        ax.plot(self.xs, y)
        plt.show
        return fig, ax