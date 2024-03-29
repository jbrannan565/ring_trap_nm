import numpy as np
from scipy.optimize import minimize

from ..schrodinger.Schrodinger1D import Schrodinger1D as Schrodinger


class Objective(Schrodinger):
    """
    A class that represents an optimization objective

    ...

    Attributes
    ----------
    objective_functions : tuple
        sting name of local methods to be used as objective functions
    desired_p: float
        the p we'd like to get close to
    y0 : np.array
        initial quantum state
    tf : float
        time to evolve to
    lambs : tuple
        list of weights for objective_functions in final weighted sum
    kwargs : dict
        kwargs for Schrodinger1D

    Methods
    -------
    compute_objective(V)
        Returns the object value for potential V.
    """
    def __init__(self, objective_functions, desired_p=2*np.pi, y0=None, tf=1.0, \
        lambs=None, **kwargs):
        super().__init__(**kwargs)

        # set objective functions
        self.objective_functions = objective_functions
    
        # set desired p
        self.desired_p = desired_p

        # handle default y0
        if y0 is None:
            self.y0 = np.zeros(self.N)
        else:
            self.y0 = y0
        
        # set tf
        self.tf = tf

        if lambs is None:
            self.lambs = np.ones(len(objective_functions))
        else:
            self.lambs = np.array(lambs)

    
    def get_V_smoothness(self, t, y):
        """return a measure of the smoothness of self.V0"""
        V = self.V0
        return np.std(np.diff(V))


    def get_dist_p_expectation(self, y, t):
        """get the distance between the expectation value of p and `desired_p`"""
        return np.abs(self.get_p_expectation(y, t) - self.desired_p)


    def compute_objective_vals(self):
        # evolve state
        res = self.evolve(self.y0, self.tf)
        ts, ys = res.t, res.y

        # get final state
        yf = ys[:,-1]
        tf = ts[-1]

        # get objective values from objective functionns
        objective_vals = np.zeros(len(self.objective_functions))
        for idx, objective in enumerate(self.objective_functions):
            objective_vals[idx] = getattr(self, objective)(tf, yf)
        
        return objective_vals


    def compute_objective(self, V):
        """
        Returns the object value for potential V.
        """
        # ensure V dims
        if V.shape[0] != self.N or len(V.shape) > 1:
            raise ValueError("provided V wrong shape in objective")
        
        # set V0
        self.V0 = V
        
        # get objective values
        objective_vals = self.compute_objective_vals()

        # weight and return
        return self.lambs.dot(objective_vals)
    

    def solve(self, method="Nelder-Mead", options=None, **kwargs):
        if options is None:
            options = {
                "maxiter": int(1e5),
                "disp": True
            }
        return minimize(fun=self.compute_objective, x0=self.y0, method=method, options=options, **kwargs)