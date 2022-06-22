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
    def __init__(self, objective_functions, y0=None, tf=1.0, \
        lambs=None, **kwargs):
        super().__init__(**kwargs)

        # set objective functions
        self.objective_functions = objective_functions

        # handle default y0
        if y0 is None:
            self.y0 = np.ones(self.N)
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
        return np.std(V) / np.abs(np.mean(np.diff(V)))

    
    def compute_objective(self, V):
        """
        Returns the object value for potential V.
        """
        # ensure V dims
        if V.shape[0] != self.N or len(V.shape) > 1:
            raise ValueError("provided V wrong shape in objective")
        
        # set V0
        self.V0 = V

        # evolve state
        ys, ts = self.evolve(self.y0, self.tf)

        # get final state
        yf = ys[-1,:]
        tf = ts[-1]

        # get objective values from objective functionns
        objective_vals = np.zeros(len(self.objective_functions))
        for idx, objective in enumerate(self.objective_functions):
            objective_vals[idx] = getattr(self, objective)(yf, tf)

        # weight and return
        return self.lambs.dot(objective_vals)
    

    def solve(self, method="Nelder-Mead", **kwargs):
        return minimize(fun=self.compute_objective, x0=self.y0)