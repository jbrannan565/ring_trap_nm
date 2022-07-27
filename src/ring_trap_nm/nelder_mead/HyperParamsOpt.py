from .Objective import Objective
from scipy.optimize import minimize

class HyperParamsOptimizer:
    def __init__(self, objective: Objective, lambs_0=None):
        self.objective = objective

        if lambs_0 is None:
            self.lambs_0 = objective.lambs
    
    def compute_objective(self, lambs, **kwargs):
        # set lambdas
        self.objective.lambs = lambs

        # solve
        res = self.objective.solve(**kwargs)

        # compute fitness
        return self.objective.compute_objective(res.x)

    def solve(self, method="Nelder-Mead", options=None, **kwargs):
        if options is None:
            options = {
                "maxiter": int(1e5),
                "disp": True
            }
        return minimize(fun=self.compute_objective, x0=self.lambs_0, method=method, options=options, **kwargs)