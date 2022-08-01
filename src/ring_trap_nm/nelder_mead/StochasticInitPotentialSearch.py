from .Objective import Objective
import numpy as np

class StochasticInitPotentialSearch:
    def __init__(self, objective: Objective, seed=2):
        # set objective
        self.objective = objective

        # set rngg
        self.rng = np.random.default_rng(seed=seed)
    

    def iter(self, *args, **kwargs):
        self.objective.y0 = self.rng.uniform(size=self.objective.xs.shape)
        return self.objective.solve(*args, **kwargs)
    

    def get_n_samples(self, n, *args, **kwargs):
        return [self.iter(*args, **kwargs) for i in range(n)]