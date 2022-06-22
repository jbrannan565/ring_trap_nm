import numpy as np
from ..schrodinger.Schrodinger1D import Schrodinger1D as Schrodinger


class Objective(Schrodinger):
    def __init__(self, y0=None, tf=1.0, **kwargs):
        super().__init__(**kwargs)
        if y0 is None:
            self.y0 = np.ones(self.N)
        else:
            self.y0 = y0

        self.tf = tf

    
    def get_objective(self, V):
        if V.shape[0] != self.N or len(V.shape) > 1:
            raise ValueError("provided V wrong shape in objective")
        
        self.V0 = V

        ys, ts = self.evolve(self.y0, self.tf)

        yf = ys[-1,:]

        raise NotImplementedError("get_objective not yet implemented!")
