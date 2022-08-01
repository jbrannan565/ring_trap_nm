#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np, matplotlib.pyplot as plt
from ring_trap_nm.nelder_mead.Objective import Objective
from ring_trap_nm.nelder_mead.StochasticInitPotentialSearch import StochasticInitPotentialSearch


# # Min Energy with Fixed Momentum random search
# Here, we search the space of starting potentials uniformly.

# In[ ]:


def generate_initial_simplex(dimensions=100):
    """
    Uses ideas from [here](https://math.stackexchange.com/questions/2739915/radius-of-inscribed-sphere-of-n-simplex)
    """
    verts = []
    for i in range(dimensions):
        tmp = np.zeros(dimensions)
        tmp[i] = dimensions
        verts.append(tmp)
    verts.append(np.zeros(dimensions))
    return verts


# In[ ]:


objective_functions = (
    "get_E_expectation",
    "get_E_deviation",
    "get_dist_p_expectation",
    "get_p_deviation",
    "get_V_smoothness"
)

lambs = (
    1.0,
    1.0,
    1.0,
    1.0,
    int(1e3)
)

options = {
    "maxiter": int(1e6),
    "disp": True,
    "adaptive": True,
    "initial_simplex": generate_initial_simplex()
}


obj = Objective(objective_functions=objective_functions, lambs=lambs)
searcher = StochasticInitPotentialSearch(objective=obj, seed=2)


# In[ ]:


res = searcher.get_n_samples(n=100*2)
res


# In[ ]:


import pickle
from datetime import datetime
now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

with open(f'results/random-init-potential-{now}', 'wb') as f:
    pickle.dump(res, f)

