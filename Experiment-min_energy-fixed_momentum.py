#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np, matplotlib.pyplot as plt
from ring_trap_nm.nelder_mead.Objective import Objective


# # Minimum Energy with Fixed Momentum

# In[2]:


objective_functions = (
    "get_E_expectation",
    "get_E_deviation",
    "get_dist_p_expectation",
    "get_p_deviation",
    "get_V_smoothness"
)

lambs = (
    1.0,
    5.0,
    1.0,
    5.0,
    int(1e3)
)

options = {
    "maxiter": int(1e6),
    "disp": True
}

obj = Objective(objective_functions=objective_functions, lambs=lambs)


# In[3]:


ret = obj.solve(options=options)
ret


# In[4]:


fig, ax = plt.subplots()
ax.plot(obj.xs, ret.x)

plt.show()


# In[6]:


obj.compute_objective_vals()


# ## Using a sin wave as the starting potential

# In[7]:


y0 = 0.5*np.sin(np.linspace(0,np.pi*4,100))

obj2 = Objective(objective_functions=objective_functions, lambs=lambs, y0=y0)


# In[7]:


ret2 = obj2.solve(options=options)
ret2


# In[8]:


fig, ax = plt.subplots()
ax.plot(obj2.xs, ret2.x)

plt.show()


# In[8]:


obj2.compute_objective_vals()


# In[9]:


from ring_trap_nm.nelder_mead.HyperParamsOpt import HyperParamsOptimizer

hyp_optimizer = HyperParamsOptimizer(objective=obj2)


# In[10]:


res = hyp_optimizer.solve()
res


# In[5]:


import pickle
from datetime import datetime

now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
with open('results/hyper_opt_res-{now}', 'wb') as fa:
    pickle.dump(res, fa)
    
with open('results/hyp_opt-{now}', 'wb') as fb:
    pickle.dump(hyp_optimizer, fb)


# In[4]:





# In[ ]:




