
# # Lorenz Attractor Declarations
# 
# #### Authors: Nico S. Gorbach and Stefan Bauer


# In[1]:

# ## Import Numpy and Symbolic Computation Modules


import numpy as np
import sympy as sym
import sys



# In[2]:

# ## Object Class Declarations

odes_path = 'Glucose_Uptake_in_Yeast_ODEs.txt'
fig_shape = (10,40)

class simulation:
    #initial_states = [1.4,0,0,0,0.73,0,0,0.005,0.005]
    initial_states = np.ones((9))
    integration_interval = 0.01
    state = None
    observations = None
    
class time_points:
    true = np.arange(0.0, 100.0, 0.01)
    observed = None
    final_observed = 100.0
    
class symbols:
    state = sym.symbols(['_x_Glce','_x_Glci','_x_EG6Pi','_x_EGlcG6Pi','_x_G6Pi','_x_EGlce','_x_EGlci','_x_Ee','_x_Ei'])
    param = sym.symbols(['_k_1','_k_1b','_k_2','_k_2b','_k_3','_k_3b','_k_4','_k_4b','_alpha','_beta'])
    
class opt_settings:
    number_of_ascending_steps = 20
    
class locally_linear_odes:
    class ode_param:
        B = None
        b = None
    class state:
        B = None
        b = None
        
class proxy:
    ode_param = None
    state = None
    
    