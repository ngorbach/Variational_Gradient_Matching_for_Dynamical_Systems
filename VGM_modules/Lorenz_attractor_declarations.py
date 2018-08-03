
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

odes_path = 'Lorenz_attractor_ODEs.txt'

class simulation:
    initial_states = [1,1,1]
    integration_interval = 0.01
    state = None
    observations = None
    
class time_points:
    true = np.arange(0.0, 20.0, 0.01)
    observed = None
    final_observed = 20.0
    
class symbols:
    state = sym.symbols(['x','y','z'])
    param = sym.symbols(['sigma','rho','alpha'])
    
class opt_settings:
    number_of_ascending_steps = 10
    
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
    
    