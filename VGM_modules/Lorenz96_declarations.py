
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

odes_path = 'Lorenz96_ODEs.txt'
fig_shape = (10,40)

class simulation:
    initial_states = np.zeros((10))
    initial_states[8] += 0.01
    integration_interval = 0.01
    state = None
    observations = None
    
class time_points:
    true = np.arange(0.0, 4.0, 0.01)
    observed = None
    final_observed = 20.0
    
class symbols:
    state = sym.symbols(['_x_1','_x_2','_x_3','_x_4','_x_5','_x_6','_x_7','_x_8','_x_9','_x_10'])
    param = sym.symbols(['_alpha'])
    
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
    
    