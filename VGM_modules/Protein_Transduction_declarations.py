
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

odes_path = 'Protein_Transduction_ODEs.txt'
fig_shape = (10,20)

class simulation:
    initial_states = [1,0,1,0,0]
    integration_interval = 0.01
    state = None
    observations = None
    
class time_points:
    true = np.arange(0.0, 100.0, 0.01)
    observed = None
    final_observed = 100.0
    
class symbols:
    state = sym.symbols(['_S','_dS','_R','_RS','_R_pp'])
    param = sym.symbols(['_k_1','_k_2','_k_3','_k_4','_V','_K_m'])
    
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
    
    