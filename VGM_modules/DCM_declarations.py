
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

odes_path = 'DCM_ODEs.txt'
ext_input_path = 'external_input.txt'

class simulation:
    #initial_states = [1.4,0,0,0,0.73,0,0,0.005,0.005]
    initial_states = 0.01*np.ones((15))     
    #initial_states = 1*np.ones((15))
    integration_interval = 0.01
    state = None
    observations = None
    
class time_points:
    true = np.arange(0.2, 1160.0, 0.2) #np.arange(0.2013, 1159.2, 0.2013)
    #true = np.arange(0.2013, 1159.2, 0.2013)
    observed = None
    final_observed = 1160.0
    
class symbols:
    state = sym.symbols(['_q1','_q3','_q2','_v1','_v3','_v2','_f1','_f3','_f2','_s1','_s3','_s2','_n1','_n3','_n2'])
    param = sym.symbols(['_a_21','_a_12','_a_32','_a_23','_b_212','_b_213','_c_11','_c_33','_a_11','_a_22','_a_33'])
    ext_input = sym.symbols(['_u1','_u2','_u3'])
    func_response = sym.symbols(['_y1','_y2','_y3'])
    
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
    
    