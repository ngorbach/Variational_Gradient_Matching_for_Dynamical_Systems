
# # Loren_z Attractor Declarations
# 
# #### Authors: Nico S. Gorbach and Stefan Bauer


# In[1]:

# ## Import Nump_y and S_ymbolic Computation Modules


import numpy as np
import sympy as sym
import sys



# In[2]:

# ## Object Class Declarations

odes_path = 'Model_Search_ODEs.txt'
fig_shape = (7,15)

class simulation:
    initial_states = [0.1,0.9,0.5,0.2,0.5] #np.random.rand(5)#[5,0,3,0,0]
    integration_interval = 0.01
    state = None
    observations = None
    
class time_points:
    #true = np.arange(0.0, 100.0, 0.01)
    true = np.arange(0.0, 20.0, 0.01)
    observed = None
    final_observed = 2.0
    
class symbols:
    state = sym.symbols(['_x','_y','_z','_w','_v'])
    param = sym.symbols(['_k_1','_k_2','_k_3','_k_4','_k_5','_k_6','_k_7','_k_8','_k_9',\
                         '_k_10','_k_11','_k_12','_k_13','_k_14','_k_15','_k_16','_k_17',\
                         '_k_18','_k_19','_k_20','_k_21','_k_22','_k_23','_k_24','_k_25',\
                         '_k_26','_k_27','_k_28','_k_29','_k_30','_k_31','_k_32','_k_33',\
                         '_k_34','_k_35','_k_36','_k_37','_k_38','_k_39','_k_40','_k_41',\
                         '_k_42','_k_43','_k_44','_k_45'])
    
#    param = sym.symbols(['_k_1','_k_2','_k_3','_k_4','_k_5','_k_6','_k_7','_k_8','_k_9',\
#                          '_k_10','_k_11','_k_12','_k_13','_k_14','_k_15','_k_16','_k_17',\
#                          '_k_18','_k_19','_k_20','_k_21','_k_22','_k_23','_k_24',\
#                          '_k_25','_k_26'])
    
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
    
    