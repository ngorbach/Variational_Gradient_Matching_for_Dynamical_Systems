
# # Proxies for ODE Parameters and States
# 
# #### Authors: Nico S. Gorbach and Stefan Bauer

# In[1]:

# ## Import Numpy and Symbolic Computation Modules


import numpy as np
import sympy as sym


# In[2]:

# ## Import ODEs


def import_odes(symbols,odes_path):
    with open(odes_path) as f:
        odes_string = f.read().splitlines()
        
    odes_symbolic = [o.factor() for o in sym.sympify(odes_string)]
    odes = sym.lambdify( [symbols.state,symbols.param],odes_symbolic)
    
    return odes

# In[3]:

# ## Find Couplings  between States across ODEs


def find_state_couplings_in_odes(odes,symbols):
    state_couplings = np.zeros((len(symbols.state),len(symbols.state)))
    for u in range(len(symbols.state)):
        odes_sym = odes(symbols.state,symbols.param)[u]
        mapping = filter(lambda v: v is not None,ismember(symbols.state,odes_sym.free_symbols))
        idx = [j for (j,val) in enumerate(mapping) if np.not_equal(val,None)]
        state_couplings[u,[idx]] = 1
        
    return state_couplings


# In[4]:


def ismember(a, b):
    bind = {}
    for i, elt in enumerate(b):
        if elt not in bind:
            bind[elt] = 1
    return [bind.get(itm, None) for itm in a]

