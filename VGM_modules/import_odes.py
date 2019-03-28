
# # Proxies for ODE Parameters and States
# 
# #### Authors: Nico S. Gorbach and Stefan Bauer

# In[1]:

# ## Import Numpy and Symbolic Computation Modules


import numpy as np
import sympy as sym


# In[2]:


def import_odes(symbols,odes_path):
    
    '''Imports ODEs from file specified by odes_path'''
    
    with open(odes_path) as f:
        odes_string = f.read().splitlines()
        
    odes_sym = [o.factor() for o in sym.sympify(odes_string)]
    
    # unpack state and parameter symbols
    symbols_all = [lst for sublist in [symbols.state,symbols.param] for lst in sublist]
    odes = sym.lambdify( symbols_all,odes_sym)
    
    return odes,odes_sym

def gradient_of_odes(symbols,odes_sym):
    
    '''Computes gradient of ODEs'''

    symbols_all = [lst for sublist in [symbols.param,symbols.state] for lst in sublist]
    symbols_all.append(sym.symbols('one_vector'))
    
    
    odes_diff_param_sym = [[o.diff(s) for s in symbols.param] for o in odes_sym]
    for j in range(len(odes_diff_param_sym)):
        for i in range(len(odes_diff_param_sym[j])):
            #if len(odes_diff_param_sym[j][i].free_symbols) == 0: odes_diff_param_sym[j][i] = sym.symbols('one_vector') 
            odes_diff_param_sym[j][i] *= sym.symbols('one_vector')
            if odes_diff_param_sym[j][i] == 0: odes_diff_param_sym[j][i] = np.finfo(float).eps * sym.symbols('one_vector') 
    odes_diff_param = sym.lambdify(symbols_all,odes_diff_param_sym)
            
    
    odes_diff_states_sym = [[] for i in range(len(symbols.state))]
    for u in range(len(symbols.state)):
        odes_diff_states_sym[u] = [o.diff(symbols.state[u]) for o in odes_sym]
    
    #odes_diff_states_sym = [[o.expand().diff(s) for s in symbols.state] for o in odes_sym]
    for j in range(len(odes_diff_states_sym)):
        for i in range(len(odes_diff_states_sym[j])):
            odes_diff_states_sym[j][i] *= sym.symbols('one_vector')
            if odes_diff_states_sym[j][i] == 0:
                odes_diff_states_sym[j][i] = np.finfo(float).eps * sym.symbols('one_vector')    
            
     
    odes_diff_states=[[] for u in range(len(symbols.state))]
    for u in range(len(symbols.state)):
        odes_diff_states[u] = sym.lambdify(symbols_all,odes_diff_states_sym[u])
    
    
    return odes_diff_param, odes_diff_states

# In[3]:



def find_state_couplings_in_odes(odes_sym,symbols):
    
    '''Find Couplings  between States across ODEs'''
    
    odes_couplings_to_states = []
    symbols_all = [lst for sublist in [symbols.state,symbols.param] for lst in sublist]
    for u in range(len(symbols.state)):

        
        membership = ismember(symbols.state,odes_sym[u].free_symbols)
        odes_couplings_to_states.append([symbols.state[i] for i,item in enumerate(membership) if item is not None])
    
    state_couplings = [[i for i,item in enumerate(odes_couplings_to_states) if s in item] for s in symbols.state]   
        
    return state_couplings,odes_couplings_to_states


# In[4]:


def ismember(a, b):
    bind = {}
    for i, elt in enumerate(b):
        if elt not in bind:
            bind[elt] = 1
    return [bind.get(itm, None) for itm in a]