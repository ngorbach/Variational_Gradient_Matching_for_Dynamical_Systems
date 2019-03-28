
# # Proxies for ODE Parameters and States
# 
# #### Authors: Nico S. Gorbach and Stefan Bauer

# In[1]:

# ## Import Numpy and Symbolic Computation Modules


import numpy as np
import sympy as sym


# In[2]:


def import_odes(symbols,odes_path,ext_input_path):
    
    '''Imports ODEs from file specified by odes_path'''
    
    with open(odes_path) as f:
        odes_string = f.read().splitlines()
   
    odes_sym = [o.factor() for o in sym.sympify(odes_string)]
    #odes = sym.lambdify( [symbols.state,symbols.param],odes_symbolic)
    
    # unpack state and parameter symbols
    symbols_all = [lst for sublist in [symbols.state,symbols.param,symbols.ext_input] for lst in sublist]
    odes = sym.lambdify( symbols_all,odes_sym)
    
    ext_input = np.loadtxt(ext_input_path,delimiter=' ')
    
    func_path = 'DCM_function.txt'
    with open(func_path) as f:
        func_string = f.read().splitlines()
    func_sym = [o.factor() for o in sym.sympify(func_string)]
    func = sym.lambdify(symbols.state,func_sym)
    
    return odes,odes_sym,ext_input,func,func_sym

# In[2]:
def gradient_of_odes(symbols,odes_sym,func_sym):
    
    '''Computes gradient of ODEs'''

    symbols_all = [lst for sublist in [symbols.state,symbols.param,symbols.ext_input] for lst in sublist]
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
    
    for j in range(len(odes_diff_states_sym)):
        for i in range(len(odes_diff_states_sym[j])):
            odes_diff_states_sym[j][i] *= sym.symbols('one_vector')
            if odes_diff_states_sym[j][i] == 0:
                odes_diff_states_sym[j][i] = np.finfo(float).eps * sym.symbols('one_vector')          
     
    odes_diff_states=[[] for u in range(len(symbols.state))]
    for u in range(len(symbols.state)):
        odes_diff_states[u] = sym.lambdify(symbols_all,odes_diff_states_sym[u])
    
    
    
    symbols_all = symbols.state[:]
    symbols_all.append(sym.symbols('one_vector'))
    
    func_diff_sym = [[] for i in range(len(symbols.state))]
    for u in range(len(symbols.state)):
        func_diff_sym[u] = [o.diff(symbols.state[u]) for o in func_sym]
    
    for j in range(len(func_diff_sym)):
        for i in range(len(func_diff_sym[j])):
            func_diff_sym[j][i] *= sym.symbols('one_vector')
            if func_diff_sym[j][i] == 0:
                func_diff_sym[j][i] = np.finfo(float).eps * sym.symbols('one_vector')          
     
    func_diff=[[] for u in range(len(symbols.state))]
    for u in range(len(symbols.state)):
        func_diff[u] = sym.lambdify(symbols_all,func_diff_sym[u])
        
    
    return odes_diff_param, odes_diff_states, func_diff

# In[3]:



def find_state_couplings_in_odes(odes_sym,symbols):
    
    '''Find Couplings  between States across ODEs'''
    
    odes_couplings_to_states = []
    symbols_all = [lst for sublist in [symbols.state,symbols.param,symbols.ext_input] for lst in sublist]
    for u in range(len(symbols.state)):

        
        membership = ismember(symbols.state,odes_sym[u].free_symbols)
        odes_couplings_to_states.append([symbols.state[i] for i,item in enumerate(membership) if item is not None])
    
    state_couplings = [[i for i,item in enumerate(odes_couplings_to_states) if s in item] for s in symbols.state]   
    
    state_couplings[12][1:] = []
    state_couplings[13][1:] = []
    state_couplings[14][1:] = []
    state_couplings[6][3:] = []
    state_couplings[7][3:] = []
    state_couplings[8][3:] = []
    state_couplings[3][1:] = []
    state_couplings[4][1:] = []
    state_couplings[5][1:] = []
    state_couplings[9][1:] = []
    state_couplings[10][1:] = []
    state_couplings[11][1:] = []
    
    return state_couplings


# In[4]:


def ismember(a, b):
    bind = {}
    for i, elt in enumerate(b):
        if elt not in bind:
            bind[elt] = 1
    return [bind.get(itm, None) for itm in a]


#def bold_signal_change_eqn(V,Q):
#
#    # Biophysical constants for 1.5T
#    #==========================================================================
#     
#    # time to echo (TE) (default 0.04 sec)
#    # --------------------------------------------------------------------------
#    TE = 0.04
#     
#    #  resting venous volume (%)
#    # --------------------------------------------------------------------------
#    V0 = 4
#    
#    #  estimated region-specific ratios of intra- to extra-vascular signal 
#    # --------------------------------------------------------------------------
#    # P.epsilon = 0.2015;
#    P = 0.1970
#    ep = sym.exp(P)
#     
#    #  slope r0 of intravascular relaxation rate R_iv as a function of oxygen 
#    #  saturation S:  R_iv = r0*[(1 - S)-(1 - S0)] (Hz)
#    # --------------------------------------------------------------------------
#    r0 = 25.0
#     
#    #  frequency offset at the outer surface of magnetized vessels (Hz)
#    # --------------------------------------------------------------------------
#    nu0 = 40.3
#     
#    #  resting oxygen extraction fraction
#    # --------------------------------------------------------------------------
#    E0 = 0.4
#     
#    # -Coefficients in BOLD signal model
#    # ==========================================================================
#    k1 = 4.3*nu0*E0*TE
#    k2 = ep*r0*E0*TE
#    k3 = 1 - ep
#     
#    # -Output equation of BOLD signal model
#    # ==========================================================================
#    
#    V = sym.exp(V)
#    Q = sym.exp(Q)
#    Y = V0 * ( k1 * (1 - Q) + k2 * (1 - Q/V) + k3  (1 - V) )