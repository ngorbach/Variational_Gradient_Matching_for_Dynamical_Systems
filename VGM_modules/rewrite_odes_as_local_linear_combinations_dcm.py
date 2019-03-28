
# # Rewrite ODEs as Local Linear Combinations
# 
# #### Authors: Nico S. Gorbach and Stefan Bauer

# In[1]:

# ## Import Numpy Module

import sympy as sym


# In[2]:

# ## Rewrite each ODE as a Linear Combination in all ODE Parameters
# 
# We rewrite the ODEs as a linear combination in all ODE parameters:
# 
# \begin{align}
# \mathbf{B}_{\boldsymbol\theta k} \boldsymbol\theta + \mathbf{b}_{\boldsymbol\theta k} \stackrel{!}{=}
# \mathbf{f}_k(\mathbf{X},\boldsymbol\theta),
# \end{align}
#     
# where matrices $\mathbf{B}_{\boldsymbol\theta k}$ and $\mathbf{b}_{\boldsymbol\theta k}$ are defined 
# such that the ODEs $\mathbf{f}_k(\mathbf{X},\boldsymbol\theta)$ are expressed as a linear combination 
# in $\boldsymbol\theta$.

def rewrite_odes_as_linear_combination_in_parameters(odes_sym,state_symbols,ode_param_symbols,ext_input_symbols): 

    '''Rewrites each ODE as a Linear Combination in all ODE Parameters'''
    
    # number of hidden states
    numb_hidden_states = len(state_symbols)
    
    # append state symbols with constant vector
    symbols_all = [lst for sublist in [state_symbols,ext_input_symbols,sym.symbols(['one_vector'])] for lst in sublist]
    
    # initialize vectors B and b
    B=[[] for k in range(numb_hidden_states)]
    b=[[] for k in range(numb_hidden_states)]
    
    # rewrite ODEs as linear combinations in parameters (locally w.r.t. individual ODE)
    for k in range(numb_hidden_states):
        expr_B,expr_b = sym.linear_eq_to_matrix([odes_sym[k].expand()],ode_param_symbols)
        expr_b = -expr_b  # see the documentation of the function "sympy.linear_eq_to_matrix"
        
        # replace scalar constant by vector populated by the same constant
        for i in range(len(expr_B)):
            if len(expr_B[i].free_symbols) == 0: expr_B[i] = sym.symbols('one_vector')
        for i in range(len(expr_b)):
            if len(expr_b[i].free_symbols) == 0: expr_b[i] = sym.symbols('one_vector')
           
        # transform symbolic expressions for B and b into functions
        B[k] = sym.lambdify(symbols_all,expr_B)
        b[k] = sym.lambdify(symbols_all,expr_b)
    
    return B,b


# In[3]:
    
# ## Rewrite each ODE as a Linear Combination in an Individual State
# 
# We rewrite each ODE $\mathbf{f}_k(\mathbf{X},\boldsymbol\theta)$ as a linear combination in an 
# individual state $\mathbf{x}_u$:
# \begin{align}
# \mathbf{R}_{uk} \mathbf{x}_u + \mathbf{r}_{uk} \stackrel{!}{=} \mathbf{f}_k(\mathbf{X},\boldsymbol\theta),
# \end{align}
# 
# where matrices $\mathbf{R}_{uk}$ and $\mathbf{r}_{uk}$ are defined such that the ODE 
# $\mathbf{f}_k(\mathbf{X},\boldsymbol\theta)$ is rewritten as a linear combination in the individual 
# state $\mathbf{x}_u$.


def rewrite_odes_as_linear_combination_in_states(odes_sym,state_symbols,ode_param_symbols,ext_input_symbols,observed_states,\
                                                 state_couplings,clamp_states_to_observation_fit=1): 

    '''Rewrite each ODE as a Linear Combination in an Individual State'''
    
    # number of hidden states
    numb_hidden_states = len(state_symbols)
    
    # number of ODEs
    numb_odes = len(odes_sym)
    
    # concatenate state, parameter and external input symbols
    symbols_all = [lst for sublist in [state_symbols,ode_param_symbols,ext_input_symbols,sym.symbols(['one_vector'])] for lst in sublist]
    
    
    # determine set of hidden states to infer
    if clamp_states_to_observation_fit==1:
    # indices of observed states
        hidden_states_to_infer = [u for u in range(len(state_symbols)) if state_symbols[u] not in observed_states]
    else:
        hidden_states_to_infer = range(len(state_symbols))   

        
    # initialize matrices R and r    
    R = [[[] for k in range(numb_odes)] for u in range(numb_hidden_states)]
    r = [[[] for k in range(numb_odes)] for u in range(numb_hidden_states)]

    # rewrite ODEs as linear combinations in individual states (locally w.r.t. individual ODE)
    for u in hidden_states_to_infer:
        for k in state_couplings[u]:
            expr_R,expr_r = sym.linear_eq_to_matrix([odes_sym[k].expand()],\
                                                     state_symbols[u])  
            expr_r = -expr_r  # see the documentation of the function "sympy.linear_eq_to_matrix"
        
            # replace scalar by vector populated by the same scalar
            for i in range(len(expr_R)):
                if len(expr_R[i].free_symbols) == 0: expr_R[i] *= sym.symbols('one_vector')
            for i in range(len(expr_r)):
                if len(expr_r[i].free_symbols) == 0: expr_r[i] *= sym.symbols('one_vector')
            
            # transform symbolic expressions for R and r into functions
            R[u][k] = sym.lambdify(symbols_all,expr_R)
            r[u][k] = sym.lambdify(symbols_all,expr_r)

    return R,r