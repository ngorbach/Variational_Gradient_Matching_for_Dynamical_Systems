
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

def rewrite_odes_as_linear_combination_in_parameters(odes,state_symbols,ode_param_symbols): 

    # append state symbols with constant vector
    state_symbols_appended = state_symbols[:]
    state_symbols_appended.append(sym.symbols('one_vector'))
    
    # initialize vectors B and b
    B=[[] for k in range(len(state_symbols))]
    b=[[] for k in range(len(state_symbols))]
    
    # rewrite ODEs as linear combinations in parameters (locally w.r.t. individual ODE)
    for k in range(len(state_symbols)):
        expr_B,expr_b = sym.linear_eq_to_matrix([odes(state_symbols,ode_param_symbols)[k].expand()],\
                                                 ode_param_symbols)
        expr_b = -expr_b  # see the documentation of the function "sympy.linear_eq_to_matrix"
        
        # replace scalar constant by vector populated by the same constant
        for i in range(len(expr_B)):
            if len(expr_B[i].free_symbols) == 0: expr_B[i] = sym.symbols('one_vector')
        for i in range(len(expr_b)):
            if len(expr_b[i].free_symbols) == 0: expr_b[i] = sym.symbols('one_vector')
           
        # transform symbolic expressions for B and b into functions
        B[k] = sym.lambdify(state_symbols_appended,expr_B)
        b[k] = sym.lambdify(state_symbols_appended,expr_b)
    
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


def rewrite_odes_as_linear_combination_in_states(odes,state_symbols,ode_param_symbols,observed_states): 

    # unpack state and parameter symbols
    symbolic_one = sym.symbols('one_vector')
    symbols_all = ode_param_symbols[:]
    symbols_all.append(state_symbols[:])
    symbols_all.append(symbolic_one)
    symbols_all = sym.flatten(symbols_all)
    
    # append state symbols with constant vector
    state_symbols_appended = state_symbols[:]
    state_symbols_appended.append(sym.symbols('one_vector'))
    
    # initialize matrices R and r
    R=[[[],[],[]] for k in range(len(state_symbols))]
    r=[[[],[],[]] for k in range(len(state_symbols))]
    
    # rewrite ODEs as linear combinations in individual states (locally w.r.t. individual ODE)
    unobserved_state_idx = [u for u in range(len(state_symbols)) if state_symbols[u] not in observed_states]
    for u in unobserved_state_idx:
        for k in range(len(state_symbols)):
            expr_R,expr_r = sym.linear_eq_to_matrix([odes(state_symbols,ode_param_symbols)[k].expand()],\
                                                     state_symbols[u])  
            expr_r = -expr_r
        
            # replace scalar by vector populated by the same scalar
            for i in range(len(expr_R)):
                if len(expr_R[i].free_symbols) == 0: expr_R[i] *= sym.symbols('one_vector')
            for i in range(len(expr_r)):
                if len(expr_r[i].free_symbols) == 0: expr_r[i] *= sym.symbols('one_vector')
            
            
            # transform symbolic expressions for R and r into functions
            R[u][k] = sym.lambdify(symbols_all,expr_R)
            r[u][k] = sym.lambdify(symbols_all,expr_r)
            
#            R[u][k] = sym.lambdify(*(ode_param_symbols,state_symbols),expr_R)
#            r[u][k] = sym.lambdify(*[ode_param_symbols,state_symbols],expr_r)

    return R,r