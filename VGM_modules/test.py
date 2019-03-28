#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 20:44:44 2018

@author: nico
"""

from sympy import hessian
import sympy as sym
symbols_all = [lst for sublist in [state_symbols,ode_param_symbols] for lst in sublist]
hessian_sym = [sym.sympify(np.diag(hessian(odes_sym[i],ode_param_symbols))) for i in range(len(odes_sym))]
hessian_func = [sym.lambdify( symbols_all,hessian_sym[i]) for i in range(len(hessian_sym))]

hess = lambda p : coumpute_gradient(p,states,dC_times_inv_C_times_state)

# In[4]:    
def coumpute_gradient(ode_param,states,dC_times_inv_C_times_state):
        
    x_dot = np.concatenate(odes(*states.T,*ode_param))
    
    cost_der = np.zeros((len(ode_param)))
    for k in range(len(hessian_func)):
        h = hessian_func[k](*states.T,*ode_param)
        cost_der += list(map(lambda h: sum(2*h*x_dot[k]),h))
    
    return np.diag(cost_der)
