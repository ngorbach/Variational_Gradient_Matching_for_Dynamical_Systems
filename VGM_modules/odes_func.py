#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 22:04:48 2018

@author: nico
"""

import numpy as np

def odes_func(state,ode_param):
    
    x_dot1 = ode_param[0] * state[:,0] - ode_param[1] * state[:,0] * state[:,1]
    x_dot2 = -ode_param[2] * state[:,1] + ode_param[3] * state[:,0] * state[:,1]
    x_dot = np.vstack([x_dot1.reshape(-1,1),x_dot2.reshape(-1,1)])
    
#    x_dot1 = ode_param[0] * (-state[:,0] + state[:,1])
#    x_dot2 = ode_param[1] * state[:,0] - state[:,0] * state[:,2] - state[:,1]
#    x_dot3 = -ode_param[2] * state[:,2] + state[:,0] * state[:,1]
#    x_dot = np.vstack([x_dot1.reshape(-1,1),x_dot2.reshape(-1,1),x_dot3.reshape(-1,1)])
        
    return np.squeeze(x_dot)


def squared_loss(ode_param,state,dC_times_invC_times_states):

    x_dot = odes_func(state,ode_param).reshape(-1,1)
    
    return sum((x_dot-dC_times_invC_times_states.reshape(-1,1))**2)