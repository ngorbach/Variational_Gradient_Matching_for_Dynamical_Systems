
# # Simulate State Dynamics
# 
# #### Authors: Nico S. Gorbach and Stefan Bauer

# In[1]:

# ## Import Numpy, Scipy and Plotting Modules


import numpy as np
import scipy.integrate as integrate
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from plotting import *

# In[2]:


def integrand(states,t,odes,ode_param):
    
    '''Evaluates state derivatives'''
    
    return odes(states,ode_param)


# In[3]:

# ## Numerical Integration
    

def numerical_integration(odes,time_points,states,param,state_symbols):
    
    '''Integrates the ODEs using numerical integration'''
    
    state = integrate.odeint(integrand, states, time_points, args=(odes,param))
    
    # pack states into pandas DataFrames
    state = pd.DataFrame(state,columns=map(str,state_symbols),index=time_points).rename_axis('time')
    
    return state


# In[5]:


def setup_simulation(simulation,time_points,symbols,odes,color_idx1=1,*args):

    '''Simulates the State Dynamics with given ODE Parameters'''
    
    
    # simulate state observations
    if len(args)==0:
        
        # simulate state trajectories by numerical integration
        state = numerical_integration(odes,time_points.true,simulation.initial_states,simulation.ode_param,symbols.state)
    
        observations = simulate_state_observations(state,simulation.interval_between_observations,\
                simulation.observed_states,time_points.final_observed,simulation.obs_variance).rename_axis('time')
        

        # write state- and observed trajectories to file in directoy './data'
        state.to_csv('./data/true_states.csv')
        observations.to_csv('./data/observations.csv')
        
        # pack simulation.ode_param into pandas DataFrame
        simulation.ode_param = pd.DataFrame(np.array(simulation.ode_param),columns=['value'],\
                                            index=map(str,symbols.param)).rename_axis('ODE parameter symbols')
            
        ## plotting ODE parameters and states
        plot_ode_parameters(simulation.ode_param)
        plot_states(state,observations)
        
        # mapping between observation- and state trajectories
        obs_to_state_relations = \
        mapping_between_observation_and_state_trajectories(time_points.for_estimation,\
                                                           np.array(observations.index),symbols.state,\
                                                           simulation.observed_states)
        
    else:
        
        # unpack ode_param into list
        ode_param = simulation.ode_param.values
        
        # simulate state trajectories by numerical integration
        state = numerical_integration(odes,time_points.true,simulation.initial_states,ode_param,symbols.state)
        
        # unpack arbitrary length arguments
        observations = args[0]
        state_proxy = args[1]
        
        ## plotting ODE parameters and states
        plot_ode_parameters(simulation.ode_param,[2,0,1])
        plot_states(state,observations,['num. int. with estimated param','VGM estimate','observed'],[2,0,1],1,state_proxy)
        obs_to_state_relations = []
        

    return state, observations, obs_to_state_relations



# In[4]:


def simulate_state_observations(state,interval_between_observations,observed_states,\
                                final_observed_time_point,obs_variance=0.1):
    
    '''Simulates observations of state trajectories by adding normally distributed noise to the true state trajectories'''
    
    #integration_time_points = np.arange(0,final_time_point,integration_interval)
    integration_time_points = np.array(state.index)
    observed_time_points = np.arange(0,final_observed_time_point,interval_between_observations)

    # indices of observed time points
    integration_interval = state.index[2] - state.index[1]
    observed_time_idx = np.round(observed_time_points / integration_interval + 1)
    
    # unpack subset of state
    state_subset = state[map(str,observed_states)]
    state_subset = state_subset.iloc[observed_time_idx.astype(int),:]
    
    # add normally distributed noise with variance $\sigma$ to the true state trajectories
    observations = state_subset + np.sqrt(obs_variance) * \
    np.random.randn(observed_time_idx.shape[0],state_subset.shape[1])
    
    observed_time_points = integration_time_points[observed_time_idx.astype(int)]
    
    # pack observations into pandas DataFrames
    observations = pd.DataFrame(observations,columns=map(str,observed_states),\
                                index=observed_time_points).rename_axis('time')
        
    return observations



# In[6]:


def mapping_between_observation_and_state_trajectories(given_time_points,observed_time_points,state_symbols,\
                                                       observed_states):

    '''Deftermines relationship between observation- and state trajectories'''
    
    # Euclidean distance between observed time_points and given time points
    if len(observed_time_points) > len(given_time_points):
        dist = cdist(given_time_points.reshape(-1,1),observed_time_points.reshape(-1,1),'euclidean')
        row_ind, col_ind = linear_sum_assignment(dist)
    else:
        dist = cdist(observed_time_points.reshape(-1,1),given_time_points.reshape(-1,1),'euclidean')
        row_ind, col_ind = linear_sum_assignment(dist)
        
    # linear relationship between observed and hidden state time points
    obs_time_to_state_time_relations = np.zeros((len(observed_time_points),len(given_time_points)))
    obs_time_to_state_time_relations[row_ind,col_ind] = 1
    
    # linear relationship between observations and state   
    observed_state_idx = [u for u in range(len(state_symbols)) if state_symbols[u] in observed_states]  
    state_mat = np.zeros((len(state_symbols),len(state_symbols)))
    state_mat[observed_state_idx,observed_state_idx] = 1
    unobserved_state_idx = [i for i in range(len(state_symbols)) if state_symbols[i] not in observed_states]
    state_mat = np.delete(state_mat,unobserved_state_idx,0)

    ## Kronecker-Delta product
    obs_to_state_relations = np.kron(state_mat,obs_time_to_state_time_relations)
    
    return obs_to_state_relations