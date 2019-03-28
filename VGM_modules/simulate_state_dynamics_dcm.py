
# # Simulate State Dynamics
# 
# #### Authors: Nico S. Gorbach and Stefan Bauer

# In[1]:

# ## Import Numpy, Scipy and Plotting Modules


import numpy as np
from scipy.integrate import odeint
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from plotting import *
#from odes_func import *



#def odes_func(state,ode_param):
#    
#    x_dot1 = ode_param[0] * (-state[:,0] + state[:,1])
#    x_dot2 = ode_param[1] * state[:,0] - state[:,0] * state[:,2] - state[:,1]
#    x_dot3 = -ode_param[2] * state[:,2] + state[:,0] * state[:,1]
#    x_dot = np.vstack([x_dot1.reshape(-1,1),x_dot2.reshape(-1,1),x_dot3.reshape(-1,1)])
#        
#    return np.squeeze(x_dot)


## In[1]:
#
## these are our constants
#N = 36  # number of variables
#F = 8  # forcing
#
#def Lorenz96(x,t):
#
#  # compute state derivatives
#  d = np.zeros(N)
#  # first the 3 edge cases: i=1,2,N
#  d[0] = (x[1] - x[N-2]) * x[N-1] - x[0]
#  d[1] = (x[2] - x[N-1]) * x[0]- x[1]
#  d[N-1] = (x[0] - x[N-3]) * x[N-2] - x[N-1]
#  # then the general case
#  for i in range(2, N-1):
#      d[i] = (x[i+1] - x[i-2]) * x[i-1] - x[i]
#  # add the forcing term
#  d = d + F
#
#  # return the state derivatives
#  return d
#
#x0 = F*np.ones(N) # initial state (equilibrium)
#x0[19] += 0.01 # add small perturbation to 20th variable
#t = np.arange(0.0, 30.0, 0.01)
#
#x = odeint(Lorenz96, x0, t)


# In[2]:


def integrand(states,t,t_lst,odes,ode_param,ext_input):
    
    '''Evaluates state derivatives'''
    
    ind = np.argmin(abs(t-t_lst))
    #states.append(ext_input[row_ind,0:-1])
    #a=np.zeros((3))
    
    return odes(*states,*ode_param,*ext_input[ind,1:])
    #return odes(*states,*ode_param,*a)

# In[3]:

# ## Numerical Integration
    

def numerical_integration(odes,time_points,states,param,state_symbols,func_response_symbols,ext_input,func):
    
    '''Integrates the ODEs using numerical integration'''
    
    state = odeint(integrand, states, time_points, args=(ext_input[:,0],odes,param,ext_input))

    # pack states into pandas DataFrames
    state = pd.DataFrame(state,columns=map(str,state_symbols),index=time_points).rename_axis('time')
    
    func_true = np.array(func(*state.values.T)).T
    func_true = pd.DataFrame(func_true,columns=map(str,func_response_symbols),index=time_points).rename_axis('time')
    
    return state,func_true


# In[5]:


def setup_simulation(simulation,time_points,symbols,odes,ext_input,func,color_idx1=1,extra_track=None,*args):

    '''Simulates the State Dynamics with given ODE Parameters'''
    
    
    # simulate state observations
    if len(args)==0:
        
        # simulate state trajectories by numerical integration
        state,func_true = numerical_integration(odes,time_points.true,simulation.initial_states,simulation.ode_param,symbols.state,symbols.func_response,ext_input,func)
    
        observations,func_obs = simulate_state_observations(state,simulation.observed_time_points,\
                simulation.observed_states,time_points.final_observed,func_true,simulation.SNR)
        

        # write state- and observed trajectories to file in directoy './data'
        state.to_csv('./data/true_states.csv')
        observations.to_csv('./data/observations.csv')
        
        # pack simulation.ode_param into pandas DataFrame
        simulation.ode_param = pd.DataFrame(np.array(simulation.ode_param),columns=['value'],\
                                            index=map(str,symbols.param)).rename_axis('ODE parameter symbols')
            
        ## plotting ODE parameters and states
        plot_ode_parameters(simulation.ode_param)
        plot_states(func_true,func_obs,(10,8),plot_name='true_BOLD_response')
        plot_states(state,observations,(10,40),plot_name='true_states')
        
    else:
        
        # unpack ode_param into list
        ode_param = np.squeeze(simulation.ode_param.values).tolist()
        
        # simulate state trajectories by numerical integration
        state,func_true = numerical_integration(odes,time_points.true,simulation.initial_states,ode_param,symbols.state,symbols.func_response,ext_input,func)

        # unpack arbitrary length arguments
        observations = args[0]
        state_proxy = args[1]
        
        ## plotting ODE parameters and states
        plot_ode_parameters(simulation.ode_param,(10,8),[2,0,1],)
        plot_states(state,observations,(10,40),label=['num. int. with estimated param','VGM estimate','observed'],color_idx=[2,0,1],traj_idx=1,sigma=None,traj_arg=state_proxy,plot_name='num_int_with_param_estimates',extra_track=extra_track)
        
        func_obs = []

    return state, observations, func_true, func_obs



# In[4]:


def simulate_state_observations(state,observed_time_points,observed_states,\
                                final_observed_time_point,func_true,SNR=1):
    
    '''Simulates observations of state trajectories by adding normally distributed noise to the true state trajectories'''
    
    #integration_time_points = np.arange(0,final_time_point,integration_interval)
    integration_time_points = np.array(state.index)
    

    # indices of observed time points
    integration_interval = state.index[2] - state.index[1]
    observed_time_idx = np.round(observed_time_points / integration_interval + 1)
    
    # unpack subset of state
    state_subset = state[list(map(str,observed_states))]
    state_subset = state_subset.iloc[observed_time_idx.astype(int),:]
    
    # determine variance of observations from SNR
    obs_variance = (np.mean(state_subset.values,axis=0) / SNR)**2
    
    # add normally distributed noise with variance $\sigma$ to the true state trajectories
    observations = state_subset + np.sqrt(obs_variance) * np.random.randn(observed_time_idx.shape[0],state_subset.shape[1])
    
    observed_time_points = integration_time_points[observed_time_idx.astype(int)]
    
    # pack observations into pandas DataFrames
    observations = pd.DataFrame(observations,columns=map(str,observed_states),index=observed_time_points).rename_axis('time')

    # determine variance of observations from SNR
    obs_variance = (np.mean(func_true.values,axis=0) / SNR)**2
    
    func_subset = func_true.iloc[observed_time_idx.astype(int),:] 
    func_obs = func_subset + np.sqrt(obs_variance) * np.random.randn(observed_time_idx.shape[0],func_subset.shape[1])
    func_obs = pd.DataFrame(func_obs,columns=list(func_true.columns),index=observed_time_points).rename_axis('time')
        
    return observations, func_obs



# In[6]:


def mapping_between_observation_and_state_trajectories(given_time_points,observed_time_points,state_symbols,observed_states,func_true_vals):

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