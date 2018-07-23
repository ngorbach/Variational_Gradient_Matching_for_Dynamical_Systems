
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

# In[2]:

# ## Integrand


def integrand(state,t,odes,ode_param):
        
    return odes(state,ode_param)

# In[3]:

# ## Numerical Integration
    

def numerical_integration(odes,t,init_val,param):
    
    return integrate.odeint(integrand, init_val, t, args=(odes,param))

# In[4]:

# ## Simulate observations of state trajectories by adding normally distributed noise to the true state trajectories


def simulate_state_observations(state,state_symbols,final_time_point,interval_between_observations,\
                                integration_interval,obs_variance,observed_states):
    
    integration_time_points = np.arange(0,final_time_point,integration_interval)
    observed_time_points = np.arange(0,final_time_point,interval_between_observations)

    # indices of observed time points
    observed_time_idx = np.round(observed_time_points / integration_interval + 1)
    
    state_true = state[observed_time_idx.astype(int),:]
    idx = [i for i in range(len(state_symbols)) if state_symbols[i] in observed_states]
    state_true = state_true[:,idx]

    # add normally distributed noise with variance $\sigma$ to the true state trajectories
    observations = state_true + np.sqrt(obs_variance) * np.random.randn(observed_time_idx.shape[0],state_true.shape[1])
    
    observed_time_points = integration_time_points[observed_time_idx.astype(int)]
    
    return observations, observed_time_points


# In[5]:

# ## Simulate the State Dynamics with given ODE Parameters


def simulate_state_dynamics(simulation,time_points,symbols,odes,color_idx,*args):
    
    # simulate state trajectories by numerical integration
    state = numerical_integration(odes,time_points.true,simulation.initial_states,simulation.ode_param)
    
    # simulate state observations
    if len(args)==0:
        observations, observed_time_points = simulate_state_observations(\
                state,symbols.state,simulation.final_time_point,simulation.interval_between_observations,\
                simulation.integration_interval,simulation.obs_variance,simulation.observed_states)
    else:
        observations = args[0]
        observed_time_points = args[1]
    
    # mapping between observation- and state trajectories
    obs_to_state_relations = mapping_between_observation_and_state_trajectories(time_points.for_estimation,\
                                                                                observed_time_points,symbols.state,\
                                                                                simulation.observed_states)
    
    
    ## Plotting
    
    # indices of observed states
    observed_state_idx = [u for u in range(len(symbols.state)) if symbols.state[u] in simulation.observed_states]    
    cmap = plt.get_cmap("tab10")
    fig = plt.figure(num=None, figsize=(10, 8), dpi=80)
    plt.subplots_adjust(hspace=0.5)
    handle=[[] for i in range(len(symbols.state))]
    for u in range(len(symbols.state)):
        handle[u] = fig.add_subplot(len(symbols.param),1,u+1)
        handle[u].plot(time_points.true, state[:,u],color=cmap(color_idx))
        plt.xlabel('time',fontsize=12), plt.title(symbols.state[u],position=(0.02,1))
    u2=0
    for u in observed_state_idx: 
        handle[u].plot(observed_time_points, observations[:,u2],'*',markersize=4,color=cmap(1))
        u2 += 1 
        
    if len(symbols.state)==3:
        fig = plt.figure(num=None, figsize=(10, 8), dpi=80)
        ax = fig.gca(projection='3d')
        ax.plot(state[:,0],state[:,1],state[:,2],color=cmap(color_idx))
        ax.set_xlabel(symbols.state[0],fontsize=25)
        ax.set_ylabel(symbols.state[1],fontsize=25)
        ax.set_zlabel(symbols.state[2],fontsize=25)
        ax.set_title('Phase Plot',fontsize=25)
    else:
        fig = plt.figure(num=None, figsize=(6, 3), dpi=80)
        ax = fig.add_subplot(111)
        ax.plot(state[:,0],state[:,1],color=cmap(color_idx))
        ax.set_xlabel(symbols.state[0],fontsize=25)
        ax.set_ylabel(symbols.state[1],fontsize=25)
        ax.set_title('Phase Plot',fontsize=25)
        
    #plt.show()
    
    return state, observations, observed_time_points, obs_to_state_relations


# In[6]:

# ## Mapping between Observations and States
    

def mapping_between_observation_and_state_trajectories(given_time_points,observed_time_points,state_symbols,\
                                                       observed_states):

    # Euclidean distance between observed time_points and given time points
    dist = cdist(given_time_points.reshape(-1,1),observed_time_points.reshape(-1,1),'euclidean')
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