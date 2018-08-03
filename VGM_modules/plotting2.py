
# # Plotting
# 
# #### Authors: Nico S. Gorbach and Stefan Bauer

import matplotlib.pyplot as plt
import numpy as np

# In[1]:


def plot_ode_parameters(ode_param1,color_idx=[1,0],*args):
    
    # number of ode parameters
    numb_ode_param = len(ode_param1.index)
    
    # generate RGB colors
    cmap = plt.get_cmap("tab10")
    
    # plot
    plt.figure(num=None, figsize=(7, 4), dpi=80)
    ax = plt.subplot(111)
    if len(args) == 0:
        ax.bar(np.array(range(numb_ode_param)),ode_param1.values.flatten(),color=cmap(color_idx[0]),width=0.2,label='true')
    else:
        ode_param2 = args[0]  # unpack args
        ax.bar(np.array(range(numb_ode_param))+0.12,ode_param1.values.flatten(),color=cmap(color_idx[0]),width=0.2,label='true')
        ax.bar(np.array(range(numb_ode_param))-0.12,ode_param2.values.flatten(),color=cmap(color_idx[1]),width=0.2,label='estimate')
    
    # customize plot
    plt.title('ODE Parameters',fontsize=18)
    plt.xticks(range(numb_ode_param),['$\%s$' % symbol for symbol in ode_param1.index],fontsize=18)
    ax.legend(fontsize=12)

    return

# In[2]:
    

def plot_states(trajectories,):
    
    numb_states = 3
    
    # generate RGB colors
    cmap = plt.get_cmap("tab10")
    
    # initialize
    fig = plt.figure(num=None, figsize=(10, 8), dpi=80)
    plt.subplots_adjust(hspace=0.5)
    handle=[[] for i in range(numb_hidden_states)]
    
    for u in range(numb_hidden_states):
        
        handle[u] = fig.add_subplot(numb_hidden_states,1,u+1)
        
        for label,color,traj in enumerate(trajectories):
            handle[u].plot(traj,color=color,label=label)
    
    return

# In[3]:
    

def plot_trajectories(time_points,*arg):

    prior_state_sample1 = arg[0]
    prior_state_sample2 = arg[1]
    
    # generate RGB colors
    cmap = plt.get_cmap("tab10")
    
    # setup figure
    fig = plt.figure(num=None, figsize=(7, 4), dpi=80)
    plt.subplots_adjust(hspace=0.5)
    handle = fig.add_subplot(111)
    
    #plot 
    handle.plot(time_points, prior_state_sample1,color=cmap(4),label='trajectory sample 1')
    handle.plot(time_points, prior_state_sample2,color=cmap(6),label='trajectory sample 2')
    plt.xlabel('time',fontsize=18), plt.title('Prior State Trajectory Samples',position=(0.5,1),fontsize=18)
    handle.legend(fontsize=12)
    
    return