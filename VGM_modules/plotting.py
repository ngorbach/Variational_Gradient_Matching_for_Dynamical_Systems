
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
    

def plot_states(trajectory1,observations,label=['numerical_intergration','VGM estimate','observed'],color_idx=[1,0,1],traj_idx=1,*args):
    
    # number of hidden states
    numb_hidden_states = trajectory1.shape[1]
    
    # generate RGB colors
    cmap = plt.get_cmap("tab10")
    
    # initialize
    fig = plt.figure(num=None, figsize=(10, 8), dpi=80)
    plt.subplots_adjust(hspace=0.5)
    handle=[[] for i in range(numb_hidden_states)]
    
    for u,key in enumerate(trajectory1.columns.values):
        handle[u] = fig.add_subplot(numb_hidden_states,1,u+1)
        
        if len(args)!=0:
            trajectory2 = args[0]  # unpack args
            handle[u].plot(trajectory1[key],color=cmap(color_idx[0]),label=label[0])
            handle[u].plot(trajectory2[key],color=cmap(color_idx[1]),label=label[1]) 
            
            # which trajectory to use in the phase plot
            if traj_idx == 1:
                traj = trajectory1
                lab = label[0]
                cidx = color_idx[0]
            else:
                traj = trajectory2
                lab = label[1]
                cidx = color_idx[1]
                
        else:
            handle[u].plot(trajectory1[key],color=cmap(color_idx[0]),label=label[0])
            traj = trajectory1
            lab = label[0]
            cidx = color_idx[0]
        
        if key in observations.columns.values:
            handle[u].plot(observations[key],'*',markersize=7,color=cmap(color_idx[2]),label=label[2])
            
        if key in observations.columns.values:
            plt.title('observed $' + key + '$',loc='left',fontsize=18)
        else:
            plt.title('unobserved $' + key + '$',loc='left',fontsize=18)
        handle[u].legend(fontsize=12)
        plt.xlabel('time',fontsize=18)
        
            
    # phase space
    if numb_hidden_states==3:
        fig = plt.figure(num=None, figsize=(10, 8), dpi=80)
        ax = fig.gca(projection='3d')
        ax.plot(traj.iloc[:,0],traj.iloc[:,1],traj.iloc[:,2],color=cmap(cidx),label=lab)
        ax.set_xlabel(traj.columns.values[0],fontsize=18)
        ax.set_ylabel(traj.columns.values[1],fontsize=18)
        ax.set_zlabel(traj.columns.values[2],fontsize=18)
        ax.set_title('Phase Space',fontsize=18)
        ax.legend(fontsize=12)
        if len(observations.columns.values) == numb_hidden_states:
            ax.plot(observations.iloc[:,0],observations.iloc[:,1],observations.iloc[:,2],'*',markersize=7,color=cmap(color_idx[2]),label=label[2])
            ax.legend(fontsize=12)
    
    else:
        fig = plt.figure(num=None, figsize=(6, 3), dpi=80)
        ax = fig.add_subplot(111)
        ax.plot(traj.iloc[:,0],traj.iloc[:,1],color=cmap(cidx),label=lab)
        ax.set_xlabel('$%s$' % traj.columns.values[0],fontsize=18)
        ax.set_ylabel('$%s$' % traj.columns.values[1],fontsize=18)
        ax.set_title('Phase Space',fontsize=18)
        ax.legend(fontsize=12)
        if len(observations.columns.values) == numb_hidden_states:
            ax.plot(observations.iloc[:,0],observations.iloc[:,1],'*',markersize=7,color=cmap(color_idx[2]),label=label[2])
            ax.legend(fontsize=12)
            
    return 

def plot_states2(trajectories,numb_subplots):
    
    
    # initialize
    fig = plt.figure(num=None, figsize=(10, 8), dpi=80)
    plt.subplots_adjust(hspace=0.5)
    handle=[[] for i in range(numb_subplots)]
    
    for u in range(numb_subplots):
        handle[u] = fig.add_subplot(numb_subplots,1,u+1)

        
    for traj,label,color,style in iter(trajectories):
        
        for u,key in enumerate(traj.columns.values):
            
            handle[u].plot(traj[key],style,color=color,label=label)
            handle[u].set_xlabel('time',fontsize=18)
            handle[u].set_title('observed $' + key + '$',loc='left',fontsize=18)
            handle[u].legend(fontsize=12)
            
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
        