
# # Plotting
# 
# #### Authors: Nico S. Gorbach and Stefan Bauer

import matplotlib.pyplot as plt
import numpy as np

# In[1]:


def plot_ode_parameters(ode_param,fig_shape=(7, 4),color_idx=[1,0],plot_name=None,ode_param_estimates=None):
    
    # number of ode parameters
    numb_ode_param = len(ode_param.index)
    
    # generate RGB colors
    cmap = plt.get_cmap("tab10")
    
    # plot
    plt.figure(num=None, figsize=fig_shape, dpi=80)
    
    ax = plt.subplot(111)
    if ode_param_estimates is None:
        ax.bar(np.array(range(numb_ode_param)),ode_param.values.flatten(),color=cmap(color_idx[0]),width=0.2,label='true')
    else:
        ax.bar(np.array(range(numb_ode_param))+0.12,ode_param.values.flatten(),color=cmap(color_idx[0]),width=0.2,label='true')
        ax.bar(np.array(range(numb_ode_param))-0.12,ode_param_estimates.values.flatten(),color=cmap(color_idx[1]),width=0.2,label='estimate')
    
    # customize plot
    plt.title('ODE Parameters',fontsize=24)
    if len(ode_param.index) < 20:
        plt.xticks(range(numb_ode_param),['$%s$' % symbol[1:] for symbol in ode_param.index])
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    ax.legend(fontsize=18)
    
    if plot_name is not None:
        plt.savefig('results/' + plot_name,transparent=True,bbox_inches='tight')

    return

# In[2]:
    

def plot_states(trajectory1,observations,fig_shape,label=['numerical_intergration','VGM estimate','observed'],color_idx=[1,0,1],traj_idx=1,sigma=None,traj_arg=None,plot_name=None,extra_track=None):
    
    # number of hidden states
    numb_hidden_states = trajectory1.shape[1]
    
    # generate RGB colors
    cmap = plt.get_cmap("tab10")
    
    # initialize
    fig = plt.figure(num=None, figsize=fig_shape, dpi=80)
    plt.subplots_adjust(hspace=0.5)
    handle=[[] for i in range(numb_hidden_states)]
    plt.ion()
    fig.show()
    fig.canvas.draw()
    for u,key in enumerate(trajectory1.columns.values):
        handle[u] = fig.add_subplot(numb_hidden_states,1,u+1)
        
        if traj_arg is not None:
            trajectory2 = traj_arg  # unpack args
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
        
        if sigma is not None:
            var = 0.1*np.diag(sigma[:,:,u])
            handle[u].fill(np.concatenate([traj.index, traj.index[::-1]]),np.concatenate([traj[key] - 5 * var,(traj[key] + 5 * var)[::-1]]),alpha=.1,fc='b', ec='None')    
        
        if key in observations.columns.values:
            handle[u].plot(observations[key],'.',markersize=10,color=cmap(color_idx[2]),label=label[2])
            
        if key in observations.columns.values:
            plt.title('observed $' + key[1:] + '$',loc='left',fontsize=18)
        else:
            plt.title('unobserved $' + key[1:] + '$',loc='left',fontsize=18)
            if extra_track is not None:
                handle[u].plot(extra_track[key],color=cmap(color_idx[2]),label=label[0])
            
        handle[u].legend(fontsize=12,loc=9,bbox_to_anchor=(1.4, 1.05))
        plt.xlabel('time',fontsize=18)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        #plt.yticks(fontsize=18)
        
        #plt.show()
        
        if plot_name is not None:
            plt.savefig('results/' + plot_name,transparent=True,bbox_inches='tight')
        
#    # phase space
#    if numb_hidden_states==3:
#        fig = plt.figure(num=None, figsize=(10, 8), dpi=80)
#        ax = fig.gca(projection='3d')
#        ax.plot(traj.iloc[:,0],traj.iloc[:,1],traj.iloc[:,2],color=cmap(cidx),label=lab)
#        ax.set_xlabel(traj.columns.values[0],fontsize=18)
#        ax.set_ylabel(traj.columns.values[1],fontsize=18)
#        ax.set_zlabel(traj.columns.values[2],fontsize=18)
#        ax.set_title('Phase Space',fontsize=18)
#        ax.legend(fontsize=12)
#        if len(observations.columns.values) == numb_hidden_states:
#            ax.plot(observations.iloc[:,0],observations.iloc[:,1],observations.iloc[:,2],'.',markersize=7,color=cmap(color_idx[2]),label=label[2])
#            ax.legend(fontsize=12)
#    
#    else:
#        fig = plt.figure(num=None, figsize=(6, 3), dpi=80)
#        ax = fig.add_subplot(111)
#        ax.plot(traj.iloc[:,0],traj.iloc[:,1],color=cmap(cidx),label=lab)
#        ax.set_xlabel('$%s$' % traj.columns.values[0],fontsize=18)
#        ax.set_ylabel('$%s$' % traj.columns.values[1],fontsize=18)
#        ax.set_title('Phase Space',fontsize=18)
#        ax.legend(fontsize=12)
#        if len(observations.columns.values) == numb_hidden_states:
#            ax.plot(observations.iloc[:,0],observations.iloc[:,1],'.',markersize=7,color=cmap(color_idx[2]),label=label[2])
#            ax.legend(fontsize=12)
            
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
    fig = plt.figure(num=None, figsize=(10, 4), dpi=80)
    plt.subplots_adjust(hspace=0.5)
    handle = fig.add_subplot(111)
    
    #plot 
    handle.plot(time_points, prior_state_sample1,color=cmap(4),label='sample 1')
    handle.plot(time_points, prior_state_sample2,color=cmap(6),label='sample 2')
    plt.xlabel('time',fontsize=18), plt.title('Prior Samples of State Trajectories',position=(0.5,1),fontsize=18)
    handle.legend(fontsize=12)
    
    plt.savefig('results/prior_samples_of_state.png',transparent=True)
    
    return
        