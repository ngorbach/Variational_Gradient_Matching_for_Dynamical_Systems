
# # Proxies for ODE Parameters and States
# 
# #### Authors: Nico S. Gorbach and Stefan Bauer

# In[1]:

# ## Import Numpy and Matplotlib Modules

import numpy as np
import matplotlib.pyplot as plt
import warnings

# In[2]:

# ## Determine proxy for each individual state

# \begin{align}
# \hat{q}~(\boldsymbol\theta) ~ \propto ~ \exp \bigg( ~E_{Q_{-\boldsymbol\theta}}  
# \ln \mathcal{N}\left(\boldsymbol\theta ; \left( \mathbf{B}_{\boldsymbol\theta}^T 
# \mathbf{B}_{\boldsymbol\theta} \right)^{-1} \left( \sum_k \mathbf{B}_{\boldsymbol\theta k}^T ~ 
# \left( {'\mathbf{C}_{\mathbf{\phi} k}} \mathbf{C}_{\mathbf{\phi} k}^{-1} \mathbf{X}_k -
# \mathbf{b}_{\boldsymbol\theta k} \right) \right), ~ \mathbf{B}_{\boldsymbol\theta}^+ ~
# (\mathbf{A} + \mathbf{I}\gamma) ~ \mathbf{B}_{\boldsymbol\theta}^{+T} \right) ~\bigg)
# \end{align}


def proxy_for_ode_parameters(state_proxy,locally_linear_odes,dC_times_inv_C,ode_param_symbols,ode_param_true):

    # initialization
    local_mean = np.zeros((len(ode_param_symbols),1))
    local_scaling = np.zeros((len(ode_param_symbols),len(ode_param_symbols)))
    
    # iterate through each ODE
    for k in range(len(locally_linear_odes.ode_param.B)):
          
        # determine vectors B and b
        B = locally_linear_odes.ode_param.B[k](*np.append(state_proxy,\
                                           np.zeros((state_proxy.shape[0],1)),axis=1).T).T.reshape(state_proxy.shape[0],-1)
        b = locally_linear_odes.ode_param.b[k](*np.append(state_proxy,\
                                           np.zeros((state_proxy.shape[0],1)),axis=1).T).T.reshape(state_proxy.shape[0],-1)
        
        # The Moore-Penrose inverse of $\mathbf{B}_{\boldsymbol\theta}$: $\mathbf{B}_{\boldsymbol\theta}^+ 
        # := \left(\mathbf{B}_{\boldsymbol\theta}^T \mathbf{B}_{\boldsymbol\theta} \right)^{-1} \mathbf{B}_{\boldsymbol\theta}^T$
        #
        # local mean: $\mathbf{B}_{\boldsymbol\theta k}^T ~ \left( {'\mathbf{C}_{\mathbf{\phi}_k}} 
        # \mathbf{C}_{\mathbf{\phi} k}^{-1} \mathbf{X}_k - \mathbf{b}_{\boldsymbol\theta k} \right)$   
        local_mean += np.dot(B.T,np.dot(dC_times_inv_C,state_proxy[:,k]).reshape(-1,1) - b)
        local_scaling += np.dot(B.T,B)
      
    # (global) mean of parameter proxy distribution:
    # $\left( \mathbf{B}_{\mathbf{\theta}}^T \mathbf{B}_{\mathbf{\theta}} \right)^{-1}
    # \left( \sum_k \mathbf{B}_{\mathbf{\theta} k}^T ~ \left( {'\mathbf{C}_{\mathbf{\phi} k}}
    # \mathbf{C}_{\mathbf{\phi} k}^{-1} \mathbf{X}_k - \mathbf{b}_{\mathbf{\theta} k} \right) \right)$
    global_mean = np.linalg.solve(local_scaling,local_mean)
    
    # plotting
    cmap = plt.get_cmap("tab10")
    plt.figure(num=None, figsize=(7, 4), dpi=80)
    ax = plt.subplot(111)
    ax.bar(np.asarray(range(len(global_mean)))+0.12,ode_param_true,color=cmap(1),width=0.2,label='true')
    ax.bar(np.asarray(range(len(global_mean)))-0.12,global_mean.T[0].tolist(),color=cmap(0),width=0.2,label='estimated')
    plt.title('ODE Parameters',fontsize=18), plt.xticks(range(len(global_mean)),ode_param_symbols,fontsize=15)
    ax.legend(fontsize=12)
    # plt.show()
    
    return global_mean

# In[4]:

# ## Determine proxy for each individual state
# 
# The proxy for an individual state is given 
# \begin{align}
# {\hat{q}} ~ (\mathbf{x}_u) ~ \propto ~ \exp\big( ~ E_{Q_{-u}} \ln
# \mathcal{N}\left(\mathbf{x}_u ; \left( \mathbf{B}_{u} \mathbf{B}_{u}^T
# \right)^{-1} \left( - \sum_k \mathbf{B}_{uk}^T \mathbf{b}_{uk} \right),
# ~\mathbf{B}_{u}^{+} ~ (\mathbf{A} + \mathbf{I}\gamma) ~ \mathbf{B}_u^{+T} \right)   
# + E_{Q_{-u}} \ln
# \mathcal{N}\left(\mathbf{x}_u ; \boldsymbol\mu_u(\mathbf{Y}), \boldsymbol\Sigma_u \right) \big)
# \end{align}
    
def proxy_for_ind_states(state_proxy,ode_param_proxy,locally_linear_odes,dC_times_inv_C,state_symbols,\
                         observed_states,state_couplings,time_points,simulation,\
                         GP_post_mean,GP_post_inv_cov,clamp_states_to_observation_fit,observations):
     
    # numer of hidden states
    numb_hidden_states = len(state_symbols)
     
    if clamp_states_to_observation_fit==1:
    # indices of observed states
        hidden_states_to_infer = [u for u in range(len(state_symbols)) if state_symbols[u] not in observed_states]
    else:
        hidden_states_to_infer = range(len(state_symbols))     
    
    # initialization
    global_mean = state_proxy[:]
    # iterate through each unobserved state
    for u in hidden_states_to_infer:
        
        # initialization
        local_mean = np.zeros((state_proxy.shape[0],1))
        local_scaling = np.zeros((state_proxy.shape[0],state_proxy.shape[0]))
        
        # iterate through each ODE
        for k in range(numb_hidden_states):
            
            # determine matrices $\mathbf{R}$ and $\mathbf{r}$
            R_tmp = locally_linear_odes.state.R[u][k](ode_param_proxy[0],ode_param_proxy[1],ode_param_proxy[2],\
                                               ode_param_proxy[3],state_proxy[:,0],state_proxy[:,1],\
                                               np.ones((state_proxy.shape[0]))).T
            #R_tmp = locally_linear_ODEs.state.R[u][k](*[ode_param_proxy,state_proxy.T])
            R = np.zeros((state_proxy.shape[0],state_proxy.shape[0]))
            np.fill_diagonal(R,R_tmp)
            
        
            r = locally_linear_odes.state.r[u][k](ode_param_proxy[0],ode_param_proxy[1],ode_param_proxy[2],\
                                           ode_param_proxy[3],state_proxy[:,0],state_proxy[:,1],\
                                           np.ones((state_proxy.shape[0]))).T
            if len(r)==1:
                tmp = r[:]
                r = np.asarray([tmp[:] for i in range(state_proxy.shape[0])])   
            r = r.reshape(state_proxy.shape[0],-1)
            
            # Define matrices B and b such that $\mathbf{B}_{uk} \mathbf{x}_u +
            # \mathbf{b}_{uk} \stackrel{!}{=} \mathbf{f}_k(\mathbf{X},\mathbf{\theta}) -
            # {'\mathbf{C}}_{\mathbf{\phi}_{k}} \mathbf{C}_{\mathbf{\phi}_{k}}^{-1} \mathbf{X}$
            if k != u:
                B = R[:]
                b = r - np.dot(dC_times_inv_C,state_proxy[:,k]).reshape(-1,1)
            else:
                B = R - dC_times_inv_C
                b = r[:]
            
            # local mean: $\mathbf{B}_{uk}^T \left(\mathbf{\epsilon_0}^{(k)} -\mathbf{b}_{uk}
            local_mean += -np.dot(B.T,b)
            local_scaling += np.dot(B.T,B)
            
        # Mean of state proxy distribution (option: Moore-penrose inverse example): 
        # $\left( \mathbf{B}_{u} \mathbf{B}_{u}^T \right)^{-1} \sum_k
        # \mathbf{B}_{uk}^T \left(\mathbf{\epsilon_0}^{(k)} -\mathbf{b}_{uk} \right)$   
        # global_mean[:,u] = np.squeeze(np.linalg.solve(local_scaling,local_mean)) 
        global_mean[:,u] = np.squeeze(np.linalg.solve(local_scaling + GP_post_inv_cov[u],local_mean\
                   + np.dot(GP_post_inv_cov[u],GP_post_mean[:,u]).reshape(-1,1)))
     
    # plotting
    warnings.simplefilter("ignore")
    cmap = plt.get_cmap("tab10")
    fig = plt.figure(num=None, figsize=(10, 8), dpi=80)
    plt.subplots_adjust(hspace=0.5)
    handle=[[] for i in range(len(state_symbols))]
    for u in range(len(state_symbols)):
        handle[u] = fig.add_subplot(len(state_symbols),1,u+1)
        plt.subplot(len(state_symbols),1,u+1)
        handle[u].plot(time_points.true, simulation.state[:,u],color=cmap(1),label='true')
        handle[u].plot(time_points.for_estimation, global_mean[:,u],color=cmap(0),label='estimated')
        plt.xlabel('time',fontsize=18), 
        if state_symbols[u] in simulation.observed_states:
            plt.title('observed %s' % state_symbols[u],loc='left')
        else:
            plt.title('unobserved %s' % state_symbols[u],loc='left')
        handle[u].legend(fontsize=12)
    observed_state_idx = [u for u in range(len(state_symbols)) if state_symbols[u] in simulation.observed_states]    
    u2=0
    for u in observed_state_idx: 
        handle[u].plot(time_points.observed, simulation.observations[:,u2],'*',markersize=7,color=cmap(1),label='observed')
        handle[u].legend(fontsize=12)
        u2 += 1
        
    # phase space    
    if numb_hidden_states==3:
        fig = plt.figure(num=None, figsize=(10, 8), dpi=80)
        ax = fig.gca(projection='3d')
        ax.plot(global_mean[:,0],global_mean[:,1],global_mean[:,2],color=cmap(0),label='estimated')
        ax.set_xlabel(state_symbols[0],fontsize=18)
        ax.set_ylabel(state_symbols[1],fontsize=18)
        ax.set_zlabel(state_symbols[2],fontsize=18)
        ax.set_title('Phase Plot',fontsize=18)
        ax.legend(fontsize=15)
        if len(simulation.observed_states) == numb_hidden_states:
            ax.plot(observations[:,0],observations[:,1],observations[:,2],'*',markersize=7,color=cmap(1),label='observed')
            ax.legend(fontsize=12)
    else:
        fig = plt.figure(num=None, figsize=(6, 3), dpi=80)
        ax = fig.add_subplot(111)
        ax.plot(global_mean[:,0],global_mean[:,1],color=cmap(0),label='estimated')
        ax.set_xlabel(state_symbols[0],fontsize=18)
        ax.set_ylabel(state_symbols[1],fontsize=18)
        ax.set_title('Phase Space',fontsize=18)
        ax.legend(fontsize=15)
        if len(simulation.observed_states) == numb_hidden_states:
            ax.plot(observations[:,0],observations[:,1],'*',markersize=7,color=cmap(1),label='observed')
            ax.legend(fontsize=12)
            
    # plt.show()
         
    return global_mean