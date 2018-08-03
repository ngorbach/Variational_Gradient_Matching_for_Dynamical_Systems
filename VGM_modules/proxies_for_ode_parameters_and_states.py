
# # Proxies for ODE Parameters and States
# 
# #### Authors: Nico S. Gorbach and Stefan Bauer

# In[1]:

# ## Import Numpy and Matplotlib Modules

import numpy as np
import matplotlib.pyplot as plt
import warnings
import pandas as pd
import scipy.integrate as integrate
from plotting import *

# In[2]:

# ## Determine proxy for ODE parameters

# \begin{align}
# \hat{q}~(\boldsymbol\theta) ~ \propto ~ \exp \bigg( ~E_{Q_{-\boldsymbol\theta}}  
# \ln \mathcal{N}\left(\boldsymbol\theta ; \left( \mathbf{B}_{\boldsymbol\theta}^T 
# \mathbf{B}_{\boldsymbol\theta} \right)^{-1} \left( \sum_k \mathbf{B}_{\boldsymbol\theta k}^T ~ 
# \left( {'\mathbf{C}_{\mathbf{\phi} k}} \mathbf{C}_{\mathbf{\phi} k}^{-1} \mathbf{X}_k -
# \mathbf{b}_{\boldsymbol\theta k} \right) \right), ~ \mathbf{B}_{\boldsymbol\theta}^+ ~
# (\mathbf{A} + \mathbf{I}\gamma) ~ \mathbf{B}_{\boldsymbol\theta}^{+T} \right) ~\bigg)
# \end{align}


def proxy_for_ode_parameters(state_proxy,locally_linear_odes,dC_times_inv_C,ode_param_symbols,ode_param_true):

    '''Estimates proxy for ODE parameters'''
    
    # error handeling
    if state_proxy.shape[0] != dC_times_inv_C.shape[0]:
        raise ValueError('Either state_proxy or dC_times_invC have the wrong shape')
    elif dC_times_inv_C.shape[0] != dC_times_inv_C.shape[1]: 
        raise ValueError('dC_times_invC is not a square matrix')
    
    # unpack state_proxy
    state_proxy = state_proxy.values
    
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
        local_mean += B.T.dot(dC_times_inv_C.dot(state_proxy[:,k]).reshape(-1,1) - b)
        local_scaling += B.T.dot(B)
      
    # perturb local_scaling if necessary
    eigval,v = np.linalg.eig(local_scaling)
    if any(eigval < 0.001):
        for i in range(local_scaling.shape[0]):
            local_scaling[i,i] += abs((max(eigval) - min(eigval)))/10000 *np.random.rand(1,1)
    
    # (global) mean of parameter proxy distribution:
    # $\left( \mathbf{B}_{\mathbf{\theta}}^T \mathbf{B}_{\mathbf{\theta}} \right)^{-1}
    # \left( \sum_k \mathbf{B}_{\mathbf{\theta} k}^T ~ \left( {'\mathbf{C}_{\mathbf{\phi} k}}
    # \mathbf{C}_{\mathbf{\phi} k}^{-1} \mathbf{X}_k - \mathbf{b}_{\mathbf{\theta} k} \right) \right)$   
    global_mean = np.linalg.solve(local_scaling,local_mean)
 
    # pack global_mean into pandas DataFrame
    ode_param_proxy = pd.DataFrame(global_mean,columns=['value'],index=map(str,ode_param_symbols)).rename_axis('ODE parameter symbols')
    
    # plotting ODE parameters
    plot_ode_parameters(ode_param_true,[1,0],ode_param_proxy)
    
    return ode_param_proxy

# In[4]:

# ## Determine proxy for each individual state
# 
# The proxy for an individual state is given 
# \begin{align}
# {\hat{q}} ~ (\mathbf{x}_u) ~ \propto ~ \exp\big( ~ E_{Q_{-u}} \ln
# \mathcal{N}\left(\mathbf{x}_u ; \left( \mathbf{B}_{u} \mathbf{B}_{u}^T
# \right)^{-1} \left( - \sum_k \mathbf{B}_{uk}^T \mathbf{b}_{uk} \right),
# ~\mathbf{B}_{u}^{+} ~ (\mathbf{A} + \mathbf{I}\gamma) ~ \mathbf{B}_u^{+T} \right) + E_{Q_{-u}} \ln
# \mathcal{N}\left(\mathbf{x}_u ; \boldsymbol\mu_u(\mathbf{Y}), \boldsymbol\Sigma_u \right) \big)
# \end{align}
    
def proxy_for_ind_states(state_proxy,ode_param_proxy,locally_linear_odes,dC_times_inv_C,\
                         state_couplings,GP_post_mean,GP_post_inv_cov,observations,true_states,\
                         clamp_states_to_observation_fit=True):
    
    '''Estimates proxy for each individual state'''
    
    
    # unpack ode_param_proxy and _state_proxy
    ode_param_proxy = ode_param_proxy.values
    time_points = np.array(state_proxy.index)
    state_symbols = state_proxy.columns.values
    state_proxy = state_proxy.values
    
    # numer of hidden states
    numb_hidden_states = len(state_symbols)
    
    if clamp_states_to_observation_fit==True:
    # indices of observed states
        hidden_states_to_infer = [u for u in range(len(state_symbols)) if state_symbols[u] not in observations.columns.values]
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
            R_vec = locally_linear_odes.state.R[u][k](ode_param_proxy[0],ode_param_proxy[1],ode_param_proxy[2],\
                                               state_proxy[:,0],state_proxy[:,1],state_proxy[:,2],\
                                               np.ones((state_proxy.shape[0]))).T
            # R_tmp = locally_linear_ODEs.state.R[u][k](*[ode_param_proxy,state_proxy.T])
            R = np.zeros((state_proxy.shape[0],state_proxy.shape[0]))
            np.fill_diagonal(R,R_vec)
            
        
            r = locally_linear_odes.state.r[u][k](ode_param_proxy[0],ode_param_proxy[1],ode_param_proxy[2],\
                                           state_proxy[:,0],state_proxy[:,1],state_proxy[:,2],\
                                           np.ones((state_proxy.shape[0]))).T
            if len(r)==1:
                r = np.array([r[:] for i in range(state_proxy.shape[0])])   
            r = r.reshape(state_proxy.shape[0],-1)
            
            # Define matrices B and b such that $\mathbf{B}_{uk} \mathbf{x}_u +
            # \mathbf{b}_{uk} \stackrel{!}{=} \mathbf{f}_k(\mathbf{X},\mathbf{\theta}) -
            # {'\mathbf{C}}_{\mathbf{\phi}_{k}} \mathbf{C}_{\mathbf{\phi}_{k}}^{-1} \mathbf{X}$
            if k != u:
                B = R[:]
                b = r - dC_times_inv_C.dot(state_proxy[:,k]).reshape(-1,1)
            else:
                B = R - dC_times_inv_C
                b = r[:]
            
            # local mean: $\mathbf{B}_{uk}^T \left(\mathbf{\epsilon_0}^{(k)} -\mathbf{b}_{uk}
            local_mean += -B.T.dot(b)
            local_scaling += B.T.dot(B)
            
        # Mean of state proxy distribution (option: Moore-penrose inverse example): 
        # $\left( \mathbf{B}_{u} \mathbf{B}_{u}^T \right)^{-1} \sum_k
        # \mathbf{B}_{uk}^T \left(\mathbf{\epsilon_0}^{(k)} -\mathbf{b}_{uk} \right)$   
        #global_mean[:,u] = np.squeeze(np.linalg.solve(local_scaling,local_mean)) 
        global_mean[:,u] = np.squeeze(np.linalg.solve(local_scaling + GP_post_inv_cov[u],local_mean\
                   + np.dot(GP_post_inv_cov[u],GP_post_mean.iloc[:,u]).reshape(-1,1)))
    
    # pack global_mean into pandas DataFrame
    state_proxy = pd.DataFrame(global_mean,columns=map(str,state_symbols),index=time_points).rename_axis('time')
        
    # plot state proxies
    plot_states(true_states,observations,['true','estimate','observed'],[1,0,1],2,state_proxy)
        
         
    return state_proxy