
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
import scipy.optimize.nnls as nnls
from sklearn.linear_model import Lasso
from plotting import *
from scipy.optimize import least_squares
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

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


def proxy_for_ode_parameters(state_proxy,ext_input,locally_linear_odes,dC_times_inv_C,eps_cov,ode_param_symbols,ode_param_true,\
                             odes,odes_gradient_param,constraints=None,optimizer='minimizer'):

    '''Estimates proxy for ODE parameters'''
    
    
    # error handeling
    if state_proxy.shape[0] != dC_times_inv_C.shape[0]:
        raise ValueError('Either state_proxy or dC_times_invC have the wrong shape')
    elif dC_times_inv_C.shape[0] != dC_times_inv_C.shape[1]: 
        raise ValueError('dC_times_invC is not a square matrix')
    
    # unpack state_proxy
    state_proxy = state_proxy.values
    
    if optimizer == 'analytical':
        
        # initialization
        local_mean = np.zeros((len(ode_param_symbols)))
        local_scaling = np.zeros((len(ode_param_symbols),len(ode_param_symbols)))
        global_cov = np.zeros((len(ode_param_symbols),len(ode_param_symbols)))
        
        state = np.append(state_proxy,ext_input,axis=1)
        state = np.append(state,np.ones((state_proxy.shape[0],1)),axis=1).T
        # iterate through each ODE
        for k in range(len(locally_linear_odes.ode_param.B)):
              
            # determine vectors B and b
            B = np.squeeze(locally_linear_odes.ode_param.B[k](*state)).T
            b = np.squeeze(locally_linear_odes.ode_param.b[k](*state))
            
            # The Moore-Penrose inverse of $\mathbf{B}_{\boldsymbol\theta}$: $\mathbf{B}_{\boldsymbol\theta}^+ 
            # := \left(\mathbf{B}_{\boldsymbol\theta}^T \mathbf{B}_{\boldsymbol\theta} \right)^{-1} \mathbf{B}_{\boldsymbol\theta}^T$
            #
            # local mean: $\mathbf{B}_{\boldsymbol\theta k}^T ~ \left( {'\mathbf{C}_{\mathbf{\phi}_k}} 
            # \mathbf{C}_{\mathbf{\phi} k}^{-1} \mathbf{X}_k - \mathbf{b}_{\boldsymbol\theta k} \right)$   
    
            local_mean += B.T.dot(dC_times_inv_C.dot(state_proxy[:,k]) - b)
            local_scaling += B.T.dot(B)
    
            global_cov += B.T.dot(eps_cov.dot(B))
         
        # perturb local_scaling if necessary
        eigval,v = np.linalg.eig(local_scaling)
        if any(eigval < 0.001):
            for i in range(local_scaling.shape[0]):
                local_scaling[i,i] += abs((max(eigval) - min(eigval)))/10000 *np.random.rand(1,1)
        
        # (global) mean of parameter proxy distribution:
        # $\left( \mathbf{B}_{\mathbf{\theta}}^T \mathbf{B}_{\mathbf{\theta}} \right)^{-1}
        # \left( \sum_k \mathbf{B}_{\mathbf{\theta} k}^T ~ \left( {'\mathbf{C}_{\mathbf{\phi} k}}
        # \mathbf{C}_{\mathbf{\phi} k}^{-1} \mathbf{X}_k - \mathbf{b}_{\mathbf{\theta} k} \right) \right)$   
        
        if constraints == None:
            #global_mean = np.linalg.solve(local_scaling,local_mean)
            linreg = LinearRegression()
            global_mean = linreg.fit(local_scaling,np.squeeze(local_mean)).coef_
            
            #x0 = np.linalg.solve(local_scaling,local_mean)                    
            #global_mean = minimize(squared_loss,x0,args=(state_proxy,dC_times_inv_C.dot(state_proxy).reshape(-1,order='F'),odes)).x
    
        elif constraints == 'nonnegative':
            #global_mean,w = nnls(local_scaling,np.squeeze(local_mean))
            lassoreg = Lasso(alpha=0.00001,positive=True)
            global_mean = lassoreg.fit(local_scaling,np.squeeze(local_mean)).coef_
            
        elif constraints == 'sparse':
            lassoreg = Lasso(alpha=1)
            global_mean = lassoreg.fit(local_scaling,np.squeeze(local_mean)).coef_
        elif constraints == 'sparse+nonnegative':
            lassoreg = Lasso(alpha=1,positive=True)
            global_mean = lassoreg.fit(local_scaling,np.squeeze(local_mean)).coef_
    
    elif optimizer == 'minimizer': 
        x0 = 0.1*np.ones(len(ode_param_symbols))
        alpha=0
        if constraints == None:
            bnds = ((None,None),(None,None),(None,None),(None,None),(None,None),(None,None),(None,None),(None,None),(-1.01,-0.99),(-1.01,-0.99),(-1.01,-0.99))
            global_mean = minimize(squared_loss,x0,bounds=bnds,args=(state_proxy.T,ext_input.T,dC_times_inv_C.dot(state_proxy),odes,odes_gradient_param,alpha),jac=squared_loss_gradient).x
        elif constraints == 'nonnegative':
            #bnds = tuple([(0,None) for i in range(len(ode_param_symbols))])
            bnds = ((0,None),(0,None),(0,None),(0,None),(0,None),(0,None),(0,None),(0,None),(-1.01,-0.99),(-1.01,-0.99),(-1.01,-0.99))
            global_mean = minimize(squared_loss,x0,bounds=bnds,args=(state_proxy.T,ext_input.T,dC_times_inv_C.dot(state_proxy),odes,odes_gradient_param,alpha),jac=squared_loss_gradient).x



    # pack global_mean into pandas DataFrame
    ode_param_proxy = pd.DataFrame(global_mean,columns=['value'],index=map(str,ode_param_symbols)).rename_axis('ODE_parameter_symbols')
    

    # plotting ODE parameters
    plot_ode_parameters(ode_param_true,fig_shape=(10,4),color_idx=[1,0],plot_name='ODE_parameter_estimation',ode_param_estimates=ode_param_proxy)

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
    
def proxy_for_ind_states(state_proxy,ext_input,ode_param_proxy,odes,odes_gradient_states,func,func_gradient,func_true_vals,func_pred_mean,locally_linear_odes,dC_times_inv_C,eps_cov,\
                         state_pred_mean,state_pred_inv_cov,observations,true_states,state_couplings,iter_idx,burnin=0,
                         clamp_states_to_observation_fit=True,constraints=None,optimizer='analytical'):
    
    '''Estimates proxy for each individual state'''
    
    
    # unpack ode_param_proxy and _state_proxy
    ode_param_proxy = ode_param_proxy.values
    time_points = np.array(state_proxy.index)
    state_symbols = state_proxy.columns.values
    state_proxy = state_proxy.values
    

    
    if clamp_states_to_observation_fit==True:
    # indices of observed states
        hidden_states_to_infer = [u for u in range(len(state_symbols)) if state_symbols[u] not in observations.columns.values]
    else:
        hidden_states_to_infer = range(len(state_symbols))   
    
    # initialization
    global_mean = state_proxy[:]
        
    if optimizer == 'analytical':
         
        # initialization
        global_cov = np.zeros((state_proxy.shape[0],state_proxy.shape[0],state_proxy.shape[1]))
        state = np.append(state_proxy,ext_input,axis=1)
        state = np.append(state,np.ones((state_proxy.shape[0],1)),axis=1).T
    
        # iterate through each unobserved state
        for u in hidden_states_to_infer:
            
            # initialization
            local_mean = np.zeros((state_proxy.shape[0]))
            local_scaling = np.zeros((state_proxy.shape[0],state_proxy.shape[0]))
            
            # iterate through each ODE
            #for k in range(len(state_symbols)):
            for k in state_couplings[u]:
    
                R_vec = np.squeeze(locally_linear_odes.state.R[u][k](*ode_param_proxy,*state).T)
                R = np.zeros((state_proxy.shape[0],state_proxy.shape[0]))
                np.fill_diagonal(R,R_vec)
                
                r = np.squeeze(locally_linear_odes.state.r[u][k](*ode_param_proxy,*state).T)
                if r.ndim==0:
                    r = np.array([r for i in range(state_proxy.shape[0])])   
                
                # Define matrices B and b such that $\mathbf{B}_{uk} \mathbf{x}_u +
                # \mathbf{b}_{uk} \stackrel{!}{=} \mathbf{f}_k(\mathbf{X},\mathbf{\theta}) -
                # {'\mathbf{C}}_{\mathbf{\phi}_{k}} \mathbf{C}_{\mathbf{\phi}_{k}}^{-1} \mathbf{X}$
                if k != u:
                    B = R[:]
                    b = r - dC_times_inv_C.dot(state_proxy[:,k])
                else:
                    B = R - dC_times_inv_C
                    b = r[:]
                
                # local mean: $\mathbf{B}_{uk}^T \left(\mathbf{\epsilon_0}^{(k)} -\mathbf{b}_{uk}
                local_mean += -B.T.dot(b)
                local_scaling += B.T.dot(B)
                
#                # perturb local_scaling if necessary
#                eigval,v = np.linalg.eig(local_scaling)
#                if any(eigval < 0.001):
#                    for i in range(local_scaling.shape[0]):
#                        local_scaling[i,i] += abs((max(eigval) - min(eigval)))/1000 *np.random.rand(1,1)
                
                
                global_cov[:,:,u] += B.T.dot(eps_cov.dot(B))
                
            # Mean of state proxy distribution (option: Moore-penrose inverse example): 
            # $\left( \mathbf{B}_{u} \mathbf{B}_{u}^T \right)^{-1} \sum_k
            # \mathbf{B}_{uk}^T \left(\mathbf{\epsilon_0}^{(k)} -\mathbf{b}_{uk} \right)$   
            if constraints == None:
                global_mean[:,u] = np.squeeze(np.linalg.solve(local_scaling,local_mean)) 
#                global_mean[:,u] = np.squeeze(np.linalg.solve(local_scaling + state_pred_inv_cov,\
#                           local_mean + np.dot(state_pred_inv_cov,state_pred_mean.iloc[:,u])))  
            elif constraints == 'nonnegative':   
                global_mean[:,u],w = nnls(local_scaling + state_pred_inv_cov,\
                         np.squeeze(local_mean + np.dot(state_pred_inv_cov,state_pred_mean.iloc[:,u])))
     
   
    elif optimizer == 'minimizer':
        
                
        for u in hidden_states_to_infer:
            #c = state_couplings[u]
            c = list(range(global_mean.shape[1]))
#            if iter_idx < burnin:
#                c = [item for item in state_couplings[u] if item in obs_idx]
#                c.append(u)
            
            x0 = global_mean[:,u]
            if constraints == None: 
                bnds = [(None,None) for i in range(global_mean.shape[0])]
                bnds[0] = (0.,0.) 
                bnds = tuple(bnds)                        
                global_mean[:,u] = minimize(squared_loss_states,x0,bounds=bnds,args=(u,global_mean,ode_param_proxy,ext_input.T,dC_times_inv_C,odes,odes_gradient_states[u],func,func_gradient[u],func_pred_mean.values,c),jac=squared_loss_states_gradient).x
            elif constraints == 'nonnegative': 
                bnds = tuple([(0,None) for i in range(global_mean.shape[0])])
                global_mean[:,u] = minimize(squared_loss_states,x0,bounds=bnds,args=(u,global_mean,ode_param_proxy,ext_input.T,dC_times_inv_C,odes,odes_gradient_states[u],func,func_gradient[u],func_pred_mean.values,c),jac=squared_loss_states_gradient).x
                
    # pack global_mean into pandas DataFrame
    state_proxy = pd.DataFrame(global_mean,columns=map(str,state_symbols),index=time_points).rename_axis('time')
        
    func_proxy = pd.DataFrame(np.array(func(*state_proxy.values.T)).T,columns=list(func_pred_mean.columns),index=time_points).rename_axis('time')
    
    # plot state proxies
    plot_states(func_true_vals,func_pred_mean,(10,8),label=['true','estimate','observed'],color_idx=[1,0,1],traj_idx=2,sigma=None,traj_arg=func_proxy,plot_name='BOLD_response_estimation')  
    plot_states(true_states,observations,(10,40),label=['true','estimate','observed'],color_idx=[1,0,1],traj_idx=2,sigma=None,traj_arg=state_proxy,plot_name='state_estimation')  

    
    return state_proxy,func_proxy


# In[5]:
def squared_loss(ode_param,states,ext_input,dC_times_inv_C_times_states,odes,odes_gradient,alpha):
    
    x_dot = np.array(odes(*states,*ode_param,*ext_input)).T
    

    return ((x_dot-dC_times_inv_C_times_states)**2).sum().sum() + alpha * (ode_param).sum()

# In[5]:
def squared_loss_gradient(ode_param,states,ext_input,dC_times_inv_C_times_states,odes,odes_gradient,alpha):
    
    ones_vector = np.ones((states.shape[1]))
    odes_grad = np.array([np.column_stack(o) for o in odes_gradient(*states,*ode_param,*ext_input,ones_vector)]).reshape(-1,len(ode_param))
    x_dot = np.array(odes(*states,*ode_param,*ext_input)).T
    
    return (2 * (x_dot.reshape(-1,1,order='F') - dC_times_inv_C_times_states.reshape(-1,1,order='F')) * odes_grad).sum(axis=0) + alpha

# In[6]:
def squared_loss_states(state,state_idx,states,ode_param,ext_input,dC_times_inv_C,odes,odes_gradient_states,func,func_gradient,func_obs,c):

    #c = list(range(states.shape[1]))
     
    states[:,state_idx] = state
    
#    dC_times_inv_C_times_states = dC_times_inv_C.dot(states)[:,c].reshape(-1,order='F')
#    x_dot = np.concatenate(odes(*states.T,*ode_param,*ext_input))
    
    dC_times_inv_C_times_states = dC_times_inv_C.dot(states[:,c])
    x_dot = np.array(odes(*states.T,*ode_param,*ext_input)).T[:,c]


    cost = ((x_dot-dC_times_inv_C_times_states)**2).sum().sum() + ((np.array(func(*states.T)).T - func_obs)**2).sum().sum()
    
    return cost

# In[7]:
def squared_loss_states_gradient(state,state_idx,states,ode_param,ext_input,dC_times_inv_C,odes,odes_gradient_states,func,func_gradient,func_obs,c):

    states[:,state_idx] = state
    
    #c = list(range(states.shape[1]))
    dC_times_inv_C_times_states = dC_times_inv_C.dot(states[:,c])
    
    x_dot =np.array(odes(*states.T,*ode_param,*ext_input)).T[:,c]
    
    one_vector = np.ones((states.shape[0]))
    
    o = odes_gradient_states(*states.T,*ode_param,*ext_input,one_vector)
    odes_grad = np.array([o[i] for i in c]).T
    
    cost = 2*(x_dot - dC_times_inv_C_times_states)
    grad1 = ( cost * odes_grad ).sum(axis=1)
    grad2 = cost[:,c.index(state_idx)].T.dot(dC_times_inv_C) 
    grad_odes = grad1 - grad2
    
    func_grad = np.array([f for f in func_gradient(*states.T,one_vector)]).T
    grad_func = (-2*(func_obs - np.array(func(*states.T)).T) * func_grad).sum(axis=1)
    
        
    return  grad_odes + grad_func