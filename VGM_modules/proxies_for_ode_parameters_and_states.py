
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
from sklearn.linear_model import LinearRegression
import scipy.optimize.nnls as nnls
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from plotting import *
from scipy.optimize import least_squares
from scipy.optimize import minimize
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy import stats
import sympy as sym

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


def proxy_for_ode_parameters(state_proxy,locally_linear_odes,dC_times_inv_C,eps_cov,ode_param_symbols,ode_param_true,\
                             odes,odes_gradient,constraints=None,optimizer='minimizer',fig_shape=(7,4),init=None):

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
        local_mean = np.zeros((len(ode_param_symbols),))
        local_scaling = np.zeros((len(ode_param_symbols),len(ode_param_symbols)))
        global_cov = np.zeros((len(ode_param_symbols),len(ode_param_symbols)))
        
        state = np.append(state_proxy,np.ones((state_proxy.shape[0],1)),axis=1).T
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
#        eigval,v = np.linalg.eig(local_scaling)
#        if any(eigval < 0.001):
#            for i in range(local_scaling.shape[0]):
#                local_scaling[i,i] += abs((max(eigval) - min(eigval)))/10000 *np.random.rand(1,1)
        
        # (global) mean of parameter proxy distribution:
        # $\left( \mathbf{B}_{\mathbf{\theta}}^T \mathbf{B}_{\mathbf{\theta}} \right)^{-1}
        # \left( \sum_k \mathbf{B}_{\mathbf{\theta} k}^T ~ \left( {'\mathbf{C}_{\mathbf{\phi} k}}
        # \mathbf{C}_{\mathbf{\phi} k}^{-1} \mathbf{X}_k - \mathbf{b}_{\mathbf{\theta} k} \right) \right)$   
        
        if constraints == None:
            global_mean = np.linalg.solve(local_scaling,local_mean)
            #linreg = LinearRegression()
            #global_mean = linreg.fit(B,np.squeeze(local_mean)).coef_
             
        elif constraints == 'nonnegative':
            #global_mean,w = nnls(local_scaling,np.squeeze(local_mean))
            lassoreg = Lasso(alpha=0.00001,positive=True)
            global_mean = lassoreg.fit(local_scaling,np.squeeze(local_mean)).coef_
        elif constraints == 'shrinkage': 
            ridgereg = Ridge(alpha=1)
            global_mean = ridgereg.fit(local_scaling,np.squeeze(local_mean)).coef_
        elif constraints == 'sparse':
            lassoreg = Lasso(alpha=1)
            global_mean = lassoreg.fit(local_scaling,np.squeeze(local_mean)).coef_
        elif constraints == 'sparse+nonnegative':
            lassoreg = Lasso(alpha=1,positive=True)
            global_mean = lassoreg.fit(local_scaling,np.squeeze(local_mean)).coef_
    
        #fig = plt.figure(num=None, figsize=(6,6), dpi=80)
        #plt.imshow(global_cov)
    
    elif optimizer == 'minimizer': 
        x0 = 0.1*np.ones(len(ode_param_symbols))
        if init is not None:
            x0 = init
         
        if constraints == None:
            alpha = 0.
            global_mean = minimize(squared_loss,x0,args=(state_proxy,dC_times_inv_C.dot(state_proxy),odes,odes_gradient,alpha),jac=squared_loss_gradient).x
        elif constraints == 'shrinkage':
            alpha = 0.3
            global_mean = minimize(squared_loss,x0,args=(state_proxy,dC_times_inv_C.dot(state_proxy),odes,odes_gradient,alpha),jac=squared_loss_gradient).x
        elif constraints == 'nonnegative':
            alpha = 0.
            bnds = tuple([(0,None) for i in range(len(ode_param_symbols))])
            global_mean = minimize(squared_loss,x0,bounds=bnds,args=(state_proxy,dC_times_inv_C.dot(state_proxy),odes,odes_gradient,alpha),jac=squared_loss_gradient).x

        

        
    # pack global_mean into pandas DataFrame
    ode_param_proxy = pd.DataFrame(global_mean,columns=['value'],index=map(str,ode_param_symbols)).rename_axis('ODE parameter symbols')
    
    # plotting ODE parameters
    plot_ode_parameters(ode_param_true,fig_shape,[1,0],plot_name='ODE_parameter_estimation',ode_param_estimates=ode_param_proxy)
    
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
    
#def proxy_for_ind_states(inferred_states,odes_couplings_to_states,state_proxy,ode_param_proxy,odes,odes_gradient_states,locally_linear_odes,dC_times_inv_C,eps_cov,\
#                         state_pred_mean,state_pred_inv_cov,observations,true_states,state_couplings,iter_idx,burnin=5,
#                         clamp_states_to_observation_fit=True,constraints=None,optimizer='analytical',fig_shape=(10,8)):
    
def proxy_for_ind_states(state_proxy,ode_param_proxy,odes,odes_gradient_states,locally_linear_odes,dC_times_inv_C,eps_cov,\
                         state_pred_mean,state_pred_inv_cov,observations,true_states,state_couplings,burnin=5,
                         clamp_states_to_observation_fit=True,constraints=None,optimizer='analytical',fig_shape=(10,8)):
    
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
    
    
    obs_idx = [u for u in range(len(state_symbols)) if state_symbols[u] in observations.columns.values]
    
    # initialization
    global_mean = state_proxy[:]
        
    if optimizer == 'analytical':
         
        # initialization
        global_cov = np.zeros((state_proxy.shape[0],state_proxy.shape[0],state_proxy.shape[1]))
        state = np.append(state_proxy,np.ones((state_proxy.shape[0],1)),axis=1).T
    
    
        # iterate through each unobserved state
        for u in hidden_states_to_infer:
    

            # initialization
            local_mean = np.zeros((state_proxy.shape[0]))
            local_scaling = np.zeros((state_proxy.shape[0],state_proxy.shape[0]))
            local_scaling_diag = local_scaling[:]
            
            # iterate through each ODE
            c = state_couplings[u]
#            if iter_idx < burnin:
#                c = [item for item in state_couplings[u] if item in obs_idx]
#                c.append(u)
            for k in c:
            #for k in state_couplings[u]:
               
                ode_param_proxy = [8]
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
                    local_mean += -np.diag(B) * b
                    local_scaling_diag += B.T.dot(B)
                else:
                    B = R - dC_times_inv_C
                    b = r[:]
                    local_mean += -B.T.dot(b)
                    local_scaling = B.T.dot(B)
            
                
                # local mean: $\mathbf{B}_{uk}^T \left(\mathbf{\epsilon_0}^{(k)} -\mathbf{b}_{uk}
                #local_mean += -B.T.dot(b)
                #local_scaling += B.T.dot(B)
                
                global_cov[:,:,u] += B.T.dot(eps_cov.dot(B))
                
                #global_mean[:,u] += np.squeeze(0.1*np.linalg.solve(B.T.dot(B)+state_pred_inv_cov,-B.T.dot(b))) 
                #linreg = LinearRegression()
                #global_mean[:,u] += linreg.fit(B.T.dot(B),np.squeeze(-B.T.dot(b))).coef_
            
            local_scaling += local_scaling_diag
            #C = np.diag(1/(np.diag(local_scaling_diag) + np.finfo(float).eps))
#            C = np.diag(1/(np.diag(local_scaling_diag)+10))
#            B_times_C = local_scaling.dot(C)
#            g = 1/(1+np.matrix.trace(B_times_C))
#            local_scaling = C - g * C.dot(B_times_C)
            
#            A = np.diag(1/(np.diag(local_scaling_diag)))
#            B = local_scaling[:]
#            local_scaling = A - A.dot(B).dot(B + B.dot(A).dot(B))
            
            # Mean of state proxy distribution (option: Moore-penrose inverse example): 
            # $\left( \mathbf{B}_{u} \mathbf{B}_{u}^T \right)^{-1} \sum_k
            # \mathbf{B}_{uk}^T \left(\mathbf{\epsilon_0}^{(k)} -\mathbf{b}_{uk} \right)$   
            if constraints == None:
                #global_mean[:,u] = np.squeeze(local_scaling.dot(local_mean)) 
                #global_mean[:,u] = np.squeeze(np.linalg.inv(local_scaling).dot(local_mean)) 
                global_mean[:,u] = np.squeeze(np.linalg.solve(local_scaling,local_mean)) 
                #global_mean[:,u] = local_mean / np.diag(local_scaling)
                #print(u)
#                global_mean[:,u] = np.squeeze(np.linalg.solve(local_scaling + state_pred_inv_cov,\
#                           local_mean + np.dot(state_pred_inv_cov,state_pred_mean.iloc[:,u])))
                #linreg = LinearRegression()
                #global_mean[:,u] = linreg.fit(local_scaling + state_pred_inv_cov,np.squeeze(local_mean)).coef_
                #global_mean[:,u] = linreg.fit(local_scaling,np.squeeze(local_mean + np.dot(state_pred_inv_cov,state_pred_mean.iloc[:,u]))).coef_
            elif constraints == 'nonnegative': 
                lassoreg = Lasso(alpha=0.00001,positive=True)
                global_mean[:,u] = lassoreg.fit(local_scaling,np.squeeze(local_mean)).coef_
#                global_mean[:,u] = lassoreg.fit(local_scaling + state_pred_inv_cov,\
#                           np.squeeze(local_mean + np.dot(state_pred_inv_cov,state_pred_mean.iloc[:,u]))).coef_
                #global_mean[:,u],w = nnls(local_scaling + state_pred_inv_cov,\
                #         np.squeeze(local_mean + np.dot(state_pred_inv_cov,state_pred_mean.iloc[:,u])))    
      
    elif optimizer == 'minimizer':
        
        
#        [state_couplings[obs_idx[i]] for i in range(len(obs_idx))]
#        s = [state_couplings[obs_idx[i]] for i in range(len(obs_idx))]
#        s2 = [item for sublst in s for item in sublst]
#        states_to_infer = [u for u in s2 if state_symbols[u] not in observations.columns.values]
#        remaining_states_to_infer = [u for u in range(len(state_symbols)) if u not in states_to_infer and u not in obs_idx]
#        if len(remaining_states_to_infer) != 0:
#            states_to_infer.append(remaining_states_to_infer)

        #ode_param_proxy = [8]

# coordinate descent using submodularity
#########################
#        c = np.argmax([sum(sublist) for sublist in inferred_states])
#        #inferred_states[c][-1] = -(iter_idx*1.1)
#        #print(inferred_states)
#        print(odes_couplings_to_states[c])
#        
#        q = []
#        for s in odes_couplings_to_states[c]:
#            #inferred_states_dummy = inferred_states.copy()
#            inferred_states_dummy = [sublist[:] for sublist in inferred_states]
#            for i in range(len(odes_couplings_to_states)):
#                if s in odes_couplings_to_states[i]:
#                    inferred_states_dummy[i][odes_couplings_to_states[i].index(s)] += 5
#            q.append(sum([sum(sublist[0:-1]) for sublist in inferred_states_dummy]))
#        
#        obs_states = odes_couplings_to_states[c]
#        obs_states_subset = [np.int(i) if (s in sym.symbols(observations.columns.values.tolist())) else None for i,s in enumerate(obs_states)]
#        obs_states_subset = [i for i in obs_states_subset if i is not None]
#        q2 = np.array(q)
#        q2[obs_states_subset] = 0
#        
#        u = state_symbols.tolist().index(str(odes_couplings_to_states[c][np.argmax(q2)]))
#        #u = state_symbols.tolist().index(str(odes_couplings_to_states[c][np.int(np.random.choice(range(len(q2)),1,p=list(q2/sum(q2))))]))
#        #print(odes_couplings_to_states[c])
#        #print(q2)
#        #print(state_symbols[u])
#        
#        unobs_states_subset = [np.int(i) if (s not in sym.symbols(observations.columns.values.tolist())) else None for i,s in enumerate(obs_states)]
#        unobs_states_subset = [i for i in unobs_states_subset if i is not None]
#        for i in unobs_states_subset:
#            u = state_symbols.tolist().index(str(odes_couplings_to_states[c][i]))
#        
#            x0 = global_mean[:,u]
#            bnds = [(None,None) for i in range(global_mean.shape[0])] 
#            bnds[0] = (1.,1.) 
#            bnds = tuple(bnds)
#            c2 = state_couplings[u]
#            global_mean[:,u] = minimize(squared_loss_states,x0,bounds=bnds,args=(u,global_mean,ode_param_proxy,dC_times_inv_C,odes,odes_gradient_states[u],c2),jac=squared_loss_states_gradient).x
#    
#            inferred_states[c][-1] = -(iter_idx*1)
##            for i in range(len(odes_couplings_to_states)):
##                if sym.symbols(state_symbols[u]) in odes_couplings_to_states[i]: 
##                    inferred_states[i][odes_couplings_to_states[i].index(sym.symbols(state_symbols[u]))] += 5
##                    inferred_states[i][-1] = -(iter_idx**2)

# vanila coordinate descent
######################
        for u in hidden_states_to_infer:
        #for i in range(20):
            
        
            #samp_prob = np.linalg.norm(state_proxy[:,hidden_states_to_infer],axis=0) + np.finfo(float).eps
            #samp_prob = samp_prob / np.sum(samp_prob)
            #print(samp_prob)
            #u = int(np.random.choice(hidden_states_to_infer, 1, p=samp_prob)[0])
            #print(u)
            c = state_couplings[u]
#            if iter_idx < burnin:
#                c = [item for item in state_couplings[u] if item in obs_idx]
#                c.append(u)

            x0 = global_mean[:,u]
            if constraints == None:
                bnds = [(None,None) for i in range(global_mean.shape[0])] 
                bnds[0] = (1.,1.) 
#                if u == 0:
#                    bnds[0] = (1.,1.)    
#                elif u == 3:
#                    bnds[0] = (0.,0.01)    
                bnds = tuple(bnds)
                global_mean[:,u] = minimize(squared_loss_states,x0,bounds=bnds,args=(u,global_mean,ode_param_proxy,dC_times_inv_C,odes,odes_gradient_states[u],c),jac=squared_loss_states_gradient).x
                #global_mean[:,u] = minimize(squared_loss_states,x0,bounds=bnds,args=(u,global_mean,ode_param_proxy,dC_times_inv_C,odes,odes_gradient_states[u],c)).x
            elif constraints == 'nonnegative': 
                bnds = tuple([(0,None) for i in range(global_mean.shape[0])])
                global_mean[:,u] = minimize(squared_loss_states,x0,bounds=bnds,args=(u,global_mean,ode_param_proxy,dC_times_inv_C,odes,odes_gradient_states[u],c),jac=squared_loss_states_gradient).x
     
       
    # pack global_mean into pandas DataFrame
    state_proxy = pd.DataFrame(global_mean,columns=map(str,state_symbols),index=time_points).rename_axis('time')
   
    # plot state proxies
    plot_states(true_states,observations,fig_shape,label=['true','estimate','observed'],color_idx=[1,0,1],traj_idx=2,sigma=None,traj_arg=state_proxy,plot_name='state_estimation')     
    
    return state_proxy


# In[5]:
def squared_loss(ode_param,states,dC_times_inv_C_times_states,odes,odes_gradient,alpha):
    
    x_dot = np.array(odes(*states.T,*ode_param)).T
    
    
    return ((x_dot-dC_times_inv_C_times_states)**2).sum().sum() + alpha * (ode_param**2).sum()

# In[5]:
def squared_loss_gradient(ode_param,states,dC_times_inv_C_times_states,odes,odes_gradient,alpha):
    
    ones_vector = np.ones((states.shape[0]))
    odes_grad = np.array([np.column_stack(o) for o in odes_gradient(*ode_param,*states.T,ones_vector)]).reshape(-1,len(ode_param))
    x_dot = np.array(odes(*states.T,*ode_param)).T
    
    return (2 * (x_dot.reshape(-1,1,order='F') - dC_times_inv_C_times_states.reshape(-1,1,order='F')) * odes_grad).sum(axis=0) + 2*alpha*ode_param
     
    
# In[6]:
def squared_loss_states(state,state_idx,states,ode_param,dC_times_inv_C,odes,odes_gradient_states,c):

    #c = list(range(states.shape[1]))
    states[:,state_idx] = state
    
    dC_times_inv_C_times_states = dC_times_inv_C.dot(states[:,c])
    
    #x_dot = np.concatenate(odes(*states.T,*ode_param))
    x_dot = np.array(odes(*states.T,*ode_param)).T[:,c]
    
    
    return ((x_dot-dC_times_inv_C_times_states)**2).sum().sum()

# In[7]:
def squared_loss_states_gradient(state,state_idx,states,ode_param,dC_times_inv_C,odes,odes_gradient_states,c):

    c = list(range(states.shape[1]))
    states[:,state_idx] = state
    
    
    dC_times_inv_C_times_states = dC_times_inv_C.dot(states[:,c])
    
    x_dot =np.array(odes(*states.T,*ode_param)).T[:,c]
    
    one_vector = np.ones((states.shape[0]))
    
    o = odes_gradient_states(*ode_param,*states.T,one_vector)
    #odes_grad = np.array([o[i] for i in c]).T.reshape(-1,order='F')
    odes_grad = np.array([o[i] for i in c]).T
    
    cost0 = 2*(x_dot - dC_times_inv_C_times_states)
    grad1 = ( cost0 * odes_grad ).sum(axis=1)
    grad2 = cost0[:,c.index(state_idx)].T.dot(dC_times_inv_C) 
    cost = grad1 - grad2
    
    #cost =  (2*(x_dot - dC_times_inv_C_times_states) * (odes_grad - tmp)).reshape(len(c),-1,order='C')
    
    return  cost
