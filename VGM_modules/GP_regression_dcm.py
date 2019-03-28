
# # GP Regression
# 
# #### Authors: Nico S. Gorbach and Stefan Bauer

# In[1]:

# ## Import Numpy, Symbolic and Plotting Modules

import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
import pandas as pd
from plotting import *

# In[2]:

# ## Fit the observations of state trajectories by standard GP regression
# 
# The GP posterior is given by:
# 
# \begin{align}
# p(\mathbf{X} \mid \mathbf{Y}, \boldsymbol\phi,\gamma) = \prod_k \mathcal{N}(\mathbf{x}_k ; 
# \boldsymbol\mu_k(\mathbf{y}_k),\boldsymbol\Sigma_k),
# \end{align}
# 
# where $\boldsymbol\mu_k(\mathbf{y}_k) := \boldsymbol\sigma_k^{-2} \left(\boldsymbol\sigma_k^{-2} \mathbf{I} + 
# \mathbf{C}_{\boldsymbol\phi_k}^{-1} \right)^{-1} \mathbf{y}_k$
# and $\boldsymbol\Sigma_k ^{-1}:=\boldsymbol\sigma_k^{-2} \mathbf{I} + \mathbf{C}_{\boldsymbol\phi_k}^{-1}$

def fitting_state_observations(observations,hidden_states,SNR,time_points,cov,cov_obs,cov_func_obs,func_obs):
    
    # determine variance of observations from SNR
    obs_variance = (np.mean(observations.values,axis=0) / SNR)
    
    state_pred_mean = pd.DataFrame(0.1*np.ones((len(time_points),len(hidden_states))),\
                                          columns=map(str,hidden_states),index=time_points).rename_axis('time')
    

    for i,state in enumerate(list(observations.columns)):
        c = cov_obs + np.diag([obs_variance[i]]*cov_obs.shape[1])
        state_pred_mean[state] = np.dot(cov_func_obs,np.linalg.solve(c,observations[state]))
       
    state_pred_cov = cov - np.dot(cov_func_obs,np.linalg.solve(cov_obs,cov_func_obs.T))
    state_pred_inv_cov = np.linalg.inv(state_pred_cov)
    state_pred_cov = np.repeat(state_pred_cov[:,:,np.newaxis],len(hidden_states),2)
      
    # determine variance of observations from SNR
    obs_variance = (np.mean(func_obs.values,axis=0) / SNR)**2
    
    func_pred_mean = pd.DataFrame(0*np.ones((len(time_points),func_obs.shape[1])),\
                                          columns=list(func_obs.columns),index=time_points).rename_axis('time')
    for i,func_sym in enumerate(list(func_obs.columns)):
        c = cov_obs + np.diag([obs_variance[i]]*cov_obs.shape[1])
        func_pred_mean[func_sym] = np.dot(cov_func_obs,np.linalg.solve(c,func_obs[func_sym]))
       
        
    #plot_states(state_pred_mean,observations,np.sqrt(np.diag(state_pred_cov)),['GP fit','','observed'],[0,0,1],2)
    plot_states(func_pred_mean,func_obs,(10,8),label=['GP fit','','observed'],color_idx=[0,0,1],traj_idx=2,sigma=state_pred_cov)
    plot_states(state_pred_mean,observations,(10,40),label=['GP fit','','observed'],color_idx=[0,0,1],traj_idx=2,sigma=state_pred_cov)
    
    return state_pred_mean, state_pred_inv_cov, func_pred_mean



# In[3]:

# ## Compute the GP covariance matrix and it's derivatives
#     
# Gradient matching with Gaussian processes assumes a joint Gaussian process prior on states and their derivatives:
# 
# \begin{align}
# \left(\begin{array}{c} \mathbf{X} \\ \dot{\mathbf{X}} \end{array}\right) \sim \mathcal{N} \left(
# \begin{array}{c} \mathbf{X} \\ \dot{\mathbf{X}} \end{array};
# \begin{array}{c}
# \mathbf{0} \\
# \mathbf{0}
# \end{array},
# \begin{array}{cc}
# \mathbf{C}_{\boldsymbol\phi} & \mathbf{C}_{\boldsymbol\phi}' \\ '\mathbf{C}_{\boldsymbol\phi} &
# \mathbf{C}_{\boldsymbol\phi}'' \end{array} \right),
# \end{align}
# 
# where:
# 
# $\mathrm{cov}(x_k(t), x_k(t)) = C_{\boldsymbol\phi_k}(t,t')$
# $\mathrm{cov}(\dot{x}_k(t), x_k(t)) = \frac{\partial C_{\boldsymbol\phi_k}(t,t') }{\partial t} =: 
# C_{\boldsymbol\phi_k}'(t,t')$
# $\mathrm{cov}(x_k(t), \dot{x}_k(t)) = \frac{\partial C_{\boldsymbol\phi_k}(t,t')}{\partial t'} =: 
# {'C_{\boldsymbol\phi_k}(t,t')}$
# $\mathrm{cov}(\dot{x}_k(t), \dot{x}_k(t)) = \frac{\partial C_{\boldsymbol\phi_k}(t,t') }{\partial t \partial t'} =: 
# C_{\boldsymbol\phi_k}''(t,t')$.

def kernel_function(time_points,time_points2,kernel_type='rbf',kernel_param=[10,0.2]):
    
    '''Populates the GP covariance matrix and it's derivatives. 
    Input time points (list of real numbers) and kernel parameters (list of real values of size 2)'''
    
    # error handeling
#    if len(kernel_param) != 3:
#        raise ValueError('kernel_param input requires two floats')
    if type(time_points) is not np.ndarray:
        raise ValueError('time points is not a numpy array')
    elif len(time_points.shape) > 1:
        raise ValueError('time points must be a one dimensional numpy array')

     
    
    t = sym.var('t0,t1')

    if kernel_type == 'rbf':
        # rbf kernel
        dist = ((t[0] - t[1])/ kernel_param[0])**2
        kernel = sym.exp(-.5 * dist)
        #kernel = kernel_param[0] * sym.exp(- dist**2)
    if kernel_type == 'periodic':
        # periodic kernel
        dist = (t[0] - t[1])/ kernel_param[1]
        kernel = sym.exp(- 2*sym.sin(np.pi * sym.sqrt(dist**2) / kernel_param[1])**2 / kernel_param[0]**2) 
    if kernel_type == 'locally_periodic':
        # locally periodic kernel
        dist = (t[0] - t[1])
        kernel = sym.exp(- dist**2 / 2*kernel_param[1]**2) * sym.exp(- 2*sym.sin(np.pi * sym.sqrt((t[0] - t[1])**2) / kernel_param[2])**2 / kernel_param[1]**2) 
    elif kernel_type == 'rbf+lin':
         # rbf kernel + linear kernel
        dist = (t[0] - t[1])
        kernel = sym.exp(- dist**2 / kernel_param[1]**2) + kernel_param[2] + kernel_param[3] * (t[0]-kernel_param[4]) * (t[1]-kernel_param[4])
    elif kernel_type == 'sigmoid':
        # sigmoid kernel
        kernel = sym.asin((kernel_param[1]+kernel_param[2]*t[0]*t[1])/sym.sqrt((kernel_param[1]+kernel_param[2]*t[0]**2+1)*(kernel_param[1]+kernel_param[2]*t[1]**2+1)));
    elif kernel_type == 'sigmoid2':
        # sigmoid kernel 2
        kernel = sym.tanh(kernel_param[0] * t[0] * t[1] + kernel_param[1])    
    elif kernel_type == 'exp_sin_squared':
        # periodic kernel
        dist = sym.sqrt((t[0] - t[1])**2)
        arg = np.pi * dist / kernel_param[1]
        sin_of_arg = sym.sin(arg)
        kernel = sym.exp(- 2 * (sin_of_arg / kernel_param[0]) ** 2)
    elif kernel_type == 'RQ':
        dist = (t[0] - t[1])**2
        tmp = dist / (2 * kernel_param[0] * kernel_param[1] ** 2)
        base = (1 + tmp)
        kernel = base ** -kernel_param[0]
    elif kernel_type == 'matern':
        # periodic kernel
        dist = sym.sqrt(((t[0] - t[1]) / kernel_param[0])**2)
        #kernel = dist * np.sqrt(3)
        #kernel = (1. + kernel) * sym.exp(-kernel)
        #kernel = kernel_param[1]**2 * (1 + kernel_param[2] * np.sqrt(3) * kernel_param[3] * dist) * sym.exp(-kernel_param[2] * np.sqrt(3) * kernel_param[3] * dist)
        if kernel_param[1] == 0.5:
            kernel = sym.exp(-dist)
        elif kernel_param[1] == 1.5:
            kernel = dist * np.sqrt(3)
            kernel = (1. + kernel) * sym.exp(-kernel)
        elif kernel_param[1] == 2.5:
            kernel = dist * np.sqrt(5)
            kernel = (1. + kernel + kernel ** 2 / 3.0) * sym.exp(-kernel)
        else:  # general case; expensive to evaluate
            kernel = dist
            kernel[kernel == 0.0] += np.finfo(float).eps  # strict zeros result in nan
            tmp = (sym.sqrt(2 * kernel_param[1]) * kernel)
            kernel.fill((2 ** (1. - kernel_param[1])) / gamma(kernel_param[1]))
            kernel *= tmp ** kernel_param[1]
            kernel *= kv(kernel_param[1], tmp)
    kernel *= kernel_param[-1]**2
    
                                                      
    # kernel derivative functions
    kernel_diff = kernel.diff(t[0])
    kernel_diff_diff = kernel_diff.diff(t[1])
    
    cov_func = sym.lambdify(t,kernel)
    cov_func_diff = sym.lambdify(t,kernel_diff)
    cov_func_diff_diff = sym.lambdify(t,kernel_diff_diff)

    cov = np.zeros((len(time_points),len(time_points)))
    cov_diff = np.zeros((len(time_points),len(time_points)))
    cov_diff_diff = np.zeros((len(time_points),len(time_points)))
    for i in range(len(time_points)):
        for j in range(len(time_points)):
            cov[i,j] = cov_func(time_points[i],time_points[j])
            cov_diff[i,j] = cov_func_diff(time_points[i],time_points[j])
            cov_diff_diff[i,j] = cov_func_diff_diff(time_points[i],time_points[j])
 
    cov_obs = np.zeros((len(time_points2),len(time_points2)))
    for i in range(len(time_points2)):
        for j in range(len(time_points2)):
            cov_obs[i,j] = cov_func(time_points2[i],time_points2[j])
    
    cov_state_obs = np.zeros((len(time_points),len(time_points2)))
    for i in range(len(time_points)):
        for j in range(len(time_points2)):
            cov_state_obs[i,j] = cov_func(time_points[i],time_points2[j])


           
    # compute $\mathbf{C}_{\boldsymbol\phi_k}' ~ \mathbf{C}_{\boldsymbol\phi_k}^{-1}$
    cov_diff_times_inv_cov = np.linalg.solve(cov.T,cov_diff.T).T
    
    # plot sample state trajectories from GP prior
    mean = np.zeros((cov.shape[0]))
    prior_state_sample1 = np.random.multivariate_normal(mean,cov)
    prior_state_sample2 = np.random.multivariate_normal(mean,cov)
    plot_trajectories(time_points,prior_state_sample1,prior_state_sample2)
   
    
    
    
    eps_cov = cov_diff_diff - cov_diff_times_inv_cov.dot(cov_diff.T)
    
    return cov_diff_times_inv_cov,eps_cov,cov,cov_obs,cov_state_obs

# In[4]:
    

def extract_block_diag(A,M,k=0):

    '''Extracts blocks of size M from the kth diagonal of square matrix A, whose size must be a multiple of M'''
    
    # Check that the matrix can be block divided
    if A.shape[0] != A.shape[1] or A.shape[0] % M != 0:
        raise StandardError('Matrix must be square and a multiple of block size')

    # Assign indices for offset from main diagonal
    if abs(k) > M - 1:
        raise StandardError('kth diagonal does not exist in matrix')
    elif k > 0:
        ro = 0
        co = abs(k)*M 
    elif k < 0:
        ro = abs(k)*M
        co = 0
    else:
        ro = 0
        co = 0

    blocks = np.array([A[i+ro:i+ro+M,i+co:i+co+M] 
                       for i in range(0,len(A)-abs(k)*M,M)])
    return blocks