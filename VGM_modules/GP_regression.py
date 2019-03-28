
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
from scipy.special import kv, gamma

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

def fitting_state_observations(observations,hidden_states,SNR,time_points,cov,cov_obs,cov_func_obs,fig_shape=(10,8)):
    
    # determine variance of observations from SNR
    obs_variance = (np.mean(observations.values,axis=0) / SNR)**2
    
    state_pred_mean = pd.DataFrame(0.*np.ones((len(time_points),len(hidden_states))),\
                                          columns=map(str,hidden_states),index=time_points).rename_axis('time')
    
    for i,state in enumerate(list(observations.columns)):
        cov_obs = cov_obs + np.diag([obs_variance[i]]*cov_obs.shape[1])
        state_pred_mean[state] = np.dot(cov_func_obs,np.linalg.solve(cov_obs,observations[state]))
       
    state_pred_cov = cov - np.dot(cov_func_obs,np.linalg.solve(cov_obs,cov_func_obs.T))
    state_pred_inv_cov = np.linalg.pinv(state_pred_cov)
    state_pred_cov = np.repeat(state_pred_cov[:,:,np.newaxis],len(hidden_states),2)
    
    plot_states(state_pred_mean,observations,fig_shape,label=['GP fit','','observed'],color_idx=[0,0,1],traj_idx=2,sigma=state_pred_cov,plot_name='GP_fit')
    
    return state_pred_mean, state_pred_inv_cov

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


def fitting_state_observations_old(observations,prior_inv_cov,hidden_states,\
                               obs_to_state_relations,obs_variance,time_points,C2,C2_star):

    '''Fits the observations of state trajectories by standard GP regression'''
    
    # number of hidden and observed states
    numb_hidden_states = len(hidden_states)
    numb_observed_states = observations.shape[1]
    # number of given and observed_time_points
    numb_given_time_points = prior_inv_cov.shape[0]
    numb_observed_time_points = observations.shape[0]
    
    # Form block-diagonal matrix out of $\mathbf{C}_{\boldmath\mathbf{\phi}_k}^{-1}$
    inv_cov_blockdiag = block_diag(*[prior_inv_cov]*numb_hidden_states)
    
    # variance of state observations
    variance = obs_variance**(-1) * np.ones((numb_observed_time_points,numb_observed_states))
    D = np.dot(np.diag(variance.reshape(1,-1)[0]),np.identity(variance.reshape(1,-1).shape[1]))
    
    # GP posterior inverse covariance matrix: $\boldmath\mathbf{\sigma}_k^{-1}:=\mathbf{\sigma}_k^{-2} \mathbf{I} + 
    # \mathbf{C}_{\boldmath\mathbf{\phi}_k}^{-1}$
    obs_to_state_relations_times_D = np.dot(obs_to_state_relations.T,D)
    A_times_D_times_A = np.dot(obs_to_state_relations_times_D,obs_to_state_relations)
    GP_post_inv_cov_flat = A_times_D_times_A + inv_cov_blockdiag
    
    # GP posterior mean: $\boldmath\mu_k(\mathbf{y}_k) := \mathbf{\sigma}_k^{-2}\left(\mathbf{\sigma}_k^{-2} \mathbf{I} + 
    # \mathbf{C}_{\boldmath\mathbf{\phi}_k}^{-1}\right)^{-1} \mathbf{y}_k$
    GP_post_mean_flat = np.linalg.solve(GP_post_inv_cov_flat,np.dot(obs_to_state_relations_times_D,\
                                                                    observations.iloc[:].values.reshape(-1,1,order='F')))
    
    # unflatten GP posterior mean and GP posterior inverse covariance matrix
    GP_post_mean = GP_post_mean_flat.reshape(1,-1).reshape(numb_given_time_points,numb_hidden_states,order='F')
    GP_post_inv_cov = extract_block_diag(GP_post_inv_cov_flat,prior_inv_cov.shape[0],k=0)
    
    # pack GP_post_mean into pandas DataFrame
    GP_post_mean = pd.DataFrame(GP_post_mean,columns=map(str,hidden_states),index=time_points).rename_axis('time')
    
    
    
#    from sklearn.gaussian_process import GaussianProcessRegressor
#    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
#    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
#    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
#    GP_post_mean2 = np.zeros((len(time_points),numb_hidden_states))
#    for i in range(numb_hidden_states):
#        gp.fit(np.array(observations.index).reshape(-1,1),np.array(observations.iloc[:,i]))
#        GP_post_mean2[:,i] = gp.predict(time_points.reshape(-1,1))
#    GP_post_mean2 = pd.DataFrame(GP_post_mean2,columns=map(str,hidden_states),index=time_points).rename_axis('time')  
   

    ## Plotting GP_post_mean
    plot_states(GP_post_mean2,observations,['GP fit','','observed'],[0,0,1],2)
    
#    traj = [GP_post_mean,observations]
#    label = ['GP_fit','observed']
#    cmap = plt.get_cmap("tab10")
#    color = [cmap(0),cmap(1)]
#    style = ['-','*']
#    plot_states2(zip(traj,label,color,style),3)
        
    return GP_post_mean, GP_post_inv_cov


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

def kernel_function(time_points,time_points2,kernel_type='RQ',kernel_param=[0.1,2,5]):
    
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
        #dist = sym.sqrt(((t[0] - t[1]) / kernel_param[0])**2)
        dist = ((t[0] - t[1]) / kernel_param[0])**2
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



#    # radial basis function (RBF) kernel
#    kernel = kernel_param[0] * sym.exp(- (t[0] - t[1])**2 / kernel_param[1]**2)
#    
#    # kernel derivative functions
#    kernel_diff = kernel.diff(t[0])
#    kernel_diff_diff = kernel_diff.diff(t[1])
#    
#    # substitute time difference
#    d = sym.var('d') 
#    kernel = kernel.subs((t[0]-t[1]),d)
#    kernel_diff = kernel_diff.subs((t[0]-t[1]),d).factor().subs((t[0]-t[1]),d)
#        
#    # make annonymous functions out of symbolic expressions
#    cov_func = sym.lambdify(d,kernel)
#    cov_func_diff = sym.lambdify(d,kernel_diff)
#    cov_func_diff_diff = sym.lambdify(t,kernel_diff_diff)
#
#    # difference between time points  
#    time_diff = time_points.reshape(-1,1) - time_points.reshape(1,-1)
#    
#    # populate covariance matrices   
#    cov = np.array(list(map(cov_func,time_diff)))
#    cov_diff = np.array(list(map(cov_func_diff,time_diff)))
#    
#    cov_diff_diff = np.zeros((len(time_points),len(time_points)))
#    for i in range(len(time_points)):
#        for j in range(len(time_points)):
#            cov_diff_diff[i,j] = cov_func_diff_diff(time_points[i],time_points[j])
# 
#
#    # for GP regression
#    time_diff = np.array(time_points2).reshape(-1,1) - np.array(time_points2).reshape(1,-1)
#    cov_obs = np.array(list(map(cov_func,time_diff)))
#    # covariance between time points for estimation and time points for observations
#    time_diff = time_points.reshape(-1,1) - np.array(time_points2).reshape(1,-1)
#    # populate covariance matrices   
#    cov_state_obs = np.array(list(map(cov_func,time_diff)))



           
    # compute $\mathbf{C}_{\boldsymbol\phi_k}' ~ \mathbf{C}_{\boldsymbol\phi_k}^{-1}$
    cov_diff_times_inv_cov = np.linalg.solve(cov.T,cov_diff.T).T
    
    # plot sample state trajectories from GP prior
    mean = np.zeros((cov.shape[0]))
    prior_state_sample1 = np.random.multivariate_normal(mean,cov_diff_diff)
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