
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


def fitting_state_observations(observations,prior_inv_cov,hidden_states,\
                               obs_to_state_relations,obs_variance,time_points):

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
    
    
    ## Plotting GP_post_mean
    plot_states(GP_post_mean,observations,['GP fit','','observed'],[0,0,1],2)
    
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

def kernel_function(time_points,kernel_param=[10,0.2]):
    
    '''Populates the GP covariance matrix and it's derivatives'''
    
    # error handeling
    if len(kernel_param) != 2:
        raise ValueError('kernel_param input requires two floats')
    elif type(time_points) is not np.ndarray:
        raise ValueError('time points is not a numpy array')
    elif len(time_points.shape) > 1:
        raise ValueError('time points must be a one dimensional numpy array')

        
    
    # radial basis function (RBF) kernel
    t = sym.var('t0,t1')
    rbf_kernel = kernel_param[0] * sym.exp(- (t[0] - t[1])**2 / kernel_param[1]**2)
    
    # kernel derivative functions
    d = sym.var('d') 
    rbf_kernel_diff = rbf_kernel.diff(t[0]).subs((t[0]-t[1]),d).factor().subs((t[0]-t[1]),d)
    rbf_kernel = rbf_kernel.subs((t[0]-t[1]),d)
    
    # make annonymous functions out of symbolic expressions
    cov_func = sym.lambdify(d,rbf_kernel)
    cov_func_diff = sym.lambdify(d,rbf_kernel_diff)

    # difference between time points   
    time_diff = time_points.reshape(-1,1) - time_points.reshape(1,-1)

    # populate covariance matrices   
    cov = np.array(map(cov_func,time_diff))
    cov_diff = np.array(map(cov_func_diff,time_diff))
 
    try:
        inv_cov = np.linalg.inv(cov)
    except:
        ValueError('unable to compute the inverse of C')
    
    # compute $\mathbf{C}_{\boldsymbol\phi_k}' ~ \mathbf{C}_{\boldsymbol\phi_k}^{-1}$
    cov_diff_times_inv_cov= np.linalg.solve(cov.T,cov_diff.T).T
    
    # plot sample state trajectories from GP prior
    mean = np.zeros((cov.shape[0]))
    prior_state_sample1 = np.random.multivariate_normal(mean,cov)
    prior_state_sample2 = np.random.multivariate_normal(mean,cov)
    plot_trajectories(time_points,prior_state_sample1,prior_state_sample2)
   
    
    return cov_diff_times_inv_cov,inv_cov

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