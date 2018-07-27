
# # GP Regression
# 
# #### Authors: Nico S. Gorbach and Stefan Bauer

# In[1]:

# ## Import Numpy, Symbolic and Plotting Modules

import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from scipy.linalg import block_diag

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


def fitting_state_observations(observations,prior_inv_cov,hidden_states,observed_states,\
                               obs_to_state_relations,obs_variance,given_time_points,observed_time_points):

    # number of hidden and observed states
    numb_hidden_states = len(hidden_states)
    numb_observed_states = len(observed_states)
    # number of given and observed_time_points
    numb_given_time_points = prior_inv_cov.shape[0]
    numb_observed_time_points = len(observed_time_points)
    
    # Form block-diagonal matrix out of $\mathbf{C}_{\boldmath\mathbf{\phi}_k}^{-1}$
    inv_C_blockdiag = block_diag(*[prior_inv_cov]*numb_hidden_states)
    
    # variance of state observations
    variance = obs_variance**(-1) * np.ones((numb_observed_time_points,numb_observed_states))
    D = np.dot(np.diag(variance.reshape(1,-1)[0]),np.identity(variance.reshape(1,-1).shape[1]))
    
    # GP posterior inverse covariance matrix: $\boldmath\mathbf{\sigma}_k^{-1}:=\mathbf{\sigma}_k^{-2} \mathbf{I} + 
    # \mathbf{C}_{\boldmath\mathbf{\phi}_k}^{-1}$
    obs_to_state_relations_times_D = np.dot(obs_to_state_relations.T,D)
    A_times_D_times_A = np.dot(obs_to_state_relations_times_D,obs_to_state_relations)
    GP_post_inv_cov_flat = A_times_D_times_A + inv_C_blockdiag
    
    # GP posterior mean: $\boldmath\mu_k(\mathbf{y}_k) := \mathbf{\sigma}_k^{-2}\left(\mathbf{\sigma}_k^{-2} \mathbf{I} + 
    # \mathbf{C}_{\boldmath\mathbf{\phi}_k}^{-1}\right)^{-1} \mathbf{y}_k$
    GP_post_mean_flat = np.linalg.solve(GP_post_inv_cov_flat,np.dot(obs_to_state_relations_times_D,\
                                                                    observations.reshape(-1,1,order='F')))
    
    # unflatten GP posterior mean
    GP_post_mean = GP_post_mean_flat.reshape(1,-1).reshape(numb_given_time_points,numb_hidden_states,order='F')
    
    # unflatten GP posterior inverse covariance matrix
    GP_post_inv_cov = extract_block_diag(GP_post_inv_cov_flat,prior_inv_cov.shape[0],k=0)
    
 
    
    # Plotting
    
    # indices of observed states
    observed_state_idx = [u for u in range(numb_hidden_states) if hidden_states[u] in observed_states]
    
    cmap = plt.get_cmap("tab10")
    fig = plt.figure(num=None, figsize=(10, 8), dpi=80)
    plt.subplots_adjust(hspace=0.5)
    handle=[[] for i in range(numb_hidden_states)]
    for u in range(numb_hidden_states):
        handle[u] = fig.add_subplot(numb_hidden_states,1,u+1)
        handle[u].plot(given_time_points, GP_post_mean[:,u],color=cmap(0),label='estimated')
        plt.xlabel('time',fontsize=18), 
        if hidden_states[u] in observed_states:
            plt.title('observed %s' % hidden_states[u],loc='left',fontsize=18)
        else:
            plt.title('unobserved %s' % hidden_states[u],loc='left',fontsize=18)
        handle[u].legend(fontsize=12)
    u2=0
    for u in observed_state_idx: 
        handle[u].plot(observed_time_points, observations[:,u2],'*',markersize=7,color=cmap(1),label='observed')
        handle[u].legend(fontsize=12)
        u2 += 1 
       
    # phase space
    if numb_hidden_states==3:
        fig = plt.figure(num=None, figsize=(10, 8), dpi=80)
        ax = fig.gca(projection='3d')
        ax.plot(GP_post_mean[:,0],GP_post_mean[:,1],GP_post_mean[:,2],color=cmap(0),label='estimated')
        ax.set_xlabel(hidden_states[0],fontsize=18)
        ax.set_ylabel(hidden_states[1],fontsize=18)
        ax.set_zlabel(hidden_states[2],fontsize=18)
        ax.set_title('Phase Space',fontsize=18)
        ax.legend(fontsize=15)
        if len(observed_states) == numb_hidden_states:
            ax.plot(observations[:,0],observations[:,1],observations[:,2],'*',markersize=7,color=cmap(1),label='observed')
            ax.legend(fontsize=12)    
    else:
        fig = plt.figure(num=None, figsize=(6, 3), dpi=80)
        ax = fig.add_subplot(111)
        ax.plot(GP_post_mean[:,0],GP_post_mean[:,1],color=cmap(0),label='estimated')
        ax.set_xlabel(hidden_states[0],fontsize=18)
        ax.set_ylabel(hidden_states[1],fontsize=18)
        ax.set_title('Phase Space',fontsize=18)
        ax.legend(fontsize=12)
        if len(observed_states) == numb_hidden_states:
            ax.plot(observations[:,0],observations[:,1],'*',markersize=7,color=cmap(1),label='observed')
            ax.legend(fontsize=12)
    
    #plt.show()
        
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

def kernel_function(given_time_points,kernel_param):
    
    # radial basis function (RBF) kernel
    t = sym.symbols(['t0','t1'])
    rbf_kernel = kernel_param[0] * sym.exp(- (t[0] - t[1])**2 / kernel_param[1]**2)
    
    # kernel derivatives
    cov_func = sym.lambdify(t,rbf_kernel)
    cov_func_d = sym.lambdify(t,rbf_kernel.diff(t[0]))
    cov_func_dd = sym.lambdify(t,rbf_kernel.diff(t[0]).diff(t[1]))
    
    # populate GP covariance matrices
    C = np.zeros((given_time_points.shape[0],given_time_points.shape[0]))
    dC = np.zeros((given_time_points.shape[0],given_time_points.shape[0]))
    Cd = np.zeros((given_time_points.shape[0],given_time_points.shape[0]))
    ddC = np.zeros((given_time_points.shape[0],given_time_points.shape[0]))
    for i in range(0,given_time_points.shape[0]):
        C[i,:] = cov_func(given_time_points[i],given_time_points)
        dC[i,:] = cov_func_d(given_time_points[i],given_time_points)
        Cd[i,:] = cov_func_d(given_time_points,given_time_points[i])
        ddC[i,:] = cov_func_dd(given_time_points[i],given_time_points)
      
    # compute inverse GP covariance matrix
    inv_C = np.linalg.inv(C)
    
    # compute $\mathbf{C}_{\boldsymbol\phi_k}' ~ \mathbf{C}_{\boldsymbol\phi_k}^{-1}$
    dC_times_inv_C = np.dot(dC,inv_C)
    
    # plotting
    cmap = plt.get_cmap("tab10")
    fig = plt.figure(num=None, figsize=(7, 4), dpi=80)
    plt.subplots_adjust(hspace=0.5)
    handle = fig.add_subplot(111)
    prior_state_sample = np.random.multivariate_normal(np.zeros((C.shape[0])),C)
    handle.plot(given_time_points, prior_state_sample,color=cmap(4),label='trajectory sample 1')
    prior_state_sample = np.random.multivariate_normal(np.zeros((C.shape[0])),C)
    handle.plot(given_time_points, prior_state_sample,color=cmap(6),label='trajectory sample 2')
    plt.xlabel('time',fontsize=18), plt.title('Prior State Trajectory Samples',position=(0.5,1),fontsize=18)
    handle.legend(fontsize=12)
    #plt.show()
    
    return dC_times_inv_C,inv_C

# In[4]:
    
# Extracts blocks of size M from the kth diagonal of square matrix A, whose size must be a multiple of M

def extract_block_diag(A,M,k=0):

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