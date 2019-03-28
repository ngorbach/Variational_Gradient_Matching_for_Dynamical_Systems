
# coding: utf-8

# # Variational Gradient Matching for Dynamical Systems: Lotka-Volterra
#                         Fast Parameter Identification for Nonlinear ODEs
#                         
# <img src="docs/logo.png">
# 
# #### Authors: Nico S. Gorbach and Stefan Bauer
# 
# 
# #### Contents
# 
# Instructional code for the NIPS (2018) paper [Scalable Variational Inference for Dynamical Systems](https://papers.nips.cc/paper/7066-scalable-variational-inference-for-dynamical-systems.pdf) by Nico S. Gorbach, Stefan Bauer and Joachim M. Buhmann. Please cite our paper if you use our program for a further publication. The derivations in this document are also given in this [doctoral thesis](https://www.research-collection.ethz.ch/handle/20.500.11850/261734) as well as in parts of [Wenk et al. (2018)](https://arxiv.org/pdf/1804.04378.pdf).
# Example dynamical system used in this code: Lorenz attractor system with the y-dimension unobserved. The ODE parameters are also unobserved.

# ## Import VGM Modules

# In[1]:


import sys
sys.path.insert(0, './VGM_modules')
from Lotka_Volterra_declarations import *
from import_odes import *
from simulate_state_dynamics import *
from GP_regression import *
from rewrite_odes_as_local_linear_combinations import *
from proxies_for_ode_parameters_and_states import *


# ## User Input

# ### Simulation Settings

# ##### True ODE parameters
# Input a row vector of real numbers of size $1$ x $4$:

# In[2]:


simulation.ode_param = [2,1,4,1]


# ##### Observed states
# Input 1, 2 symbolic variables from the set $(x_1,x_2)$:

# In[3]:


simulation.observed_states = sym.symbols(['_x_1','_x_2'])


# ##### Final time for simulation
# Input a positive real number:

# In[4]:


simulation.final_time_point = 4.0


# ##### Observation noise
# Input a positive real number:

# In[5]:


simulation.SNR = 10


# ##### Time interval between observations
# Input a positive real number:

# In[6]:


simulation.observed_time_points = [0,0.1,0.5,0.6,0.8,1.0,1.4,1.6,2.0]


# ### Estimation Settings

# ##### Positivity constraint on ODE parameters
# Input Boolean Expression:

# In[7]:


opt_settings.ode_param_constraint = None


# ##### Positivity constraint on state trajectories
# Input Boolean Expression:

# In[8]:


opt_settings.state_constraint = 'nonnegative'


# ##### Time points used to estimate the state trajectories
# Input a row vector of positive real numbers in ascending order:

# In[9]:


time_points.for_estimation = np.arange(0,4.0,0.08)


# ## Import Lotka-Volterra ODEs
# 
# The Lotka-Volterra ODEs are given by:
# 
# \begin{align}
#   \dot{x}_1 &= \theta_1 ~ x_1 - \theta_2 ~ x_1 ~ x_2\\
#   \dot{x}_2 &= -\theta_3 ~ x_2 + \theta_4 ~ x_1 ~ x_2
# \end{align}

# In[10]:


odes,odes_sym = import_odes(symbols,odes_path)
state_couplings = find_state_couplings_in_odes(odes_sym,symbols)


# ## Simulate Trajectories

# In[11]:


simulation.state, simulation.observations = setup_simulation(simulation,time_points,symbols,odes,fig_shape,1)


# ## Prior on States and State Derivatives
# Gradient matching with Gaussian processes assumes a joint Gaussian process prior on states and their derivatives:
# 
# \begin{align}
# \begin{bmatrix}
# \mathbf{X} \\ \dot{\mathbf{X}}
# \end{bmatrix} 
#  \sim \mathcal{N} \left(\begin{bmatrix}
# \mathbf{X} \\ \dot{\mathbf{X}}
# \end{bmatrix} ; 
#  \begin{bmatrix}
#  \mathbf{0} \\ \mathbf{0}
#  \end{bmatrix},
#  \begin{bmatrix}
#  \mathbf{K}(\mathbf{t},\mathbf{t}') & \frac{\partial \mathbf{K}(\mathbf{t},\mathbf{t}')}{\partial \mathbf{t}} \\
#  \frac{\partial \mathbf{K}(\mathbf{t},\mathbf{t})}{\partial \mathbf{t}} & \frac{\partial \mathbf{K}(\mathbf{t},\mathbf{t}')}{\partial \mathbf{t} ~ \partial \mathbf{t}'} 
#  \end{bmatrix}
#  \right), \qquad \qquad \qquad (2)
#  \label{eqn:joint_state_and_derivatives}
# \end{align}

# ## Matching Gradients
# 
# Given the joint distribution over states and their derivatives (2) as well as the ODEs (1), we therefore have two expressions for the state derivatives:
# 
# \begin{align}
# &\dot{\mathbf{X}} = \mathbf{F} \nonumber \\
# &\dot{\mathbf{X}} = \frac{\partial \mathbf{K}(\mathbf{t},\mathbf{t}')}{\partial \mathbf{t}}~~ \mathbf{K}^{-1}(\mathbf{t},\mathbf{t}') ~~ \mathbf{X} ~~ + ~~ \boldsymbol\epsilon, &&\boldsymbol\epsilon \sim \mathcal{N}\left(\boldsymbol\epsilon ~ ; ~ \mathbf{0}, ~ \frac{\partial \mathbf{K}(\mathbf{t},\mathbf{t}')}{\partial \mathbf{t}~\partial\mathbf{t}'} ~~ -  ~~ \frac{\partial \mathbf{K}(\mathbf{t},\mathbf{t}')}{\partial \mathbf{t}} ~~ \mathbf{K}^{-1}(\mathbf{t},\mathbf{t}') ~~ \frac{\partial \mathbf{K}(\mathbf{t},\mathbf{t}')}{\partial \mathbf{t}'} \right) \qquad \qquad \qquad (3)
# \label{eqn:state_derivative_expressions}
# \end{align}
# 
# where $\mathbf{F} := \mathbf{f}(\mathbf{X},\boldsymbol\theta)$. The second equation in (3) is obtained by deriving the conditional distribution for $\dot{\mathbf{X}}$ from the joint distribution in (2). Equating the two expressions in (3) we can eliminate the unknown state derivatives $\dot{\mathbf{X}}$:
# 
# \begin{align}
# \mathbf{F} = \frac{\partial \mathbf{K}(\mathbf{t},\mathbf{t}')}{\partial \mathbf{t}}~~ \mathbf{K}^{-1}(\mathbf{t},\mathbf{t}') ~~ \mathbf{X} ~~ + ~~ \boldsymbol\epsilon. \qquad \qquad \qquad (4)
# \end{align}

# In[12]:


dC_times_inv_C,eps_cov,cov,cov_obs,cov_state_obs = kernel_function(time_points.for_estimation,simulation.observations.index)


# ## Rewrite ODE's as Linear Combination in Parameters
# 
# Since, according to the mass action dynamics, the ODEs are linear in the parameters, we can rewrite the ODEs in equation (1) as a linear combination in the parameters:
#     
# \begin{align}
# \mathbf{B}_{\boldsymbol\theta} ~ \boldsymbol\theta + \mathbf{b}_{\boldsymbol\theta} \stackrel{!}{=} \mathbf{f}(\mathbf{X},\boldsymbol\theta), \qquad \qquad \qquad (5)
# \label{eqn:lin_comb_param}
# \end{align}
# 
# where matrices $\mathbf{B}_{\boldsymbol\theta}$ and $\mathbf{b}_{\boldsymbol\theta}$ are defined such that the ODEs $\mathbf{f}(\mathbf{X},\boldsymbol\theta)$ are expressed as a linear combination in $\boldsymbol\theta$.

# In[13]:


locally_linear_odes.ode_param.B,locally_linear_odes.ode_param.b = rewrite_odes_as_linear_combination_in_parameters(odes,symbols.state,symbols.param)


# ## Posterior over ODE Parameters
# 
# Inserting (5) into (4) and solving for $\boldsymbol\theta$ yields:
# 
# \begin{align}
# &\boldsymbol\theta = \mathbf{B}_{\boldsymbol\theta}^+ \left( \frac{\partial \mathbf{K}(\mathbf{t},\mathbf{t}')}{\partial \mathbf{t}} ~~ \mathbf{K}^{-1}(\mathbf{t},\mathbf{t}') ~~ \mathbf{X} ~~ - ~~ \mathbf{b}_{\boldsymbol\theta} ~ + ~ \boldsymbol\epsilon \right),
# \end{align}
# 
# where $\mathbf{B}_{\boldsymbol\theta}^+$ denotes the pseudo-inverse of $\mathbf{B}_{\boldsymbol\theta}$. Since 
# $\mathbf{K}$ is block diagonal we can rewrite the expression above as:
# 
# \begin{align}
# \boldsymbol\theta &= \left(\mathbf{B}_{\boldsymbol\theta}^T \mathbf{B}_{\boldsymbol\theta} \right)^{-1} \mathbf{B}_{\boldsymbol\theta}^T ~ \sum_k ~ \frac{\partial \mathbf{K}_k(\mathbf{t},\mathbf{t}')}{\partial \mathbf{t}} ~~ \mathbf{K}_k^{-1}(\mathbf{t},\mathbf{t}') ~~ \mathbf{X}_k ~~ - ~~  \mathbf{b}_{\boldsymbol\theta}^{(k)} ~ + ~ \boldsymbol\epsilon_k  \\
# &= \left(\mathbf{B}_{\boldsymbol\theta}^T \mathbf{B}_{\boldsymbol\theta} \right)^{-1} \sum_k ~ {\mathbf{B}_{\boldsymbol\theta}^{(k)}}^T ~ \left( ~ \frac{\partial \mathbf{K}_k(\mathbf{t},\mathbf{t}')}{\partial \mathbf{t}} ~~ \mathbf{K}_k^{-1}(\mathbf{t},\mathbf{t}') ~~ \mathbf{X}_k ~~ - ~~ \mathbf{b}_{\boldsymbol\theta}^{(k)} ~ + ~ \boldsymbol\epsilon_k \right)
# \end{align}
# 
# where we substitute the Moore-Penrose inverse for the pseudo inverse ($~$ i.e. $~$ $\mathbf{B}_{\boldsymbol\theta}^+ := \left(\mathbf{B}_{\boldsymbol\theta}^T \mathbf{B}_{\boldsymbol\theta} \right)^{-1} \mathbf{B}_{\boldsymbol\theta}^T ~$). 
# 
# We can therefore derive the posterior distribution over ODE parameters:
#                                                               
# \begin{align}
# p(\boldsymbol\theta \mid \mathbf{X}, \boldsymbol\phi) = \mathcal{N}\left(~ \boldsymbol\theta ~  ; \left(\mathbf{B}_{\boldsymbol\theta}^T \mathbf{B}_{\boldsymbol\theta} \right)^{-1} ~ \sum_k ~ { \mathbf{B}_{\boldsymbol\theta}^{(k)} }^T ~ \left(\frac{\partial \mathbf{K}_k(\mathbf{t},\mathbf{t}')}{\partial \mathbf{t}} ~~ \mathbf{K}_k^{-1}(\mathbf{t},\mathbf{t}') ~~ \mathbf{X}_k - \mathbf{b}_{\boldsymbol\theta}^{(k)} \right), ~ \mathbf{B}_{\boldsymbol\theta}^+ ~ \left(~ \frac{\partial \mathbf{K}(\mathbf{t},\mathbf{t}')}{\partial \mathbf{t}~\partial\mathbf{t}'} ~~ -  ~~ \frac{\partial \mathbf{K}(\mathbf{t},\mathbf{t}')}{\partial \mathbf{t}} ~~ \mathbf{K}^{-1}(\mathbf{t},\mathbf{t}') ~~ \frac{\partial \mathbf{K}(\mathbf{t},\mathbf{t}')}{\partial \mathbf{t}'} ~ \right) ~  ~ \mathbf{B}_{\boldsymbol\theta}^{+T} \right). \qquad \qquad \qquad (6)
# \label{eqn:posterior_over_param}
# \end{align}

# ## Rewrite ODEs as Linear Combination in Individual States
# 
# Since, according to the mass action dynamics, the ODEs are linear in the individual state  $\mathbf{x}_u$, we can rewrite the ODE $\mathbf{f}_k(\mathbf{X},\boldsymbol\theta)$ as a linear combination in the individual state $\mathbf{x}_u$:
# 
# \begin{align}
# \mathbf{R}_{uk} ~ \mathbf{x}_u + \mathbf{r}_{uk} ~~ \stackrel{!}{=} ~~ \mathbf{f}_k(\mathbf{X},\boldsymbol\theta), \qquad \qquad \qquad (7)
# \label{eqn:lin_comb_states}
# \end{align}
# 
# where matrices $\mathbf{R}_{uk}$ and $\mathbf{r}_{uk}$ are defined such that the ODE $\mathbf{f}_k(\mathbf{X},\boldsymbol\theta)$ is expressed as a linear combination in the individual state $\mathbf{x}_u$.

# In[14]:


locally_linear_odes.state.R,locally_linear_odes.state.r = rewrite_odes_as_linear_combination_in_states(odes,symbols.state,symbols.param,simulation.observed_states,state_couplings,opt_settings.clamp_states_to_observation_fit)


# ## Posterior over Individual States
# 
# Given the linear combination of the ODEs w.r.t. an individual state, we define the matrices $\mathbf{B}_u$ and $\mathbf{b}_u$ such that the expression $\mathbf{f}(\mathbf{X},\boldsymbol\theta) ~~ - ~~ \frac{\partial \mathbf{K}_k(\mathbf{t},\mathbf{t}')}{\partial \mathbf{t}} ~~ \mathbf{K}_k^{-1}(\mathbf{t},\mathbf{t}') ~~ \mathbf{X}_k$ is rewritten as a linear combination in an individual state $\mathbf{x}_u$:
# 
# \begin{align}
# \mathbf{B}_{u} ~ \mathbf{x}_u + \mathbf{b}_{u} ~~ \stackrel{!}{=} ~~ \mathbf{f}(\mathbf{X},\boldsymbol\theta) ~~ - ~~ \frac{\partial \mathbf{K}_k(\mathbf{t},\mathbf{t}')}{\partial \mathbf{t}} ~~ \mathbf{K}_k^{-1}(\mathbf{t},\mathbf{t}') ~~ \mathbf{X}_k \qquad \qquad \qquad (8)
# \end{align}
# 
# Inserting (8) into (4) and solving for $\mathbf{x}_u$ yields:
# 
# \begin{align}
# \mathbf{x}_u = \mathbf{B}_{u}^+ \left( \boldsymbol\epsilon -\mathbf{b}_{u} \right),
# \end{align}
# 
# where $\mathbf{B}_{u}^+$ denotes the pseudo-inverse of $\mathbf{B}_{u}$. Since $\mathbf{K}$ is block diagonal we can rewrite the expression above as:
# 
# \begin{align}
# \mathbf{x}_u &= \left( \mathbf{B}_{u} \mathbf{B}_{u}^T \right)^{-1}\mathbf{B}_{u}^T \sum_k \left(\boldsymbol{\epsilon}_k -\mathbf{b}_{u}^{(k)} \right)\\ 
# &= \left( \mathbf{B}_{u} \mathbf{B}_{u}^T \right)^{-1} \sum_k {\mathbf{B}_{u}^{(k)}}^T \left(\boldsymbol{\epsilon}_k -\mathbf{b}_{u}^{(k)} \right),
# \end{align}
# 
# where we subsitute the Moore-Penrose inverse for the pseudo-inverse (i.e. $\mathbf{B}_{u}^+ := \left( \mathbf{B}_{u}^T \mathbf{B}_{u}\right)^{-1} \mathbf{B}_{u}^T$ ).
# 
# We can therefore derive the posterior distribution over an individual state $\mathbf{x}_u$:
# 
# \begin{align}
# p(\mathbf{x}_u \mid \mathbf{X}_{-u}, \boldsymbol\phi)= \mathcal{N}\left(~ \mathbf{x}_u ~ ; ~ -\left( \mathbf{B}_{u} \mathbf{B}_{u}^T\right)^{-1} \sum_k {\mathbf{B}_{u}^{(k)}}^T \mathbf{b}_{u}^{(k)} ~~
#  ,~~ \mathbf{B}_{u}^{+} ~ \left(~ \frac{\partial \mathbf{K}(\mathbf{t},\mathbf{t}')}{\partial \mathbf{t}~\partial\mathbf{t}'} ~~ -  ~~ \frac{\partial \mathbf{K}(\mathbf{t},\mathbf{t}')}{\partial \mathbf{t}} ~~ \mathbf{K}^{-1}(\mathbf{t},\mathbf{t}') ~~ \frac{\partial \mathbf{K}(\mathbf{t},\mathbf{t}')}{\partial \mathbf{t}'} ~ \right) ~ \mathbf{B}_u^{+T}\right) \qquad \qquad \qquad (9)
# \end{align}
# 
# with $\mathbf{X}_{-u}$ denoting the set of all states except state $\mathbf{x}_u$.

# ## Mean-field Variational Inference
# 
# To infer the parameters $\boldsymbol\theta$, we want to find the maximum a posteriori estimate (MAP):
# 
# \begin{align}
# \boldsymbol\theta^\star :&=\mathrm{arg}\max_{\boldsymbol\theta} ~ \ln p(\boldsymbol\theta \mid \mathbf{Y},\boldsymbol\phi,\boldsymbol\sigma) \nonumber \\
# &= \mathrm{arg}\max_{\boldsymbol\theta} ~ \ln \int  p(\boldsymbol\theta,\mathbf{X} \mid \mathbf{Y},\boldsymbol\phi,\boldsymbol\sigma) ~ d\mathbf{X} \nonumber \\
# &= \mathrm{arg}\max_{\boldsymbol\theta} ~ \ln \int  \underbrace{p(\boldsymbol\theta \mid \mathbf{X},\boldsymbol\phi)}_{\textrm{ODE-informed}} ~ \underbrace{p(\mathbf{X} \mid \mathbf{Y}, \boldsymbol\phi, \boldsymbol\sigma)}_{\textrm{data-informed}} ~ d\mathbf{X}.
# \label{eq:map_param}
# \end{align}
# 
# However, the integral above is intractable due to the strong couplings induced by the nonlinear ODEs $\mathbf{f}$ which appear in the term $p(\boldsymbol\theta \mid \mathbf{X},\boldsymbol\phi)$. 
# 
# We use mean-field variational inference to establish variational lower bounds that are analytically tractable by decoupling state variables from the ODE parameters as well as decoupling the state variables from each other. We first note that, since the ODEs (1) are locally linear, both conditional distributions $p(\boldsymbol\theta \mid \mathbf{X},\mathbf{Y},\boldsymbol\phi,\boldsymbol\sigma)$ and $p(\mathbf{x}_u \mid \boldsymbol\theta, \mathbf{X}_{-u},\mathbf{Y},\boldsymbol\phi,\boldsymbol\sigma)$ are analytically tractable and Gaussian distributed as mentioned previously. 
# 
# The decoupling is induced by designing a variational distribution $Q(\boldsymbol\theta,\mathbf{X})$ which is restricted to the family of factorial distributions:
# 
# \begin{align}
# \mathcal{Q} := \bigg{\{} Q : Q(\boldsymbol\theta,\mathbf{X}) = q(\boldsymbol\theta) \prod_u q(\mathbf{x}_u) \bigg{\}}.
# \label{eqn:proxy_family}
# \end{align}
# 
# The particular form of $q(\boldsymbol\theta)$ and $q(\mathbf{x}_u)$ are designed to be Gaussian distributed which places them in the same family as the true full conditional distributions. To find the optimal factorial distribution we minimize the Kullback-Leibler divergence between the variational and the true posterior distribution:
# 
# \begin{align}
# \widehat{Q} :&= \mathrm{arg}\min_{Q(\boldsymbol\theta,\mathbf{X}) \in \mathcal{Q}} \mathrm{KL} \left[ Q(\boldsymbol\theta,\mathbf{X}) ~ \big{|}\big{|} ~ p(\boldsymbol\theta,\mathbf{X} \mid \mathbf{Y},\boldsymbol\phi, \boldsymbol\sigma) \right], \qquad \qquad \qquad (10)
# \label{eqn:proxy_objective}
# \end{align}
# 
# where $\widehat{Q}$ is the proxy distribution. The proxy distribution that minimizes the KL-divergence (10) depends on the true full conditionals and is given by:
# 
# \begin{align}
# &\widehat{q}(\boldsymbol\theta) \propto \exp \left( ~  \mathbb{E}_{Q_{-\boldsymbol\theta}} \ln p(\boldsymbol\theta \mid \mathbf{X},\mathbf{Y},\boldsymbol\phi,\boldsymbol\sigma) ~ \right) \qquad \qquad \qquad ~~~ (11) \\
# &\widehat{q}(\mathbf{x}_u) \propto \exp\left( ~ \mathbb{E}_{Q_{-u}} \ln p(\mathbf{x}_u \mid \boldsymbol\theta, \mathbf{X}_{-u},\mathbf{Y},\boldsymbol\phi,\boldsymbol\sigma) ~ \right). \qquad \qquad (12)
# \label{eqn:proxies}
# \end{align}
# 
# with $Q_{-\boldsymbol\theta} := \prod_u \widehat{q}(\mathbf{x}_u)$ and $Q_{-u} := \widehat{q}(\boldsymbol\theta) \prod_{l\neq u} \widehat{q}(\mathbf{x}_l)$. Further expanding the optimal proxy distribution in (11) for $\boldsymbol\theta$ yields:
# 
# \begin{align}
# &\widehat{q}(\boldsymbol\theta) \stackrel{(a)}{\propto} \exp \left( ~  \mathbb{E}_{Q_{-\boldsymbol\theta}} \ln p(\boldsymbol\theta \mid \mathbf{X},\mathbf{Y},\boldsymbol\phi,\boldsymbol\sigma) ~ \right)
#  \nonumber \\
# &\stackrel{(b)}{=} \exp \left( ~E_{Q_{-\boldsymbol\theta}} \ln \mathcal{N}\left(\boldsymbol\theta
# ; { \mathbf{B}_{\boldsymbol\theta}^T \mathbf{B}_{\boldsymbol\theta} }^{-1}
#  \sum_k    { \mathbf{B}_{\boldsymbol\theta}^{(k)} }^T ~ \left( \frac{\partial \mathbf{K}_k(\mathbf{t},\mathbf{t}')}{\partial \mathbf{t}} ~~ \mathbf{K}_k^{-1}(\mathbf{t},\mathbf{t}') ~~ \mathbf{X}_k ~~ -~~ \mathbf{b}_{\boldsymbol{\theta}
# }^{(k)}    \right), ~ \mathbf{B}_{\boldsymbol\theta}^+ ~ \left(~ \frac{\partial \mathbf{K}(\mathbf{t},\mathbf{t}')}{\partial \mathbf{t}~\partial\mathbf{t}'} ~~ -  ~~ \frac{\partial \mathbf{K}(\mathbf{t},\mathbf{t}')}{\partial \mathbf{t}} ~~ \mathbf{K}^{-1}(\mathbf{t},\mathbf{t}') ~~ \frac{\partial \mathbf{K}(\mathbf{t},\mathbf{t}')}{\partial \mathbf{t}'} ~\right)
# ~    \mathbf{B}_{\boldsymbol\theta}^{+T} \right) ~\right)
# \end{align}
# 
# which can be normalized analytically due to its exponential quadratic form. In (a) we recall that the ODE parameters depend only indirectly on the observations $\mathbf{Y}$ through the states $\mathbf{X}$ and in (b) we substitute $p(\boldsymbol\theta \mid \mathbf{X},\boldsymbol\phi)$ by its density given in (6).
# 
# Similarly, we expand the proxy over the individual state $\mathbf{x}_u$ in equation (12):
# 
# \begin{align}
# &\widehat{q}(\mathbf{x}_u) \stackrel{(a)}{\propto} \exp\left( ~ \mathbb{E}_{Q_{-u}} \ln \left( p(\mathbf{x}_u \mid \boldsymbol\theta, \mathbf{X}_{-u},\boldsymbol\phi) ~ p(\mathbf{x}_u \mid\mathbf{Y},\boldsymbol\phi,\boldsymbol\sigma) ~ \right) \right) \nonumber  \\
# &\stackrel{(b)}{=} \exp\left( ~ \mathbb{E}_{Q_{-u}} \ln \mathcal{N}\left(\mathbf{x}_u ; - \left(    \mathbf{B}_{u}^T \mathbf{B}_{u} \right)^{-1} \sum_k {\mathbf{B}_{u}^{(k)}}^T \mathbf{b}_{u}^{(k)}, ~\mathbf{B}_u^{+} ~ \left(~ \frac{\partial \mathbf{K}(\mathbf{t},\mathbf{t}')}{\partial \mathbf{t}~\partial\mathbf{t}'} ~~ -  ~~ \frac{\partial \mathbf{K}(\mathbf{t},\mathbf{t}')}{\partial \mathbf{t}} ~~ \mathbf{K}^{-1}(\mathbf{t},\mathbf{t}') ~~ \frac{\partial \mathbf{K}(\mathbf{t},\mathbf{t}')}{\partial \mathbf{t}'} ~\right) ~ \mathbf{B}_u^{+T} \right) + \mathbb{E}_{Q_{-u}}  \ln \mathcal{N}\left(\mathbf{x}_u ; \boldsymbol\mu_u(\mathbf{Y}), \boldsymbol\Sigma_u \right) \right)
# \end{align}
# 
# In (a) we decompose the full conditional into an ODE-informed distribution and a data-informed distribution and in (b) we substitute the ODE-informed
# distribution $p(\mathbf{x}_u \mid \boldsymbol\theta, \mathbf{X}_{-u},\boldsymbol\phi)$
# with its density given by equation (9).

# ## Initialization of State Trajectories
# 
# We initialize the states by fitting the observations of state trajectories using classical GP regression. The data-informed distribution $p(\mathbf{X} \mid \mathbf{Y}, \boldsymbol\phi,\boldsymbol\sigma)$ in equation (9) can be determined analytically using Gaussian process regression with the GP prior $p(\mathbf{X} \mid \boldsymbol\phi) = \prod_k \mathcal{N}(\mathbf{x}_k ~ ; ~ \mathbf{0},\mathbf{K}_k)$:
# 
# \begin{align}
# p(\mathbf{X} \mid \mathbf{Y}, \boldsymbol\phi) = \prod_k\mathcal{N}(\mathbf{x}_k ~~ ;~ \boldsymbol\mu_k(\mathbf{y}_k),\boldsymbol\Sigma_k),
# \end{align}
# 
# where $\boldsymbol\mu_k(\mathbf{y}_k) := \sigma_k^{-2} \left(\mathbf{\sigma}_k^{-2}\mathbf{I} + \mathbf{K}_k^{-1} \right)^{-1} \mathbf{y}_k$ and $\boldsymbol\Sigma_k ^{-1}:=\mathbf{\sigma}_k^{-2} \mathbf{I} +\mathbf{K}_k^{-1}$.

# In[15]:


state_pred_mean,state_pred_inv_cov = fitting_state_observations(simulation.observations,symbols.state,simulation.SNR,time_points.for_estimation,cov,cov_obs,cov_state_obs,fig_shape=fig_shape)
proxy.state = state_pred_mean


# ## Variational Coordinate Ascent
# 
# We minimize the KL-divergence in equation (10) by variational coordinate descent (where each step is analytically tractable) by iterating between determining the proxy for the distribution over ODE parameters $\widehat{q}(\boldsymbol\theta)$ and the proxies for the distribution over individual states $\widehat{q}(\mathbf{x}_u)$.

# In[16]:


for i in range(80):
    proxy.param = proxy_for_ode_parameters(proxy.state,locally_linear_odes,dC_times_inv_C,eps_cov,symbols.param,simulation.ode_param,odes,opt_settings.ode_param_constraint)
    proxy.state = proxy_for_ind_states(proxy.state,proxy.param,odes,locally_linear_odes,dC_times_inv_C,eps_cov,state_pred_mean,state_pred_inv_cov,simulation.observations,simulation.state,state_couplings,i,clamp_states_to_observation_fit=False,constraints=opt_settings.state_constraint,optimizer='analytical',fig_shape=fig_shape)


# ## Numerical Integration with Estimated ODE Parameters
# 
# We plug the estimated ODE parameters into a numerical integrator and observe it's trajectories (in green).

# In[17]:


simulation_with_est_param = simulation
simulation_with_est_param.ode_param = proxy.param
setup_simulation(simulation_with_est_param,time_points,symbols,odes,fig_shape,2,simulation.observations,proxy.state);

