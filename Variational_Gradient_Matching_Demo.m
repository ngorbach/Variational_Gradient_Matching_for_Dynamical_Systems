%% Variational Gradient Matching for Dynamical Systems
%
% <<cover_pic.png>>
%
% *Author: Nico Stephan Gorbach*, Institute of Machine Learning, ETHZ, email: nico.gorbach@gmail.com
%
% Instructional code for " *Scalable Variational Inference for Dynamical Systems* "
% by Nico S. Gorbach, Stefan Bauer and Joachim M. Buhmann.
% Paper available at <https://papers.nips.cc/paper/7066-scalable-variational-inference-for-dynamical-systems.pdf>.
% Please cite our paper if you use our program for a further publication.

%% Overview
% This document presents a workflow code for applying gradient matching with
% Gaussian processes to Dynamic Causal Models. We start by introducing
% the gradient matching framework, followed by an introduction to dynamic
% causal models. The code for a simulated three-state system is presented thereafter, starting with
% the inputs and ending with the inferred couplings between the neuronal
% populations in the brain. Discussion and future work followed by the appendix is given
% at the end.
%
%% Introduction to Variational Gradient Matching
% The essential idea of gradient matching (Calderhead et al., 2002) is to match the gradient
% governed by the ODE with that inferred from the observations. In contrast
% to previous approaches gradient matching introduces a prior over states
% instead of a prior over ODE parameters. The advantages of gradients matching
% two-fold:
%%
%
% # A prior over the functional form of state dynamics as opposed to ODE parameters facilitates a
% more expert-aware estimation of ODE parameters since experts can provide
% a better _a priori_ description of state dynamics than ODE parameters.
% # Gradient matching yields a global gradient as opposed to a local one which
% offers significant computational advantages and provides access to a rich
% source of sophisticated optimization tools.
%
% <html><h4> Deterministic Dynamical Systems </h4></html>
%
% A deterministic dynamical system is represented by a set of $K$ ordinary differential equations (ODEs) with model parameters $\theta \in R^d$ that describe the evolution of $K$ states $\mathbf{x}(t) = [x_1(t),\ldots, x_K(t)]^T$ such that:
% 
% $\dot{\mathbf{x}}(t) = \frac{d \mathbf{x}(t)}{d t} = \mathbf{f}(\mathbf{x}(t),\theta)$.
% 
% A sequence of observations, $\mathbf{y}(t)$, is usually contaminated by some measurement error which we assume to be normally distributed with zero mean and variance for each of the $K$ states, i.e. $\mathbf{E}\sim \mathcal{N}(\mathbf{E};\mathbf{0},\mathbf{D})$, with $\mathbf{D}_{ik}=\sigma_k ^2 \delta_{ik}$. For $N$ distinct time points the overall system may therefore be summarized as:
% 
% $\mathbf{Y} = \mathbf{X} + \mathbf{E}$,
% 
% where 
%
% $\mathbf{X} = [\mathbf{x}(t_1),\ldots,\mathbf{x}(t_N)] = [\mathbf{x}_1,\ldots,\mathbf{x}_K]^T$,
%
% $\mathbf{Y} = [\mathbf{y}(t_1),\ldots,\mathbf{y}(t_N)] = [\mathbf{y}_1,\ldots,\mathbf{y}_K]^T$,
% 
% and $\mathbf{x}_k = [x_k(t_1),\ldots,x_k(t_N)]^T$ is the $k$'th state sequence and $\mathbf{y}_k = [y_k(t_1),$ $\ldots,y_k(t_N)]^T$ are the observations. Given the observations $\mathbf{Y}$ and the description of the dynamical system \eqref{eqn:ODE}, the aim is to estimate both state variables $\mathbf{X}$ and parameters $\theta$. While numerical integration can be used for both problems, its computational cost is prohibitive for large systems and motivates the grid free method outlined in the following section. 
%
% <html><h4> Gaussian Process based Gradient Matching </h4></html>
%
% Gaussian processes based gradient matching was originally motivated in Calderhead et al (2008) and further developed in Dondelinger et al (2013). In this section we provide a novel derivation for gradient matching with Gaussian processes. Formally, gradient matching with Gaussian processes assumes a joint Gaussian process prior on states and their derivatives:
%
% $\left(\begin{array}{c}
% \mathbf{X} \\ \dot{\mathbf{X}}
% \end{array}\right)
%  \sim \mathcal{N} \left(
% \begin{array}{c}
% \mathbf{X} \\ \dot{\mathbf{X}}
% \end{array}; 
% \begin{array}{c}
%  \mathbf{0} \\ 
% \mathbf{0}
%  \end{array},
% \begin{array}{cc}
%  \mathbf{C}_{\phi} & \mathbf{C}_{\phi}' \\
%  '\mathbf{C}_{\phi} & \mathbf{C}_{\phi}'' 
%  \end{array}
%  \right)$,
%
% with $\mathbf{C}_{\phi}$ denoting the covariance matrix defined by a given kernel with hyperparameters $\phi$. $'\mathbf{C}_{\phi}$ and $\mathbf{C}_{\phi}'$ are the cross-covariances between states and their derivatives and $\mathbf{C}_{\phi}''$ denotes the autocovariance for state derivatives. Since Gaussian processes are closed under differentiation, we can determine the cross-covariances $'\mathbf{C}_{\phi}$ and $\mathbf{C}_{\phi}'$ between states and their derivatives as well as the autocovariance $\mathbf{C}_{\phi}''$ between state derivatives. In particular, the entries of $'\mathbf{C}_{\phi}$, $\mathbf{C}_{\phi}'$ and $\mathbf{C}_{\phi}''$ are given by:
%
% $\mathrm{cov}(x_k(t), x_k(t)) = C_{\phi_k}(t,t')$
%
% $\mathrm{cov}(\dot{x}_k(t), x_k(t)) = \frac{\partial C_{\phi_k}(t,t') }{\partial t} =: C_{\phi_k}'(t,t')$
%
% $\mathrm{cov}(x_k(t), \dot{x}_k(t)) = \frac{\partial C_{\phi_k}(t,t') }{\partial t'} =: {'C_{\phi_k}(t,t')}$
%
% $\mathrm{cov}(\dot{x}_k(t), \dot{x}_k(t)) = \frac{\partial C_{\phi_k}(t,t') }{\partial t \partial t'} =: C_{\phi_k}''(t,t')$.
% 
% Given the joint distribution over states and their derivatives \ref{eqn:joint_state_and_derivatives} as well as the ODEs \ref{eqn:ODE}, we therefore have two expressions for the state derivatives:
%
% $\dot{\mathbf{X}} = \mathbf{F} + \epsilon_1, \epsilon_1 \sim \mathcal{N}\left(\epsilon_1;\mathbf{0}, \mathbf{I}\gamma \right)$
%
% $\dot{\mathbf{X}} = {'\mathbf{C}_{\phi}} \mathbf{C}_{\phi}^{-1} ~\mathbf{X} + \epsilon_2, \epsilon_2 \sim \mathcal{N}\left(\epsilon_2;\mathbf{0}, \mathbf{A} \right)$
%
% where $\mathbf{F} := \mathbf{f}(\mathbf{X},\theta)$, $\mathbf{A} := \mathbf{C}_{\phi}'' -  {'\mathbf{C}_{\phi}} \mathbf{C}_{\phi}^{-1} \mathbf{C}_{\phi}'$ and $\gamma$ is the error variance in the ODEs. Note that, in a deterministic system, the output of the ODEs $\mathbf{F}$ should equal the state derivatives $\dot{\mathbf{X}}$. However, in the first equation of \ref{eqn:state_derivative_expressions} we relax this contraint by adding stochasticity to the state derivatives $\dot{\mathbf{X}}$ in order to compensate for a potential model mismatch. The second equation in \ref{eqn:state_derivative_expressions} is obtained by deriving the conditional distribution for $\dot{\mathbf{X}}$ from the joint distribution in \ref{eqn:joint_state_and_derivatives}. Equating the two expressions in \ref{eqn:state_derivative_expressions} we can eliminate the unknown state derivatives $\dot{\mathbf{X}}$:
%
% $\mathbf{F} = {'\mathbf{C}_{\phi}} \mathbf{C}_{\phi}^{-1} ~\mathbf{X} + \epsilon_0$,
%
% with $\epsilon_0 := \epsilon_2 - \epsilon_1$.
%
% <html><h4> Variational Inference for Gradient Matching by exploiting Local Linearity in ODEs </h4></html>
% 
% For subsequent sections in this chapter, we consider only dynamical systems that are locally linear with respect to ODE parameters $\boldmath\theta$ and individual states $\mathbf{x}_u$. Such ODEs include mass-action kinetics and are given by: 
%
% $f_{k}(\mathbf{x}(t),\boldmath\theta) = \sum_{i=1} \theta_{ki} \prod_{j \in \mathcal{M}_{ki}} x_j$,
%
% with $\mathcal{M}_{ki} \subseteq \{ 1, \dots, K\}$ describing the state variables in each factor of the equation i.e. the functions are linear in parameters and contain arbitrary large products of monomials of the states. This formulation includes models that exhibit periodicity as well as high nonlinearity and especially physically realistic reactions in systems biology \citep{schillings2015efficient}.
% 
% A crucial aspect to notice in equation \ref{eqn:equating_derivative_eqns} and one of the main contributions of this chapter is that, for a restricted class of ODEs such as those described by equation \ref{eqn:ode_spec}, the conditional distributions $p(\boldmath\theta \mid \mathbf{X},\boldmath\phi,\gamma)$ and $p(\mathbf{x}_u\mid \mathbf{X}_{-u},\boldmath\theta,\boldmath{\phi},\gamma)$ are tractable even when the ODEs are nonlinear in all of the states. To see this, we first rewrite the ODEs as a linear combination in the parameters:
%
% $\mathbf{B}_{\boldmath\theta} \boldmath\theta + \mathbf{b}_{\boldmath\theta} \stackrel{!}{=} \mathbf{f}(\mathbf{X},\boldmath\theta)$,
%
% where matrices $\mathbf{B}_{\boldmath\theta}$ and $\mathbf{b}_{\boldmath\theta}$ are defined such that the ODEs $\mathbf{f}(\mathbf{X},\boldmath\theta)$ are expressed as a linear combination in $\boldmath\theta$. Inserting \ref{eqn:lin_comb_param} into \ref{eqn:equating_derivative_eqns} and solving for $\boldmath\theta$ yields:
%
% $\boldmath\theta = \mathbf{B}_{\boldmath\theta}^+ \left( {'\mathbf{C}_{\boldmath\phi}} \mathbf{C}_{\boldmath\phi}^{-1} \mathbf{X} - \mathbf{b}_{\boldmath\theta} + \boldmath\epsilon_0 \right)$,
% 
% where $\mathbf{B}_{\boldmath\theta}^+$ denotes the pseudo-inverse of $\mathbf{B}_{\boldmath\theta}$. We can therefore derive the posterior distribution over ODE parameters:
%
% $p(\boldmath\theta \mid \mathbf{X}, \boldmath\phi, \gamma) = \mathcal{N}\left(\boldmath\theta ; \mathbf{B}_{\boldmath\theta}^+ ~ \left( {'\mathbf{C}_{\boldmath\phi}} \mathbf{C}_{\boldmath\phi}^{-1} \mathbf{X} - \mathbf{b}_{\boldmath\theta} \right), ~ \mathbf{B}_{\boldmath\theta}^+ ~ (\mathbf{A} + \mathbf{I}\gamma) ~ \mathbf{B}_{\boldmath\theta}^{+T} \right)$.
% 
% Similarly, to derive the posterior over an individual state $\mathbf{x}_u$, we rewrite the expression $\mathbf{f}(\mathbf{X},\boldmath\theta) - {'\mathbf{C}}_{\boldmath\phi} \mathbf{C}_{\boldmath\phi}^{-1} \mathbf{X}$ in equation \ref{eqn:equating_derivative_eqns} as a linear combination in the individual state $\mathbf{x}_u$:
%
% $\mathbf{B}_{u} \mathbf{x}_u + \mathbf{b}_{u} \stackrel{!}{=} \mathbf{f}(\mathbf{X},\boldmath\theta) - {'\mathbf{C}}_{\boldmath\phi} \mathbf{C}_{\boldmath\phi}^{-1} \mathbf{X}$.
% 
% Inserting \ref{eqn:lin_comb_states} into \ref{eqn:equating_derivative_eqns} and solving for $\mathbf{x}_u$ yields:
%
% $\mathbf{x}_u = \mathbf{B}_{u}^+ \left( \boldmath\epsilon_0 -\mathbf{b}_{u} \right)$,
%
% where $\mathbf{B}_{u}^+$ denotes the pseudo-inverse of $\mathbf{B}_{u}$. We can therefore derive the posterior distribution over an individual state $\mathbf{x}_u$:
%
% $p(\mathbf{x}_u \mid \mathbf{X}_{-u}, \boldmath\phi, \gamma) = \mathcal{N}\left(\mathbf{x}_u ; -\mathbf{B}_{u}^+ \mathbf{b}_u, ~\mathbf{B}_u^{+} ~ (\mathbf{A} + \mathbf{I}\gamma) ~ \mathbf{B}_u^{+T} \right)$,
%
% with $\mathbf{X}_{-u}$ denoting the set of all states except state $\mathbf{x}_u$.
% 
% Mean-field gradient matching (mean-field GM) at its core can be interpreted as iterating between computing the distribution $p(\boldmath\theta \mid \mathbf{X}, \boldmath\phi, \gamma)$  given the moments of the state distributions and $p(\mathbf{x}_u \mid \mathbf{X}_{-u},\boldmath\theta, \boldmath\phi, \gamma)$ given the moments of the ODE parameter distribution. Such an iterative scheme is derived by coordinate ascent mean-field variational inference described in section \ref{sec:mean_field}.
%
% <html><h4> Mean-field Variational Inference </h4></html>
%
% To infer the parameters $\boldmath\theta$, we want to find the maximum \textit{a posteriori} estimate (MAP): 
%
% $\boldmath\theta^* := arg \max_{\boldmath\theta} ~ \ln p(\boldmath\theta \mid \mathbf{Y},\boldmath\phi,\boldmath\gamma, \boldmath\sigma)$
%
% $= arg\max_{\boldmath\theta} ~ \ln \int  p(\boldmath\theta,\mathbf{X} \mid \mathbf{Y},\boldmath\phi,\boldmath\gamma,\boldmath\sigma) d\mathbf{X}$
%
% $= arg\max_{\boldmath\theta} ~ \ln \int p(\boldmath\theta \mid \mathbf{X},\boldmath\phi,\boldmath\gamma) p(\mathbf{X} \mid \mathbf{Y}, \boldmath\phi,\boldmath\sigma) d\mathbf{X}$.
% 
% However, the integral above is intractable in most cases due to the strong couplings induced by the nonlinear ODEs $\mathbf{f}$ which appear in the term $p(\boldmath\theta \mid \mathbf{X},\boldmath\phi,\boldmath\gamma)$. Notice that, since the ``ODE-informed'' distribution $p(\boldmath\theta \mid \mathbf{X},\boldmath\phi,\boldmath\gamma)$ in the equation above does not depend on the observations $\mathbf{Y}$ and the ``data-informed'' distribution $p(\mathbf{X} \mid \mathbf{Y}, \boldmath\phi, \boldmath\sigma)$ does not depend on the ODEs (i.e. independent of $\gamma$), the ODE parameters $\boldmath\theta$ depend only \textit{indirectly} on the observations $\mathbf{Y}$ through the states $\mathbf{X}$. In other words, given the states $\mathbf{X}$, the ODE parameters $\boldmath\theta$ are conditionally independent of the observations $\mathbf{Y}$. The disadvantage of such a modeling assumption is that it demands a reasonably good estimation of the state trajectories $\mathbf{X}$ since the state trajectories provide the only mechanistic link between the observations $\mathbf{Y}$ and the ODE parameters $\boldmath\theta$.
% 
% The data-informed distribution p(\mathbf{X} \mid \mathbf{Y}, \boldmath\phi,\boldmath\sigma) in the equation above can be  determined analytically using Gaussian process regression with the GP prior $p(\mathbf{X} \mid \boldmath\phi) = \prod_k \mathcal{N}(\mathbf{x}_k ; \mathbf{0},\mathbf{C}_{\boldmath\phi})$:
%
% $p(\mathbf{X} \mid \mathbf{Y}, \boldmath\phi,\gamma) = \prod_k \mathcal{N}(\mathbf{x}_k ; \boldmath\mu_k(\mathbf{y}_k),\boldmath\Sigma_k)$,
%
% where $\boldmath\mu_k(\mathbf{y}_k) := \sigma_k^{-2} \left(\sigma_k^{-2} \mathbf{I} + \mathbf{C}_{\boldmath\phi_k}^{-1} \right)^{-1} \mathbf{y}_k$ and $\boldmath\Sigma_k ^{-1}:=\sigma_k^{-2} \mathbf{I} + \mathbf{C}_{\boldmath\phi_k}^{-1}$.
% 
% We use mean-field variational inference to establish variational lower bounds that are analytically tractable by decoupling state variables from the ODE parameters as well as decoupling the state variables from each other. We first note that, since the ODEs described by \ref{eqn:ode_spec} are \textit{locally linear}, both conditional distributions $p(\boldmath\theta \mid \mathbf{X},\mathbf{Y},\boldmath\phi,\boldmath\gamma,\boldmath\sigma)$ and $p(\mathbf{x}_u \mid \boldmath\theta, \mathbf{X}_{-u},\mathbf{Y},\boldmath\phi,\boldmath\gamma,\boldmath\sigma)$ are analytically tractable and Gaussian distributed as mentioned previously in section \ref{sec:variational_inference_for_gradient_matching}. 
% 
% The decoupling is induced by designing a variational distribution $Q(\boldmath\theta,\mathbf{X})$ which is restricted to the family of factorial distributions:
%
% $\mathcal{Q} := \bigg{\{} Q : Q(\boldmath\theta,\mathbf{X}) = q(\boldmath\theta) \prod_u q(\mathbf{x}_u) \bigg{\}}$.
% 
% The particular form of $q(\boldmath\theta)$ and $q(\mathbf{x}_u)$ are designed to be Gaussian distributed which places them in the same family as the true full conditional distributions. To find the optimal factorial distribution we minimize the Kullback-Leibler divergence between the variational and the true posterior distribution:
%
% $\hat{Q} := arg \min_{Q(\boldmath\theta,\mathbf{X}) \in \mathcal{Q}} \mathrm{KL} \left[ Q(\theta,\mathbf{X}) \mid \mid p(\boldmath\theta,\mathbf{X} \mid \mathbf{Y},\boldmath\phi, \boldmath\gamma,\boldmath\sigma) \right]$,
%
% where $\hat{Q}$ is the proxy distribution. The proxy distribution that minimizes the KL-divergence \ref{eqn:proxy_objective} depends on the true full conditionals and is given by:
%
% $\hat{q}({\boldmath\theta}) \propto \exp \left(~ E_Q \ln p(\boldmath\theta \mid \mathbf{X},\mathbf{Y},\boldmath\phi,\boldmath\gamma,\boldmath\sigma) ~ \right)$
% 
% $\hat{q}(\mathbf{x}_u) \propto \exp\left( ~ E_Q \ln p(\mathbf{x}_u \mid \theta, \mathbf{X}_{-u},\mathbf{Y},\phi,\gamma,\sigma) ~ \right)$.
% 
% Further expanding the optimal proxy distribution in \ref{eqn:proxies} for $\boldmath\theta$ yields:
%
% $\hat{q}(\theta) \stackrel{(a)}{\propto} \exp \left( ~E_Q \ln p(\theta \mid \mathbf{X},\mathbf{Y},\phi,\gamma,\sigma) ~ \right)$
%
% $\stackrel{(b)}{\propto} \exp \left( ~E_Q  \ln \mathcal{N} \left( \theta; \mathbf{B}_{\theta}^+ ~ \left( '\mathbf{C}_{\phi} \mathbf{C}_{\phi}^{-1} \mathbf{X} - \mathbf{b}_{\theta} \right), ~ \mathbf{B}_{\theta}^+ ~ (\mathbf{A} + \mathbf{I}\gamma) ~ \mathbf{B}_{\theta}^{+T} \right) ~\right)$,
%
% which can be normalized analytically due to its exponential quadratic form. In (a) we recall that the ODE parameters depend only indirectly on the observations $\mathbf{Y}$ through the states $\mathbf{X}$ and in (b) we substitute $p(\boldmath\theta \mid \mathbf{X},\boldmath\phi,\boldmath\gamma)$ by its density given in \ref{eqn:posterior_over_param}.
% 
% Similarly, we expand the proxy over the individual state $\mathbf{x}_u$:
%
% $\hat{q}(\mathbf{x}_u) \stackrel{(a)}{\propto} \exp\left( ~ E_Q  \ln p(\mathbf{x}_u \mid \theta, \mathbf{X}_{-u},\phi,\gamma) p(\mathbf{x}_u \mid\mathbf{Y},\phi,\sigma) ~ \right)$
%
% $\stackrel{(b)}{\propto} \exp\big( ~ E_Q \ln \mathcal{N}\left(\mathbf{x}_u ; -\mathbf{B}_{u}^+ \mathbf{b}_u, ~\mathbf{B}_u^{+} ~ (\mathbf{A} + \mathbf{I}\gamma) ~ \mathbf{B}_u^{+T} \right) + E_Q \ln \mathcal{N}\left(\mathbf{x}_u ; \boldmath\mu_u(\mathbf{Y}), \Sigma_u \right) \big)$,
%
% which, once more, can be normalized analytically due to its exponential quadratic form. In (a) we decompose the full conditional into an ODE-informed distribution and a data-informed distribution and in (b) we substitute the ODE-informed distribution $p(\mathbf{x}_u \mid \boldmath\theta, \mathbf{X}_{-u},\boldmath\phi,\boldmath\gamma)$ with its density given by \ref{eqn:posterior_over_state}.
% 
% We can therefore minimize the KL-divergence \ref{eqn:proxy_objective} by coordinate descent (where each step is analytically tractable) by iterating between determining the proxy for the distribution over ODE parameters $\hat{q}(\boldmath\theta)$ and the proxies for the distribution over individual states $\hat{q}(\mathbf{x}_u)$. 


% Clear workspace and close figures
clear all; close all;

%% Simulation Settings

simulation.state_obs_variance = @(mean)(bsxfun(@times,[0.5^2,0.5^2],...
    ones(size(mean))));                                                    % observation noise
simulation.ode_param = [2 1 4 1];                                          % true ODE parameters [2 1 4 1] is used as a benchmark in many publications;
simulation.final_time = 2;                                                 % end time for integration
simulation.int_interval = 0.01;                                            % integration interval
simulation.time_samp = 0:0.1:simulation.final_time;                        % sample times for observations
simulation.init_val = [5 3];                                               % state values at first time point

%% User Input
%
% <html><h4> Kernel </h4></html>

kernel.param_sym = sym(['rbf_param%d'],[1,2]); assume(kernel.param_sym,'real');
kernel.time1 = sym('time1'); assume(kernel.time1,'real'); kernel.time2 = sym('time2'); assume(kernel.time2,'real');
kernel.func = kernel.param_sym(1).*exp(-(kernel.time1-kernel.time2).^2./(kernel.param_sym(2).^2));
kernel.param = [10,0.5];                                                   % set kernel parameter values
kernel.name = 'rbf';                                                       % kernel name
%%
% <html><h4> Estimation </h4></html>
state.derivative_variance = [6,6];                                         % gamma for gradient matching model
time.est = 0:0.1:4;                                                        % estimation times
coord_ascent_numb_iter = 200;                                               % number of coordinate ascent iterations

%%
% <html><h4> Symbols </h4></html>
symbols.state = {'[prey]','[predator]'};                                   % symbols of states in 'ODEs.txt' file
symbols.param = {'[\theta_1]','[\theta_2]','[\theta_3]','[\theta_4]'};     % symbols of parameters in 'ODEs.txt' file

%%
% <html><h4> Path to ODEs </h4></html>
path.ode = './ODEs.txt';                                                   % path to system of ODEs

%%
% <html><h4> Import ODEs </h4></html>
ode = import_odes(path.ode,symbols);

%% Simulate Data
%%
% <html><h4> Generate ground truth by numerical integration </h4></html>
[state,time,ode] = generate_ground_truth(time,state,ode,symbols,simulation);

%%
% <html><h4> Generate state observations </h4></html>
[state,time,obs_to_state_relation] = generate_state_obs(state,time,simulation);

%%
% <html><h4> Symbols </h4></html>
state.sym.mean = sym('x%d%d',[length(time.est),length(ode.system)]);
state.sym.variance = sym('sigma%d%d',[length(time.est),length(ode.system)]);
%assume(state.sym.mean,'real'); assume(state.sym.variance,'real');
ode_param.sym.mean = sym('param%d',[length(symbols.param),1]); assume(ode_param.sym.mean,'real');

%%
% <html><h4> Setup plots </h4></html>
[h,h2] = setup_plots(state,time,simulation,symbols);

%% Prior on States and State Derivatives
[Lambda,dC_times_invC,inv_Cxx,time.est] = kernel_function(kernel,state,time.est);

%% Preprocessing
% <html><h4> Observation mean and covariance </h4></html>
[mu,inv_sigma] = GP_regression(state,inv_Cxx,obs_to_state_relation,simulation);
%%
% <html><h4> Couplings </h4></html>
coupling_idx = find_couplings_in_odes(ode,symbols);
%%
% <html><h4> Rewrite ODEs as linear combination in parameters </h4></html>
[ode_param.B,ode_param.b,ode_param.r,ode_param.B_times_Lambda_times_B] = rewrite_odes_as_linear_combination_in_parameters(ode,symbols);
%%
% <html><h4> Rewrite ODEs as linear combination in individual states </h4></html>
state = rewrite_odes_as_linear_combination_in_ind_states(state,ode,symbols,coupling_idx.states);

%% Proxies for ODE Parameters and Individual States
state.proxy.mean = mu;
for i = 1:coord_ascent_numb_iter
    %%
    % <html><h4> Proxy for ODE parameters </h4></html>
    [param_proxy_mean,param_proxy_inv_cov] = proxy_for_ode_parameters(state.proxy.mean,Lambda,dC_times_invC,ode_param,symbols);
    if ~mod(i,50); plot_results(h,h2,state,time,simulation,param_proxy_mean,'not_final'); end
    %%
    % <html><h4> Proxy for individual states </h4></html>
    state.proxy.mean = proxy_for_ind_states(state.lin_comb,state.proxy.mean,param_proxy_mean',...
        dC_times_invC,coupling_idx.states,symbols,mu,inv_sigma);
end

%%
% <html><h4> Final result </h4></html>
plot_results(h,h2,state,time,simulation,param_proxy_mean,'final');

%% Subroutines
%
% <html><h4> Import ODEs </h4></html>
function ode = import_odes(path_ode,symbols)

tmp = importdata(path_ode);
for k = 1:length(tmp)
for u = 1:length(symbols.state); tmp{k} = strrep(tmp{k},[symbols.state{u}],['state(:,' num2str(u) ')']); end 
for j = 1:length(symbols.param); tmp{k} = strrep(tmp{k},symbols.param{j},['param(' num2str(j) ')']); end
end
for k = 1:length(tmp); ode.system{k} = str2func(['@(state,param)(' tmp{k} ')']); end

end

%%
% <html><h4> Generate ground truth </h4></html>
function [state,time,ode] = generate_ground_truth(time,state,ode,symbols,simulation)

time.true=0:simulation.int_interval:simulation.final_time;                 % true times

Tindex=length(time.true);                                                  % index time
TTT=length(simulation.time_samp);                                          % number of sampled points
itrue=round(simulation.time_samp./simulation.int_interval+ones(1,TTT));    % Index of sample time in the true time

param_sym = sym(['param%d'],[1,length(symbols.param)]); assume(param_sym,'real');
state_sym = sym(['state%d'],[1,length(symbols.state)]); assume(state_sym,'real');
for i = 1:length(ode.system)
    ode.system_sym(i) = ode.system{i}(state_sym,param_sym);
end

ode_system_mat = matlabFunction(ode.system_sym','Vars',{state_sym',param_sym'});
[~,OutX_solver]=ode45(@(t,x) ode_system_mat(x,simulation.ode_param'), time.true, simulation.init_val);
state.true_all=OutX_solver;
state.true=state.true_all(itrue,:);

end

%%
% <html><h4> Generate observations of states </h4></html>
function [state,time,obs_to_state_relation] = generate_state_obs(state,time,simulation)

% State observations
state_obs_variance = simulation.state_obs_variance(state.true);
state.obs = state.true + sqrt(state_obs_variance) .* randn(size(state.true));

% Relationship between states and observations
if length(simulation.time_samp) < length(time.est)
    time.idx = munkres(pdist2(simulation.time_samp',time.est'));
    time.ind = sub2ind([length(simulation.time_samp),length(time.est)],1:length(simulation.time_samp),time.idx);
else
    time.idx = munkres(pdist2(time.est',simulation.time_samp'));
    time.ind = sub2ind([length(time.est),length(simulation.time_samp)],1:length(time.est),time.idx);
end

time.obs_time_to_state_time_relation = zeros(length(simulation.time_samp),length(time.est)); time.obs_time_to_state_time_relation(time.ind) = 1;
state_mat = eye(size(state.true,2));
obs_to_state_relation = sparse(kron(state_mat,time.obs_time_to_state_time_relation));
time.samp = simulation.time_samp;

end

%%
% <html><h4> Kernel function </h4></html>
function [Lambda,dC_times_invC,inv_Cxx,time_est] = kernel_function(kernel,state,time_est)

% kernel derivatives
for i = 1:length(kernel)
    kernel.func_d = diff(kernel.func,kernel.time1);
    kernel.func_dd = diff(kernel.func_d,kernel.time2);
    GP.fun = matlabFunction(kernel.func,'Vars',{kernel.time1,kernel.time2,kernel.param_sym});
    GP.fun_d = matlabFunction(kernel.func_d,'Vars',{kernel.time1,kernel.time2,kernel.param_sym});
    GP.fun_dd = matlabFunction(kernel.func_dd,'Vars',{kernel.time1,kernel.time2,kernel.param_sym});
end

% populate GP covariance matrix
for t=1:length(time_est)
    C(t,:)=GP.fun(time_est(t),time_est,kernel.param);
    dC(t,:)=GP.fun_d(time_est(t),time_est,kernel.param);
    Cd(t,:)=GP.fun_d(time_est,time_est(t),kernel.param);
    ddC(t,:)=GP.fun_dd(time_est(t),time_est,kernel.param);
end

% GP covariance scaling
[~,D] = eig(C); perturb = abs(max(diag(D))-min(diag(D))) / 10000;
if any(diag(D)<1e-6); C(logical(eye(size(C,1)))) = C(logical(eye(size(C,1)))) + perturb.*rand(size(C,1),1); end
[~,D] = eig(C);
if any(diag(D)<0); error('C has negative eigenvalues!'); elseif any(diag(D)<1e-6); warning('C is badly scaled'); end
inv_Cxx = inv_chol(chol(C,'lower'));

dC_times_invC = dC * inv_Cxx;

% plot GP prior samples
figure(3); 
hold on; plot(time_est,mvnrnd(zeros(1,length(time_est)),C(:,:,1),3),'LineWidth',2);
h1 = gca; h1.FontSize = 20; h1.XLabel.String = 'time (s)'; h1.YLabel.String = 'state value';
h1.Title.String = [kernel.name ' kernel'];

% determine \Lambda:
A = ddC - dC_times_invC * Cd;
inv_Lambda = A + state.derivative_variance(1) .* eye(size(A));
inv_Lambda = 0.5.*(inv_Lambda+inv_Lambda');
Lambda = inv_chol(chol(inv_Lambda,'lower'));

end

%%
% <html><h4> GP regression </h4></html>
function [mu_u,inv_sigma_u,state] = GP_regression(state,inv_Cxx,obs_to_state_relation,simulation)

state_obs_variance = simulation.state_obs_variance(state.obs); 

numb_states = size(state.sym.mean,2);
numb_time_points = size(state.sym.mean,1);

inv_Cxx_tmp = num2cell(inv_Cxx(:,:,[1,1]),[1,2]);
inv_Cxx_blkdiag = sparse(blkdiag(inv_Cxx_tmp{:})); 

dim = size(state_obs_variance,1)*size(state_obs_variance,2);
D = spdiags(reshape(state_obs_variance.^(-1),[],1),0,dim,dim) * speye(dim); % covariance matrix of error term (big E)
A_times_D_times_A = obs_to_state_relation' * D * obs_to_state_relation;
inv_sigma = A_times_D_times_A + inv_Cxx_blkdiag;

mu = inv_sigma \ obs_to_state_relation' * D * reshape(state.obs,[],1);

mu_u = zeros(numb_time_points,numb_states);
for u = 1:numb_states
    idx = (u-1)*numb_time_points+1:(u-1)*numb_time_points+numb_time_points;
    mu_u(:,u) = mu(idx);
end

inv_sigma_u = zeros(numb_time_points,numb_time_points,numb_states);
for i = 1:numb_states
    idx = [(i-1)*numb_time_points+1:(i-1)*numb_time_points+numb_time_points];
    inv_sigma_u(:,:,i) = inv_sigma(idx,idx);
end

end

%%
% <html><h4> Setup plots </h4></html>
function [h,h2] = setup_plots(state,time,simulation,symbols)

for i = 1:length(symbols.param); symbols.param{i} = symbols.param{i}(2:end-1); end

figure(1); set(1, 'Position', [0, 200, 1200, 500]);

h2 = subplot(1,3,1); h2.FontSize = 20; h2.Title.String = 'ODE parameters';
set(gca,'XTick',[1:length(symbols.param)]); set(gca,'XTickLabel',symbols.param);
hold on; drawnow

for u = 1:2
    h{u} = subplot(1,3,u+1); cla; plot(time.true,state.true_all(:,u),'LineWidth',2,'Color',[217,95,2]./255); 
    hold on; plot(simulation.time_samp,state.obs(:,u),'*','Color',[217,95,2]./255,'MarkerSize',10);
    h{u}.FontSize = 20; h{u}.Title.String = symbols.state{u}(2:end-1);
    hold on;
end

end

%%
% <html><h4> Find ODE couplings </h4></html>
function coupling_idx = find_couplings_in_odes(ode,symbols)

% state couplings
state_sym = sym(['state%d'],[1,length(ode.system)]); assume(state_sym,'real');
for k = 1:length(ode.system)
    tmp_idx = ismember(state_sym,symvar(ode.system_sym(k))); tmp_idx(:,k) = 1;
    ode_couplings_states(k,tmp_idx) = 1; 
end

for u = 1:length(symbols.state)
    coupling_idx_tmp = find(ode_couplings_states(:,u));
    coupling_idx.states{u} = coupling_idx_tmp;    
end

end

%%
% <html><h4> Rewrite ODEs as linear combination in parameters </h4></html>
function [B,b,r,B_times_Lambda_times_B] = rewrite_odes_as_linear_combination_in_parameters(ode,symbols)

param_sym = sym(['param%d'],[1,length(symbols.param)]); assume(param_sym,'real');
state_sym = sym(['state%d'],[1,length(symbols.state)]); assume(state_sym,'real');
state0_sym = sym(['state0']); assume(state0_sym,'real');
state_const_sym = sym(['state_const']); assume(state_const_sym,'real');

% Rewrite ODEs as linear combinations in parameters
[B_sym,b_sym] = equationsToMatrix(ode.system_sym,param_sym);

% Product of ODE factors (product of Gaussians)
for k = 1:length(ode.system)
    B_sym(k,B_sym(k,:)=='0') = state0_sym;
    for i = 1:length(B_sym(k,:))
        sym_var = symvar(B_sym(k,i));
        if isempty(sym_var)
            B_sym(k,i) = B_sym(k,i) + state0_sym;
        end
    end
    B{k} = matlabFunction(B_sym(k,:),'Vars',{state_sym,state0_sym,state_const_sym});
    b{k} = matlabFunction(b_sym(k,:),'Vars',{state_sym,state0_sym,state_const_sym});
end

B_times_Lambda_times_B = @(B,Lambda)(B' * B);
r = @(B,Lambda,dC_times_invC,state,b)(B' * (dC_times_invC * state + b));

end

%%
% <html><h4> Rewrite ODEs as linear combination in individual states </h4></html>
function state = rewrite_odes_as_linear_combination_in_ind_states(state,ode,symbols,coupling_idx)

state_sym = sym('state%d',[1,length(symbols.state)]); assume(state_sym,'real');
param_sym = sym('param%d',[1,length(symbols.param)]); assume(param_sym,'real');

for u = 1:length(symbols.state)
    for k = coupling_idx{u}'
        [B,b] = equationsToMatrix(ode.system{k}(state_sym,param_sym'),state_sym(:,u));
        state.lin_comb{u,k}.B = matlabFunction(B,'Vars',{state_sym,param_sym});
        state.lin_comb{u,k}.b = matlabFunction(b,'Vars',{state_sym,param_sym});
    end
end

end

%%
% <html><h4> Proxy for ODE parameters </h4></html>
function [param_proxy_mean,param_inv_cov] = proxy_for_ode_parameters(state_proxy_mean,Lambda,dC_times_invC,ode_param,symbols)

B_global = []; b_global = [];
state0 = zeros(size(dC_times_invC,1),1);
param_inv_cov = zeros(length(symbols.param));
local_mean_sum = zeros(length(symbols.param),1);
for k = 1:length(symbols.state)
    B = ode_param.B{k}(state_proxy_mean,state0,...
        ones(size(state_proxy_mean,1),1));
    local_inv_cov = ode_param.B_times_Lambda_times_B(B,Lambda);
    b = ode_param.b{k}(state_proxy_mean,state0,ones(size(state_proxy_mean,1),1));
    local_mean = ode_param.r(B,Lambda,dC_times_invC,state_proxy_mean(:,k),b);
    param_inv_cov = param_inv_cov + local_inv_cov;
    local_mean_sum = local_mean_sum + local_mean;
    
    B_global = [B_global;B];
    b_tmp = b; if length(b_tmp)==1; b_tmp=zeros(size(dC_times_invC,1),1);end
    b_global = [b_global;b_tmp];
end

[~,D] = eig(param_inv_cov);
if any(diag(D)<0)
    warning('param_inv_cov has negative eigenvalues!');
elseif any(diag(D)<1e-3)
    warning('param_inv_cov is badly scaled')
    disp('perturbing diagonal of param_inv_cov')
    perturb = abs(max(diag(D))-min(diag(D))) / 10000;
    param_inv_cov(logical(eye(size(param_inv_cov,1)))) = param_inv_cov(logical(eye(size(param_inv_cov,1)))) ...
        + perturb.*rand(size(param_inv_cov,1),1);
end
param_proxy_mean = pinv(param_inv_cov) * local_mean_sum;

end

%%
% <html><h4> Proxy for individual states </h4></html>
function [state_mean,state_inv_cov] = proxy_for_ind_states(lin_comb,state,...
    ode_param,dC_times_invC,coupling_idx,symbols,mu,inv_sigma)

for u = 1:length(symbols.state)
    
    state_inv_cov(:,:,u) = zeros(size(dC_times_invC));
    local_mean_sum = zeros(size(dC_times_invC,1),1);
    for k = coupling_idx{u}'
        if k~=u
            B = diag(lin_comb{u,k}.B(state,ode_param));
            if size(B,1) == 1; B = B.*eye(size(dC_times_invC,1)); end

            state_inv_cov(:,:,u) = state_inv_cov(:,:,u) + B' * B;
            local_mean_sum = local_mean_sum + B' * (dC_times_invC * state(:,k) ...
                + lin_comb{u,k}.b(state,ode_param));
        else
            B = diag(lin_comb{u,k}.B(state,ode_param));
            if size(B,1) == 1; B = B.*eye(size(dC_times_invC,1)); end
            B = B - dC_times_invC;

            state_inv_cov(:,:,u) = state_inv_cov(:,:,u) + B' * B;
            
            l = lin_comb{u,k}.b(state,ode_param); if length(l)==1; l = zeros(length(local_mean_sum),1); end
            local_mean_sum = local_mean_sum + B' * l;
        end
    end
    
    state_mean(:,u) = (state_inv_cov(:,:,u) + inv_sigma(:,:,u)) \ (local_mean_sum + (inv_sigma(:,:,u) * mu(:,u)));
end

end

%%
% <html><h4> Plot results </h4></html>
function plot_results(h,h2,state,time,simulation,param_proxy_mean,plot_type)

for u = 1:2
    if strcmp(plot_type,'final')
        hold on; plot(h{u},time.est,state.proxy.mean(:,u),'Color',[117,112,179]./255,'LineWidth',2);
    else
        hold on; plot(h{u},time.est,state.proxy.mean(:,u),'LineWidth',0.1,'Color',[0.8,0.8,0.8]); 
    end
end
cla(h2); b = bar(h2,[1:length(param_proxy_mean)],[simulation.ode_param',param_proxy_mean]);
b(1).FaceColor = [217,95,2]./255; b(2).FaceColor = [117,112,179]./255;
h2.XLim = [0.5,length(param_proxy_mean)+0.5]; h2.YLimMode = 'auto';   

end