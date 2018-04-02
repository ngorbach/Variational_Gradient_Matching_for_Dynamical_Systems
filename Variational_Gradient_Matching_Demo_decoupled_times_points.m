%% ANTICIPATE: EFFICIENT AND FLEXIBLE MODELLING FOR DYNAMICAL SYSTEMS
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
% Gaussian processes to Dynamic Causal Models. The name "Anticipate" refers to
% the variational analysis applied to gradient matching. We start by introducing
% the gradient matching framework, followed by an introduction to dynamic
% causal models. The code for a simulated three-state system is presented thereafter, starting with
% the inputs and ending with the inferred couplings between the neuronal
% populations in the brain. Discussion and future work followed by the appendix is given
% at the end.
%
%% Introduction to Anticipate
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
clear all; close all; %clc; %cd('../../');

%% Simulation settings

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
time.est = 0:0.1:2;                                                        % estimation times
coord_ascent_numb_iter = 100;                                              % number of coordinate ascent iterations

%%
% <html><h4> Symbols </h4></html>
symbols.state = {'[prey]','[predator]'};                                   % symbols of states in 'ODEs.txt' file
symbols.param = {'[\theta_1]','[\theta_2]','[\theta_3]','[\theta_4]'};     % symbols of parameters in 'ODEs.txt' file

%%
% <html><h4> Path to ODEs </h4></html>
path.ode = './ODEs.txt';                                      % path to system of ODEs

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

%% Prior on states and state derivatives
[Lambda,dC_times_invC,inv_Cxx,time.est] = kernel_function(kernel,state,time.est);

%% Preprocessing
% <html><h4> Observation mean and covariance </h4></html>
[mu,inv_sigma,state] = GP_regression(state,inv_Cxx,obs_to_state_relation,simulation);

%%
% <html><h4> State coefficients </h4></html>
[state_coeff,ode] = gather_state_coeffs(ode,ode_param.sym.mean,symbols);

%%
% <html><h4> Coeff functions </h4></html>
coeff = coeff_functions;

%%
% <html><h4> ODE parameters Gaussian sufficient statistics </h4></html>
ode_param_suff_stat = ode_param_Gaussian_suff_stat(ode,symbols);

%% Coordinate ascent mean-field variational inference
[state,ode_param] = mean_field(state,ode,state_coeff,coeff,ode_param,...
    Lambda,dC_times_invC,mu,inv_sigma,time,h,h2,ode_param_suff_stat,...
    coord_ascent_numb_iter,simulation);

%% Plot proxies
plot_results(h,state,time);

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
%
time.obs_time_to_state_time_relation = zeros(length(simulation.time_samp),length(time.est)); time.obs_time_to_state_time_relation(time.ind) = 1;
state_mat = eye(size(state.true,2));
obs_to_state_relation = sparse(kron(state_mat,time.obs_time_to_state_time_relation));
time.samp = simulation.time_samp;
%
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

% kernel initialization
tmp = zeros(length(time_est),length(time_est),length(kernel));
Cxx = tmp; dC = tmp; Cd = tmp; ddC = tmp;

% populate GP covariance matrix
for t=1:length(time_est)
    Cxx(t,:)=GP.fun(time_est(t),time_est,kernel.param);
    dC(t,:)=GP.fun_d(time_est(t),time_est,kernel.param);
    Cd(t,:)=GP.fun_d(time_est,time_est(t),kernel.param);
    ddC(t,:)=GP.fun_dd(time_est(t),time_est,kernel.param);
end
Cxx=0.5*(Cxx+Cxx');

% GP covariance scaling
[~,D] = eig(Cxx); perturb = abs(max(diag(D))-min(diag(D))) / 10000;
if any(diag(D)<1e-6)
    disp('perturbing Cxx');
    Cxx(logical(eye(size(Cxx,1)))) = Cxx(logical(eye(size(Cxx,1)))) + perturb.*rand(size(Cxx,1),1);
end
[~,D] = eig(Cxx);
if any(diag(D)<0)
    error('Cxx has negative eigenvalues!');
elseif any(diag(D)<1e-6)
    warning('Cxx is badly scaled')
end
inv_Cxx = inv_chol(chol(Cxx,'lower'));

dC_times_invC = dC * inv_Cxx;

% plot GP prior samples
figure(3); 
hold on; plot(time_est,mvnrnd(zeros(1,length(time_est)),Cxx(:,:,1),3),'LineWidth',2);
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

for i = 1:length(symbols.param)
    symbols.param{i} = symbols.param{i}(2:end-1);
end

figure(1); set(1, 'Position', [0, 200, 1200, 500]);

h2 = subplot(1,3,1);
h2.FontSize = 20; h2.YLabel.String = 'ODE parameters'; h2.YLabel.Rotation = 0; 
h2.YLabel.Units = 'normalized'; h2.YLabel.Position(1) = h2.Title.Position(1);
h2.YLabel.Position(2) = h2.Title.Position(2);
set(gca,'XTick',[1:length(symbols.param)]); set(gca,'XTickLabel',symbols.param);
if length(symbols.param)>9; h2.XTickLabelRotation = 90; end; hold on; drawnow

for u = 1:2
    h{u} = subplot(1,3,u+1); cla
    plot(time.true,state.true_all(:,u),'LineWidth',2,'Color',[217,95,2]./255); hold on;    
    hold on; plot(simulation.time_samp,state.obs(:,u),'*','Color',[217,95,2]./255,'MarkerSize',10);
    h{u}.FontSize = 20; h{u}.Title.Units = 'normalized'; h{u}.YLabel.String = symbols.state{u}(2:end-1);
    h{u}.YLabel.Rotation = 0; h{u}.YLabel.Units = 'normalized'; h{u}.YLabel.Position(1) = 0.1;
    h{u}.YLabel.Position(2) = h2.Title.Position(2); h{u}.XLim = [time.est(1), time.est(end)];
    hold on;
end

end

%%
% <html><h4> Gather state coeffs </h4></html>
function [state_coeff,ode] = gather_state_coeffs(ode,ode_param,symbols)

    
    state_tmp = sym('x%d',[1,length(symbols.state)]);
    variance_tmp = sym('sigma%d',[1,length(symbols.state)]);
    assume(state_tmp,'real'); assume(variance_tmp,'real');
    d = sym('d'); assume(d,'real');
    
    symbols.state_sym = state_tmp;
    symbols.param_sym = ode_param;
    
    ode_system_sym = sym('a',[1,length(ode.system)]);
    for k = 1:length(ode.system); ode_system_sym(k) = ode.system{k}(symbols.state_sym,symbols.param_sym); end
    
    state_coeff = cell(length(ode.system),size(state_tmp,2));
    for k = 1:length(ode.system)
        [~,ode.state_idx{k}] = ismember(symvar(ode.system{k}(state_tmp,ode_param)),state_tmp); ode.state_idx{k}(ode.state_idx{k}==0) = [];
        if ~mod(k,10); fprintf('.'); end
        state_local = state_tmp(ode.state_idx{k});
        variance_local = variance_tmp(ode.state_idx{k});
        for u = ode.state_idx{k}
            [ode_coeff_tmp,ode_const_tmp] = equationsToMatrix(ode.system{k}(state_tmp,ode_param),state_tmp(u));
            state_coeff{u,k}.coeff = matlabFunction(ode_coeff_tmp,'Vars',{state_tmp,ode_param});
            state_coeff{u,k}.square = matlabFunction(substitute_second_order_moments(ode_coeff_tmp.^2,...
                state_local,variance_local),'Vars',{state_tmp,variance_tmp,ode_param});
            state_coeff{u,k}.coeff_times_state = matlabFunction(substitute_second_order_moments(ode_coeff_tmp.*state_tmp(u),...
                state_local,variance_local),'Vars',{state_tmp,variance_tmp,ode_param});
            state_coeff{u,k}.coeff_minus_d_square = matlabFunction(substitute_second_order_moments((ode_coeff_tmp-d).^2,...
                state_local,variance_local),'Vars',{state_tmp,variance_tmp,ode_param,d});
            state_coeff{u,k}.const = matlabFunction(-ode_const_tmp,'Vars',{state_tmp,ode_param});
        end
    end   
end

%%
% <html><h4> Coeff functions </h4></html>
function coeff = coeff_functions


% Please look at the file "state_coefficients.pdf" for further details.

%%
% For u not equal k:
%
% rewrite ODE as linear combination in $x_u$ ($x_u$ is a vector): Define
% $B_{uk}$ and $b_{uk}$ such that: $B_{uk} x_u + b_{uk} = f_k(X,\theta)$
% where:
%
% $B_{uk} :=
% \left(\begin{array}{cccc}
% w_{uk}(1)  &  0     &    0     &    0 \\
%           0    &   w_{uk}(2)  &   0     &      0   \\
%             0    &       0     &    \ddots  & \vdots \\
%            0     &      0    &     \ldots    &   w_{uk}(T)
% \end{array}\right)$,
% $x_u :=
% \left(\begin{array}{c}
% x_u(1) \\
% x_u(2) \\
% \vdots \\
% x_u(T)
% \end{array}\right)$,
% $b_u :=
% \left(\begin{array}{c}
% b(1) \\
% b(2) \\
% \vdots \\
% b(T)
% \end{array}\right)$.
%
% $\log N(f_k(X,\theta) | D*x_k, \Lambda_k) = \log N(B_{uk} x_u | m_k - b_{uk}, \Lambda_k)
%   \propto -0.5 * (B_{uk} x_u)^T \Lambda_k B_{uk} x_u + (B_{uk} x_u)^T \Lambda_k (m_k - b_{uk})$
%
% $w_{uk}$ is called "ode_coeff" in the code
% $b_{uk}$ is called "ode_const" in the code
% $w_{uk}$ is called "ode_coeff_times_state" in the code
% $D$ is called "dC_times_invC" in the code
%
% coefficient of squared x_u(\alpha) ("coeff.u_not_k.square"):
% ( $B_{uk}^T \Lambda_k B_{uk} )_{\alpha,\alpha} = \Lambda_{\alpha,\alpha} w_{uk}(\alpha)^2
% w_{uk}(\alpha)^2$ := "ode_coeff_square"

coeff.u_not_k.square = @(state,variance,ode_param,Lambda,ode_coeff_square)(-0.5.*diag(Lambda) .* ...
    ode_coeff_square(state,variance,ode_param));
%%
%
% coefficients of monomial $x_u(\alpha)$ ("coeff.u_not_k.mon.term1"):
% $\sum_{t \neq \alpha} (B_{uk}^T \Lambda_k B_{uk})_{\alpha t} x_u(t) =
% w_{uk}(\alpha) * \sum_{t\neq \alpha} w_{uk}(t) x_u(t):$

coeff.u_not_k.mon.term1 = @(ode_param,Lambda,dC_times_invC,ode_coeff,ode_coeff_times_state)(ode_coeff .* ...
    (Lambda * ode_coeff_times_state - diag(Lambda) .* ode_coeff_times_state));
%%
%
% coefficients of monomial $x_u(\alpha)$ ("coeff.u_not_k.mon.term2"):
% $( B_{uk}^T \Lambda_k (D x_k - b_{uk}) )_{\alpha} = w_{uk}(\alpha) *
% \sum_t \Lambda_k(\alpha,t) * ( \sum_{t'} d_{t t'} x_u(t') - b_{u}(t') ):$

coeff.u_not_k.mon.term2 = @(state,ode_param,Lambda,dC_times_invC,ode_coeff,ode_const)(ode_coeff .* Lambda * ...
    (dC_times_invC * state - ode_const));
coeff.u_not_k.mon.term1_plus_term2 = @(mon_coeff_term1,mon_coeff_term2)(-mon_coeff_term1 + mon_coeff_term2);

%%
%
% For u equal k:
%
% rewrite ODE as linear combination in $x_u$ ($x_u$ is a vector): $B_{uu} x_u + b_{uu} \stackrel{!}{=} f_u(X,\theta)$
%
% $\log N(f_u(X,\theta) | D*x_u, \Lambda_u) = \log N(B_{uu} x_u | D*x_u -
% b_{uu}, \Lambda_u) = \log N(B_{uu} x_u - D x_u | 0, \Lambda^{-1}_u)$
% Define $G_u$ such that: $G_u x_u = B_{uu} x_u - D x_u$
%
% $G_{uu} :=
% \left(\begin{array}{cccc}
% w_{uu}(1) - d_{uu}(1) & 0 & 0 & 0 \\
% 0 & w_{uu}(2) - d_{uu}(2) & 0 & 0 \\
% 0 & 0 & \ddots & \vdots \\
% 0 & 0 & \ldots & w_{uu}(T) - d_{uu}(T)
% \end{array}\right)$
%
% $\log N(B_{uu} x_u - D x_u | 0, \Lambda^{-1}_u) = \log N(G_u x_u | 0, \Lambda^{-1}_u)
% \propto -0.5 * x_u(\alpha)^2 * ( G_u^T \Lambda_u G_u )_{\alpha \alpha} - x_u(\alpha) \sum_{t\neq \alpha} ( G_u^T \Lambda G_u )_{\alpha t} x_u(t)$
%
% $w_{uk}$ is called "ode_coeff" in the code
% $b_{uk}$ is called "ode_const" in the code
% $w_{uk}$ is called "ode_coeff_times_state" in the code
% $D$ is called "dC_times_invC" in the code

coeff.u_equal_k.B = @(ode_coeff)(diag(ode_coeff));
coeff.u_equal_k.B_times_Lambda_times_B = @(B,Lambda)(B'*Lambda*B);
%%
%
% $G_u^T \Lambda_u G_u = B_{uu}^T \Lambda_u B_{uu} - B_{uu}^T \Lambda_u D_u
% - D_u^T \Lambda_u B_{uu} + D_u^T \Lambda D_u:$
coeff.u_equal_k.G_times_Lambda_times_G = @(Lambda,B,B_times_Lambda_times_B,dC_times_invC)(B_times_Lambda_times_B - ...
    B'*Lambda*dC_times_invC - dC_times_invC'*Lambda*B + dC_times_invC'*Lambda*dC_times_invC);
%%
%
% coefficients of squared $x_u(\alpha)$
coeff.u_equal_k.square = @(G_times_Lambda_times_G)(-0.5 * diag(G_times_Lambda_times_G));
%%
%
% $-\sum_{t\neq \alpha} ( G_u^T \Lambda G_u )_{\alpha t} x_u(t) = -( G_u^T \Lambda_u G_u x_u - trace(G_u^T \Lambda_u G_u) .* x_u )_{\alpha}:$
coeff.u_equal_k.mon = @(state,G_times_Lambda_times_G)(-(G_times_Lambda_times_G * state - diag(G_times_Lambda_times_G) .* state));

% Observations
coeff.obs.square = @(inv_sigma_u)(-0.5 .* diag(inv_sigma_u));
coeff.obs.mon = @(state,inv_sigma_u,mu_u)(-(inv_sigma_u * state - diag(inv_sigma_u) .* state) + inv_sigma_u * mu_u);

end

%%
% <html><h4> Sufficient Statistics for ODE parameters </h4></html>
function ode_param_suff_stat = ode_param_Gaussian_suff_stat(ode,symbols)

param_sym = sym(['param%d'],[1,4]); assume(param_sym,'real');
state_sym = sym(['state%d'],[1,length(symbols.state)]); assume(state_sym,'real');
state0_sym = sym(['state0']); assume(state0_sym,'real');
state_const_sym = sym(['state_const']); assume(state_const_sym,'real');

[B_sym,b_sym] = equationsToMatrix(ode.system_sym,param_sym);
for k = 1:length(ode.system)
    B_sym(k,B_sym(k,:)=='0') = state0_sym;
    for i = 1:length(B_sym(k,:))
        sym_var = symvar(B_sym(k,i));
        if isempty(sym_var)
            B_sym(k,i) = B_sym(k,i) + state0_sym;
        end
    end
    ode_param_suff_stat.B{k} = matlabFunction(B_sym(k,:),'Vars',{state_sym,state0_sym,state_const_sym});
    ode_param_suff_stat.b{k} = matlabFunction(b_sym(k,:),'Vars',{state_sym,state0_sym,state_const_sym});
end

ode_param_suff_stat.B_times_Lambda_times_B = @(B,Lambda)(B' * B);
ode_param_suff_stat.r = @(B,Lambda,dC_times_invC,state,b)(B' * (dC_times_invC * state + b));

end

%%
% <html><h4> Coordinate ascent variational mean-field </h4></html>
function [state,ode_param] = mean_field(state,ode,state_coeff,coeff,ode_param,...
    Lambda,dC_times_invC,mu,inv_sigma,time,h,h2,ode_param_suff_stat,coord_ascent_numb_iter,simulation)

% ---initialize proxies
state_proxy_mean = mu;
for u = 1:size(state.sym.mean,2)
    state_proxy_variance(:,u) = 1./diag(inv_sigma(:,:,u));
end

% proxy for ODE parameters
param_proxy_mean = compute_ode_param_proxy(state_proxy_mean,Lambda,dC_times_invC,...
    ode_param_suff_stat,ode_param);

% plotting
figure(1);
for u = 1:2
    hold on; plot(h{u},time.est,state_proxy_mean(:,u),'LineWidth',0.5,'Color',[0.4,0.4,0.4]);
end
b = bar(h2,[1:length(param_proxy_mean)],[simulation.ode_param',param_proxy_mean]);
b(1).EdgeColor = 'none'; b(1).FaceAlpha = 1; b(1).FaceColor = [217,95,2]./255;
b(2).EdgeColor = 'none'; b(2).FaceAlpha = 1; b(2).FaceColor = [117,112,179]./255;
h2.XLim = [0.5,length(param_proxy_mean)+0.5];
drawnow

% coordinate ascent
for i=1:coord_ascent_numb_iter

    
    % proxy for state
    for u = 1:2
        for alpha = 1:length(time.est)
            square_coeff_global = zeros(size(state_proxy_mean,1),1); mon_coeff_global = zeros(size(state_proxy_mean,1),1);
            
            % determine coefficients per ODE
            for k = 1:length(ode.system)
                if any(u==ode.state_idx{k})
                    if k==u
                        [square_coeff_local, mon_coeff_local] = coeffs_for_u_equal_k(state_proxy_mean,state_proxy_variance,...
                            param_proxy_mean,Lambda,dC_times_invC,state_coeff,coeff,u,k);
                        
                    else
                        [square_coeff_local, mon_coeff_local] = coeffs_for_u_not_k(state_proxy_mean,state_proxy_variance,...
                            param_proxy_mean,Lambda,dC_times_invC,state_coeff,coeff,u,k);
                        
                    end
                    square_coeff_global = square_coeff_global + square_coeff_local;
                    mon_coeff_global = mon_coeff_global + mon_coeff_local;
                end
            end
            
            % add coefficients from observation component only if the state is observed!!!
            square_coeff_global = square_coeff_global + coeff.obs.square(inv_sigma(:,:,u));
            mon_coeff_global = mon_coeff_global + coeff.obs.mon(state_proxy_mean(:,u),inv_sigma(:,:,u),mu(:,u));
            
            % determine mean and variance
            state_proxy_mean(alpha,u) = -mon_coeff_global(alpha)./(2*square_coeff_global(alpha));
            state_proxy_variance(alpha,u) = -1/(2*square_coeff_global(alpha));
        end
    end

    % proxy for ODE parameters
    param_proxy_mean = compute_ode_param_proxy(state_proxy_mean,Lambda,dC_times_invC,...
        ode_param_suff_stat,ode_param);
    
    % plotting
    for u = 1:2
        hold on; plot(h{u},time.est,state_proxy_mean(:,u),'LineWidth',0.5,'Color',[0.4,0.4,0.4]);
    end
    hold on; cla(h2);
    b = bar(h2,[1:length(param_proxy_mean)],[simulation.ode_param',param_proxy_mean]);
    b(1).EdgeColor = 'none'; b(1).FaceAlpha = 1; b(1).FaceColor = [217,95,2]./255;
    b(2).EdgeColor = 'none'; b(2).FaceAlpha = 1; b(2).FaceColor = [117,112,179]./255;
    h2.XLim = [0.5,length(param_proxy_mean)+0.5]; h2.YLimMode = 'auto';
    
    % write to structure
    ode_param.proxy.mean = param_proxy_mean;
    state.proxy.mean = state_proxy_mean;
    state.proxy.variance = state_proxy_variance;
end

end

%%
% <html><h4> Proxy for ODE parameters </h4></html>
function [ode_param_mean,ode_param_inv_cov] = compute_ode_param_proxy(state_proxy_mean,...
    Lambda,dC_times_invC,ode_param_suff_stat,ode_param)

B_global = []; b_global = [];
state0 = zeros(size(state_proxy_mean,1),1);
ode_param_inv_cov = zeros(length(ode_param.sym.mean));
local_mean_sum = zeros(length(ode_param.sym.mean),1);
for k = 1:size(state_proxy_mean,2)
    B = ode_param_suff_stat.B{k}(state_proxy_mean,state0,...
        ones(size(state_proxy_mean,1),1));
    local_inv_cov = ode_param_suff_stat.B_times_Lambda_times_B(B,Lambda);
    b = ode_param_suff_stat.b{k}(state_proxy_mean,state0,ones(size(state_proxy_mean,1),1));
    local_mean = ode_param_suff_stat.r(B,Lambda,dC_times_invC,state_proxy_mean(:,k),b);
    ode_param_inv_cov = ode_param_inv_cov + local_inv_cov;
    local_mean_sum = local_mean_sum + local_mean;
    
    B_global = [B_global;B];
    if length(b)==1; b=zeros(size(dC_times_invC,1),1);end
    b_global = [b_global;b];
end


[~,D] = eig(ode_param_inv_cov);
if any(diag(D)<0)
    error('ode_param_inv_cov has negative eigenvalues!');
elseif any(diag(D)<1e-6)
    warning('ode_param_inv_cov is badly scaled')
    disp('perturbing diagonal of ode_param_inv_cov')
    perturb = abs(max(diag(D))-min(diag(D))) / 10000;
    ode_param_inv_cov(logical(eye(size(ode_param_inv_cov,1)))) = ode_param_inv_cov(logical(eye(size(ode_param_inv_cov,1)))) ...
        + perturb.*rand(size(ode_param_inv_cov,1),1);
end
ode_param_mean = abs(ode_param_inv_cov \ local_mean_sum);

end

%%
% <html><h4>  coeffs for u=k </h4></html>
function [square_coeff, mon_coeff] = coeffs_for_u_equal_k(state_proxy_mean,...
    state_proxy_variance,param_proxy_mean,Lambda,dC_times_invC,state_coeff,coeff,u,k)


state_coeff_tmp = state_coeff{u,k}.coeff(state_proxy_mean,param_proxy_mean);

B = coeff.u_equal_k.B(state_coeff_tmp);

B_times_Lambda_times_B = ...
    coeff.u_equal_k.B_times_Lambda_times_B(B,Lambda);

B_times_Lambda_times_B(logical(eye(size(B_times_Lambda_times_B)))) = ...
    state_coeff{u,k}.square(state_proxy_mean,state_proxy_variance,param_proxy_mean) .* ...
    diag(Lambda);

G_times_Lambda_times_G = ...
    coeff.u_equal_k.G_times_Lambda_times_G(Lambda,...
    B,B_times_Lambda_times_B,dC_times_invC);

square_coeff = coeff.u_equal_k.square(G_times_Lambda_times_G);

mon_coeff = coeff.u_equal_k.mon(state_proxy_mean(:,k),G_times_Lambda_times_G);

end

%%
% <html><h4> Coeffs for $u not equal k$ </h4></html>
function [square_coeff, mon_coeff] = coeffs_for_u_not_k(state_proxy_mean,...
    state_proxy_variance,param_proxy_mean,Lambda,dC_times_invC,state_coeff,coeff,u,k)


square_coeff = coeff.u_not_k.square(state_proxy_mean,...
    state_proxy_variance,param_proxy_mean,Lambda,state_coeff{u,k}.square);

ode_coeff_times_state = ...
    state_coeff{u,k}.coeff_times_state(state_proxy_mean,state_proxy_variance,param_proxy_mean);

state_coeff_tmp = state_coeff{u,k}.coeff(state_proxy_mean,param_proxy_mean);

state_const_tmp = state_coeff{u,k}.const(state_proxy_mean,param_proxy_mean);

mon_term1 = coeff.u_not_k.mon.term1(param_proxy_mean,...
    Lambda,dC_times_invC,...
    state_coeff_tmp,ode_coeff_times_state);

mon_term2 = coeff.u_not_k.mon.term2(state_proxy_mean(:,k),param_proxy_mean,...
    Lambda,dC_times_invC,...
    state_coeff_tmp,state_const_tmp);

mon_coeff = coeff.u_not_k.mon.term1_plus_term2(mon_term1,mon_term2);

end

%%
% <html><h4> Plot results </h4></html>
function plot_results(h,state,time)

for u = 1:2
    shaded_region = [state.proxy.mean(:,u)+1*sqrt(state.proxy.variance(:,u));...
        flipdim(state.proxy.mean(:,u)-1*sqrt(state.proxy.variance(:,u)),1)];
    f = fill(h{u},[time.est'; flipdim(time.est',1)], shaded_region, [222,235,247]/255); set(f,'EdgeColor','None');
    hold on; plot(h{u},time.est,state.proxy.mean(:,u),'Color',[117,112,179]./255,'LineWidth',2);
end
drawnow

end

%%
% <html><h4> Sustitute second order moments </h4></html>
function f = substitute_second_order_moments(f,state,variance)    

f = expand(f);
f = subs(expand(f),state.^2,state.^2+variance);

end