%% Proxy for ODE parameters
% Authors: Nico Stephan Gorbach and Stefan Bauer
%
% $\hat{q}(\mathbf{\theta}) {\propto} \exp \bigg( ~E_{Q_{-\mathbf{\theta}}}  \ln
% \mathcal{N}\left(\mathbf{\theta} ; \left( \mathbf{B}_{\mathbf{\theta}}^T
% \mathbf{B}_{\mathbf{\theta}} \right)^{-1} \left( \sum_k \mathbf{B}_{\mathbf{\theta} k}^T ~
% \left( {'\mathbf{C}_{\mathbf{\phi} k}} \mathbf{C}_{\mathbf{\phi} k}^{-1} \mathbf{X}_k -
% \mathbf{b}_{\mathbf{\theta} k} \right) \right), ~ \mathbf{B}_{\mathbf{\theta}}^+ ~
% (\mathbf{A} + \mathbf{I}\gamma) ~ \mathbf{B}_{\mathbf{\theta}}^{+T} \right)
% ~\bigg)$,

function [param_proxy_mean,param_proxy_inv_cov] = ...
    proxy_for_ode_parameters(state_proxy_mean,dC_times_invC,lin_comb,symbols,...
    A_plus_gamma_inv,opt_settings)

%%
% Initialization
state0 = zeros(size(dC_times_invC,1),1);
param_proxy_inv_cov = zeros(length(symbols.param));
global_scaling = zeros(length(symbols.param));
global_mean = zeros(length(symbols.param),1);

%%
% Iteratate through ODEs
%for k = 1:sum(cellfun(@(x) ~strcmp(x([1:3,end]),'[u_]'),symbols.state_string))
for k = 1:sum(cellfun(@(x) ~strcmp(x(1),'u'),symbols.state_string))
    
    %%
    % unpack matrices $\mathbf{B}$ and $\mathbf{b}$
    B = lin_comb.B{k}(state_proxy_mean,state0,ones(size(state_proxy_mean,1),1));
    b = lin_comb.b{k}(state_proxy_mean,state0,ones(size(state_proxy_mean,1),1));
    
    %%
    % Local operations
    if strcmp(opt_settings.pseudo_inv_type,'Moore-Penrose')
        %%
        % The Moore-Penrose inverse of $\mathbf{B}_{\mathbf{\theta}}$ is given by:
        % $\mathbf{B}_{\mathbf{\theta}}$ is given by: $\mathbf{B}_{\mathbf{\theta}}^+ :=
        % \left(\mathbf{B}_{\mathbf{\theta}}^T \mathbf{B}_{\mathbf{\theta}} \right)^{-1}
        % \mathbf{B}_{\mathbf{\theta}}^T$
        %
        % local mean: $\mathbf{B}_{\mathbf{\theta} k}^T ~ \left(
        % {'\mathbf{C}_{\mathbf{\phi}_k}} \mathbf{C}_{\mathbf{\phi} k}^{-1} \mathbf{X}_k -
        % \mathbf{b}_{\mathbf{\theta} k} \right)$
        local_mean = B' * (dC_times_invC * state_proxy_mean(:,k) - b);
        local_scaling = B' * B;
        local_inv_cov = B' * A_plus_gamma_inv * B;
    elseif strcmp(opt_settings.pseudo_inv_type,'modified Moore-Penrose')
        %%
        % The modified Moore-Penrose inverse of $\mathbf{B}_{\mathbf{\theta}}$ is
        % given by: $\mathbf{B}_{\mathbf{\theta}}$ is given by:
        % $\mathbf{B}_{\mathbf{\theta}}^+ := \left(\mathbf{B}_{\mathbf{\theta}}^T (\mathbf{A}
        % + \mathbf{I}\gamma) \mathbf{B}_{\mathbf{\theta}} \right)^{-1}
        % \mathbf{B}_{\mathbf{\theta}}^T (\mathbf{A} + \mathbf{I}\gamma)$
        %
        % local mean: $\mathbf{B}_{\mathbf{\theta} k}^T (\mathbf{A} +
        % \mathbf{I}\gamma) ~ \left( {'\mathbf{C}_{\mathbf{\phi}_k}}
        % \mathbf{C}_{\mathbf{\phi} k}^{-1} \mathbf{X}_k - \mathbf{b}_{\mathbf{\theta} k}
        % \right)$
        local_mean = B' * A_plus_gamma_inv * (dC_times_invC * state_proxy_mean(:,k) - b);
        local_scaling = B' * A_plus_gamma_inv * B;
        local_inv_cov = local_scaling;
    end
    
    %%
    % Global operations
    global_mean = global_mean + local_mean;
    global_scaling = global_scaling + local_scaling;
    
    % Inverse covariance of ODE param proxy distribution
    param_proxy_inv_cov = param_proxy_inv_cov + local_inv_cov;
end

%%
% Include prior on ODE parameters
if isfield(opt_settings,'ode_param_prior')
    global_mean = global_mean + opt_settings.ode_param_prior.inv_cov * ...
        opt_settings.ode_param_prior.mean;
    global_scaling = global_scaling + opt_settings.ode_param_prior.inv_cov;
    param_proxy_inv_cov = param_proxy_inv_cov + opt_settings.ode_param_prior.inv_cov;
end

%%
% Check scaling of covariance matrix
[~,D] = eig(global_scaling);
if any(diag(D)<0)
    warning('scaling has negative eigenvalues!');
elseif any(diag(D)<1e-6)
    disp('Scaling is badly scaled. We therefore perturb diagonal values of scaling.')
    perturb = abs(max(diag(D))-min(diag(D))) / 10000;
    global_scaling(logical(eye(size(global_scaling,1)))) = ...
        global_scaling(logical(eye(size(global_scaling,1)))) ...
        + perturb.*rand(size(global_scaling,1),1);
end

%%
% Mean of parameter proxy distribution (option: Moore-penrose inverse
% example):
%
% $\left( \mathbf{B}_{\mathbf{\theta}}^T \mathbf{B}_{\mathbf{\theta}} \right)^{-1}
% \left( \sum_k \mathbf{B}_{\mathbf{\theta} k}^T ~ \left( {'\mathbf{C}_{\mathbf{\phi} k}}
% \mathbf{C}_{\mathbf{\phi} k}^{-1} \mathbf{X}_k - \mathbf{b}_{\mathbf{\theta} k} \right)
% \right)$

param_proxy_mean = global_scaling \ global_mean;