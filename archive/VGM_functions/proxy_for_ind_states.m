%% Proxy for individual states
% Authors: Nico Stephan Gorbach and Stefan Bauer
%
% $\hat{q}(\mathbf{x}_u) \propto \exp\big( ~ E_{Q_{-u}} \ln
% \mathcal{N}\left(\mathbf{x}_u ; \left( \mathbf{B}_{u} \mathbf{B}_{u}^T
% \right)^{-1} \left( - \sum_k \mathbf{B}_{uk}^T \mathbf{b}_{uk} \right),
% ~\mathbf{B}_{u}^{+} ~ (\mathbf{A} + \mathbf{I}\gamma) ~ \mathbf{B}_u^{+T}
% \right)$
%
% $\qquad \qquad \qquad \qquad \qquad + E_{Q_{-u}} \ln
% \mathcal{N}\left(\mathbf{x}_u ; \boldmath\mu_u(\mathbf{Y}), \mathbf{\sigma}_u
% \right) \big)$,

function [state_proxy_mean,state_proxy_inv_cov] = proxy_for_ind_states(lin_comb,...
    state_proxy_mean,ode_param,dC_times_invC,coupling_idx,symbols,mu,inv_sigma,...
    observed_states,A_plus_gamma_inv,opt_settings)

%%
% Indices of observed states
tmp = cellfun(@(x) {strcmp(x,observed_states)},symbols.state_string);
state_obs_idx = cellfun(@(x) any(x),tmp);

%%
% Clamp observed states to GP fit
if opt_settings.clamp_obs_state_to_GP_fit
    state_enumeration = find(~state_obs_idx);
else
    state_enumeration = 1:length(symbols.state);
end

%%
% Iterate through states
for u = state_enumeration
    
    %%
    % Initialization
    state_proxy_inv_cov(:,:,u) = zeros(size(dC_times_invC));
    global_scaling = zeros(size(dC_times_invC));
    global_mean = zeros(size(dC_times_invC,1),1);
    
    %%
    % Iteratate through ODEs
    for k = coupling_idx{u}'
        
        %%
        % unpack matrices $\mathbf{R}$ and $\mathbf{r}$
        R = diag(lin_comb.R{u,k}(state_proxy_mean,ode_param));
        r = lin_comb.r{u,k}(state_proxy_mean,ode_param);
        if size(R,1) == 1; R = R.*eye(size(dC_times_invC,1)); end
        if length(r)==1; r = zeros(length(global_mean),1); end
        
        %%
        % Define matrices B and b such that $\mathbf{B}_{uk} \mathbf{x}_u +
        % \mathbf{b}_{uk} \stackrel{!}{=} \mathbf{f}_k(\mathbf{X},\mathbf{\theta}) -
        % {'\mathbf{C}}_{\mathbf{\phi}_{k}} \mathbf{C}_{\mathbf{\phi}_{k}}^{-1} \mathbf{X}$
        if k~=u
            B = R;
            b = r - dC_times_invC * state_proxy_mean(:,k);
        else
            B = R - dC_times_invC;
            b = r;
        end
        
        %%
        % Local operations
        if strcmp(opt_settings.pseudo_inv_type,'Moore-Penrose')
            % local mean: $\mathbf{B}_{uk}^T \left(\mathbf{\epsilon_0}^{(k)}
            % -\mathbf{b}_{uk}
            local_mean = -B' * b;
            local_scaling = B' * B;
            local_inv_cov = B' * A_plus_gamma_inv * B;
        elseif strcmp(opt_settings.pseudo_inv_type,'modified Moore-Penrose')
            local_mean = -B' * A_plus_gamma_inv * b;
            local_scaling = B' * A_plus_gamma_inv * B;
            local_inv_cov = local_scaling;
        end
        
        %%
        % Global operations
        global_mean = global_mean + local_mean;
        global_scaling = global_scaling + local_scaling;
        
        %%
        % Inverse covariance for state proxy distribution
        state_proxy_inv_cov(:,:,u) = state_proxy_inv_cov(:,:,u) + local_inv_cov;
    end
    
    %%
    % Mean of state proxy distribution (option: Moore-penrose inverse
    % example): $\left( \mathbf{B}_{u} \mathbf{B}_{u}^T \right)^{-1} \sum_k
    % \mathbf{B}_{uk}^T \left(\mathbf{\epsilon_0}^{(k)} -\mathbf{b}_{uk} \right)$
    state_proxy_mean(:,u) = (global_scaling + inv_sigma(:,:,u)) \ (global_mean + ...
        (inv_sigma(:,:,u) * mu(:,u)));
end