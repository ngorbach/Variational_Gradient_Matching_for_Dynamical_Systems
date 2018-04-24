%% Proxy for vasosignalling
% Authors: Nico Stephan Gorbach and Stefan Bauer

function [vaso_proxy_mean,vaso_proxy_inv_cov] = proxy_for_vasosignalling(vaso,dC_times_invC,...
    state_proxy_mean,ode_param,symbols,A_plus_gamma_inv,opt_settings)

%% 
% find index of vasosignalling states
state_idx = find(cellfun(@(x) strcmp(x(1),'s'),symbols.state_string));
state_partner_idx = find(cellfun(@(x) strcmp(x(1),'f'),symbols.state_string));

j = 0;
for u = state_idx
    
    j = j+1;
    
    %% 
    % Initialization
    global_scaling = zeros(size(dC_times_invC,1),1);
    global_mean = zeros(size(dC_times_invC,1),1);
    
    %%
    % unpack matrices B and b for vasosignalling ODE
    R = diag(vaso.R{u}.vaso(state_proxy_mean,ode_param'));
    r = vaso.r{u}.vaso(state_proxy_mean,ode_param);
    if size(R,1) == 1; R = R.*eye(size(dC_times_invC,1)); end
    if size(r,1) == 1; r = r.*zeros(size(dC_times_invC,1),1); end
    
    %%
    % define matrices B and b such that $\mathbf{B}_{uk} \mathbf{x}_u +
    % \mathbf{b}_{uk} \stackrel{!}{=}
    % \mathbf{f}_k(\mathbf{X},\boldmath\theta) - {'\mathbf{C}}_{\phi_{k}}
    % \mathbf{C}_{\phi_{k}}^{-1} \mathbf{X}$
    B = R - dC_times_invC;
    b = r;
    %%
    if strcmp(opt_settings.pseudo_inv_type,'Moore-Penrose')
        local_mean.vaso = -B' * b;
        local_scaling.vaso = B' * B;
        local_inv_cov.vaso = B' * A_plus_gamma_inv * B;
    elseif strcmp(opt_settings.pseudo_inv_type,'modified Moore-Penrose')
        local_mean.vaso = -B' * b;
        local_scaling.vaso = B' * A_plus_gamma_inv * B;
        local_inv_cov.vaso = local_scaling.vaso;
    end
    
    %%
    % unpack matrices B and b for blood flow ODE
    R = diag(vaso.R{u}.flow(state_proxy_mean,ode_param'));
    r = vaso.r{u}.flow(state_proxy_mean,ode_param);
    if size(R,1) == 1; R = R.*eye(size(dC_times_invC,1)); end
    if size(r,1) == 1; r = r.*zeros(size(dC_times_invC,1),1); end
    
    %%
    % Define matrices B and b such that $\mathbf{B}_{uk} \mathbf{x}_u +
    % \mathbf{b}_{uk} \stackrel{!}{=}
    % \mathbf{f}_k(\mathbf{X},\boldmath\theta) - {'\mathbf{C}}_{\phi_{k}}
    % \mathbf{C}_{\phi_{k}}^{-1} \mathbf{X}$
    B = R;
    b = r - dC_times_invC * state_proxy_mean(:,state_partner_idx(j));
    
    %%
    % local operations
    if strcmp(opt_settings.pseudo_inv_type,'Moore-Penrose')
        local_mean.flow = -B' * b;
        local_scaling.flow = B' * B;
        local_inv_cov.flow = B' * A_plus_gamma_inv * B;
    elseif strcmp(opt_settings.pseudo_inv_type,'modified Moore-Penrose')
        local_mean.flow = -B' * A_plus_gamma_inv * b;
        local_scaling.flow = B' * A_plus_gamma_inv * B;
        local_inv_cov.flow = local_scaling.flow;
    end
    
    %%
    % global mean
    global_mean = local_mean.vaso + local_mean.flow;
    global_scaling = local_scaling.vaso + local_scaling.flow;
      
    %% 
    % mean of state proxy distribution
    vaso_proxy_mean(:,j) = global_scaling \ global_mean;
    
    %% 
    % Inverse covariance for state proxy distribution
    vaso_proxy_inv_cov(:,:,u) = local_inv_cov.vaso + local_inv_cov.flow;
end