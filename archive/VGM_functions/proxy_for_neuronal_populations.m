%% Proxy for neuronal populations
% Authors: Nico Stephan Gorbach and Stefan Bauer

function [neuronal_proxy_mean,neuronal_proxy_inv_cov] = proxy_for_neuronal_populations(neuronal,...
state_proxy_mean,ode_param,dC_times_invC,coupling_idx,symbols,A_plus_gamma_inv,opt_settings)

state_idx = find(cellfun(@(x) strcmp(x(1),'n'),symbols.state_string));
j = 0;
for u = state_idx
    
    j = j+1;
    
    % Initialize
    neuronal_proxy_inv_cov(:,:,u) = zeros(size(dC_times_invC));
    global_scaling = zeros(size(dC_times_invC,1),1);
    global_mean = zeros(size(dC_times_invC,1),1);
    
    for k = coupling_idx{u}'
        
        % unpack matrices B and b
        R = diag(neuronal.R{u,k}(state_proxy_mean,ode_param));
        r = neuronal.r{u,k}(state_proxy_mean,ode_param);
        if size(R,1) == 1; R = R.*eye(size(dC_times_invC,1)); end
        
        %%
        % Define matrices B and b such that $\mathbf{B}_{uk} \mathbf{x}_u + \mathbf{b}_{uk} \stackrel{!}{=} \mathbf{f}_k(\mathbf{X},\boldmath\theta)  - {'\mathbf{C}}_{\phi_{k}} \mathbf{C}_{\phi_{k}}^{-1} \mathbf{X}$
        if k~=u
            B = R;
            b = r - dC_times_invC * state_proxy_mean(:,k);
        else
            B = R - dC_times_invC;
            b = r;
        end
        
        % local
        if strcmp(opt_settings.pseudo_inv_type,'Moore-Penrose')
            local_mean = -B' * b;
            local_scaling = B' * B;
            local_inv_cov = B' * A_plus_gamma_inv * B;
        elseif strcmp(opt_settings.pseudo_inv_type,'modified Moore-Penrose')
            local_mean = -B' * A_plus_gamma_inv * b;
            local_scaling = B' * A_plus_gamma_inv * B;
            local_inv_cov = local_scaling;
        end
        
        % global
        global_mean = global_mean + local_mean;
        global_scaling = global_scaling + local_scaling;
        
        % Inverse covariance for state proxy distribution
        neuronal_proxy_inv_cov(:,:,u) = neuronal_proxy_inv_cov(:,:,u) + local_inv_cov;
    end
    
    % Mean of state proxy distribution
    neuronal_proxy_mean(:,j) = global_scaling \ global_mean;
end