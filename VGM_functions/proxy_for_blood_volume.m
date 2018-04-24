%% Proxy for blood volume
% Authors: Nico Stephan Gorbach and Stefan Bauer

function vol_proxy_mean = proxy_for_blood_volume(vol,dC_times_invC,state_proxy_mean,ode_param,symbols,...
    A_plus_gamma_inv,opt_settings)

state_idx = find(cellfun(@(x) strcmp(x(1),'v'),symbols.state_string));
state_partner_idx = find(cellfun(@(x) strcmp(x(1),'q'),symbols.state_string));

j = 0;
% Iteratate through states
for u = state_idx

    % unpack matrices B and b
    j = j+1;
    R = diag(vol.R{u}(state_proxy_mean,ode_param'));
    r = vol.r{u}(state_proxy_mean,ode_param);
    if size(R,1) == 1; R = R.*eye(size(dC_times_invC,1)); end
    
    %%
    % Define matrices B and b such that $\mathbf{B}_{uk} \mathbf{x}_u + \mathbf{b}_{uk} \stackrel{!}{=} \mathbf{f}_k(\mathbf{X},\boldmath\theta) - {'\mathbf{C}}_{\phi_{k}} \mathbf{C}_{\phi_{k}}^{-1} \mathbf{X}$
    B = R;
    b = r - dC_times_invC * state_proxy_mean(:,state_partner_idx(j));
    
    if strcmp(opt_settings.pseudo_inv_type,'Moore-Penrose')
        local_mean = -B' * b;
        local_scaling = B' * B;
        local_inv_cov = B' * A_plus_gamma_inv * B;
    elseif strcmp(opt_settings.pseudo_inv_type,'modified Moore-Penrose')
        local_mean = -B' * A_plus_gamma_inv * b;
        local_scaling = B' * A_plus_gamma_inv * B;
        local_inv_cov = local_scaling;
    end
        
    vol_proxy_mean(:,j) = (8/17) * log(local_scaling \ local_mean);
    % Check if blood volume is positive
    if any(~isreal(vol_proxy_mean(:,j)))
        disp('warning: blood volume is not positive')
        vol_proxy_mean(:,j) = real(vol_proxy_mean(:,j));
    end
end