%% Proxy for blood flow
% Authors: Nico Stephan Gorbach and Stefan Bauer

function flow_proxy_mean = proxy_for_blood_flow(flow,dC_times_invC,...
    state_proxy_mean,ode_param,symbols,A_plus_gamma_inv,opt_settings)

state_idx = find(cellfun(@(x) strcmp(x(1),'f'),symbols.state_string));
state_partner_idx = find(cellfun(@(x) strcmp(x(1),'s'),symbols.state_string));

j = 0;
for u = state_idx
    
    % unpack matrices B and b
    j = j+1;
    R = diag(flow.R{u}(state_proxy_mean,ode_param'));
    r = flow.r{u}(state_proxy_mean,ode_param);
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
    
    flow_proxy_mean(:,j) = log(local_scaling \ local_mean);
    % Check if the blood flow is positive
    if any(~isreal(flow_proxy_mean))
        disp('warning: blood flow is not positive')
        flow_proxy_mean(:,j) = real(flow_proxy_mean(:,j));
    end
end