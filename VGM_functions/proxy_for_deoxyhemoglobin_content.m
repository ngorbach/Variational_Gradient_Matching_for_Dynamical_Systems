%% Proxy for deoxyhemoglobin content
% Authors: Nico Stephan Gorbach and Stefan Bauer

function deoxyhemo_proxy_mean = proxy_for_deoxyhemoglobin_content(deoxyhemo,state,...
    bold_response,symbols,A_plus_gamma_inv,opt_settings)

%% 
% find index of deoxyhemoglobin content states
state_idx = find(cellfun(@(x) strcmp(x(1),'q'),symbols.state_string));
state_partner_idx = find(cellfun(@(x) strcmp(x(1),'v'),symbols.state_string));

j = 0;
%% 
% Iterate through deoxyhemglobin content states
for u = state_idx

    %%
    % unpack matrices B and b
    j = j+1;
    R = diag(deoxyhemo.R(state(:,state_partner_idx(j))));
    r = deoxyhemo.r(state(:,state_partner_idx(j)));
     
    %%
    % define matrices $B_{uk}$ and $b_{uk}$ such that
    % $B_{uk} \mathbf{q} + b_{uk} \stackrel{!}{=}
    % \mathbf{f}_k(\mathbf{X},\mathbf{\theta})$
    B = R;
    b = r - bold_response(:,u);
     
    %% 
    % local operations
    if strcmp(opt_settings.pseudo_inv_type,'Moore-Penrose')
        local_mean =  -B' * b;
        local_scaling = B' * B;
        local_inv_cov = B' * A_plus_gamma_inv * B;
    elseif strcmp(opt_settings.pseudo_inv_type,'modified Moore-Penrose')
        local_mean =  -B' * A_plus_gamma_inv * b;
        local_scaling = B' * A_plus_gamma_inv * B;
        local_inv_cov = local_scaling;
    end
     
    %% 
    % proxy mean
    deoxyhemo_proxy_mean(:,u) = log(local_scaling \ local_mean);
    
    %% 
    % check if deoxyhemoglobin content is positive
    if any(~isreal(deoxyhemo_proxy_mean(:,u)))
        % disp('warning: deoxyhemoglobin content is not positive')
        deoxyhemo_proxy_mean(:,u) = real(deoxyhemo_proxy_mean(:,u));
    end
end
