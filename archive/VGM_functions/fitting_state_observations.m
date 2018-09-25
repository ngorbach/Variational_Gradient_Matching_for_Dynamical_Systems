%% Fitting state observations
% Authors: NicoStephan Gorbach and Stefan Bauer
%
% We fit the observations of state trajectories by standard GP regression.
%
% $p(\mathbf{X} \mid \mathbf{Y}, \mathbf{\phi},\gamma) = \prod_k
% \mathcal{N}(\mathbf{x}_k ;
% \boldmath\mu_k(\mathbf{y}_k),\boldmath\mathbf{\sigma}_k)$,
%
% where $\boldmath\mu_k(\mathbf{y}_k) := \mathbf{\sigma}_k^{-2} \left(\mathbf{\sigma}_k^{-2}
% \mathbf{I} + \mathbf{C}_{\boldmath\mathbf{\phi}_k}^{-1} \right)^{-1} \mathbf{y}_k$
% and $\boldmath\mathbf{\sigma}_k ^{-1}:=\mathbf{\sigma}_k^{-2} \mathbf{I} +
% \mathbf{C}_{\boldmath\mathbf{\phi}_k}^{-1}$.

function [mu_reshaped,inv_sigma_reshaped] = fitting_state_observations(inv_C,...
    obs_to_state_relation,simulation,symbols)

%%
% Dimensions
numb_states = sum(cellfun(@(x) ~strcmp(x(1),'u'),symbols.state_string));
numb_time_points = size(inv_C,1);

%%
% Variance of state observations
state_obs_variance = simulation.state_obs_variance(...
    simulation.observations{:,simulation.observed_states});

%%
% Form block-diagonal matrix out of $\mathbf{C}_{\boldmath\mathbf{\phi}_k}^{-1}$
inv_C_replicas = num2cell(inv_C(:,:,ones(1,numb_states)),[1,2]);
inv_C_blkdiag = sparse(blkdiag(inv_C_replicas{:}));

%%
% GP posterior inverse covariance matrix: $\boldmath\mathbf{\sigma}_k
% ^{-1}:=\mathbf{\sigma}_k^{-2} \mathbf{I} + \mathbf{C}_{\boldmath\mathbf{\phi}_k}^{-1}$
dim = size(state_obs_variance,1)*size(state_obs_variance,2);
% covariance matrix of error term (big E):
D = spdiags(reshape(state_obs_variance.^(-1),[],1),0,dim,dim) * speye(dim);
A_times_D_times_A = obs_to_state_relation' * D * obs_to_state_relation;
inv_sigma = A_times_D_times_A + inv_C_blkdiag;

%%
% GP posterior mean: $\boldmath\mu_k(\mathbf{y}_k) := \mathbf{\sigma}_k^{-2}
% \left(\mathbf{\sigma}_k^{-2} \mathbf{I} + \mathbf{C}_{\boldmath\mathbf{\phi}_k}^{-1}
% \right)^{-1} \mathbf{y}_k$
mu = inv_sigma \ obs_to_state_relation' * D * reshape(...
    simulation.observations{:,simulation.observed_states},[],1);

%%
% Reshape GP mean
mu_reshaped = zeros(numb_time_points,numb_states);
for u = 1:numb_states
    idx = (u-1)*numb_time_points+1:(u-1)*numb_time_points+numb_time_points;
    mu_reshaped(:,u) = mu(idx);
end

% Add external input to mu
state_idx = cellfun(@(x) strcmp(x(1),'u'),symbols.state_string);
mu_reshaped(:,state_idx) = simulation.state{:,symbols.state_string(state_idx)};

%%
% Reshape GP inverse covariance matrix
inv_sigma_reshaped = zeros(numb_time_points,numb_time_points,numb_states);
for i = 1:numb_states
    idx = (i-1)*numb_time_points+1:(i-1)*numb_time_points+numb_time_points;
    inv_sigma_reshaped(:,:,i) = inv_sigma(idx,idx);
end
