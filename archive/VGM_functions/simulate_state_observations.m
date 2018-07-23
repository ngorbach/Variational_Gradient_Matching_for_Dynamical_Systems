%% Simulate observations of states
% Authors: Nico Stephan Gorbach and Stefan Bauer

function [simulation,obs_to_state_relation,fig_handle,plot_handle] = ...
    simulate_state_observations(time,simulation,symbols,plot_settings)

%%
% State observations
integration_time_points = 0:simulation.integration_interval:simulation.final_time;
observed_time_idx = round(integration_time_points ./ ...
    simulation.integration_interval + ones(1,length(integration_time_points)));

state_true = simulation.state{observed_time_idx,simulation.observed_states};
state_obs_variance = simulation.state_obs_variance(state_true);

observed_states = state_true + sqrt(state_obs_variance) .* randn(size(state_true));
observed_time_points = integration_time_points(observed_time_idx);

% Pack into table
simulation.observations = array2table([observed_time_points',observed_states],...
    'VariableNames',['time',simulation.observed_states]);

%%
% Mapping between states and observations
if length(simulation.observations{:,'time'}) < length(time.est)
    time.idx = munkres(pdist2(simulation.observed_time_points',time.est'));
    time.ind = sub2ind([length(simulation.observed_time_points),length(time.est)],...
        1:length(simulation.observed_time_points),time.idx);
else
    time.idx = munkres(pdist2(time.est',simulation.observations{:,'time'}));
    time.ind = sub2ind([length(time.est),length(simulation.observations{:,'time'})],...
        1:length(time.est),time.idx);
end
obs_time_to_state_time_relation = zeros(length(simulation.observations{:,'time'}),...
    length(time.est));
obs_time_to_state_time_relation(time.ind) = 1;
state_mat = eye(size(simulation.state{:,symbols.state_string},2));

tmp = cellfun(@(x) {strcmp(x,simulation.observed_states)},symbols.state_string);
state_obs_idx = cellfun(@(x) any(x),tmp);
state_mat(~state_obs_idx,:) = [];
obs_to_state_relation = sparse(kron(state_mat,obs_time_to_state_time_relation));

%%
% Plot the observations
% Only the state dynamics are (partially) observed.
[fig_handle.states,fig_handle.param,plot_handle] = setup_plots(time,simulation,symbols,plot_settings);