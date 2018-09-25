%% Simulate State Observations
% Authors: Nico Stephan Gorbach and Stefan Bauer

function [simulation,obs_to_state_relation,fig_handle,plot_handle] = simulate_state_dynamics(simulation,state,symbols,ode,odes_path,time,plot_settings)

%%
% Simulate state trajectories by numerical integration
simulation = simulate_state_trajectories(state.derivative_variance,ode,symbols,simulation,odes_path);

%%
% Simulate state observations
[simulation,obs_to_state_relation,fig_handle,plot_handle] = simulate_state_observations(time,simulation,symbols,plot_settings);
end

%% Simulate state trajectories by numerical integration
% Authors: Nico Stephan Gorbach and Stefan Bauer

function simulation = simulate_state_trajectories(derivative_variance,...
    ode,symbols,simulation,odes_path)

%%
% Integration times
time_true=0:simulation.integration_interval:simulation.final_time;

%% 
% Define symbolic variables
param_sym = sym('param%d',[1,length(symbols.param)]); assume(param_sym,'real');
state_sym = sym('state%d',[1,length(symbols.state)]); assume(state_sym,'real');

%%
% Numerical integration
if ~strcmp(odes_path,'Lorenz96_ODEs.txt')
    % Fourth order Runge-Kutta (numerical) integration
    ode_system_mat = matlabFunction(ode.system_sym,'Vars',{state_sym',param_sym'});
    [~,OutX_solver]=ode45(@(t,x) ode_system_mat(x,simulation.ode_param'),time_true,...
        simulation.init_val');
else
    OutX_solver = create_Lorenz96(min(time_true),max(time_true),time_true(2)-time_true(1),...
        derivative_variance',[simulation.ode_param, length(symbols.state)])';
end

%%
% Pack into table
simulation.state = array2table([time_true',OutX_solver],'VariableNames',...
    ['time',symbols.state_string]);
end

%% Simulate observations of states
% Authors: Nico Stephan Gorbach and Stefan Bauer

function [simulation,obs_to_state_relation,fig_handle,plot_handle] = ...
    simulate_state_observations(time,simulation,symbols,plot_settings)

%%
% State observations
integration_time_points = 0:simulation.integration_interval:simulation.final_time;
observation_time_points = 0:simulation.interval_between_observations:simulation.final_time;
observed_time_idx = round(observation_time_points ./ ...
    simulation.integration_interval + ones(1,length(observation_time_points)));

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
end