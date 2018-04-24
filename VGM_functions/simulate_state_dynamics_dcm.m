%% Simulate State Observations
% Authors: Nico Stephan Gorbach and Stefan Bauer

function [simulation,obs_to_state_relation,fig_handle,plot_handle] = simulate_state_dynamics_dcm(simulation,symbols,ode,time,plot_settings,ext_input,plot_string)

%%
% Simulate state trajectories by numerical integration
simulation = simulate_state_trajectories(ode,symbols,simulation,simulation.ode_param,ext_input,time);

%%
% Simulate state observations
[simulation,obs_to_state_relation,fig_handle,plot_handle] = simulate_state_observations(time,simulation,symbols,plot_settings,ext_input,plot_string);

end

%% Simulate state trajectories by numerical integration
% Authors: Nico Stephan Gorbach and Stefan Bauer

function simulation = simulate_state_trajectories(ode,symbols,simulation,...
    ode_param,ext_input,time)

%% 
% Define symbolic variables
param_sym = sym('param%d',[1,length(symbols.param)]); assume(param_sym,'real');
state_sym = sym('state%d',[1,length(symbols.state)]); assume(state_sym,'real');

%%
% Fourth order Runge-Kutta (numerical) integration
ext_input_idx = cellfun(@(x) strcmp(x(1),'u'),symbols.state_string);
ode_system_mat = matlabFunction(ode.system_sym,'Vars',{state_sym(~ext_input_idx),...
        param_sym,state_sym(ext_input_idx)});

warning ('off','all');      
[OutT,OutX_solver] = ode113(@(t,x) ode_function(t,x,ode_system_mat,ode_param,...
    ext_input(:,2:end),ext_input(:,1)),ext_input(:,1),simulation.init_val);
warning ('on','all');
OutX_solver(1:5,:) = 0;

[~,ext_input_to_bold_response_mapping_idx] = min(pdist2(ext_input(:,1),time.est'),[],1);
state_true = OutX_solver(ext_input_to_bold_response_mapping_idx,:);
time_samp = OutT(ext_input_to_bold_response_mapping_idx)';
ext_input = ext_input(ext_input_to_bold_response_mapping_idx,:);
%%
% Pack into table
simulation.state = array2table([time_samp',state_true,ext_input(:,2:end)],...
    'VariableNames',['time',symbols.state_string]);
end

%% Simulate observations of states
% Authors: Nico Stephan Gorbach and Stefan Bauer

function [simulation,obs_to_state_relation,fig_handle,plot_handle] = ...
    simulate_state_observations(time,simulation,symbols,plot_settings,ext_input,...
    plot_string)

%%
% State observations
integration_time_points = 0:simulation.integration_interval:simulation.final_time;
observation_time_points = 0:simulation.interval_between_observations:simulation.final_time;
observed_time_idx = round(observation_time_points ./ ...
    simulation.integration_interval + ones(1,length(observation_time_points)));

state_true = simulation.state{:,simulation.observed_states};
state_obs_variance = simulation.state_obs_variance(state_true);

observed_states = state_true + sqrt(state_obs_variance) .* randn(size(state_true));
observed_time_points = simulation.state{:,'time'}';

% Pack into table
simulation.observations = array2table([observed_time_points',...
    observed_states],'VariableNames',['time',simulation.observed_states]);

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
state_mat = eye(size(simulation.state{:,symbols.state_string(cellfun(@(x) ~strcmp(x(1),'u'),...
    symbols.state_string))},2));

tmp = cellfun(@(x) {strcmp(x,simulation.observed_states)},...
    symbols.state_string(cellfun(@(x) ~strcmp(x(1),'u'),symbols.state_string)));
state_obs_idx = cellfun(@(x) any(x),tmp);
state_mat(~state_obs_idx,:) = [];
obs_to_state_relation = sparse(kron(state_mat,obs_time_to_state_time_relation));

%%
% Simulate BOLD responses
simulation = simulate_BOLD_response(simulation,symbols);

if strcmp(plot_string,'plot')
    %%
    % Plot the observations
    % Only the state dynamics are (partially) observed.
    %
    % Setup plots for BOLD responses
    fig_handle.bold = setup_plots_for_bold_response_and_ext_input(ext_input,simulation,time,symbols);
    
    % Setup plots for states
    [fig_handle.states,fig_handle.param,plot_handle] = setup_plots(time,simulation,symbols,plot_settings);
else
    fig_handle = []; plot_handle = [];
end
end

%% ODE function
function state_derivatives = ode_function(time,states,ode_system_mat,ode_param,ext_input,time_lst)

[~,idx] = min(pdist2(time,time_lst));
u = ext_input(idx,:);
state_derivatives = ode_system_mat(states',ode_param,u);
end

%% Simulate BOLD responses
% Authors: Nico Stephan Gorbach and Stefan Bauer

function simulation = simulate_BOLD_response(simulation,symbols)

% true bold responses
bold_response.true = bold_signal_change_eqn(simulation.state{:,{'v_1','v_3','v_2'}},...
    simulation.state{:,{'q_1','q_3','q_2'}});

% mean correction
bold_response.true = bsxfun(@minus,bold_response.true,mean(bold_response.true,1));

% observed bold responses
bold_response.variance = simulation.state_obs_variance(bold_response.true);
bold_response.obs = bold_response.true + sqrt(bold_response.variance) .* ...
    randn(size(bold_response.true));

simulation.X0 = importdata('dcm/confounding_effects_X0.txt');

% Pack into table
simulation.bold_response_true = array2table([simulation.state{:,'time'},bold_response.true],...
    'VariableNames',['time',symbols.state_string(cellfun(@(x) strcmp(x(1),'n'),...
    symbols.state_string))]);
simulation.bold_response = array2table([simulation.state{:,'time'},bold_response.obs],...
    'VariableNames',['time',symbols.state_string(cellfun(@(x) strcmp(x(1),'n'),...
    symbols.state_string))]);
end

%% Setup plots for BOLD response
% Authors: Nico Stephan Gorbach and Stefan Bauer

function [h_bold,h_ext_input] = setup_plots_for_bold_response_and_ext_input(ext_input,simulation,time,symbols)

figure(2); set(2, 'Position', [0, 200, 1600, 800]);

plot_titles_idx = find(cellfun(@(x) strcmp(x(1),'n'),symbols.state_string));
plot_idx = [1:2:3*2];
for u = 1:3
    h_bold{u} = subplot(3,2,plot_idx(u)); cla; 
    plot(h_bold{u},simulation.bold_response{:,'time'},...
        simulation.bold_response{:,symbols.state_string(plot_titles_idx(u))},'LineWidth',2,'Color',...
            [217,95,2]./255,'MarkerSize',3);
    h_bold{u}.FontSize = 20; h_bold{u}.Title.String = ['observed BOLD response ' symbols.state_string{plot_titles_idx(u)}];
    h_bold{u}.Title.FontWeight = 'Normal';
    h_bold{u}.XLim = [min(time.est),max(time.est)];
    h_bold{u}.XLabel.String = 'time (s)'; hold on;
end

plot_titles_idx = flipdim(find(cellfun(@(x) strcmp(x(1),'u'),symbols.state_string)),2);
plot_idx = [2:2:3*2];
for i = 1:3
    h_ext_input{i} = subplot(3,2,plot_idx(i));
    plot(h_ext_input{i},ext_input(:,1),ext_input(:,i+1),'LineWidth',2,'Color',[217,95,2]./255); hold on;
    h_ext_input{i}.FontSize = 20; h_ext_input{i}.Title.String = ['given external input ' symbols.state_string{plot_titles_idx(i)}];
    h_ext_input{i}.Title.FontWeight = 'Normal';
    h_ext_input{i}.XLim = [min(time.est),max(time.est)];
    h_ext_input{i}.XLabel.String = 'time (s)'; hold on;
end
drawnow
end