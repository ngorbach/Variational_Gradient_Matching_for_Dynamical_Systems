%% Preprocessing for Lorenz 96
% Authors: Nico Stephan Gorbach and Stefan Bauer
function [symbols,ode,plot_settings,state,simulation,odes_path,coupling_idx,...
    opt_settings] = preprocessing_dynamic_causal_modeling(simulation,...
    candidate_odes,state)

%%
% Path to ODEs
odes_path = ['dcm/ODEs/' candidate_odes '.txt'];

%%
% Symbols: symbols of states and parameters in the '_ODEs.txt' file without delimiters '[' and ']':
symbols_raw = importdata(['dcm/ODEs/' candidate_odes '_symbols.mat']);

% Refine symbols
symbols.state = cell2sym(cellfun(@(x) sym(x(2:end-1)),symbols_raw.state,'UniformOutput',false));
symbols.state_string = cellfun(@(x) x(2:end-1),symbols_raw.state,'UniformOutput',false);
symbols.param = cell2sym(cellfun(@(x) sym(x(2:end-1)),symbols_raw.param,'UniformOutput',false));

%%
% observed time points:
simulation.observed_time_points = 0:0.1:simulation.final_time;

%%
% initial state values:
simulation.init_val = 0.01*ones(1,sum(cellfun(@(x) ~strcmp(x(1),'u'),...
    symbols.state_string)));

%%
% Integration interval:
simulation.integration_interval = 0.01;

%%
% Integration interval:
simulation.interval_between_observations = simulation.integration_interval;

%%
% Type of pseudo inverse
% options: 'Moore-Penrose' or 'modified Moore-Penrose'
opt_settings.pseudo_inv_type = 'Moore-Penrose';

%%
% Optimization settings

% Number of coordinate ascent iterations
opt_settings.coord_ascent_numb_iter = 40;

% The observed state trajectories are clamped to the trajectories determined by standard GP regression (Boolean):
opt_settings.clamp_obs_state_to_GP_fit = true;

% Damping for Hemodynamic States
opt_settings.damping = 0.1; 

% Prior on ODE parameters
opt_settings.ode_param_prior = prior_on_neuronal_couplings(symbols.param);

%%
% External input 
state.ext_input = importdata('dcm/external_input.txt');                   

%%
% Plot settings: layout and size and symbols of states for plotting
plot_settings.size = [1600, 800];
plot_settings.layout = [3,2];
plot_settings.plot_states = {'q_2','v_2','f_2','s_2','n_2'};

%%
% Import ODEs
[ode,coupling_idx] = import_odes(symbols,odes_path);

end