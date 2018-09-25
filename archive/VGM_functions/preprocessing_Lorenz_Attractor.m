%% Preprocessing for Lorenz 96
% Authors: Nico Stephan Gorbach and Stefan Bauer
function [symbols,simulation,ode,odes_path,coupling_idx,opt_settings,plot_settings] = preprocessing_Lorenz_Attractor(simulation)

%%
% observed time points:
simulation.observed_time_points = 0:0.1:simulation.final_time;

%%
% initial state values:
simulation.init_val = [-7 -7 -7];

%%
% Path to ODEs
odes_path = 'Lorenz_attractor_ODEs.txt';

%%
% Symbols: symbols of states and parameters in the '_ODEs.txt' file without delimiters '[' and ']':

% States :
symbols.state = [sym('x'),sym('y'),sym('z')]; 

% ODE parameters :
symbols.param = [sym('sigma'),sym('rho'),sym('lambda')];

%%
% Refine symbols
symbols.state_string = cellfun(@(x) char(x),sym2cell(symbols.state),'UniformOutput',false);

%%
% Observed states
simulation.observed_states = symbols.state_string([1,3]);

%%
% Integration interval:
simulation.integration_interval = 0.01;

%%
% Type of pseudo inverse
% options: 'Moore-Penrose' or 'modified Moore-Penrose'
opt_settings.pseudo_inv_type = 'Moore-Penrose';

%%
% Optimization settings

% Number of coordinate ascent iterations
opt_settings.coord_ascent_numb_iter = 25;

% The observed state trajectories are clamped to the trajectories determined by standard GP regression (Boolean):
opt_settings.clamp_obs_state_to_GP_fit = true;

%%
% Plot settings: layout and size and symbols of states for plotting
plot_settings.size = [1600, 800];
plot_settings.layout = [2,2];
plot_settings.plot_states = symbols.state;


%%
% Import ODEs
[ode,coupling_idx] = import_odes(symbols,odes_path);

end