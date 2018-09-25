%% Preprocessing for Lorenz 96
% Authors: Nico Stephan Gorbach and Stefan Bauer
function [symbols,simulation,ode,odes_path,coupling_idx,opt_settings,plot_settings] = preprocessing_Lotka_Volterra(simulation)

%%
% observed time points:
simulation.observed_time_points = 0:0.1:simulation.final_time;

%%
% initial state values:
simulation.init_val = [5 3];

%%
% Path to ODEs
odes_path = 'Lotka_Volterra_ODEs.txt';

%%
% Symbols: symbols of states and parameters in the '_ODEs.txt' file without delimiters '[' and ']':

% States :
symbols.state = [sym('x_1'),sym('x_2')]; 

% ODE parameters :
symbols.param = [sym('theta_1'),sym('theta_2'),sym('theta_3'),sym('theta_4')];

%%
% Refine symbols
symbols.state_string = cellfun(@(x) char(x),sym2cell(symbols.state),'UniformOutput',false);

%%
% Observed states
simulation.observed_states = symbols.state_string;

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
opt_settings.coord_ascent_numb_iter = 40;

% The observed state trajectories are clamped to the trajectories determined by standard GP regression (Boolean):
opt_settings.clamp_obs_state_to_GP_fit = false;

%%
% Plot settings: layout and size and symbols of states for plotting
plot_settings.size = [1200, 500];
plot_settings.layout = [1,3];
plot_settings.plot_states = symbols.state;

%%
% Import ODEs
[ode,coupling_idx] = import_odes(symbols,odes_path);

end