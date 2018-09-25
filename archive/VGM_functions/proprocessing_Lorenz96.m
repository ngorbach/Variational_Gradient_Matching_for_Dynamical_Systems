%% Preprocessing for Lorenz 96
% Authors: Nico Stephan Gorbach and Stefan Bauer
function [symbols,simulation,ode,odes_path,coupling_idx,opt_settings,plot_settings] = proprocessing_Lorenz96(simulation)

%%
% final time for integration:
simulation.final_time = 4;

%%
% observed time points:
simulation.observed_time_points = 0:0.125:simulation.final_time;

%%
% initial state values:
simulation.init_val = zeros(1,simulation.numb_odes);

%%
% Path to ODEs
odes_path = 'Lorenz96_ODEs.txt';

%%
% Symbols: symbols of states and parameters in the '_ODEs.txt' file without delimiters '[' and ']':

% States :
for u = 1:simulation.numb_odes; symbols.state(u) = sym(['x_' num2str(u) ]); end

% ODE parameters :
symbols.param = sym('alpha');

%%
% Refine symbols
symbols.state_string = cellfun(@(x) char(x),sym2cell(symbols.state),'UniformOutput',false);

%%
% Type of pseudo inverse
% options: 'Moore-Penrose' or 'modified Moore-Penrose'
opt_settings.pseudo_inv_type = 'Moore-Penrose';

%%
% Optimization settings

% Number of coordinate ascent iterations
opt_settings.coord_ascent_numb_iter = 20;

% The observed state trajectories are clamped to the trajectories determined by standard GP regression (Boolean):
opt_settings.clamp_obs_state_to_GP_fit = true;

%%
% Plot settings: layout and size and symbols of states for plotting
plot_settings.size = [1600, 800];
plot_settings.layout = [3,3];
idx = randperm(length(symbols.state)); 
plot_settings.plot_states = symbols.state(idx(1:8));

%%
% Integration interval:
simulation.integration_interval = 0.01;

%%
% Generate Lorenz 96 ODEs
generate_Lorenz96_ODEs(simulation.numb_odes,odes_path)

if ~iscell(simulation.observed_states)
    ratio_observed = simulation.observed_states;
    state_obs_idx = zeros(1,simulation.numb_odes,'logical');
    idx = randperm(simulation.numb_odes);
    idx = idx(1:floor(simulation.numb_odes * ratio_observed));
    state_obs_idx(idx) = 1;
    simulation.observed_states = cellfun(@(x) char(x),sym2cell(symbols.state(state_obs_idx)),'UniformOutput',false);
end

%%
% Import ODEs
[ode,coupling_idx] = import_odes(symbols,odes_path);

end

%% Generate Lorenz 96 ODEs
% Authors: Nico Stephan Gorbach and Stefan Bauer

function generate_Lorenz96_ODEs(numb_odes,odes_path)

for i = 1:numb_odes
    state{i} = ['[x_' num2str(i) ']'];
end
param = '[alpha]';

ode{1} = ['(' state{2} ' - ' state{end-1} ') .* ' state{end} ' - ' state{1} ...
    ' + ' param];
ode{2} = ['(' state{3} ' - ' state{end} ') .* ' state{1} ' - ' state{2} ...
    ' + ' param];
for i = 3:numb_odes-1
    ode{i} = [ '(' state{i+1} ' - ' state{i-2} ') .* ' state{i-1} ' - ' state{i}...
        ' + ' param];
end
ode{numb_odes} = ['(' state{1} ' - ' state{end-2} ') .* ' state{end-1} ' - ' ...
    state{end} ' + ' param];

dlmwrite(odes_path,[])

for i = 1:numb_odes
    dlmwrite(odes_path,char(ode{i}),'delimiter','','-append')
end
end