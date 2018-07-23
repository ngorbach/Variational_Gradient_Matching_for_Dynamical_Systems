%% Import ODEs
% Authors: Nico Stephan Gorbach and Stefan Bauer
function [ode,coupling_idx] = import_odes(symbols,odes_path)

%%
% Import ODEs expressions
ode.system_string = importdata(odes_path);

%%
% Refine ODEs
assume(symbols.state,'real'); assume(symbols.param,'real');
for k = 1:length(ode.system_string)
    for u = 1:length(symbols.state)
        ode.system_string{k} = strrep(ode.system_string{k},...
            ['[' char(symbols.state(u)) ']'],['state(:,' num2str(u) ')']);
    end
    for j = 1:length(symbols.param)
        ode.system_string{k} = strrep(ode.system_string{k},...
            ['[' char(symbols.param(j)) ']'],['param(' num2str(j) ')']);
    end
end
ode.system = cellfun(@(x) str2func(['@(state,param)(' x ')']),ode.system_string,...
    'UniformOutput',false);

ode.system_sym_unpacked = cell2sym(cellfun(@(x) x(symbols.state,symbols.param),ode.system,...
    'UniformOutput',false));

%%
% Packing
param_sym = sym('param%d',[1,length(symbols.param)]); assume(param_sym,'real');
state_sym = sym('state%d',[1,length(symbols.state)]); assume(state_sym,'real');
ode.system_sym = cell2sym(cellfun(@(x) x(state_sym,param_sym),ode.system,'UniformOutput',false));

coupling_idx = find_state_couplings_in_odes(ode);

disp(' '); disp('ODEs:'); disp(' '); 
state_idx = cellfun(@(x) ~strcmp(x(1),'u'),symbols.state_string);
d = sym('d'); dt = sym('dt'); assume(symbols.state,'real');
assume(d,'real'); assume(dt,'real');
pretty(d * symbols.state(state_idx)'/dt == ode.system_sym_unpacked);

end

%% Find state couplings in ODEs
% Authors: Nico Stephan Gorbach and Stefan Bauer

function coupling_idx = find_state_couplings_in_odes(ode)

state_sym = sym('state%d',[1,length(ode.system)]); assume(state_sym,'real');
for k = 1:length(ode.system)
    tmp_idx = ismember(state_sym,symvar(ode.system_sym(k))); tmp_idx(:,k) = 1;
    ode_couplings_states(k,tmp_idx) = 1;
end

for u = 1:size(ode_couplings_states,1)
    coupling_idx.states{u} = find(ode_couplings_states(:,u));
end
end