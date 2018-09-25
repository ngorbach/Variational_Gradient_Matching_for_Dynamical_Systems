%% Rewrite ODEs as linear combination in vasosignalling s
%
% $\mathbf{R}_{s\dot{s}} \mathbf{s} + \mathbf{r}_{s\dot{s}} \stackrel{!}{=} \mathbf{f}_{\dot{s}}(\mathbf{X},\boldmath\theta)$
%
% $\mathbf{R}_{s\dot{f}} \mathbf{s} + \mathbf{r}_{s\dot{f}} \stackrel{!}{=} \mathbf{f}_{\dot{f}}(\mathbf{X},\boldmath\theta)$

function [R,r] = rewrite_vaso_and_flow_odes_as_linear_combination_in_vaso(ode,symbols)

% define symbolic variables
param_sym = sym('param%d',[1,length(symbols.param)]); assume(param_sym,'real');
state_sym = sym('state%d',[1,length(symbols.state)]); assume(state_sym,'real');

state_idx = find(cellfun(@(x) strcmp(x(1),'s'),symbols.state_string));

% vasosignaling ODE
ode_idx = find(cellfun(@(x) strcmp(x(1),'s'),symbols.state_string));
j = 0;
for u = state_idx
    j = j+1;
    [R_sym,r_sym] = equationsToMatrix(ode.system_sym(ode_idx(j)),state_sym(u));
    r_sym = -r_sym; % See the documentation of the function "equationsToMatrix"
    
    R{u}.vaso = matlabFunction(R_sym,'Vars',{state_sym,param_sym});
    r{u}.vaso = matlabFunction(r_sym,'Vars',{state_sym,param_sym});
end

% blood flow ODE
ode_idx = find(cellfun(@(x) strcmp(x(1),'f'),symbols.state_string));
j = 0;
for u = state_idx
    j = j+1;
    [R_sym,r_sym] = equationsToMatrix(ode.system_sym(ode_idx(j)),state_sym(u));
    r_sym = -r_sym; % See the documentation of the function "equationsToMatrix"
    
    R{u}.flow = matlabFunction(R_sym,'Vars',{state_sym,param_sym});
    r{u}.flow = matlabFunction(r_sym,'Vars',{state_sym,param_sym});
end
end