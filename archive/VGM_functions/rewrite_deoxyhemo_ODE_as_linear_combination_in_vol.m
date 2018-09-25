%% Rewrite deoxyhemoglobin content ODE as linear combination in monotonic function of blood volume e^v
%
% $\mathbf{R}_{v\dot{q}} e^{\mathbf{v}} + \mathbf{r}_{v\dot{q}} \stackrel{!}{=} \mathbf{f}_{\dot{q}}(\mathbf{X},\boldmath\theta)$.

function [R,r] = rewrite_deoxyhemo_ODE_as_linear_combination_in_vol(ode,symbols)

% define symbolic variables
param_sym = sym('param%d',[1,length(symbols.param)]); assume(param_sym,'real');
state_sym = sym('state%d',[1,length(symbols.state)]); assume(state_sym,'real');
exp_v = sym('exp_v'); assume(exp_v,'real');

state_idx = find(cellfun(@(x) strcmp(x(1),'v'),symbols.state_string));

% deoxyhemoglobin ODE
ode_idx = find(cellfun(@(x) strcmp(x(1),'q'),symbols.state_string));
j = 0;
for u = state_idx
    j = j+1;
    [R_sym,r_sym] = equationsToMatrix(subs(ode.system_sym(ode_idx(j)),exp((17*state_sym(u)/8)),exp_v),exp_v);
    r_sym = -r_sym; % See the documentation of the function "equationsToMatrix"
    
    R{u} = matlabFunction(R_sym,'Vars',{state_sym,param_sym});
    r{u} = matlabFunction(r_sym,'Vars',{state_sym,param_sym});
end