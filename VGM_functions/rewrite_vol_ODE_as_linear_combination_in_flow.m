%% Rewrite ODEs as linear combination in monotonic function of blood flow e^f
%
% $\mathbf{R}_{f\dot{v}} e^{\mathbf{f}} + \mathbf{r}_{f\dot{v}} \stackrel{!}{=} \mathbf{f}_{\dot{v}}(\mathbf{X},\boldmath\theta)$

function [R,r] = rewrite_vol_ODE_as_linear_combination_in_flow(ode,symbols)

% define symbolic variables
param_sym = sym('param%d',[1,length(symbols.param)]); assume(param_sym,'real');
state_sym = sym('state%d',[1,length(symbols.state)]); assume(state_sym,'real');
exp_f = sym('exp_f'); assume(exp_f,'real');

state_idx = find(cellfun(@(x) strcmp(x(1),'f'),symbols.state_string));

% blood volume ODE
ode_idx = find(cellfun(@(x) strcmp(x(1),'v'),symbols.state_string));

j = 0;
for u = state_idx
    j = j+1;
    [R_sym,r_sym] = equationsToMatrix(subs(ode.system_sym(ode_idx(j)),exp(state_sym(u)),exp_f),exp_f);
    r_sym = -r_sym; % See the documentation of the function "equationsToMatrix"
    
    R{u} = matlabFunction(R_sym,'Vars',{state_sym,param_sym});
    r{u} = matlabFunction(r_sym,'Vars',{state_sym,param_sym});
end

end