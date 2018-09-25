%% Rewrite ODEs as linear combination in parameterS
% Authors: Nico Stephan Gorbach and Stefan Bauer
%
% $\mathbf{B}_{\mathbf{\theta} k} \mathbf{\theta} + \mathbf{b}_{\mathbf{\theta} k} \stackrel{!}{=}
% \mathbf{f}_k(\mathbf{X},\mathbf{\theta})$,
%
% where matrices $\mathbf{B}_{\mathbf{\theta} k}$ and $\mathbf{b}_{\mathbf{\theta} k}$ are
% defined such that the ODEs $\mathbf{f}_k(\mathbf{X},\mathbf{\theta})$ are
% expressed as a linear combination in $\mathbf{\theta}$.

function [B,b] = rewrite_odes_as_linear_combination_in_parameters(ode,symbols)

%%
% Symbolic computations
param_sym = sym('param%d',[1,length(symbols.param)]); assume(param_sym,'real');
state_sym = sym('state%d',[1,length(symbols.state)]); assume(state_sym,'real');
state0_sym = sym('state0'); assume(state0_sym,'real');
state_const_sym = sym('state_const'); assume(state_const_sym,'real');

%%
% Rewrite ODEs as linear combinations in parameters (global)
[B_sym,b_sym] = equationsToMatrix(ode.system_sym,param_sym);
b_sym = -b_sym; % See the documentation of the function "equationsToMatrix"

%%
% Operations locally w.r.t. ODEs
for k = 1:length(ode.system)
    B_sym(k,B_sym(k,:)=='0') = state0_sym;
    for i = 1:length(B_sym(k,:))
        sym_var = symvar(B_sym(k,i));
        if isempty(sym_var)
            B_sym(k,i) = B_sym(k,i) + state0_sym;
        end
    end
    B{k} = matlabFunction(B_sym(k,:),'Vars',{state_sym,state0_sym,state_const_sym});
    b{k} = matlabFunction(b_sym(k,:),'Vars',{state_sym,state0_sym,state_const_sym});
end
end