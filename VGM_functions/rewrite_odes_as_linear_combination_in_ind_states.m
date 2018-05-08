%% Rewrite ODEs as linear combination in individual states
% Authors: Nico Stephan Gorbach and Stefan Bauer
%
% $\mathbf{R}_{uk} \mathbf{x}_u + \mathbf{r}_{uk} \stackrel{!}{=}
% \mathbf{f}_k(\mathbf{X},\mathbf{\theta})$.
%
% where matrices $\mathbf{R}_{uk}$ and $\mathbf{r}_{uk}$ are defined such
% that the ODEs $\mathbf{f}_k(\mathbf{X},\mathbf{\theta})$ is rewritten as a linear
% combination in the individual state $\mathbf{x}_u$.

function [R,r] = rewrite_odes_as_linear_combination_in_ind_states(ode,symbols,coupling_idx)

%%
% Symbolic computations
param_sym = sym('param%d',[1,length(symbols.param)]); assume(param_sym,'real');
state_sym = sym('state%d',[1,length(symbols.state)]); assume(state_sym,'real');
state0_sym = sym('state0'); assume(state0_sym,'real');
state_const_sym = sym('state_const'); assume(state_const_sym,'real');

%%
% Rewrite ODEs as linear combinations in parameters (locally)
for u = 1:length(symbols.state)
    for k = coupling_idx{u}'
        [R_sym,r_sym] = equationsToMatrix(ode.system_sym(k),...
            state_sym(:,u));
        r_sym = -r_sym; % See the documentation of the function "equationsToMatrix"
        
        R{u,k} = matlabFunction(R_sym,'Vars',{state_sym,param_sym});
        r{u,k} = matlabFunction(r_sym,'Vars',{state_sym,param_sym});
    end
end