%% Rewrite ODEs as linear combination in neuronal states n
%
% $\mathbf{R}_{uk} \mathbf{x} + \mathbf{r}_{uk} \stackrel{!}{=} \mathbf{f}_{k}(\mathbf{X},\boldmath\theta)$

function [R,r]= rewrite_odes_as_linear_combination_in_ind_neuronal_states(ode,symbols,coupling_idx)

state_sym = sym('state%d',[1,length(symbols.state)]); assume(state_sym,'real');
param_sym = sym('param%d',[1,length(symbols.param)]); assume(param_sym,'real');

state_idx = find(cellfun(@(x) strcmp(x(1),'n'),symbols.state_string));

for u = state_idx
    for k = coupling_idx{u}'
        [R_sym,r_sym] = equationsToMatrix(ode.system_sym(k),state_sym(:,u));
        r_sym = -r_sym; % See the documentation of the function "equationsToMatrix"
        
        R{u,k} = matlabFunction(R_sym,'Vars',{state_sym,param_sym});
        r{u,k} = matlabFunction(r_sym,'Vars',{state_sym,param_sym});
    end
end
end