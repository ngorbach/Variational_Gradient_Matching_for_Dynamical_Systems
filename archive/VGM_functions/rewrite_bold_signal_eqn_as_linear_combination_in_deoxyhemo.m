%% Rewrite BOLD signal change equation as linear combination in monotonic function of deoxyhemoglobin content e^q
% Authors: Nico Stephan Gorbach and Stefan Bauer
%
% $\mathbf{R}_{q\lambda} e^{\mathbf{q}} + \mathbf{r}_{v\lambda} \stackrel{!}{=} \lambda(q,v)$.

function [R,r] = rewrite_bold_signal_eqn_as_linear_combination_in_deoxyhemo(symbols)

%%
% define symbolic variables
param_sym = sym('param%d',[1,length(symbols.param)]); assume(param_sym,'real');
state_sym = sym('state%d',[1,length(symbols.state)]); assume(state_sym,'real');
v = sym('v'); assume(v,'real');
q = sym('q'); assume(q,'real');
exp_q = sym('exp_q'); assume(exp_q,'real');

%%
% bold signal change equation
bold_signal_change = bold_signal_change_eqn(v,q);
[R_sym,r_sym] = equationsToMatrix(subs(bold_signal_change,exp(q),exp_q),exp_q);
r_sym = -r_sym; % See the documentation of the function "equationsToMatrix"

R = matlabFunction(R_sym,'Vars',{v,q});
r = matlabFunction(r_sym,'Vars',{v,q});