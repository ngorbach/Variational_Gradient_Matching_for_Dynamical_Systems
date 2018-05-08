%% Prior on neuronal couplings
% Authors: Nico Stephan Gorbach and Stefan Bauer
%
% The prior variance on all non-selfinhibitory neuronal couplings is infinity.

function prior = prior_on_neuronal_couplings(ode_param_symbols)

non_selfinhibitory_prior_variance = realmax;

numb_states = 3;
prior.mean = zeros(length(ode_param_symbols),1);
prior.mean(end-numb_states+1:end) = -1;                          
tmp = non_selfinhibitory_prior_variance * ones(1,length(ode_param_symbols));
tmp(end-numb_states+1:end) = 1e-9;
prior.inv_cov = diag(tmp.^(-1)); 

end