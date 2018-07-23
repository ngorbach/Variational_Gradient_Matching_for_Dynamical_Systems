%% Simulate state trajectories by numerical integration
% Authors: Nico Stephan Gorbach and Stefan Bauer

function [state,time,ode,bold_response] = simulate_dynamics_by_numerical_integration_dcm(state,time,ode,simulation,symbols)

param_sym = sym('param%d',[1,length(symbols.param)]); assume(param_sym,'real');
state_sym = sym('state%d',[1,length(symbols.state)]); assume(state_sym,'real');
for i = 1:length(ode.system)
    ode.system_sym(i) = ode.system{i}(state_sym,param_sym);
end

idx0 = cellfun(@(x) ~strcmp(x(1),'u'),symbols.state_string);
learn_method.state(idx0) = {'Laplace mean-field'};
learn_method.state(~idx0) = {'external input'};

state.obs_idx = zeros(1,sum(idx0));
state.init_val = zeros(1,sum(idx0));
%
init_val = 0.01*ones(1,sum(idx0));

%
dt = state.ext_input(end,1) - state.ext_input(end-1,1);
ode_system_mat = matlabFunction(ode.system_sym,'Vars',{state_sym(~strcmp(learn_method.state,'external input'))',...
        param_sym',state_sym(strcmp(learn_method.state,'external input'))'});
  
ode_param_true = simulation.ode_param';

% warning ('off','all');    
[ToutX,OutX_solver] = ode113(@(t,n) ode_function(t,n,ode_system_mat,ode_param_true,state.ext_input(:,2:end),state.ext_input(:,1)),...
    state.ext_input(:,1), init_val');
% warning ('on','all');

[~,idx] = min(pdist2(ToutX,state.ext_input(:,1)),[],1);
ToutX = ToutX(idx); OutX_solver = OutX_solver(idx,:);

% pack
[~,state.ext_input_to_bold_response_mapping_idx] = min(pdist2(state.ext_input(:,1),time.est'),[],1);
state.true = OutX_solver(state.ext_input_to_bold_response_mapping_idx,:);
state.true(1:5,:) = 0;

time.true = ToutX';
time.samp = time.true(state.ext_input_to_bold_response_mapping_idx);

% true bold responses
bold_response.true = bold_signal_change_eqn(state.true(:,cellfun(@(x) strcmp(x(1),'v'),...
    symbols.state_string)),state.true(:,cellfun(@(x) strcmp(x(1),'q'),symbols.state_string)));
% mean correction
% bold_response.confounding_effects.intercept = mean(bold_response.true,1);
% bold_response.true = bsxfun(@minus,bold_response.true,mean(bold_response.true,1));
% % bold_response.confounding_effects.X0 = ones(size(bold_response.true));

% observed bold responses
bold_response.obs = bold_response.true + bsxfun(@times,sqrt(var(bold_response.true) ./ simulation.SNR),randn(size(bold_response.true)));
bold_response.confounding_effects.intercept = mean(bold_response.obs,1);
bold_response.variance = (repmat(max(bold_response.obs,[],1),size(bold_response.obs,1),1)./simulation.SNR).^2;

% align externel input with observations
shift_num = 1;
e = state.ext_input;
e(shift_num+1:end,2:end) = state.ext_input(1:end-shift_num,2:end);
e(1:shift_num,2:end) = zeros(shift_num,size(state.ext_input,2)-1);
state.ext_input = e;
end

%% ODE function
function state_derivatives = ode_function(time,states,ode_system_mat,ode_param,ext_input,time_lst)

[~,idx] = min(pdist2(time,time_lst));
u = ext_input(idx,:);

state_derivatives = ode_system_mat(states,ode_param,u');

end