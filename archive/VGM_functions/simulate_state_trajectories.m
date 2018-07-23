%% Simulate state trajectories by numerical integration
% Authors: Nico Stephan Gorbach and Stefan Bauer

function simulation = simulate_state_trajectories(derivative_variance,...
    ode,symbols,simulation,odes_path)

%%
% Integration times
time_true=0:simulation.integration_interval:simulation.final_time;

%%
% Numerical integration
if ~strcmp(odes_path,'Lorenz96_ODEs.txt')
    % Fourth order Runge-Kutta (numerical) integration
    ode_system_mat = matlabFunction(ode.system_sym','Vars',{state_sym',param_sym'});
    [~,OutX_solver]=ode45(@(t,x) ode_system_mat(x,simulation.ode_param'),time_true,...
        simulation.init_val);
else
    OutX_solver = create_Lorenz96(min(time_true),max(time_true),time_true(2)-time_true(1),...
        derivative_variance',[simulation.ode_param, length(symbols.state)])';
end

%%
% Pack into table
simulation.state = array2table([time_true',OutX_solver],'VariableNames',...
    ['time',symbols.state_string]);