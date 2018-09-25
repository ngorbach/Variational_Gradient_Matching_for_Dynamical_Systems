%% Setup plots
% Authors: Nico Stephan Gorbach and Stefan Bauer

function [h_states,h_param,p] = setup_plots(time,simulation,symbols,plot_settings)

%%
% Figure size and position setup
figure(1); set(1,'Position',[0,200,plot_settings.size(1),plot_settings.size(2)]);
%%
% ODE parameters
h_param = subplot(plot_settings.layout(1),plot_settings.layout(2),1);
h_param.FontSize = 20; h_param.Title.String = 'ODE parameters';
h_param.Title.FontWeight = 'Normal';
h_param.XTick = 1:length(symbols.param); 
h_param.XTickLabel = cellfun(@(x) latex(x),sym2cell(symbols.param),...
    'UniformOutput',false);
hold on;

%%
% States
for j = 1:length(plot_settings.plot_states)
    
    % Index of state for plotting
    u = find(cellfun(@(x) strcmp(x,char(plot_settings.plot_states(j))),symbols.state_string));
    
    h_states{u} = subplot(plot_settings.layout(1),plot_settings.layout(2),j+1); cla;
    p.true = plot(simulation.state{:,'time'},simulation.state{:,symbols.state_string(u)},...
        'LineWidth',2,'Color',[217,95,2]./255);
    h_states{u}.Title.String = latex(symbols.state(u));
    hold on;
    try
        p.obs = plot(simulation.observations{:,'time'},...
            simulation.observations{:,symbols.state_string(u)},'*','Color',...
            [217,95,2]./255,'MarkerSize',3);
        h_states{u}.Title.String = ['observed ' latex(symbols.state(u))];
    catch
        h_states{u}.Title.String = ['unobserved ' latex(symbols.state(u))];
    end
    h_states{u}.FontSize = 20;
    h_states{u}.Title.FontWeight = 'Normal';
    h_states{u}.XLim = [min(time.est),max(time.est)];
    h_states{u}.XLabel.String = 'time'; hold on;
end
drawnow