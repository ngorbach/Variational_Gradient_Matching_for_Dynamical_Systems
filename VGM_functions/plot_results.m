%% Plot results
% Authors: Nico Stephan Gorbach and Stefan Bauer

function plot_results(fig_handle,state_proxy,simulation,ode_param_proxy_mean,plot_handle,...
    symbols,plot_settings,plot_type,varargin)

%%
% Indices of observed states
tmp = cellfun(@(x) {strcmp(x,simulation.observed_states)},symbols.state_string);
state_obs_idx = cellfun(@(x) any(x),tmp);
obs_ind = find(state_obs_idx);

for j = 1:length(plot_settings.plot_states)
    
    %%
    % Index of state for plotting
    u = find(cellfun(@(x) strcmp(x,char(plot_settings.plot_states(j))),symbols.state_string));
    
    if strcmp(plot_type,'final')
        
        %%
        % State proxy variance
        try
            if ~any(obs_ind==u)
                state_proxy_variance = diag(state_proxy.inv_cov(:,:,u)^(-1));
                shaded_region = [state_proxy.mean{:,symbols.state_string(u)}+1*sqrt(state_proxy_variance);
                    flip(state_proxy.mean{:,symbols.state_string(u)})-1*sqrt(state_proxy_variance)];
                f = fill(fig_handle.states{u},[state_proxy.mean{:,'time'};
                    flip(state_proxy.mean{:,'time'},1)], shaded_region,[222,235,247]/255);
                set(f,'EdgeColor','None');
            end
        end
        
        %%
        % Replot true states
        plot_handle.true = plot(fig_handle.states{u},simulation.state{:,'time'},simulation.state{:,symbols.state_string(u)},...
            'LineWidth',2,'Color',[217,95,2]./255);
        
        %%
        % Replot state observations
        try
            plot_handle.obs = plot(fig_handle.states{u},simulation.observations{:,'time'},...
                simulation.observations{:,symbols.state_string(u)},'*','Color',[217,95,2]./255,...
                'MarkerSize',6);
        end
        %%
        % State proxy mean (final)
        hold on;
        plot_handle.est = plot(fig_handle.states{u},state_proxy.mean{:,'time'},state_proxy.mean{:,symbols.state_string(u)},'Color',...
            [117,112,179]./255,'LineWidth',2);
        
        try; plot_handle.num_int = plot(fig_handle.states{u},state_proxy.num_int{:,'time'},state_proxy.num_int{:,symbols.state_string(u)},'Color',[0,0,0],'LineWidth',1); end
    else
        %%
        % state proxy mean (not final)
        hold on; plot_handle.est = plot(fig_handle.states{u},state_proxy.mean{:,'time'},state_proxy.mean{:,symbols.state_string(u)},...
            'LineWidth',0.1,'Color',[0.6,0.6,0.6]);
    end
    
    %%
    % Specify legend entries
    if ~isfield(plot_handle,'num_int')
        if state_obs_idx(u)
            legend(fig_handle.states{u},[plot_handle.true,plot_handle.obs,plot_handle.est],{'true','observed','estimate'},...
                'Location','southwest','FontSize',12,'Orientation','horizontal');
        else
            legend(fig_handle.states{u},[plot_handle.true,plot_handle.est],{'true','estimate'},'Location',...
                'southwest','FontSize',12,'Orientation','horizontal');
        end
    else
        if state_obs_idx(u)
            legend(fig_handle.states{u},[plot_handle.true,plot_handle.obs,...
                plot_handle.est,plot_handle.num_int],{'true','observed',...
                'estimate','numerical int. with est. param.'},'Location',...
                'southwest','FontSize',12,'Orientation','horizontal');
        else
            legend(fig_handle.states{u},[plot_handle.true,plot_handle.est,...
                plot_handle.num_int],{'true','estimate','numerical int. with est. param.'},...
                'Location','southwest','FontSize',12,'Orientation','horizontal');
        end
    end
end

%%
% ODE parameters
cla(fig_handle.param);
try
    b = bar(fig_handle.param,1:length(ode_param_proxy_mean),[simulation.ode_param',...
        ode_param_proxy_mean]);
    legend(fig_handle.param,{'true','estimate'},'Location','northwest','FontSize',12,'Box','off','Orientation','horizontal');
catch
    b = bar(fig_handle.param,1:length(ode_param_proxy_mean)+1,[[simulation.ode_param';0],...
        [ode_param_proxy_mean;0]]);
    legend(fig_handle.param,{'true','estimate'},'Location','northwest','FontSize',12,'Box','off','Orientation','horizontal');
end
b(1).FaceColor = [217,95,2]./255; b(2).FaceColor = [117,112,179]./255;
fig_handle.param.XLim = [0.5,length(ode_param_proxy_mean)+0.5]; fig_handle.param.YLimMode = 'auto';

%%
% For dynamic causal modeling
if ~isempty(varargin)
    bold_response = varargin{1};
    plot_titles_idx = find(cellfun(@(x) strcmp(x(1),'n'),symbols.state_string));
    for j = 1:3
        plot(fig_handle.bold{j},bold_response{:,'time'},bold_response{:,symbols.state_string(plot_titles_idx(j))},...
            'LineWidth',1,'Color',[0,0,0]); hold on;
        legend(fig_handle.bold{j},{'observed BOLD response','numerical int. with est. param.'},...
            'Location','southwest','FontSize',10,'Orientation','horizontal');
    end
    if ~strcmp(varargin{2},varargin{3})
        cla(fig_handle.param);
        b = bar(fig_handle.param,1:length(ode_param_proxy_mean),ode_param_proxy_mean);
        legend(fig_handle.param,{'estimate'},'Location','northwest','FontSize',12,'Orientation','horizontal');
        b.FaceColor = [117,112,179]./255;
        fig_handle.param.XLim = [0.5,length(ode_param_proxy_mean)+0.5]; fig_handle.param.YLimMode = 'auto';
    end
    text(-0.6,4.75,{['true:          ' varargin{2}],['candidate: ' varargin{3}]},'Units','normalized',...
        'FontSize',18,'Interpreter','none')
end
drawnow