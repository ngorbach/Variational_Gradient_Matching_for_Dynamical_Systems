%% Kernel Function
% Authors: Nico Stephan Gorbach and Stefan Bauer

% Gradient matching with Gaussian processes assumes a joint Gaussian
% process prior on states and their derivatives:
%
% $\left(\begin{array}{c} \mathbf{X} \\ \dot{\mathbf{X}} \end{array}\right)
%  \sim \mathcal{N} \left(
% \begin{array}{c} \mathbf{X} \\ \dot{\mathbf{X}} \end{array};
% \begin{array}{c}
%  \mathbf{0} \\
% \mathbf{0}
%  \end{array},
% \begin{array}{cc}
%  \mathbf{C}_{\mathbf{\phi}} & \mathbf{C}_{\mathbf{\phi}}' \\ '\mathbf{C}_{\mathbf{\phi}} &
%  \mathbf{C}_{\mathbf{\phi}}'' \end{array} \right)$,
%
% $\mathrm{cov}(x_k(t), x_k(t)) = C_{\mathbf{\phi}_k}(t,t')$
%
% $\mathrm{cov}(\dot{x}_k(t), x_k(t)) = \frac{\partial C_{\mathbf{\phi}_k}(t,t')
% }{\partial t} =: C_{\mathbf{\phi}_k}'(t,t')$
%
% $\mathrm{cov}(x_k(t), \dot{x}_k(t)) = \frac{\partial C_{\mathbf{\phi}_k}(t,t')
% }{\partial t'} =: {'C_{\mathbf{\phi}_k}(t,t')}$
%
% $\mathrm{cov}(\dot{x}_k(t), \dot{x}_k(t)) = \frac{\partial
% C_{\mathbf{\phi}_k}(t,t') }{\partial t \partial t'} =: C_{\mathbf{\phi}_k}''(t,t')$.


function [dC_times_invC,inv_C,A_plus_gamma_inv,C_samp,C_samp_est] = kernel_function2(kernel,state,time_est,time_samp)

kernel.param_sym = sym('rbf_param%d',[1,2]); assume(kernel.param_sym,'real');
kernel.t1 = sym('time1'); assume(kernel.t1,'real'); kernel.t2 = sym('time2');
assume(kernel.t2,'real');
% RBF kernel
kernel.func = kernel.param_sym(1).*exp(-(kernel.t1-kernel.t2).^2./...
    (kernel.param_sym(2).^2));
kernel.name = 'rbf';

%%
% kernel derivatives
for i = 1:length(kernel)
    kernel.func_d = diff(kernel.func,kernel.t1);
    kernel.func_dd = diff(kernel.func_d,kernel.t2);
    GP.fun = matlabFunction(kernel.func,'Vars',{kernel.t1,kernel.t2,kernel.param_sym});
    GP.fun_d = matlabFunction(kernel.func_d,'Vars',{kernel.t1,kernel.t2,kernel.param_sym});
    GP.fun_dd = matlabFunction(kernel.func_dd,'Vars',{kernel.t1,kernel.t2,...
        kernel.param_sym});
end

%%
% populate GP covariance matrix
for t=1:length(time_est)
    C(t,:)=GP.fun(time_est(t),time_est,kernel.param);
    dC(t,:)=GP.fun_d(time_est(t),time_est,kernel.param);
    Cd(t,:)=GP.fun_d(time_est,time_est(t),kernel.param);
    ddC(t,:)=GP.fun_dd(time_est(t),time_est,kernel.param);
end

%%
% populate GP covariance matrix
for t=1:length(time_samp)
    C_samp(t,:)=GP.fun(time_samp(t),time_samp,kernel.param);
end

%%
% populate GP covariance matrix
for t1=1:length(time_est)
    for t2=1:length(time_samp)
        C_samp_est(t1,t2)=GP.fun(time_est(t1),time_samp(t2),kernel.param);
    end
end

%%
% GP covariance scaling
[~,D] = eig(C); perturb = abs(max(diag(D))-min(diag(D))) / 10000;
if any(diag(D)<1e-6)
    C(logical(eye(size(C,1)))) = C(logical(eye(size(C,1)))) + perturb.*rand(size(C,1),1);
end
[~,D] = eig(C);
if any(diag(D)<0)
    error('GP prior covariance between states has negative eigenvalues!');
elseif any(diag(D)<1e-6)
    warning('GP prior covariance between states is badly scaled');
end
inv_C = inv_chol(chol(C,'lower'));

dC_times_invC = dC * inv_C;

%%
% determine $\mathbf{A} + \mathbf{I}\gamma$:
A = ddC - dC_times_invC * Cd;
A_plus_gamma = A + state.derivative_variance(1) .* eye(size(A));
A_plus_gamma = 0.5.*(A_plus_gamma+A_plus_gamma');  % ensure that A plus gamma is symmetric
A_plus_gamma_inv = inv_chol(chol(A_plus_gamma,'lower'));

%%
% plot samples from GP prior
% figure(3);
% hold on; plot(time_est,mvnrnd(zeros(1,length(time_est)),C(:,:,1),3),'LineWidth',2);
% set(gca,'FontSize',20); xlabel('time');ylabel('state value');
% title([kernel.name ' kernel']);