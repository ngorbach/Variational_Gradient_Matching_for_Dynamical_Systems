function [X] = create_Lorenz96(t0, tf, dt, K, theta)
% Forcing parameters (e.g. F = 8 - default value -).
F = theta(1);

% Number of variables (e.g. M = 40 - default value -).
if(length(theta)>1)
    D = theta(2);
else
    D = 40;
end

% Number of actual trajectory samples.
N = length(t0:dt:tf);

% Preallocate Matrix for efficiency.
X = zeros(D,N);

% Default starting point.
x0 = F*ones(D,1);

% Purturb the L-th dimension by 1/1000.
x0(round(D/2)) = 1.001*F;

% Time-step for initial discretisation.
dtau = 1.0E-3;

% BURN IN: Using the deterministic equations run forwards in time.
for t = 1:50000
	x0 = x0 + lorenz96([],x0,F)*dtau;
end

% Start with the new point.
X(:,1) = x0;

% Noise variance coefficient.
K = sqrt(K*dt);

% Create the path by solving the "stochastic" Diff.Eq. iteratively.
for t = 2:N
    X(:,t) = X(:,t-1) + lorenz96([],X(:,t-1),F)*dt + K.*randn(size(K));
end

function [dx] = lorenz96(~,x,u)
% function [dx] = lorenz96(~,x,u)
%
% Differential equations for the Lorenz 96 system.
% 
% [INPUT]
% t - unused in this system of equations (but needed for compatibility with
%     ode45 function)
% x - D-dimensional state vector
% u - additional parameters (forcing parameter - theta)
% 
% [OUTPUT]
% dx - return vector.
% 
% Copyright (c) Michail D. Vrettas - September 2012.
% 
% Last Update: September 2012.
% 
% Non-linearity and Complexity Research Group (NCRG)
% Dept. of Mathematics, Aston University, Birmingham, UK.
%

% Make sure x is column vector.
x = x(:);

% Generate the shifted state vectors.
xf1 = circshift(x,-1);
xb1 = circshift(x,+1);
xb2 = circshift(x,+2);

% Differential equations.
dx = (xf1 - xb2).*xb1 - x + u;

% == END OF FILE ==