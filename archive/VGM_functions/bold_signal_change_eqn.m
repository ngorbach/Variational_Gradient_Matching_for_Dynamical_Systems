function Y = bold_signal_change_eqn(V,Q)

% Biophysical constants for 1.5T
%==========================================================================
 
% time to echo (TE) (default 0.04 sec)
%--------------------------------------------------------------------------
TE  = 0.04;
 
% resting venous volume (%)
%--------------------------------------------------------------------------
V0  = 4;

% estimated region-specific ratios of intra- to extra-vascular signal 
%--------------------------------------------------------------------------
%P.epsilon = 0.2015;
P.epsilon = 0.1970;
ep  = exp(P.epsilon);
 
% slope r0 of intravascular relaxation rate R_iv as a function of oxygen 
% saturation S:  R_iv = r0*[(1 - S)-(1 - S0)] (Hz)
%--------------------------------------------------------------------------
r0  = 25;
 
% frequency offset at the outer surface of magnetized vessels (Hz)
%--------------------------------------------------------------------------
nu0 = 40.3; 
 
% resting oxygen extraction fraction
%--------------------------------------------------------------------------
E0  = 0.4;
 
%-Coefficients in BOLD signal model
%==========================================================================
k1  = 4.3*nu0*E0*TE;
k2  = ep*r0*E0*TE;
k3  = 1 - ep;
 
%-Output equation of BOLD signal model
%==========================================================================

V = exp(V);
Q = exp(Q);
Y = V0 .* ( k1.*(1 - Q) + k2.*(1 - Q./V) + k3.*(1 - V) );