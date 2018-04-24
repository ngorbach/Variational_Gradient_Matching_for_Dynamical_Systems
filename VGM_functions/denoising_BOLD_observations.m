%% Denoising BOLD observations
% Authors: Nico Stephan Gorbach and Stefan Bauer
%
% We denoise the BOLD observation by standard GP regression.
%
% $p(\mathbf{X} \mid \mathbf{Y}, \boldmath\phi,\gamma) = \prod_k \mathcal{X}(\mathbf{n}_k ; \boldmath\mu_k(\mathbf{y}_k),\boldmath\Sigma_k)$,
%
% where $\boldmath\mu_k(\mathbf{y}_k) := \sigma_k^{-2} \left(\sigma_k^{-2} \mathbf{I} + \mathbf{C}_{\boldmath\phi_k}^{-1} \right)^{-1} \mathbf{y}_k$ and $\boldmath\Sigma_k ^{-1}:=\sigma_k^{-2} \mathbf{I} + \mathbf{C}_{\boldmath\phi_k}^{-1}$.

function [mu,inv_sigma] = denoising_BOLD_observations(bold_response,inv_C,symbols,simulation)

inv_C_cell = num2cell(inv_C(:,:,ones(1,sum(cellfun(@(x) strcmp(x(1),'n'),symbols.state_string)))),[1,2]);
inv_C_blkdiag = blkdiag(inv_C_cell{:});

b = repmat(var(bold_response)./5,size(bold_response,1),1);
%b = repmat(simulation.state_obs_variance(bold_response),size(bold_response,1),1);
dim = size(inv_C_blkdiag,1);
D = spdiags(reshape(b.^(-1),[],1),0,dim,dim) * speye(dim); % covariance matrix of error term (big E)
inv_sigma = D + inv_C_blkdiag;

mu = inv_sigma \ D * reshape(bold_response,[],1);
mu = reshape(mu,[],size(bold_response,2));
end