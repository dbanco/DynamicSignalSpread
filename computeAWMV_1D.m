function awmv = computeAWMV_1D(x,sigmas)
%computeAWMV_1D
% Inputs
% x-      (N x K) array of fitted coefficients
% sigmas- (K x 1) array of dictionary sigmas

% Outputs
% awmv_az- amplitude weighted mean variance (azimuthal)

K = numel(sigmas);

% row vector
sigmas = sigmas(:)';

sigma_signal = squeeze(sum(x,1));
[~,n2] = size(sigma_signal);
if n2 == K
    sigma_signal = sigma_signal';
end
total = sum(sigma_signal,1);

awmv = (sigmas*sigma_signal./total)';
