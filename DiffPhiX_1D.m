function DiffPhi = DiffPhiX_1D(X)
%DiffPhiX_1D Compute the difference between unnormalized variance
%distribution functions
%
% Inputs:
% X - Coefficients [N,K,T]
%
% Outputs:
% DiffPhi - UVDF difference [K,T-1]

[~,K,T] = size(X);
DiffPhi = zeros(K,T-1);
PhiX = squeeze(sum(X,1));
for t = 1:(T-1)
    DiffPhi(:,t) = PhiX(:,t+1)-PhiX(:,t);
end

