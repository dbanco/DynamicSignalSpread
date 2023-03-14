function DiffPhi = DiffPhiX_1D(X,N,K,T)
%DiffPhiX_1D Apply summing over space and temporal difference matrix 

% reshape X if it is a vector
if nargin == 1
    [~,K,T] = size(X);
else
    X = reshape(X,[N,K,T]);
end

PhiX = squeeze(sum(X,1));
DiffPhi = PhiX(:,2:T)-PhiX(:,1:(T-1));

