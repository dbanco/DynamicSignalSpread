function AtB = AtB_ft_1D_Time(A0,B,Bnorms)
%conjGradResidual Compute residual for conjugate gradient that includes 
% difference matrix
%   Detailed explanation goes here

[N,K] = size(A0);
T = size(B,2);
AtB = zeros(N,K,T);
for t = 1:T
    AtB(:,:,t) = AtR_ft_1D(A0/Bnorms(t),B(:,t));
end