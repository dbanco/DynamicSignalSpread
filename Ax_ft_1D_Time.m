function Y = Ax_ft_1D_Time(A0,X,Bnorms)
%conjGradResidual Compute residual for conjugate gradient that includes 
% difference matrix

T = size(X,3);
[N,~] = size(A0);

Y = zeros(N,T);
if nargin > 2
    for t = 1:T
        Y(:,t) = Ax_ft_1D(A0/Bnorms(t),X(:,:,t));
    end
else
    for t = 1:T
        Y(:,t) = Ax_ft_1D(A0,X(:,:,t));
    end
end