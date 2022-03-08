function Y = Ax_ft_1D_Time(A0ft_stack,X,Bnorms)
%Ax_ft_1D_Time Compute a sequence of matrix vector products using FFT
% with the option to normalize 
% Inputs: 
% A0ft_stack - FFT of dictionary atoms [N,K]
% X - Sequence of sparse coefficients [N,K,T]
% Bnorms (optional) - Normalization factors [T,1]
% 
% Outputs:
% Y - Sequeunce of data reconstructions

T = size(X,3);
[N,M] = size(A0ft_stack);
Y = zeros(N,T);

if nargin > 2 % With normalization
    for t = 1:T
        Y(:,t) = Ax_ft_1D(A0ft_stack/Bnorms(t),X(:,:,t));
    end
else % Without normalization
    for t = 1:T
        Y(:,t) = Ax_ft_1D(A0ft_stack,X(:,:,t));
    end
end
