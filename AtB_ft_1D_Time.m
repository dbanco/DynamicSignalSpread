function AtB = AtB_ft_1D_Time(A0ft_stack,B,Bnorms)
%AtB_ft_1D_Time Compute a sequence of transposed matrix vector products 
%using FFT with the option to normalize 
% Inputs: 
% A0ft_stack - FFT of dictionary atoms [N,K]
% X - Sequence of sparse coefficients [N,K,T]
% Bnorms (optional) - Normalization factors [T,1]
% 
% Outputs:
% Y - Sequeunce of data reconstructions

[N,M] = size(A0ft_stack);
T = size(B,2);
AtB = zeros(N,M,T);

if nargin > 2 % With normalization
    for t = 1:T
        AtB(:,:,t) = AtR_ft_1D(A0ft_stack/Bnorms(t),B(:,t));
    end
else % Without normalization
    for t = 1:T
        AtB(:,:,t) = AtR_ft_1D(A0ft_stack,B(:,t));
    end
end