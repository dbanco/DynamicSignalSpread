function D = dictionary(P)
%dictionaryFFT Generates FFT of zero mean gaussian  
% basis function vectors with unit 2-norm
%
% Inputs:
% P.N 1D signal length
% P.K Number of dictionary entries
% P.var_theta -vector of theta variances
% P.dtheta - difference in theta between each pixel
% P.num_theta - image size in theta direction 
%
% Outputs:
% A0ft_stack - FFT of dictionary atoms [N,K]

D = zeros(P.N,P.K);
for k = 1:numel(P.N)
    D(:,k) = gaussian_basis_wrap_1D(P.N, 1, P.sigmas(k),'2-norm');                    
end

end

