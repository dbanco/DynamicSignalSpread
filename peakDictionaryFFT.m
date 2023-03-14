function D = peakDictionaryFFT(P)
%peakDictionaryFFT Generates FFT of zero mean gaussian  
% basis function vectors with unit 2-norm
%
% Inputs:
% P.N 1D signal length
% P.K Number of dictionary entries
% P.sigmas Vector of width parameters for dictionary 
%
% Outputs:
% D - FFT of dictionary atoms [N,K]

D = zeros(P.N,P.K);
for k = 1:P.K
    D(:,k) = fft(gaussian_basis_wrap_1D(P.N, floor(P.N/2), P.sigmas(k),'2-norm'));                    
end

end

