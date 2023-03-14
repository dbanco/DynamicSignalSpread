function AtR = AtR_ft_1D( A0, R )
%AtR_ft_1D Computes matrix vector product between A transposed and R
% Elementwise multiplies each basis function of A0ft_stack with fft(R)
%
% Inputs:
% A0 - (N x K) array
% R - (N x 1) array
%
% Outputs:
% AtR - (N x K) array


AtR = zeros(size(A0));

R_ft = fft(R);

for k = 1:size(A0,2)
        y = ifft(conj(A0(:,k)).*R_ft(:));
        AtR(:,k) = y;
end


end

