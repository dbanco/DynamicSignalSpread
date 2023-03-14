function Ax = Ax_ft_1D( A0, x )
%Ax_ft_2D Computes matrix vector product between A and x
% Elementwise multiplies each basis function of A0ft_stack with fft2(x)
%
% Inputs:
% A0ft_stack - (N x K) array
% x - (N x K) array
%
% Outputs:
% Ax - (N x 1) array

Ax = zeros(size(A0,1),1);

x_ft = fft(x);

for k = 1:size(x,2)
        Ax = Ax + real(ifft(A0(:,k).*x_ft(:,k)));
end

end

