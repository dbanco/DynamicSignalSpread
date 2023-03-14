 function PtDtR = PhiTranDiffTran_1D(R,N,K,T)
%conjGradResidual Compute residual for conjugate gradient that includes 
% difference matrix

% reshape R if it is a vector
if nargin == 2
    [K,T] = size(R);
    T = T+1;
else
    R = reshape(R,[K,T]);
    T = T+1;
end

DtR = zeros(K,T);
DtR(:,1) = -R(:,1);
DtR(:,T) =  R(:,T-1);
DtR(:,2:(T-1)) = R(:,1:(T-2)) - R(:,2:(T-1));

DtR = reshape(DtR,1,size(DtR,1),size(DtR,2));
PtDtR = repmat(DtR, [N,1,1]);

if nargin == 4
    PtDtR = reshape(PtDtR,[1,size(PtDtR)]);
end