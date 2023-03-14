function [Xk,cgIters] = conjGrad_TVphi_1D(A0,B,Bnorms,X_init,YV,ZU,params)
%conjGrad_TVx_1D Solves linear system:
% At A x + rho1 I + rho2(Phit Phi X) = At b + rho1(Y-V) + rho2(Z-U) 
%
% Inputs:
% A0 (N x K) fft of dictionary  
% B (N x T) data
% Bnorms (T x 1) norm values of data
% X_init (N x K x T) initial solution
% YV (N x K x T) Y-V
% ZU (K x T-1) Z-U
% params:
%   rho1 dual variable 1
%   rho2 dual variable 2
%   conjGradIter max number of conjugate gradient iterations
%   cgEpsilon stopping threshold
%   
% Outputs:
% Xk (N x K x T) solution
% cgIters final number of iterations

% ADMM penalty parameter
rho1 = params.rho1;
rho2 = params.rho2;
N = size(A0,1);

% Coefficeint Vectors
Xk = X_init;

% Target Vectors
AtB = AtB_ft_1D_Time(A0,B,Bnorms);
PtDtZ = PhiTranDiffTran_1D(ZU,N);

% Initial Residual
Rk = AtB - AtAx(A0,Xk,Bnorms) +...
     rho2*PtDtZ - rho2*PtDtDPx(Xk) +...
     rho1*YV - rho1*Xk;
Pk = Rk;

for i = 1:params.conjGradIter
    Apk = AtAx(A0,Pk,Bnorms) + rho2*PtDtDPx(Pk) + rho1*Pk;
    RkRk = sum(Rk(:).*Rk(:));
    alphak = RkRk/sum(Pk(:).*Apk(:));
    Xk = Xk + alphak*Pk;
    Rkp1 = Rk - alphak*Apk;
    if norm(Rkp1(:)) < params.cgEpsilon
        break;
    end
    betak = sum(Rkp1(:).*Rkp1(:))/RkRk;
    Pk = Rkp1 + betak*Pk;
    Rk = Rkp1;
end
cgIters = i;
end

function y = AtAx(A0ft_stack,X,Bnorms)
    y = AtB_ft_1D_Time(A0ft_stack,Ax_ft_1D_Time(A0ft_stack,X,Bnorms),Bnorms);
end

function y = PtDtDPx(X)
    N = size(X,1);
    y = PhiTranDiffTran_1D(DiffPhiX_1D(X),N);
end