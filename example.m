%% Generate example data
% Gaussian peak functions with Poisson statistics
close all

T = 10;
N = 101;
[B,B_poiss,awmv_true] = generateExampleData(N,T);

figure(1)
subplot(2,1,1)
waterfall(B')
subplot(2,1,2)
waterfall(B_poiss')

%% Define parameters

% Length of intensity data (theta coordinate)
P.N = size(B,1); 

% Define dictionary of Gaussian basis functions
P.K = 20;   % Number of different basis functions 
P.sigmas = linspace(1/2,25,P.K); % Sigmas of basis functions

% ADMM parameters
params.adaptRho = 1; % binary flag for adaptive rho
params.mu = 2;       % tolerated factor between primal/dual residual
params.tau = 1.05;   % rho update factor
params.alpha = 1.8; % over-relaxation paramter
params.isNonnegative = 1; % flag to enforce nonnegativity
params.normData = 1; % flag to normalize b(t)

params.stoppingCriterion = 'OBJECTIVE_VALUE';
params.maxIter = 100;

% Conjugate gradient parameters
params.conjGradIter = 100;
params.tolerance = 1e-8;
params.cgEpsilon = 1e-6;

params.plotProgress = 0; % flag to plot intermediate solution at each iteration 
params.verbose = 1;      % flag to print objective values at each iteration 
P.params = params;

% Construct dictionary
A0ft = dictionaryFFT(P);
A0 = dictionary(P);

%% Setup and solve
% Without temporal coupling
params.rho1 = 1;                % ADMM parameter 1
params.lambda = 2e-2*ones(T,1); % Sparsity parameters
params.rho2 = 0;                % ADMM parameter 2
params.gamma = 0;               % Smoothness parameter
X_hat_indep = convADMM_LASSO_CG_TVphi_1D(A0ft,B_poiss,zeros(N,P.K,T),params);
B_hat_indep = Ax_ft_1D_Time(A0ft,X_hat_indep);

% With temporal coupling
params.rho2 = 1;     % ADMM parameter 2
params.gamma = 18e-4;% Smoothness parameter
X_hat = convADMM_LASSO_CG_TVphi_1D(A0ft,B_poiss,X_hat_indep,params);
B_hat = Ax_ft_1D_Time(A0ft,X_hat);

%% Plot awmv recovery
fig3 = figure(3);
subplot(1,4,1)
waterfall(B')
title('truth')

subplot(1,4,2)
waterfall(B_hat_indep')
err1 = norm(B_hat_indep(:)-B(:))/norm(B(:))/T;
title(['recon: ',sprintf('%0.3f',err1)])

subplot(1,4,3)
waterfall(B_hat')
err2 = norm(B_hat(:)-B(:))/norm(B(:))/T;
title(['coupled recon: ',sprintf('%0.3f',err2)])

subplot(1,4,4)
waterfall(B_poiss')
title('poisson data')
fig3.Position = [16 745 1250 240];

fig4 = figure(4);
awmv1 = computeAWMV_1D(X_hat_indep,P.sigmas);
awmv2 = computeAWMV_1D(X_hat,P.sigmas);
awmv_err3 = norm(awmv1-awmv_true)/norm(awmv_true);
awmv_err4 = norm(awmv2-awmv_true)/norm(awmv_true);
plot(awmv_true,'Linewidth',2)
hold on
plot(awmv1,'Linewidth',2)
plot(awmv2,'Linewidth',2)
ylabel('AWMV')
xlabel('t')
title('AWMV')
legend('true',...
       sprintf('indep: %0.3f',awmv_err3),...
       sprintf('coupled: %0.3f',awmv_err4),...
       'Location','Best')
fig4.Position = [1276 562 560 420];
