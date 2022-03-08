%% Generate example data
% Gaussian peak functions with additive white Gaussian noise
close all

T = 10;
noise = 0.18;
N = 200;
[B,B_noise,awmv_true] = generateExampleData(T,noise);

figure(1)
subplot(2,1,1)
waterfall(B')
subplot(2,1,2)
waterfall(B_noise')

%% Define parameters

% Length of intensity data (theta coordinate)
P.num_theta = size(B,1); 

% Define dictionary of Gaussian basis functions
P.num_var_t = 20;   % Number of different basis functions 
P.var_theta = linspace(1/2,35,P.num_var_t).^2; % Variances of basis functions

% ADMM parameters
params.adaptRho = 1; % binary flag for adaptive rho
params.mu = 2;       % tolerated factor between primal/dual residual
params.tau = 1.05;   % rho update factor
params.alpha = 1.8; % over-relaxation paramter
params.isNonnegative = 1; % flag to enforce nonnegativity

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
D0ft = dictionaryFFT(P);
D0 = dictionary(P);

%% Setup and solve
% Without temporal coupling
params.rho1 = 1;  % ADMM parameter
params.lambda = [0.9*ones(T/2,1),1.4*ones(T/2,1)];

params.rho2 = 0;
params.gamma = 0;
X_hat_indep = convADMM_LASSO_CG_TVphi_1D(D0ft,B_noise,zeros(N,P.num_var_t,T),params);
B_hat_indep = Ax_ft_1D_Time(D0ft,X_hat_indep);

% With temporal coupling
params.rho2 = 1;
params.gamma = 0.3;
X_hat = convADMM_LASSO_CG_TVphi_1D(D0ft,B_noise,X_hat_indep,params);
B_hat = Ax_ft_1D_Time(D0ft,X_hat);

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
waterfall(B_noise')
title('noisy data')
fig3.Position = [16 745 1250 240];

fig4 = figure(4);
awmv3 = computeAWMV(X_hat_indep,P.var_theta);
awmv4 = computeAWMV(X_hat,P.var_theta);
awmv_err3 = norm(awmv3-awmv_true)/norm(awmv_true);
awmv_err4 = norm(awmv4-awmv_true)/norm(awmv_true);
plot(awmv_true,'Linewidth',2)
hold on
plot(awmv3,'Linewidth',2)
plot(awmv4,'Linewidth',2)
ylabel('AWMV')
xlabel('t')
title('AWMV')
legend('true',...
       sprintf('indep: %0.3f',awmv_err3),...
       sprintf('coupled: %0.3f',awmv_err4),...
       'Location','Best')
fig4.Position = [1276 562 560 420];
