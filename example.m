%% Generate example data
% Gaussian peak functions with additive white Gaussian noise
close all

T = 10;
noise = 0.03;
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
P.basis = 'norm2';

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
A0ft = dictionaryFFT(P);
A0 = dictionary(P);

%% Setup and solve
% Parameters here need to be set
lambda = 1.1;
params.lambda = lambda; % sparsity penalty
params.rho1 = 1;  % ADMM parameter

% Without temporal coupling
params.rho2 = 0;
params.gamma = 0;
params.lambda = lambda*ones(T,1);
X_hat_indep = convADMM_LASSO_CG_TVphi_1D(A0ft,B_noise,zeros(N,P.num_var_t,T),params);
B_hat_indep = Ax_ft_1D_Time(A0ft,X_hat_indep);

% With temporal coupling
params.rho2 = 1;
params.gamma = 0.01;
params.lambda = lambda*ones(T,1);
X_hat = convADMM_LASSO_CG_TVphi_1D(A0ft,B_noise,zeros(N,P.num_var_t,T),params);
B_hat = Ax_ft_1D_Time(A0ft,X_hat);

%% Plot awmv recovery and 

fig3 = figure(3)
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

% Plot variance distribution function
% vdf3 = sum(x_hat3,1)/sum(x_hat3,'all');
% subplot(1,2,2)
% bar(vdf3)
% xlabel('narrow --> \sigma index --> wide')
% ylabel('\Sigma x_i / \Sigma x')
% title('VDF')
%{
Notes:

Changing the lambda parameter will render different results:
- lambda = 1 fits the peak well and not the noise (this is ideal)
- lambda = 0.1 fits the peak and the noise well
- lambda = 0.01 fits everything almost exactly
- lambda = 10 fits peak poorly and fits mean of noise with wide basis
            function
- lambda = 100 fit is worse

We can look at which basis functions were used and how that varies with
selection of lambda:
- lambda = 1 
    Basis function 2 with sigma ~3.1 contributes ~40% of the signal
    Basis function 3 with sigma ~5.7 contributes ~53% of the signal
    True sigma is sqrt(20) = 4.47
    3.1*0.40 + 5.7*0.53 = 4.3
    The 2 basis functions nearest in size to the peak in the data
    contribute 93% of the signal
- lambda = 0.01 Biases solution towards using the smallest basis function 
    which allows the algorithm to fit everything almost exactly
- lambda = 100 Biases solution towards using the larger basis functions
    than are necessary because they tend to be cheaper in terms of l1-norm
    cost.

%}

