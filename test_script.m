%% Generate example data
% Gaussian peak functions with additive white Gaussian noise
% close all
% N = 200;
% T = 10;
% numSpots = 2;
% B = zeros(N,T);
% B_noise = zeros(N,T);
% noise = 0.01;
% amplitude = 2*rand(numSpots,1)+1;
% mean_param = N*rand(numSpots,1);
% std_param = 7*rand(numSpots,1)+1;
% std_param2 = 10*rand(numSpots,1)+10;
% std_time = zeros(numSpots,T);
% awmv_true = zeros(T,1);
% for i = 1:numSpots
%    std_time(i,:) = linspace(std_param(i),std_param2(i),T); 
% end
% for t = 1:T
%     for i = 1:numSpots
%         b = gaussian_basis_wrap_1D(N,...
%                                    mean_param(i),...
%                                    std_time(i,t),...
%                                    '2-norm');
%        awmv_true(t) = awmv_true(t) + amplitude(i)*std_time(i,t);
%        B(:,t) = B(:,t) + amplitude(i)*b;
%     end
%     awmv_true(t) = awmv_true(t)/sum(amplitude(:));
%     B_noise(:,t) = B(:,t) + noise*randn(N,1);
% end
% 
% figure(1)
% subplot(2,1,1)
% waterfall(B')
% subplot(2,1,2)
% waterfall(B_noise')

%% Define parameters

% Length of intensity data (theta coordinate)
P.num_theta = size(B,1); 

% Define dictionary of Gaussian basis functions
P.num_var_t = 20;   % Number of different basis functions 
P.var_theta = linspace(1/2,35,P.num_var_t).^2; % Variances of basis functions
P.basis = 'norm2';
% Zero padding and mask (just ignore this)
zPad = [0,0];
zMask = [];

% ADMM parameters
lambda1 = 1.1;
params.lambda1 = lambda1; % sparsity penalty
params.rho1 = 1;  % initial ADMM
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

params.zeroPad = zPad; % number of [rows,columns]of padding to add to data
params.zeroMask = zMask; % specifies columns known to be zero

params.plotProgress = 0; % flag to plot intermediate solution at each iteration 
params.verbose = 1;      % flag to print objective values at each iteration 
P.params = params;

% Construct dictionary
A0ft_stack = unshifted_basis_vector_ft_stack_zpad(P);
A0_stack = unshifted_basis_vector_stack_zpad(P);

%% Setup and solve

% % Initialize solution
% x_init = zeros(size(A0ft_stack));
% 
% % Solve
% x_hat1 = convADMM_LASSO_Sherman_1D(A0ft_stack/norm(b),b/norm(b),x_init,params);
% 
% % Compute result
% b_hat = Ax_ft_1D(A0ft_stack,x_hat1);
% 
% 
% % Plot fit
% figure(1)
% subplot(1,2,1)
% plot(b)
% hold on
% plot(b_hat)
% xlabel('\theta')
% ylabel('Intensity')
% title('Data fit')
% legend('data','fit')
% 
% % Plot variance distribution function
% vdf = sum(x_hat1,1)/sum(x_hat1,'all');
% subplot(1,2,2)
% bar(vdf)
% xlabel('narrow --> \sigma index --> wide')
% ylabel('\Sigma x_i / \Sigma x')
% title('VDF')

%% Indep CG
% params.lambda1 = lambda1; % sparsity penalty
% params.rho1 = 0.1;  % initial ADMM
% x_hat2 = convADMM_LASSO_CG_1D(A0ft_stack/norm(b),b/norm(b),x_init,params);
% 
% % Compute result
% b_hat2 = Ax_ft_1D(A0ft_stack,x_hat2);
% % Plot fit
% figure(2)
% subplot(1,2,1)
% plot(b)
% hold on
% plot(b_hat2)
% xlabel('\theta')
% ylabel('Intensity')
% title('Data fit')
% legend('data','fit')
% % Plot variance distribution function
% vdf2 = sum(x_hat2,1)/sum(x_hat2,'all');
% subplot(1,2,2)
% bar(vdf2)
% xlabel('narrow --> \sigma index --> wide')
% ylabel('\Sigma x_i / \Sigma x')
% title('VDF')

%% Coupled CG
params.rho2 = 0;
params.lambda2 = 0;
params.lambda1 = lambda1*ones(T,1);
X_hat3 = convADMM_LASSO_CG_TVphi_1D(A0ft_stack,B_noise,zeros(N,P.num_var_t,T),params);

%% Coupled CG
params.rho2 = 1;
params.lambda2 = 0.01;
params.lambda1 = lambda1*ones(T,1);
X_hat4 = convADMM_LASSO_CG_TVphi_1D(A0ft_stack,B_noise,zeros(N,P.num_var_t,T),params);


%% Plot results
B_hat3 = Ax_ft_1D_Time(A0ft_stack,X_hat3);
B_hat4 = Ax_ft_1D_Time(A0ft_stack,X_hat4);

% Plot fit
fig3 = figure(3)
subplot(1,4,1)
waterfall(B')
title('truth')

subplot(1,4,2)
waterfall(B_hat3')
err1 = norm(B_hat3(:)-B(:))/norm(B(:))/T;
title(['recon: ',sprintf('%0.3f',err1)])

subplot(1,4,3)
waterfall(B_hat4')
err2 = norm(B_hat4(:)-B(:))/norm(B(:))/T;
title(['coupled recon: ',sprintf('%0.3f',err2)])

subplot(1,4,4)
waterfall(B_noise')
title('noisy data')
fig3.Position = [16 745 1250 240];

fig4 = figure(4);
awmv3 = computeAWMV(X_hat3,P.var_theta);
awmv4 = computeAWMV(X_hat4,P.var_theta);
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

Changing the lambda1 parameter will render different results:
- lambda1 = 1 fits the peak well and not the noise (this is ideal)
- lambda1 = 0.1 fits the peak and the noise well
- lambda1 = 0.01 fits everything almost exactly
- lambda1 = 10 fits peak poorly and fits mean of noise with wide basis
            function
- lambda1 = 100 fit is worse

We can look at which basis functions were used and how that varies with
selection of lambda1:
- lambda1 = 1 
    Basis function 2 with sigma ~3.1 contributes ~40% of the signal
    Basis function 3 with sigma ~5.7 contributes ~53% of the signal
    True sigma is sqrt(20) = 4.47
    3.1*0.40 + 5.7*0.53 = 4.3
    The 2 basis functions nearest in size to the peak in the data
    contribute 93% of the signal
- lambda1 = 0.01 Biases solution towards using the smallest basis function 
    which allows the algorithm to fit everything almost exactly
- lambda1 = 100 Biases solution towards using the larger basis functions
    than are necessary because they tend to be cheaper in terms of l1-norm
    cost.

%}

