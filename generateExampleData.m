function [B,B_noise,awmv_true] = generateExampleData(T,noise)
%generateExampleData Generate example noisy Gaussian intensity measurements

N = 200;
numSpots = 2;
B = zeros(N,T);
B_noise = zeros(N,T);
amplitude = 2*[0.4 0.7]+1;
mean_param = N*[0.3 0.7];
std_param = 7*[0.1 0.9]+1;
std_param2 = 10*[0.5 0.55]+10;
% amplitude = 2*rand(numSpots,1)+1;
% mean_param = N*rand(numSpots,1);
% std_param = 7*rand(numSpots,1)+1;
% std_param2 = 10*rand(numSpots,1)+10;
std_time = zeros(numSpots,T);
awmv_true = zeros(T,1);
% for i = 1:numSpots
%    std_time(i,:) = linspace(std_param(i),std_param2(i),T); 
% end
for i = 1:numSpots
   std_time(i,:) = [ones(1,floor(T/2))*std_param(i),...
                    ones(1,ceil(T/2))*std_param2(i)]; 
end
for t = 1:T
    for i = 1:numSpots
        b = gaussian_basis_wrap_1D(N,...
                                   mean_param(i),...
                                   std_time(i,t),...
                                   '2-norm');
       awmv_true(t) = awmv_true(t) + amplitude(i)*std_time(i,t);
       B(:,t) = B(:,t) + amplitude(i)*b;
    end
    awmv_true(t) = awmv_true(t)/sum(amplitude(:));
    B_noise(:,t) = B(:,t) + noise*randn(N,1);
end
end

