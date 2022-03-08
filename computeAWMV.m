function awmv = computeAWMV(x,dict_variances)
%computeAWMV Computes Amplitude-Weighted Mean Variance (AWMV
%given coefficients and dictionary variances. The coefficients
%can correspond to one or multiple time instances
% Inputs: 
% x - coefficients [N,K] or [N,K,T]
% dict_variances - variances of dictionary atoms [K,1]
%
% Outputs:
% awmv - Spread metric scalar or [T,1]


if size(x) < 3 % Single instance
    uvdf = squeeze(sum(x,1));
    awmv = sum( uvdf(:).*sqrt(dict_variances(:)) )/sum(x,'all');
else % Sequence
    T = size(x,3);
    awmv = zeros(T,1);
    uvdf = squeeze(sum(x,1));
    for t = 1:T
        awmv(t) = sum( uvdf(:,t).*sqrt(dict_variances(:)) )/sum(uvdf(:,t),'all');    
    end
end
