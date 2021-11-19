function awmv = computeAWMV(x,dict_variances)
%computeAWMV Computes AWMV given coefficients and dictionary variances

if size(x) < 3
    uvdf = squeeze(sum(x,1));
    awmv = sum( uvdf(:).*sqrt(dict_variances(:)) )/sum(x,'all');
else
    T = size(x,3);
    awmv = zeros(T,1);
    uvdf = squeeze(sum(x,1));
    for t = 1:T
        awmv(t) = sum( uvdf(:,t).*sqrt(dict_variances(:)) )/sum(uvdf(:,t),'all');    
    end
end
