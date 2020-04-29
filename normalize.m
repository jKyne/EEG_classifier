function [norm_x]=normalize(x,mode)
% x: the data ready for normalization
% mode: 0 (data-mean)/std  1: (data-min)/(max-min)

if mode==0
mu=mean(x,1);
sigma=std(x,1);
mu=repmat(mu,size(x,1),1);
sigma=repmat(sigma,size(x,1),1);
norm_x=(x-mu)./sigma;
else
    minimum=min(x);
    maximum=max(x);
    diff=maximum-minimum;
    diff=repmat(diff,size(x,1),1);
    minval=repmat(minimum,size(x,1),1);
    norm_x=(x-minval)./diff;
end



end