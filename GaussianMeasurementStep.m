function v=GaussianMeasurementStep(v,a,q,z)
%GAUSSIANMEASUREMENTSTEP Exact Hermitian Gaussian instrument in its eigenbasis.
% q = integral Gamma/hbar dN; z is one standard-normal quantile, not dW.
% The outcome law is the Born-weighted mixture N(2*q*a_j,q).
if q==0, return; end
assert(q>0&&isfinite(q)&&isfinite(z));
a=a(:); weights=abs(v).^2; weights=weights/sum(weights); centers=2*sqrt(q)*a;
lo=z+min(centers); hi=z+max(centers); upper=z>0;
if upper, target=.5*erfc(z/sqrt(2)); else, target=.5*erfc(-z/sqrt(2)); end
for k=1:80
    mid=.5*lo+.5*hi;
    if upper, total=sum(weights.*(.5*erfc((mid-centers)/sqrt(2))));
    else, total=sum(weights.*(.5*erfc((centers-mid)/sqrt(2)))); end
    if upper&&total>target || ~upper&&total<target, lo=mid; else, hi=mid; end
    if hi-lo<8*eps*max(1,abs(mid)), break; end
end
y=sqrt(q)*(.5*lo+.5*hi); mask=abs(v)>0;
logs=log(abs(v(mask)))+a(mask)*y-q*a(mask).^2;
v(mask)=v(mask)./abs(v(mask)).*exp(logs-max(logs)); v=v/norm(v);
end
