function [kinetic, potentialScale, Gamma, Nstar] = PositionMonitoringCoefficients(N, mu, lambda, Hb, volume)
% Coefficients of arXiv:2512.14204v2, Eqs. (2.7), (3.14), (3.15), hbar=1.
% K = kinetic*P^2 + potentialScale*V(X), L = sqrt(Gamma)*X.
% Gamma*D[X]rho = -(Gamma/2)*[X,[X,rho]].
% N is the time coordinate supplied by the caller: no extra shift is applied.
% For mu=.5 and volume=4*sqrt(2), the characteristic crossover Nstar is zero.
if nargin < 5, volume = 4*sqrt(2); end
validateattributes(volume, {'numeric'}, {'scalar','real','finite','positive'});
validateattributes(mu, {'numeric'}, {'scalar','real','finite','positive'});
validateattributes(Hb, {'numeric'}, {'scalar','real','finite','positive'});
validateattributes(lambda, {'numeric'}, {'scalar','real','finite','nonnegative'});
kinetic = exp(-3*N)/(2*Hb*volume);
potentialScale = exp(3*N)*volume/Hb;
Gamma = 131*pi*lambda^2*exp(6*N)/(256*mu^5*volume);
Nstar = log(1/(2*volume^2*mu^6))/6;
end
