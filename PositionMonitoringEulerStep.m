function psiNew = PositionMonitoringEulerStep(psi, dt, dW, H, L)
% Normalized Euler-Maruyama step for the Hermitian X-monitoring SSE (hbar=1).
% Normalize the NEW state: dividing by norm(psi) does not correct step error.
psi = psi/norm(psi);
meanL = real(psi'*(L*psi));
fluctuation = L*psi - meanL*psi;
drift = -1i*(H*psi) - 0.5*(L*fluctuation - meanL*fluctuation);
psiNew = psi + dt*drift + dW*fluctuation;
newNorm = norm(psiNew);
if ~isfinite(newNorm) || newNorm == 0
    error('CosmicLockdown:InvalidState', 'SSE step produced an invalid state; reduce dN.');
end
psiNew = psiNew/newNorm;
end
