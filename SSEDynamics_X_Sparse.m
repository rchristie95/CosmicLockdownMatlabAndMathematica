function [tvals, PsiOut, RhoOut] = SSEDynamics_X_Sparse( ...
    bSize, hbar, mu, beta3, beta4, lambda, Hb, PlotSpan, dNs, volume)

% SSE in big basis (2*bSize), storage in small basis (bSize).
% Only X-channel Lindblad operator retained. No P-channel Lindblad,
% no cross-noise terms.

% Paper coefficients, with monitoring active from the supplied initial time.
if nargin < 10, volume = 4*sqrt(2); end
assert(hbar == 1, 'The paper X-monitoring model uses hbar=1.');
[kineticScale,potentialScale,GammaScale] = PositionMonitoringCoefficients(0,mu,lambda,Hb,volume);

% ---------- setup ----------
N0      = PlotSpan(1);
KnotN   = numel(PlotSpan);
tvals   = PlotSpan;

% ---------- big-basis (size 2N) operators ----------
BigbSize = 2*bSize;

ab_big = zeros(BigbSize);
for n = 1:BigbSize-1
    ab_big(n,n+1) = sqrt(n);
end
adag_big = ab_big';

Xhat_b = sqrt(hbar/2) * (adag_big + ab_big);
Phat_b = 1i * sqrt(hbar/2) * (adag_big - ab_big);

Xb = sparse(Xhat_b); Xb = 0.5*(Xb + Xb');    % Hermitian
Pb = sparse(Phat_b); Pb = 0.5*(Pb + Pb');    % Hermitian

Xpow2b = Xb*Xb;
Xpow3b = Xpow2b*Xb;
Xpow4b = Xpow3b*Xb;
P2b    = Pb'*Pb;

Vchi_b = -(mu^2/2)*Xpow2b + (2*beta3*mu/3)*Xpow3b ...
         + ((beta4^2-beta3^2)/4)*Xpow4b;

H1b = kineticScale * P2b;
H2b = potentialScale * Vchi_b;

% ---------- initial state: ground state of H_eff(N0) in big basis -------

a2_0   = exp(3*N0);
a1_0   = 1/a2_0;
H_eff0 = a1_0*H1b + a2_0*H2b;

[Vec_big, E_big] = eig(full(H_eff0));
[~, idx0]        = min(diag(E_big));     % ground-state index
psi_big          = Vec_big(:, idx0);
psi_big          = psi_big / norm(psi_big);
clear Vec_big E_big

% ---------- small-basis objects for storage / diagnostics ---------------

% project to small basis and normalise for output
psi_small0 = psi_big(1:bSize);
PsiCurr    = psi_small0 / max(norm(psi_small0), eps);

PsiOut        = zeros(bSize, KnotN);
RhoOut        = zeros(bSize, bSize, KnotN);
PsiOut(:,1)   = PsiCurr;
RhoOut(:,:,1) = PsiCurr*PsiCurr';   % Hermitian outer product

t     = N0;
tol   = 1e-10;

% ---------- main SSE loop: big basis evolution only ---------------------
for s = 1:KnotN-1
    t_next = PlotSpan(s+1);

    dt_loc = max(dNs(s), eps);  % local adaptive dN step

    while t < t_next - tol
        dt  = min(dt_loc, t_next - t);
        dW1 = sqrt(dt) * randn;       % only one noise now

        % Cache parameter validation and constant prefactors outside the fine-step loop.
        H0 = exp(-3*t)*H1b + exp(3*t)*H2b;
        Gamma = GammaScale*exp(6*t);
        if lambda == 0
            psi_big = expmv(-1i*H0, psi_big, dt);
            psi_big = psi_big/norm(psi_big);
        else
            L = sqrt(Gamma)*Xb;
            psi_big = PositionMonitoringEulerStep(psi_big,dt,dW1,H0,L);
        end
        t = t + dt;
    end

    % project and store on the small basis at the knot
    psi_small = psi_big(1:bSize);
    PsiCurr   = psi_small / max(norm(psi_small), eps);

    PsiOut(:,s+1)   = PsiCurr;
    RhoOut(:,:,s+1) = PsiCurr*PsiCurr';

    phi  = real(psi_big' * Xb * psi_big);
    phiV = real(psi_big' * Xpow2b * psi_big) - phi^2;

    fprintf('Markov-SSE (big→small): φ=%+6.3f Var(φ)=%.3f N=%+.3f dN=%+.3g snap %d\n', ...
            phi, phiV, t, dNs(s), s+1);
end

end
