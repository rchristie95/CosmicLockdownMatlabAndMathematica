function [plotSpan, RhoStore] = Markov_LindbladP_ExpStep_1000( ...
    TSpan, bSize, hbar, mu, beta3, beta4, Hb, displayFlag)
% Markov_LindbladP_ExpStep_1000
%
%   1000-step fixed-time exponential integrator:
%     - Evolution is purely Lindblad with P-dissipator only.
%     - Dynamics is carried out in a BIG basis of size 2*bSize.
%     - Output is stored in the SMALL basis (first bSize states).
%
%   No Schrödinger-only phase; the whole evolution is via expmv on vec(ρ).
%
%   Inputs:
%       TSpan       - [t0, tF]
%       bSize       - small-basis size; large basis is 2*bSize
%       hbar, mu, beta3, beta4, Hb - parameters
%       displayFlag - if true, print observables each step
%
%   Outputs:
%       plotSpan    - stored times (1 x nSaves)
%       RhoStore    - density matrices in the small basis (bSize x bSize x nSaves)

tic

% ---- basic time grid ---------------------------------------------------
t0 = TSpan(1);
tF = TSpan(end);

nSteps = 1000;
if tF == t0
    dt = 0;
else
    dt = (tF - t0)/nSteps;
end

nSaves   = nSteps + 1;
plotSpan = zeros(1, nSaves);
RhoStore = zeros(bSize, bSize, nSaves);

% ---- BIG-basis (size 2*bSize) operators for Lindblad evolution ---------
BigbSize = 2*bSize;

ab_big = zeros(BigbSize);
for n = 1:BigbSize-1
    ab_big(n,n+1) = sqrt(n);
end
adag_big  = ab_big';

Xhat_b = sqrt(hbar/2) * (adag_big + ab_big);
Phat_b = 1i * sqrt(hbar/2) * (adag_big - ab_big);

Xb = sparse(Xhat_b); Xb = 0.5*(Xb + Xb');
Pb = sparse(Phat_b); Pb = 0.5*(Pb + Pb');

P2b = Pb' * Pb;

Xpow2b = Xb * Xb;
Xpow3b = Xpow2b * Xb;
Xpow4b = Xpow3b * Xb;

Vchi_b = -(mu^2/2) * Xpow2b + (2*beta3*mu/3) * Xpow3b ...
       + ((beta4^2 - beta3^2)/4) * Xpow4b;

H1b = (0.5 / Hb) * P2b;
H2b = (1.0 / Hb) * Vchi_b;

% ---- initial state: ground state of H_eff(t0) in big basis ------------
a2_0   = exp(3*t0);
a1_0   = 1 / a2_0;
H_eff0 = a1_0*H1b + a2_0*H2b;

[Vec, E] = eig(full(H_eff0));
[~, idx] = min(diag(E));       % ground state
psi_big  = Vec(:, idx);
psi_big  = psi_big / norm(psi_big);
clear Vec E

% initial BIG-basis density matrix
Rbig = psi_big * psi_big';
trB  = real(trace(Rbig));
if trB ~= 0
    Rbig = Rbig / trB;
end

% ---- Dissipation prefactors (P-only) -----------------------------------

cP  = (Hb / (2*pi));
cP2 = cP^2;

% ---- BIG-basis Lindblad superoperators ---------------------------------

Ibig    = speye(BigbSize);
vecIbig = Ibig(:);

LH1b = (-1i/hbar) * (kron(Ibig, H1b) - kron(H1b.', Ibig));
LH2b = (-1i/hbar) * (kron(Ibig, H2b) - kron(H2b.', Ibig));

DPb  = dissipator_super(Pb, P2b, Ibig);   % P-only dissipator in big basis

% vectorised initial state in big basis
ybig = Rbig(:);

% ---- store initial SMALL-basis projection ------------------------------

t           = t0;
storeCount  = 1;
plotSpan(1) = t;

R_small = Rbig(1:bSize, 1:bSize);
trS     = real(trace(R_small));
if trS ~= 0
    R_small = R_small / trS;
end

RhoStore(:,:,1) = full(R_small);

if displayFlag
    phi    = real(trace(Rbig * Xb));
    varPhi = real(trace(Rbig * Xpow2b)) - phi^2;
    purity = real(trace(Rbig * Rbig));
    fprintf(['[exp P-only BIG] Hb=%+6.3f  φ=%+6.3f  Var(φ)=%.3f  Purity=%.3f  ', ...
             'N=%.3g  dN=%1.1e  time=%.3g\n'], ...
            Hb, phi, varPhi, purity, t, dt, toc);
end

% ---- main exponential Lindblad loop in BIG basis -----------------------

for k = 1:nSteps
    if dt == 0 || t >= tF
        break;
    end

    tm = t + 0.5*dt;     % midpoint for time-dependent coefficients
    a2 = exp(3*tm);
    a1 = 1 / a2;

    % Liouvillian in big basis (P-only dissipator)
    Lbig = a1*LH1b + a2*LH2b + cP2*DPb;

    % exponential step on vec(ρ_big)
    ybig = expmv(Lbig, ybig, dt);

    % trace renormalisation in big basis
    trB = real(vecIbig' * ybig);
    if trB ~= 0
        ybig = ybig / trB;
    end

    Rbig = reshape(ybig, BigbSize, BigbSize);

    % project to small basis and renormalise
    R_small = Rbig(1:bSize, 1:bSize);
    trS     = real(trace(R_small));
    if trS ~= 0
        R_small = R_small / trS;
    end

    % advance time and store
    t = t + dt;

    storeCount = storeCount + 1;
    plotSpan(storeCount)     = t;
    RhoStore(:,:,storeCount) = full(R_small);

    if displayFlag
        phi    = real(trace(Rbig * Xb));
        varPhi = real(trace(Rbig * Xpow2b)) - phi^2;
        purity = real(trace(Rbig * Rbig));
        fprintf(['[exp P-only BIG] Hb=%+6.3f  φ=%+6.3f  Var(φ)=%.3f  Purity=%.3f  ', ...
                 'N=%.3g  dN=%1.1e  time=%.3g\n'], ...
                Hb, phi, varPhi, purity, t, dt, toc);
    end
end

% ---- trim storage ------------------------------------------------------
plotSpan = plotSpan(1:storeCount);
RhoStore = RhoStore(:,:,1:storeCount);

end % main function


% ---- helper ------------------------------------------------------------
function D = dissipator_super(L, L2, I) %#ok<INUSD>
% D_L[ρ] = LρL† - 1/2(L†L ρ + ρ L†L)
% vec(D_L[ρ]) = (L* ⊗ L) vec(ρ) - 1/2( I ⊗ L†L + (L†L)^T ⊗ I ) vec(ρ)
Lc    = conj(L);
LdagL = L' * L;
D = kron(Lc, L) - 0.5*( kron(I, LdagL) + kron(LdagL.', I) );
end
