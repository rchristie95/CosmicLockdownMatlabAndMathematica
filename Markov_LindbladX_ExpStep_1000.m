function [plotSpan, RhoStore] = Markov_LindbladX_ExpStep_1000( ...
    TSpan, bSize, hbar, mu, beta3, beta4, lambda, Hb, displayFlag, volume)
% 1000 midpoint-exponential steps for the paper's position-monitoring GKLS equation.
% Monitoring acts from TSpan(1), including N < -1. hbar=1.
% The initial ground state is computed in 3*bSize, projected and normalized;
% density-matrix propagation uses bSize throughout. Check basis convergence.
% Optional volume defaults to 4*sqrt(2), giving Nstar=0 when mu=.5.
if nargin < 10, volume = 4*sqrt(2); end
assert(hbar == 1, 'The paper X-monitoring model uses hbar=1.');
[kineticScale,potentialScale,GammaScale] = PositionMonitoringCoefficients(0,mu,lambda,Hb,volume);

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

% ---- big-basis (size 3N) operators for initial preparation ----------
BigbSize = 3*bSize;

ab = zeros(BigbSize);
for n = 1:BigbSize-1
    ab(n,n+1) = sqrt(n);
end
adagb  = ab';
Xhat_b = sqrt(hbar/2) * (adagb + ab);
Phat_b = 1i * sqrt(hbar/2) * (adagb - ab);

Xb = sparse(Xhat_b); Xb = 0.5*(Xb + Xb');
Pb = sparse(Phat_b); Pb = 0.5*(Pb + Pb');

X2b = Xb'*Xb;
P2b = Pb'*Pb;

Xpow2b = Xb*Xb;
Xpow3b = Xpow2b*Xb;
Xpow4b = Xpow3b*Xb;
Vchi_b = -(mu^2/2)*Xpow2b + (2*beta3*mu/3)*Xpow3b ...
         + ((beta4^2-beta3^2)/4)*Xpow4b;

H1b = kineticScale*P2b;
H2b = potentialScale*Vchi_b;

% ---- initial state: ground state of H_eff(t0) in big basis ------------
a2_0   = exp(3*t0);
a1_0   = 1/a2_0;
H_eff0 = a1_0*H1b + a2_0*H2b;

% smallest eigenvector (ground state); 'sa' = smallest algebraic
[Vec, E] = eig(full(H_eff0));                 % E is diagonal
[~, idx] = min(diag(E));                 % position of lowest eigen-value
psi_big = Vec(:, idx);
psi_big = psi_big / norm(psi_big);             % normalise
clear Vec E

% ---- small-basis (size N) operators for Lindblad evolution ------------

a = zeros(bSize);
for n = 1:bSize-1
    a(n,n+1) = sqrt(n);
end
adag = a';
Xhat = sqrt(hbar/2) * (adag + a);
Phat = 1i * sqrt(hbar/2) * (adag - a);

X = sparse(Xhat); X = 0.5*(X + X');
P = sparse(Phat); P = 0.5*(P + P');

X2 = X'*X;
P2 = P'*P;

Xpow2 = X*X;
Xpow3 = Xpow2*X;
Xpow4 = Xpow3*X;
Vchi  = -(mu^2/2)*Xpow2 + (2*beta3*mu/3)*Xpow3 ...
        + ((beta4^2-beta3^2)/4)*Xpow4;

H1 = kineticScale*P2;
H2 = potentialScale*Vchi;

% ---- Lindblad superoperators (small basis) -----------------------------
N2   = bSize*bSize;
I    = speye(bSize);
vecI = reshape(I, N2, 1);  % for fast trace using vec form

LH1 = (-1i/hbar) * (kron(I, H1) - kron(H1.', I));
LH2 = (-1i/hbar) * (kron(I, H2) - kron(H2.', I));

DX  = dissipator_super(X, X2, I);
% DP removed (no P Lindblad)

% ---- storage (store every step) ----------------------------------------
nSaves   = nSteps + 1;          % include initial state
plotSpan = zeros(1, nSaves);
RhoStore = zeros(bSize, bSize, nSaves);

% ---- initial density in small basis from psi_big -----------------------
t = t0;
storeCount = 1;

psi_small = psi_big(1:bSize);
R         = psi_small * psi_small';   % ρ = |ψ⟩⟨ψ|

% normalize trace
tr = real(trace(R));
if tr ~= 0
    R = R / tr;
end

RhoStore(:,:,storeCount) = full(R);
plotSpan(storeCount)     = t;

% vectorized density for Lindblad part
y = R(:);

% ---- main loop: unified Schrödinger + Lindblad -------------------------
for k = 1:nSteps
    if dt == 0 || t >= tF
        break;
    end

    tm = t + 0.5*dt;          % midpoint for time-dependent coefficients
    a2 = exp(3*tm);
    a1 = 1/a2;

    Gamma = GammaScale*exp(6*tm);
    % D[X] = -[X,[X,rho]]/2: Gamma has denominator 256, not 512.
    L = a1*LH1 + a2*LH2 + Gamma*DX;
    y = expmv(L, y, dt);
    tr = real(vecI' * y);
    if ~isfinite(tr) || tr <= 0
        error('CosmicLockdown:InvalidTrace', 'Invalid GKLS trace; check the exponential solver.');
    end
    y = y/tr;
    R = reshape(y,bSize,bSize);
    if displayFlag
        fprintf('X-GKLS: N=%+.3f trace=%.12g purity=%.6g\n',t+dt,real(trace(R)),real(trace(R*R)));
    end

    % advance time and store
    t = t + dt;

    storeCount = storeCount + 1;
    plotSpan(storeCount)     = t;
    RhoStore(:,:,storeCount) = full(R);
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
