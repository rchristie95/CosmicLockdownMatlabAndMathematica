function [plotSpan, RhoStore] = Markov_2Lindblads_ExpStep_1000( ...
    TSpan, bSize, hbar, mu, beta3, beta4, lambda, Hb, displayFlag)
% 1000-step fixed-time integrator with split evolution:
%   - For t < -1: Schrödinger evolution on a state vector in a
%                 Hilbert space of size 2N.
%   - For t >= -1: Lindblad evolution on a density matrix of size N.
%
% Initial state: ground state (minimum eigenvector) of the
% time-dependent Hamiltonian at t = TSpan(1), in the large basis (2N).
%
%   Inputs:
%       TSpan       - [t0, tF]
%       N           - small-basis size; large basis is 2N
%       hbar, mu, beta3, beta4, lambda, Hb - parameters
%       displayFlag - if true, print observables each step
%
%   Outputs:
%       plotSpan    - stored times
%       RhoStore    - density matrices in the small basis (N x N x nSaves)

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

% ---- big-basis (size 2N) operators for Schrödinger evolution ----------
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

H1b = (0.5/Hb)*P2b;
H2b = (1.0/Hb)*Vchi_b;

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

H1 = (0.5/Hb)*P2;
H2 = (1.0/Hb)*Vchi;

% Dissipation prefactors
cP  = (Hb/(2*pi));
cP2 = cP^2;

% Lindblad superoperators (small basis)
N2   = bSize*bSize;
I    = speye(bSize);
vecI = reshape(I, N2, 1);  % for fast trace using vec form

LH1 = (-1i/hbar) * (kron(I, H1) - kron(H1.', I));
LH2 = (-1i/hbar) * (kron(I, H2) - kron(H2.', I));

DX  = dissipator_super(X, X2, I);
DP  = dissipator_super(P, P2, I);

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

    if t < -1
        % ----- Schrödinger evolution in big basis ----------------------
        H_eff = a1*H1b + a2*H2b;
        Lpsi  = (-1i/hbar) * H_eff;

        % expmv convention: expmv(A, v, t)
        psi_big = expmv(Lpsi, psi_big, dt);
        psi_big = psi_big / norm(psi_big);

        % reduced density on small basis
        psi_small = psi_big(1:bSize);
        R = psi_small * psi_small';

        % normalize trace and sync y
        tr = real(trace(R));
        if tr ~= 0
            R = R / tr;
        end
        y = R(:);

        % observables w.r.t the large basis for output
        if displayFlag
            phi_big    = real(psi_big' * Xb      * psi_big);
            varPhi_big = real(psi_big' * Xpow2b  * psi_big) - phi_big^2;
            purity_big = real((psi_big' * psi_big)^2);  % ~1 if normalized

            fprintf(['λ=%+6.3f  Hb=%+6.3f  φ=%+6.3f  Var(φ)=%.3f  Purity=%.3f  ', ...
                     'N=%.3g  dN=%1.1e  time=%.3g\n'], ...
                     lambda, Hb, phi_big, varPhi_big, purity_big, t+dt, dt, toc);
        end
    else
        % ----- Lindblad evolution in small basis -----------------------
        GGamma = (131*pi*lambda^2 * (a2*a2)) / (512*mu^5);

        % Liouvillian: L = a1*LH1 + a2*LH2 + GGamma*DX + cP2*DP
        L = a1*LH1 + a2*LH2 + GGamma*DX + cP2*DP;

        y = expmv(L, y, dt);

        % trace re-normalize
        tr = real(vecI' * y);
        if tr ~= 0
            y = y / tr;
        end

        R = reshape(y, bSize, bSize);

        if displayFlag
            phi    = real(trace(R*X));
            varPhi = real(trace(R*Xpow2)) - phi^2;
            purity = real(trace(R*R));
            fprintf(['λ=%+6.3f  Hb=%+6.3f  φ=%+6.3f  Var(φ)=%.3f  Purity=%.3f  ', ...
                     'N=%.3g  dN=%1.1e  time=%.3g\n'], ...
                     lambda, Hb, phi, varPhi, purity, t+dt, dt, toc);
        end
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
