function [plotSpan, PsiStore, RhoStore] = SchrodingerSingleTrajectory_ExpStep_1000( ...
    TSpan, bSize, hbar, mu, beta3, beta4, Hb)

% SchrodingerSingleTrajectory_ExpStep_1000
%   Fixed-step propagator for
%       dψ/dN = -(i/ħ) H(N) ψ ,
%   with
%       H(N) = ½ P̂² e^{-3N} + Vχ e^{+3N}
%
% Entire evolution is done in a big basis of size 3*bSize.
% Initial state is the ground state of the effective Hamiltonian
%   H_eff(N0) = a1(N0) H1b + a2(N0) H2b
% in the big basis. Outputs are projected to the first bSize components.

%% ---------- basic time grid -------------------------------------------
N0   = TSpan(1);
Nend = TSpan(end);

nSteps = 1000;
if Nend == N0
    dt = 0;
else
    dt = (Nend - N0)/nSteps;
end

%% ---------- big-basis construction ------------------------------------
BigbSize = 3*bSize;

ab_big = zeros(BigbSize);
for n = 1:BigbSize-1
    ab_big(n,n+1) = sqrt(n);
end
adag_big = ab_big';

Xhat_b = sqrt(hbar/2) * (adag_big + ab_big);
Phat_b = 1i * sqrt(hbar/2) * (adag_big - ab_big);

Xb = sparse(Xhat_b); Xb = 0.5*(Xb + Xb');   % Hermitian
Pb = sparse(Phat_b); Pb = 0.5*(Pb + Pb');   % Hermitian

X2b  = Xb*Xb;
X3b  = X2b*Xb;
X4b  = X3b*Xb;
P2b  = Pb'*Pb;

Vchi_b = -(mu^2/2)*X2b + (2*beta3*mu/3)*X3b ...
         + ((beta4^2 - beta3^2)/4)*X4b;

H1b = (0.5/Hb)*P2b;
H2b = (1.0/Hb)*Vchi_b;

%% ---- initial state: ground state of H_eff(N0) in big basis -----------
a2_0   = exp(3*N0);
a1_0   = 1/a2_0;
H_eff0 = a1_0*H1b + a2_0*H2b;

[Vec_big, E_big] = eig(full(H_eff0));
[~, idx0]        = min(diag(E_big));        % ground state
psi_big          = Vec_big(:, idx0);
psi_big          = psi_big / norm(psi_big);
clear Vec_big E_big

%% ---------- storage ----------------------------------------------------
nSaves   = nSteps + 1;       % include initial state
plotSpan = zeros(1, nSaves);
PsiStore = zeros(bSize, nSaves);
RhoStore = zeros(bSize, bSize, nSaves);

t = N0;
storeCount = 1;

psi_small = psi_big(1:bSize);
PsiStore(:,storeCount)     = psi_small;
RhoStore(:,:,storeCount)   = psi_small*psi_small';
plotSpan(storeCount)       = t;

%% ---------- main loop: big-basis Schrödinger evolution ----------------
for k = 1:nSteps
    if dt == 0 || t >= Nend
        break;
    end

    tm = t + 0.5*dt;              % midpoint for time-dependent coeffs
    a2 = exp(3*tm);
    a1 = 1/a2;

    % effective Hamiltonian at midpoint
    H_eff = a1*H1b + a2*H2b;

    % generator for expmv: dψ/dN = Lpsi ψ
    Lpsi = (-1i/hbar) * H_eff;

    % propagate one step
    psi_big = expmv(Lpsi, psi_big, dt);
    psi_big = psi_big / norm(psi_big);

    % project to small basis for storage
    psi_small = psi_big(1:bSize);

    t = t + dt;
    storeCount = storeCount + 1;

    plotSpan(storeCount)       = t;
    PsiStore(:,storeCount)     = psi_small;
    RhoStore(:,:,storeCount)   = psi_small*psi_small';
end

% trim storage
plotSpan = plotSpan(1:storeCount);
PsiStore = PsiStore(:,1:storeCount);
RhoStore = RhoStore(:,:,1:storeCount);

end
