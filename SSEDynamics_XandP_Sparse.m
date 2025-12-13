function [tvals, PsiOut, RhoOut] = SSEDynamics_XandP_Sparse( ...
    bSize, hbar, mu, beta3, beta4, lambda, Hb, PlotSpan, dNs)

% SSE in big basis (2*bSize), storage in small basis (bSize).

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

H1b = (0.5/Hb) * P2b;
H2b = (1.0/Hb) * Vchi_b;

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

% precompute big-basis Phat^2 needed by opPack
Phat2b = Pb*Pb';

% ---------- main SSE loop: big basis evolution only ---------------------
for s = 1:KnotN-1
    t_next = PlotSpan(s+1);

    dt_loc = max(dNs(s), eps);  % local adaptive dN step

    while t < t_next - tol
        dt  = min(dt_loc, t_next - t);
        dW1 = sqrt(dt) * randn;
        dW2 = sqrt(dt) * randn;

        % operators at t and t+dt in big basis
        if t_next<-1 % phi lindblad irrelavant at this time
            [H0, L1_0, L2_0] = opPackBig(t+dt/2,      Xb, Pb, Phat2b, lambda, mu, Vchi_b, Hb);
            psi_big = ExpP_Step(psi_big, dt, dW2,H0,Hb,Pb, hbar);

        else % phi both lindblads
            [H0, L1_0, L2_0] = opPackBig(t,      Xb, Pb, Phat2b, lambda, mu, Vchi_b, Hb);

            % [H1, L1_1, L2_1] = opPackBig(t+dt,   Xb, Pb, Phat2b, lambda, mu, Vchi_b, Hb);
            % psi_big = SRK_Step(psi_big, dt, dW1, dW2, H0, H1, L1_0, L1_1, L2_0, L2_1, BigbSize, hbar);

            psi_big = Euler_Step(psi_big, dt, dW1, dW2, H0, L1_0, L2_0, BigbSize, hbar);

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

end % ==== main ====


%% ====================================================================
%%  Stochastic RK integrator (Rößler SRK(1.0) with full 2D I^{ij})
%% ====================================================================
function Psi_new = SRK_Step(Psi, dtLoc, dW1, dW2, ...
    H0, H1, L1_now, L1_next, L2_now, L2_next, ...
    bSz, hbarLoc)

% Drift and stochastic terms at t_k
DPsi0 = Drift(Psi, H0, L1_now, L2_now, bSz, hbarLoc);   % f(Ψ_k, t_k)
G1_0  = Stoch1(Psi, L1_now, bSz, hbarLoc);              % G1(Ψ_k, t_k)
G2_0  = Stoch2(Psi, L2_now, bSz, hbarLoc);              % G2(Ψ_k, t_k)

% 2D Itô iterated integrals on this step
[I11, I22, I12, I21] = ito_iterated_2d(dtLoc, dW1, dW2);

sqdt = sqrt(dtLoc);

% Supporting state for the drift (n = 0):
% x_2^(0) = Ψ_k + f(Ψ_k)*Δt
Psi1 = Psi + DPsi0*dtLoc;

% Supporting states for n = 1:
% Xi_1 = (G1*I11 + G2*I21) / √Δt
Xi_1   = ( G1_0*I11 + G2_0*I21 ) / sqdt;
Psi2_1 = Psi + DPsi0*dtLoc + Xi_1;
Psi3_1 = Psi + DPsi0*dtLoc - Xi_1;

% Supporting states for n = 2:
% Xi_2 = (G1*I12 + G2*I22) / √Δt
Xi_2   = ( G1_0*I12 + G2_0*I22 ) / sqdt;
Psi2_2 = Psi + DPsi0*dtLoc + Xi_2;
Psi3_2 = Psi + DPsi0*dtLoc - Xi_2;

% Rößler SRK(1.0) update
% Drift part: 1/2 (f(Ψ_k) + f(x_2^(0))) Δt
T1 = Psi + 0.5*(DPsi0 + Drift(Psi1, H1, L1_next, L2_next, bSz, hbarLoc))*dtLoc;

% Diffusion part: ∑_n G_n(Ψ_k) ΔW_n
T2 = G1_0*dW1 + G2_0*dW2;

% Stochastic RK correction: 1/2√Δt [ G_n(x_2^(n)) - G_n(x_3^(n)) ]
T3 = 0.5*sqdt * ( ...
    (Stoch1(Psi2_1, L1_next, bSz, hbarLoc) - Stoch1(Psi3_1, L1_next, bSz, hbarLoc)) + ...
    (Stoch2(Psi2_2, L2_next, bSz, hbarLoc) - Stoch2(Psi3_2, L2_next, bSz, hbarLoc)) );

Psi_new = T1 + T2 + T3;
Psi_new = Psi_new / max(norm(Psi_new), eps); % renormalize
end

function Psi_new = Euler_Step(Psi, dtLoc, dW1, dW2, ...
    H0, L1_now, L2_now, ...
    bSz, hbarLoc)

% Drift and stochastic terms at t_k
Norm=norm(Psi);

DPsi0 = Drift(Psi, H0, L1_now, L2_now, bSz, hbarLoc);   % f(Ψ_k, t_k)
G1_0  = Stoch1(Psi, L1_now, bSz, hbarLoc);              % G1(Ψ_k, t_k)
G2_0  = Stoch2(Psi, L2_now, bSz, hbarLoc);              % G2(Ψ_k, t_k)

% Supporting state for the drift (n = 0):
% x_2^(0) = Ψ_k + f(Ψ_k)*Δt
Psi_new = (Psi + DPsi0*dtLoc+G1_0*dW1+G2_0*dW2)/Norm;

end

function Psi_new = ExpP_Step(Psi, dt, dW2, ...
    H0,Hb,Pb, hbar)
dEta=dW2/dt;
G  =H0+(Hb/2/pi)*Pb*dEta;
Psi_new=expmv((-1i/hbar)*G, Psi, dt);
end


function out = Drift(PsiV, H, L1, L2, bSz, hbarV)
Id  = speye(bSz);   % identity for Drift1
out = (-1i/hbarV) * (H*PsiV) ...
    + Drift1(PsiV, L1, Id, hbarV) ...
    + Drift2(PsiV, L2, [], hbarV);
end

% Your LSSE-style drift trick for the X-channel
function out = Drift1(PsiV, L1, Id, hbarV)
EPsi = L1 - (PsiV' * L1 * PsiV)*Id;   % operator with mean subtracted
out  = (-0.5/hbarV) * (EPsi' * EPsi) * PsiV;
end

function out = Drift2(PsiV, L2, ~, hbarV)
out  = (1/hbarV) * (-0.5*(L2'*L2)) * PsiV;
end

function out = Stoch1(PsiV, L1, bSz, hbarV)
out = ( L1 - (PsiV' * L1 * PsiV)*eye(bSz) ) * PsiV / sqrt(hbarV);
end

function out = Stoch2(PsiV, L2, ~, hbarV)
out = (-1i * L2) * PsiV / sqrt(hbarV);
end


%% ====================================================================
%%  2D Itô iterated integrals: I^{11}, I^{22}, I^{12}, I^{21}
%%  (Davie-style Gaussian Lévy area approximation)
%% ====================================================================
function [I11, I22, I12, I21] = ito_iterated_2d(dt, dW1, dW2)
% Diagonal Itô integrals (exact)
I11 = 0.5*(dW1^2 - dt);
I22 = 0.5*(dW2^2 - dt);

% Symmetric cross part
cross_sym = 0.5*dW1*dW2;

% Lévy area A ~ N(0, dt^2/12)
% (Davie-type approximation; good enough for strong order 1 SRK)
A = (dt / sqrt(12)) * randn;

% Off-diagonal Itô integrals
I12 = cross_sym + A;
I21 = cross_sym - A;
end


%% ====================================================================
%%  Big-basis operators for SSE
%% ====================================================================
function [H, L1, L2] = opPackBig(Ncur, Xhat_b, Phat_b, Phat2_b, lambda, mu, Vchi_b, Hb)
Gamma = (131*pi*lambda^2 * exp(6*Ncur)) / (512*mu^5);

H  = (0.5*Phat2_b * exp(-3*Ncur) + Vchi_b * exp(3*Ncur)) / Hb;
L1 = sqrt(Gamma) * Xhat_b;
L2 = (Hb/(2*pi)) * Phat_b;     % constant in time
end


