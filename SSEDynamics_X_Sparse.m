function [tvals, PsiOut, RhoOut] = SSEDynamics_X_Sparse( ...
    bSize, hbar, mu, beta3, beta4, lambda, Hb, PlotSpan, dNs)

% SSE in big basis (2*bSize), storage in small basis (bSize).
% Only X-channel Lindblad operator retained. No P-channel Lindblad,
% no cross-noise terms.

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

% precompute big-basis P^2 needed by opPackBig (Hamiltonian only)
Phat2b = Pb*Pb';

% ---------- main SSE loop: big basis evolution only ---------------------
for s = 1:KnotN-1
    t_next = PlotSpan(s+1);

    dt_loc = max(dNs(s), eps);  % local adaptive dN step

    while t < t_next - tol
        dt  = min(dt_loc, t_next - t);
        dW1 = sqrt(dt) * randn;       % only one noise now

        % operators at t and t+dt in big basis (X-channel only)
        [H0, L1_0] = opPackBig(t,      Xb, Phat2b, lambda, mu, Vchi_b, Hb);
        if t_next>-1
            % [H1, L1_1] = opPackBig(t+dt,   Xb, Phat2b, lambda, mu, Vchi_b, Hb);
            % psi_big = SRK_Step(psi_big, dt, dW1, ...
            %     H0, H1, L1_0, L1_1, ...
            %     BigbSize, hbar);
            psi_big = Euler_Step(psi_big, dt, dW1, H0, L1_0,BigbSize, hbar);
        else
            psi_big=Ham_Step(psi_big, dt, H0,hbar);
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
%%  1D Stochastic RK integrator (Rößler SRK(1.0) for single noise)
%% ====================================================================
function Psi_new = SRK_Step(Psi, dtLoc, dW1, ...
                            H0, H1, L1_now, L1_next, ...
                            bSz, hbarLoc)

    % Drift and stochastic term at t_k
    DPsi0 = Drift(Psi, H0, L1_now, bSz, hbarLoc);   % f(Ψ_k, t_k)
    G1_0  = Stoch1(Psi, L1_now, bSz, hbarLoc);      % G(Ψ_k, t_k)

    % 1D Itô iterated integral on this step
    I11 = ito_iterated_1d(dtLoc, dW1);

    sqdt = sqrt(dtLoc);

    % Supporting state for the drift:
    % x_2^(0) = Ψ_k + f(Ψ_k)*Δt
    Psi1 = Psi + DPsi0*dtLoc;

    % Supporting states for n = 1:
    % Xi_1 = (G1*I11) / √Δt
    Xi_1   = ( G1_0*I11 ) / sqdt;
    Psi2_1 = Psi + DPsi0*dtLoc + Xi_1;
    Psi3_1 = Psi + DPsi0*dtLoc - Xi_1;

    % Rößler SRK(1.0) update (single noise)
    % Drift part: 1/2 (f(Ψ_k) + f(x_2^(0))) Δt
    T1 = Psi + 0.5*(DPsi0 + Drift(Psi1, H1, L1_next, bSz, hbarLoc))*dtLoc;

    % Diffusion part: G(Ψ_k) ΔW
    T2 = G1_0*dW1;

    % Stochastic RK correction: 1/2√Δt [ G(x_2^(1)) - G(x_3^(1)) ]
    T3 = 0.5*sqdt * ( ...
         Stoch1(Psi2_1, L1_next, bSz, hbarLoc) - ...
         Stoch1(Psi3_1, L1_next, bSz, hbarLoc) );

    Psi_new = T1 + T2 + T3;
    Psi_new = Psi_new / max(norm(Psi_new), eps); % renormalize
end

function Psi_new = Euler_Step(Psi, dtLoc, dW1, ...
                            H0, L1_now, ...
                            bSz, hbarLoc)

    % Drift and stochastic term at t_k
    DPsi0 = Drift(Psi, H0, L1_now, bSz, hbarLoc);   % f(Ψ_k, t_k)
    G1_0  = Stoch1(Psi, L1_now, bSz, hbarLoc);      % G(Ψ_k, t_k)



    % Supporting state for the drift:
    % x_2^(0) = Ψ_k + f(Ψ_k)*Δt
    Psi_new =( Psi + DPsi0*dtLoc+G1_0*dW1)/norm(Psi);

end


function Psi_new = Ham_Step(Psi, dtLoc, H0,hbarV)
        Psi_new=expmv( (-1i/hbarV)*H0,Psi,dtLoc);
        Psi_new=Psi_new/norm(Psi_new);
end

function out = Drift(PsiV, H, L1, bSz, hbarV)
    Id  = speye(bSz);   % identity for Drift1
    out = (-1i/hbarV) * (H*PsiV) ...
        + Drift1(PsiV, L1, Id, hbarV);
end

% LSSE-style drift trick for the X-channel
function out = Drift1(PsiV, L1, Id, hbarV)
    EPsi = L1 - (PsiV' * L1 * PsiV)*Id;   % operator with mean subtracted
    out  = (-0.5/hbarV) * (EPsi' * EPsi) * PsiV;
end

function out = Stoch1(PsiV, L1, bSz, hbarV)
    out = ( L1 - (PsiV' * L1 * PsiV)*eye(bSz) ) * PsiV / sqrt(hbarV);
end


%% ====================================================================
%%  1D Itô iterated integral: I^{11}
%% ====================================================================
function I11 = ito_iterated_1d(dt, dW)
    % Diagonal Itô integral (exact for scalar Wiener):
    % I^{11} = ∫∫ dW_s dW_t = 0.5(ΔW^2 - Δt)
    I11 = 0.5*(dW^2 - dt);
end


%% ====================================================================
%%  Big-basis operators for SSE (X-Lindblad only)
%% ====================================================================
function [H, L1] = opPackBig(Ncur, Xhat_b, Phat2_b, lambda, mu, Vchi_b, Hb)
    % X-channel rate
    Gamma = (131*pi*lambda^2 * exp(6*Ncur)) / (512*mu^5);

    % Hamiltonian (still contains P^2 kinetic term via Phat2_b)
    H  = (0.5*Phat2_b * exp(-3*Ncur) + Vchi_b * exp(3*Ncur)) / Hb;

    % Only X Lindblad operator
    L1 = sqrt(Gamma) * Xhat_b;
end
