function [plotSpan, PsiStore, RhoStore] = NMQSD_SingleTrajectory_Hybrid( ...
    bSize, etaC, Lfac, TSpan, plotSpan, hbar, ...
    mu, beta3, beta4, lambda, Hb)
% NMQSD_SingleTrajectory  -- hybrid:
%   - N < -1: Schrödinger evolution in a BIG basis (2*bSize), no bath/memory.
%   - N >= -1: Non-Markovian NMQSD in the SMALL basis (bSize) with coloured noise.
%
%   Initial state is the ground state of the effective Hamiltonian at N = TSpan(1),
%   found in the BIG basis and projected down to the SMALL basis.

% Split the supplied sampled bath/noise grid exactly at the switch.
% New samples use linear interpolation of the supplied discrete process.
assert(all(diff(TSpan)>0),'Time grid must increase.');
if TSpan(1)<-1 && TSpan(end)>-1 && ~any(abs(TSpan+1)<1e-13)
    newGrid=sort([TSpan(:);-1]);
    etaC=interp1(TSpan,etaC,newGrid,'linear');
    Lfac=interp1(TSpan,Lfac,newGrid,'linear');
    TSpan=newGrid.';
end

% ---------- time grid -------------------------------------------------
Nsteps = numel(TSpan) - 1;

% ---------- map plotSpan -> nearest indices in TSpan ------------------
Nplot    = numel(plotSpan);
tIdxReal = interp1(TSpan, 1:numel(TSpan), plotSpan, 'nearest', 'extrap');
plotIdx  = round(tIdxReal);                         % integer indices
plotIdx  = max(1, min(numel(TSpan), plotIdx));      % clamp to [1, end]

% snap the output times to the actual grid we will use
plotSpan = TSpan(plotIdx);

% ---------- dimensions & storage --------------------------------------
Ndim     = bSize;
Id       = eye(Ndim);

PsiStore = zeros(Ndim, Nplot);
RhoStore = zeros(Ndim, Ndim, Nplot);
nextPlot = 1;

% =====================================================================
%   BUILD BIG-BASIS OPERATORS AND INITIAL GROUND STATE
% =====================================================================
BigbSize = 2 * bSize;

% --- big-basis ladder operators ---------------------------------------
ab_big = zeros(BigbSize);
for n = 1:BigbSize-1
    ab_big(n,n+1) = sqrt(n);
end
adag_big = ab_big';

Xb = sqrt(hbar/2) * (adag_big + ab_big);
Pb = 1i * sqrt(hbar/2) * (adag_big - ab_big);

Xb = sparse(Xb); Xb = 0.5*(Xb + Xb');     % Hermitian
Pb = sparse(Pb); Pb = 0.5*(Pb + Pb');     % Hermitian

% powers and potential in BIG basis
X2b    = Xb * Xb;
X3b    = X2b * Xb;
X4b    = X3b * Xb;
P2b    = Pb' * Pb;

Vchi_b = -(mu^2/2)          * X2b ...
       + (2*beta3*mu/3)     * X3b ...
       + ((beta4^2-beta3^2)/4) * X4b;

H1b = (0.5/Hb) * P2b;
H2b = (1.0/Hb) * Vchi_b;

% --- initial effective Hamiltonian and ground state -------------------
N0    = TSpan(1);
a2_0  = exp(3*N0);
a1_0  = 1/a2_0;
H_eff0 = a1_0 * H1b + a2_0 * H2b;

[Vec_big, E_big] = eig(full(H_eff0));
[~, idx0]        = min(diag(E_big));     % ground-state index
psi_big          = Vec_big(:, idx0);
psi_big          = psi_big / norm(psi_big);
clear Vec_big E_big

% project to small basis and normalise
psi0_small = psi_big(1:bSize);
psi0_small = psi0_small / max(norm(psi0_small), eps);

% ---------- SMALL-basis operators for NMQSD stage ---------------------
Xhat = sparse(Xb(1:bSize,1:bSize)); Xhat = 0.5*(Xhat + Xhat');
Phat = sparse(Pb(1:bSize,1:bSize)); Phat = 0.5*(Phat + Phat');
P2   = Phat * Phat';

X2s  = Xhat * Xhat;
X3s  = X2s * Xhat;
X4s  = X3s * Xhat;

Vchi = -(mu^2/2)             * X2s ...
     + (2*beta3*mu/3)        * X3s ...
     + ((beta4^2-beta3^2)/4) * X4s;

% ---------- store any snapshots at the initial time -------------------
while nextPlot <= Nplot && plotIdx(nextPlot) == 1
    PsiStore(:,nextPlot)   = psi0_small;
    RhoStore(:,:,nextPlot) = psi0_small * psi0_small';
    nextPlot = nextPlot + 1;
end

% ---------- determine switch index N = -1 -----------------------------
Nswitch   = -1;
idxSwitch = find(TSpan <= Nswitch, 1, 'last');   % last gridpoint ≤ -1

if isempty(idxSwitch)
    % no N < -1 region on the TSpan grid: pure NMQSD from the start
    idxSwitch = 1;
end

% =====================================================================
%   STAGE 1: Schrödinger evolution in BIG basis for N < -1
% =====================================================================
if idxSwitch > 1
    % start from ground state in BIG basis
    psi_big = psi_big;   % already set above

    for n = 1:(idxSwitch-1)
        Nn  = TSpan(n);
        Nn1 = TSpan(n+1);
        dN  = Nn1 - Nn;

        % BIG-basis Hamiltonian at Nn (same structure as Markovian code)
        H_big_n = (0.5*P2b*exp(-3*Nn) + Vchi_b*exp(3*Nn)) / Hb;

        psi_big = Ham_Step_Big(psi_big, dN, H_big_n, hbar);



        % project back to small basis and normalise
        psi_small = psi_big(1:bSize);
        psi_small = psi_small / max(norm(psi_small), eps);

        % store any snapshots at TSpan(n+1)
        while nextPlot <= Nplot && plotIdx(nextPlot) == (n+1)
            % console output for Schrödinger stage
            phi_big  = real(psi_big' * Xb  * psi_big);
            phiV_big = real(psi_big' * X2b * psi_big) - phi_big^2;
            fprintf('Schro (big): φ=%+6.3f Var(φ)=%.3f N=%+.3f dN=%.3g step %d\n', ...
                phi_big, phiV_big, Nn1, dN, n+1);
            PsiStore(:,nextPlot)   = psi_small;
            RhoStore(:,:,nextPlot) = psi_small * psi_small';
            nextPlot = nextPlot + 1;
        end
    end

    % initial state for NMQSD stage
    psi = psi_small;

else
    % no BIG-basis stage; start NMQSD directly from ground state in small basis
    psi = psi0_small;
end

% =====================================================================
%   STAGE 2: Non-Markovian NMQSD for N >= -1 in SMALL basis
% =====================================================================

% nothing to do if idxSwitch is at the final gridpoint
if idxSwitch < numel(TSpan)

    % ---------- auxiliary G-matrix (Ndim × r) -------------------------
    r = size(Lfac,2);
    G = zeros(Ndim, r);                 % columns G_k

    % initial functional derivative F at TSpan(idxSwitch)
    dN0   = TSpan(idxSwitch+1) - TSpan(idxSwitch);
    L0_op = (lambda/Hb)*exp(3*TSpan(idxSwitch))*Xhat;
    phiExp = real(psi' * L0_op * psi);
    Fnew   = (L0_op - phiExp*Id) * psi * dN0;

    % add L_{idxSwitch,k} F to every G_k  (row idxSwitch of Lfac)
    G = Fnew * Lfac(idxSwitch,:);       % Ndim×1 times 1×r -> Ndim×r

    % ------------------ MAIN NMQSD LOOP (Heun) ------------------------
    for n = idxSwitch:Nsteps

        % ---- grid points and local step -----------------------------
        Nn   = TSpan(n);
        Nn1  = TSpan(n+1);
        dN   = Nn1 - Nn;

        % ---- operators at Nn ----------------------------------------
        H_n = (0.5*P2*exp(-3*Nn ) + Vchi*exp(3*Nn ))  / Hb;
        L_n = (lambda/Hb)*exp(3*Nn) * Xhat;

        % ---- coloured noise at N_{n+1} ------------------------------
        eta_n1 = etaC(n+1);

        % ---- memory term at Nn (rank-r contraction) -----------------
        Lrow  = Lfac(n+1,:).';          % r×1, row n+1 of Lfac
        Mem_n = G * Lrow;               % Ndim×1

        % ---- drift operator at Nn -----------------------------------
        D_n     = -1i/hbar*H_n + (L_n - phiExp*Id)*conj(eta_n1);
        drift_n = D_n*psi - (L_n - phiExp*Id)*Mem_n;

        % ---- predictor (Heun) ---------------------------------------
        psi_pred = psi + dN * drift_n;
        psi_pred = psi_pred / max(norm(psi_pred), eps);

        % propagate G homogeneously with the same D_n
        G_pred = G + dN * (D_n * G);

        % ---- operators at Nn1 (for corrector) -----------------------
        H_n1 = (0.5*P2*exp(-3*Nn1) + Vchi*exp(3*Nn1)) / Hb;
        L_n1 = (lambda/Hb)*exp(3*Nn1) * Xhat;

        % ---- predictor expectations and memory at Nn1 ---------------
        phi_pred = real(psi_pred' * L_n1 * psi_pred);
        Mem_pred = G_pred * Lrow;       % still uses row n+1 of Lfac

        % ---- drift at Nn1 (using predictor state) -------------------
        D_n1     = -1i/hbar*H_n1 + (L_n1 - phi_pred*Id)*conj(eta_n1);
        drift_n1 = D_n1*psi_pred - (L_n1 - phi_pred*Id)*Mem_pred;

        % ---- corrector step for psi ---------------------------------
        psi = psi + 0.5*dN*(drift_n + drift_n1);
        psi = psi / max(norm(psi), eps);

        % ---- corrector step for homogeneous G -----------------------
        G_hom = G + 0.5*dN*(D_n*G + D_n1*G_pred);

        % ---- new functional derivative F_{n+1} ----------------------
        L_np1_op = L_n1;                        % same matrix
        phiExp   = real(psi' * L_np1_op * psi); % this φ is for next step
        Fnew     = (L_np1_op - phiExp*Id) * psi * dN;

        % ---- update G with source term (add L_{n+1,k} F_{n+1}) ------
        G = G_hom + Fnew * Lfac(n+1,:);        % outer product

        % ---- snapshots: store at requested indices ------------------
        % we have just advanced to time index (n+1)
        while nextPlot <= Nplot && plotIdx(nextPlot) == (n+1)
            PsiStore(:,nextPlot)   = psi;
            RhoStore(:,:,nextPlot) = psi * psi';

            phi    = real(psi' * Xhat * psi);
            phiVar = real(psi' * (X2s) * psi) - phi^2;
            fprintf('NM-SSE (hybrid): φ = %+6.3f   Var(φ) = %.3f   N = %.3f   snap %d\n', ...
                phi, phiVar, Nn1, nextPlot);

            nextPlot = nextPlot + 1;
        end
    end
end
end  % ==== main function ====
