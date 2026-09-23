function [plotSpan, RhoStore] = NMQSD_MasterEquation_lowrank( ...
    rho, Lfac, Xhat, Phat, TSpan, hbar, ...
    mu, beta3, beta4, lambda, Hb)

% INPUT
%   rho    (Ndim×Ndim)       initial density matrix
%   Lfac   ((Nsteps+1)×r)    low–rank factor so that  C ≈ Lfac*Lfac'
%   Xhat,Phat,TSpan,hbar,mu,beta3,beta4,lambda,Hb  as before

% ---------- time grid -------------------------------------------------
N0        = TSpan(1);
Nsteps    = numel(TSpan)-1;
assert(numel(TSpan)>=2 && all(diff(TSpan)>0),'Time grid must increase.');
saveEvery = max(1,floor(Nsteps/999));
plotSpan  = N0;

% ---------- basic checks ----------------------------------------------
if size(Lfac,1) ~= Nsteps+1
    error('Lfac must have size (Nsteps+1)×r with Nsteps = numel(TSpan)-1.');
end

% ---------- system operators -----------------------------------------
Ndim  = size(rho,1);
Id    = eye(Ndim,'like',rho);

Vchi = -(mu^2/2)*(Xhat*Xhat') ...
    + (2*beta3*mu/3)*Xhat^3 ...
    + ((beta4^2 - beta3^2)/4)*(Xhat*Xhat')^2;

% ---------- storage ---------------------------------------------------
RhoStore   = rho;                     % first snapshot
storeCount = 1;

% ---------- low-rank memory accumulator -------------------------------
% G(:,:,k) ≈ ∑_{m ≤ n} Lfac(m,k) * K_m   with K_m = [L_m, ρ_m]
r   = size(Lfac,2);
G   = zeros(Ndim,Ndim,r,'like',rho);  % pages G_k

% ---------- main loop -------------------------------------------------
for n = 1:Nsteps
    % grid points
    Nn   = TSpan(n);
    Nn1  = TSpan(n+1);
    dN = Nn1-Nn;
    Nmid = 0.5*(Nn+Nn1);

    % operators at Nn and Nmid
    H_n   = (0.5*Phat^2*exp(-3*Nn ) + Vchi*exp(3*Nn )) / Hb;
    H_mid = (0.5*Phat^2*exp(-3*Nmid) + Vchi*exp(3*Nmid)) / Hb;

    L_n   = (lambda/Hb)*exp(3*Nn )  * Xhat;
    L_mid = (lambda/Hb)*exp(3*Nmid) * Xhat;

    % --- current commutator K_n --------------------------------------
    K_n = L_n*rho - rho*L_n;

    % --- update low-rank history G with K_n --------------------------
    % G_k ← G_k + Lfac(n,k) * K_n
    Lpast = reshape(Lfac(n,:),1,1,[]);   % 1×1×r
    G     = G + dN*K_n .* Lpast;

    % --- low-rank memory contraction at Nn ---------------------------
    % S = ∑_{m≤n} C(n+1,m) K_m  ≈  ∑_k Lfac(n+1,k) G_k
    Lrow = reshape(Lfac(n+1,:),1,1,[]);  % 1×1×r (row n+1)
    S    = sum( G .* Lrow , 3 );         % Ndim×Ndim

    M_n  = ( L_n*S - S*L_n );       % memory term at Nn

    % --- predictor Heun half-step ------------------------------------
    drho_n   = -1i/hbar*(H_n*rho - rho*H_n) - M_n;
    rho_half = rho + 0.5*dN*drho_n;

    % --- corrector: reuse S, but with L_mid --------------------------
    M_mid   = ( L_mid*S - S*L_mid );
    drho_mid = -1i/hbar*(H_mid*rho_half - rho_half*H_mid) - M_mid;
    rho      = rho + dN*drho_mid;

    % enforce Hermiticity and trace 1 (numerical hygiene)
    rho = 0.5*(rho + rho');
    rho = rho / trace(rho);

    % ---------- diagnostics / storage --------------------------------
    if mod(n,saveEvery)==0 || n==Nsteps
        storeCount               = storeCount + 1;
        phi    = real(trace(  Xhat * rho));
        phiVar = real(trace(  Xhat^2 * rho)) - phi^2;
        plotSpan(storeCount)     = Nn1;
        RhoStore(:,:,storeCount) = rho;
        fprintf('NM-Master (low-rank): φ = %+6.3f   Var(φ) = %.3f   N = %.3f   snap %d\n', ...
            phi, phiVar, Nn, storeCount);
    end
end
end





%% --- panel maker (local function) ---
