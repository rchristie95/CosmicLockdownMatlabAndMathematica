function [plotSpan, RhoStore] = NMQSD_MasterEquation_pagemat( ...
    rho,   Cmat, Xhat, Phat, TSpan, hbar, ...
    mu, beta3, beta4, lambda, Hb)

% ---------- time grid -------------------------------------------------
N0        = TSpan(1);
Nsteps    = numel(TSpan)-1;
assert(numel(TSpan)>=2 && all(diff(TSpan)>0),'Time grid must increase.');
saveEvery = max(1,floor(Nsteps/999));
plotSpan  = N0;

% ---------- system operators -----------------------------------------
Ndim  = size(rho,1);
Id    = eye(Ndim,'like',rho);
Vchi = -(mu^2/2)*(Xhat*Xhat')+(2*beta3*mu/3)*Xhat^3+((beta4^2-beta3^2)/4)*(Xhat*Xhat')^2 ;

RhoStore   = rho;                     % first snapshot
storeCount = 1;

% ---------- history buffers for commutators --------------------------
Khist      = zeros(Ndim,Ndim,Nsteps+1,'like',rho);  % K_m pages

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

    % --- commutator page for current step ----------------------------
    K_n                 = L_n*rho - rho*L_n;
    Khist(:,:,n)        = dN*K_n;               % store for future use

    % --- vectorised memory contraction -------------------------------
    % weights row   w(1,1,m)  for  m = 1..n
    w = reshape( Cmat(n+1,1:n) , 1,1,[] );   % implicit expansion
    S = sum( Khist(:,:,1:n) .* w , 3 );      %   Σ_m C(n,m) K_m

    M_n  = ( L_n*S - S*L_n );           % memory term at Nn

    % --- predictor Heun half-step ------------------------------------
    drho_n   = -1i/hbar*(H_n*rho - rho*H_n) - M_n;
    rho_half = rho + 0.5*dN*drho_n;

    % --- corrector: reuse the same Σ_m C(n,m)K_m but with L_mid ------
    M_mid = ( L_mid*S - S*L_mid );

    drho_mid = -1i/hbar*(H_mid*rho_half - rho_half*H_mid) - M_mid;
    rho      = rho + dN*drho_mid;

    % enforce Hermiticity and trace 1 (numerical hygiene)
    rho = 0.5*(rho+rho');
    rho = rho/trace(rho);

    % ---------- diagnostics / storage --------------------------------
    if mod(n,saveEvery)==0 || n==Nsteps
        storeCount               = storeCount + 1;
        phi    = real(trace(  Xhat * rho));
        phiVar = real(trace(  Xhat^2 * rho)) - phi^2;
        plotSpan(storeCount)     = Nn1;
        RhoStore(:,:,storeCount) = rho;
        fprintf('NM-Master: φ = %+6.3f   Var(φ) = %.3f   N = %.3f   snap %d\n', phi, phiVar, Nn, storeCount);

    end
end
end
%======================================================================
%  NMQSD_MasterEquation_pagemat_lowrank.m
%----------------------------------------------------------------------
%  Vectorised master-equation integrator using page-tensor algebra
%  and a low-rank factorisation of the bath correlator:
%       C ≈ Lfac * Lfac'
%
%  Requires MATLAB R2020b+ for implicit-expansion / pagemtimes.
%======================================================================

