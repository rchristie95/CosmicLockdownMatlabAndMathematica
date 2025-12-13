function [PsiSmallInst, RhoSmallInst] = AdiabaticGroundStates( ...
    bSize, hbar, mu, beta3, beta4, Hb, plotSpanInst)

    % ensure row vector
    plotSpanInst = plotSpanInst(:).';
    nPts = numel(plotSpanInst);

    % ----- big basis = 3*bSize ----------------------------------------
    BigbSize = 2*bSize;

    a = zeros(BigbSize);
    for n = 1:BigbSize-1
        a(n,n+1) = sqrt(n);
    end
    adag = a';

    Xhat = sqrt(hbar/2) * (adag + a);
    Phat = 1i * sqrt(hbar/2) * (adag - a);

    X = sparse(Xhat); X = 0.5*(X + X');  % Hermitian
    P = sparse(Phat); P = 0.5*(P + P');  % Hermitian

    X2    = X'*X;
    P2    = P'*P;

    Xpow2 = X*X;
    Xpow3 = Xpow2*X;
    Xpow4 = Xpow3*X;

    Vchi  = -(mu^2/2)*Xpow2 + (2*beta3*mu/3)*Xpow3 ...
            + ((beta4^2 - beta3^2)/4)*Xpow4;

    H1 = sparse((0.5/Hb)*P2);
    H2 = sparse((1.0/Hb)*Vchi);

    % small-basis X operator for observables
    Xsmall = X(1:bSize, 1:bSize);

    % ----- allocate outputs in small basis -----------------------------
    PsiSmallInst = zeros(bSize, nPts);
    RhoSmallInst = zeros(bSize, bSize, nPts);

    % eigs options (for numeric matrix)
    opts.tol   = 1e-8;
    opts.maxit = 1e4;

    % ----- loop over N -------------------------------------------------
    for k = 1:nPts
        Ne = plotSpanInst(k);

        a2    = exp(3*Ne);
        a1    = 1/a2;
        Hcurr = a1*H1 + a2*H2;      % sparse Hermitian

        usedDense = false;

        % try sparse ground state first
        try
            [psi_big, ~, flag] = eigs(Hcurr, 1, 'sa', opts);
            if flag ~= 0 || any(~isfinite(psi_big))
                error('eigs did not converge');
            end
        catch
            % fall back to dense eig if eigs fails
            usedDense = true;
            [Vfull, Dfull] = eig(full(Hcurr));
            [~, idx]       = min(real(diag(Dfull)));
            psi_big        = Vfull(:, idx);
        end

        psi_big = psi_big / norm(psi_big);

        psi_small = psi_big(1:bSize);
        psi_small = psi_small / norm(psi_small);

        PsiSmallInst(:,k)   = psi_small;
        RhoSmallInst(:,:,k) = psi_small * psi_small';

        % ----- terminal output like previous solvers -------------------
        phi    = real(psi_small' * Xsmall * psi_small);
        varPhi = real(psi_small' * (Xsmall^2) * psi_small) - phi^2;

        if usedDense
            solverTag = 'eig ';
        else
            solverTag = 'eigs';
        end

        fprintf('AdiabaticGS (%s): k=%4d/%4d  N=%+7.3f  φ=%+8.4f  Var(φ)=%8.4f\n', ...
                solverTag, k, nPts, Ne, phi, varPhi);
    end
end
