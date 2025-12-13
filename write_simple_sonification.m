function [audioPath, total_duration] = write_simple_sonification(basename, RhoSeq, plotSpan, ...
        Xhat, Phat, Hb, mu, beta3, beta4, BSound, f0, fs)
    % Xhat = Xhat(1:BSound, 1:BSound, :);
    % Phat = Phat(1:BSound, 1:BSound, :);
    % RhoSeq=RhoSeq(1:BSound, 1:BSound, :);

    % 1) Instantaneous eigen-representation with TIME TRACKING + PHASE FIX
    [outRho, Ener] = HamiltonianEigenrep(RhoSeq, plotSpan, Xhat, Phat, Hb, mu, beta3, beta4);
    outRho=outRho(1:BSound, 1:BSound, :);
    Ener=Ener(1:BSound,:);
    % 2) Render 30 s stereo audio using FIXED f0
    total_duration = 30.0;  % seconds (fixed target)
    audioLR = render_simple_binaural_from_plotspan(outRho, Ener, plotSpan, f0, fs, total_duration, BSound);

    % 3) Save WAV
    audioPath = [basename '_EigenbasisSimple.wav'];
    audiowrite(audioPath, audioLR, fs);

    fprintf('Wrote %s (%.3f s)\n', audioPath, total_duration);
end


function [outRho, EnergiesOut] = HamiltonianEigenrep(RhoIn, plotSpan, Xhat, Phat, Hb,mu, beta3, beta4)
% TIME-TRACKED instantaneous eigen-representation.
% - Sorts the FIRST frame by ascending energy.
% - For subsequent frames: assigns modes by maximum overlap with previous eigenbasis.
% - Fixes eigenvector phases so overlaps with previous frame are real-positive.
%
% Inputs:
%   RhoIn:    (b x b x T) density matrices in Fock basis
%   plotSpan: (1 x T)
%   Xhat, Phat, Hb, Height, Separation: operators/params for H(N)
%
% Outputs (both are in the tracked instantaneous eigenbasis at each t):
%   outRho:       (b x b x T)
%   EnergiesOut:  (b x T), with the initial frame ascending & labels tracked

    bSize = size(RhoIn,1);
    T     = length(plotSpan);

    outRho       = zeros(bSize,bSize,T);
    EnergiesOut  = zeros(bSize,T);

    Id  = eye(bSize);
    H0     = @(Ne) (0.5/Hb) * (Phat^2 * exp(-3*Ne) ...
    + (-(mu^2/2)*Xhat^2 + (2*mu*beta3/3)*Xhat^3 + (beta4^2-beta3^2)*Xhat^4/4) * exp(3*Ne));
    % --- Frame 1: eigendecomposition, sort by ascending energy
    [V_prev, D_prev] = eig(full(H0(plotSpan(1))));
    E_prev = real(diag(D_prev));
    [E_prev_sorted, perm0] = sort(E_prev, 'ascend');
    V_prev = V_prev(:, perm0);

    % Phase gauge for the very first frame: make largest component real-positive
    for i = 1:bSize
        [~,ix] = max(abs(V_prev(:,i)));
        s = V_prev(ix,i) / max(abs(V_prev(ix,i)), eps);
        V_prev(:,i) = V_prev(:,i) * conj(s) / max(abs(s), eps);
    end

    % Project first-frame rho
    EnergiesOut(:,1) = E_prev_sorted - min(E_prev_sorted) + 1;
    R1 = V_prev' * RhoIn(:,:,1) * V_prev;
    R1 = 0.5*(R1+R1'); R1 = R1 / max(trace(R1), eps);
    outRho(:,:,1) = R1;

    % --- Subsequent frames: track by maximum overlap + phase fix
    for t = 2:T
        Ht = sparse(H0(plotSpan(t)));
        Ht=0.5*(Ht+Ht');
        [V_curr, D_curr] = eig(full(Ht));
        E_curr = real(diag(D_curr));
        % [E_curr, perm] = sort(E_curr, 'ascend');

        % Compute overlaps O_ij = <v_i(prev) | v_j(curr)>
        O = V_prev' * V_curr;

        % Assignment: maximize |O| (greedy if matchpairs unavailable)
        perm = assign_max_overlap(O);   % size bSize, maps i(prev) -> perm(i) in curr

        % Permute current eigenvectors and energies
        V_curr = V_curr(:, perm);
        E_curr = E_curr(perm);

        % Phase gauge: make diagonal overlaps real-positive
        diagOv = diag(V_prev' * V_curr);
        ph = exp(-1i * angle(diagOv + (diagOv==0)));  % avoid NaN if zero
        V_curr = V_curr * diag(ph);

        % Project Rho into the TRACKED curr basis
        Rt = V_curr' * RhoIn(:,:,t) * V_curr;
        Rt = 0.5*(Rt+Rt');
        % Rt=Rt/trace(Rt);

        EnergiesOut(:,t) = E_curr - min(E_curr) + 1;
        
        outRho(:,:,t)    = Rt;

        % Advance
        V_prev = V_curr;
    end
end

function perm = assign_max_overlap(O)
% Heuristic maximum-overlap assignment:
% - If Optimization Toolbox is available, use matchpairs on -abs(O)
% - Else, fall back to a greedy row-wise argmax with column exclusion
    try
        % Requires Optimization Toolbox (R2016b+)
        cost = -abs(O);
        pairs = matchpairs(cost, -Inf);  % returns [rowIdx colIdx]
        perm = zeros(1,size(O,1));
        perm(pairs(:,1)) = pairs(:,2);
    catch
        % Greedy fallback (works well when tracking is near-diagonal)
        n = size(O,1);
        perm = zeros(1,n);
        used = false(1,n);
        for i = 1:n
            [~, jlist] = sort(abs(O(i,:)),'descend');
            j = jlist(find(~used(jlist),1,'first'));
            if isempty(j), j = find(~used,1,'first'); end
            perm(i) = j;
            used(j) = true;
        end
    end
end

function audio_stereo = render_simple_binaural_from_plotspan(RhoSeq, ESeq, plotSpan, f0, fs, total_dur, BSound)
    % RhoSeq: (b x b x T), in a CONSISTENT tracked eigenbasis, Hermitian, trace~1
    % ESeq  : (b x T), tracked energies (E0=1 so f_n = E_n * f0)
    % plotSpan: (1 x T) physical timestamps (nonuniform)
    % total_dur: target audio length in seconds (e.g. 30)

    [bFull, T] = size(ESeq);
    if T < 2
        audio_stereo = zeros(0,2);
        return;
    end

    % Hard cap on how many bands we will ever consider
    bBase = min(BSound, bFull);

    % Frequencies for the first bBase tracked bands
    Ffull = ESeq(1:bBase,:) * f0;      % (bBase x T)

    % Time allocation from plotSpan
    dN   = diff(plotSpan(:));
    span = plotSpan(end) - plotSpan(1);
    frac = dN / max(span, eps);
    L    = round(frac * total_dur * fs);
    need = total_dur*fs - sum(L);
    if need ~= 0
        sgn = sign(need);
        idxs = 1:(T-1);
        k = 1;
        while need ~= 0
            L(idxs(k)) = L(idxs(k)) + sgn;
            need = need - sgn;
            k = k + 1;
            if k > numel(idxs), k = 1; end
        end
    end
    L = max(L,1);

    starts = [1; 1 + cumsum(L(1:end-1))];
    Ntot   = sum(L);
    audio_stereo = zeros(Ntot,2);

    % Precompute magnitudes and phases for the first bBase x bBase block
    Abs = abs(RhoSeq(1:bBase,1:bBase,:));
    Arg = angle(RhoSeq(1:bBase,1:bBase,:));

    % Unwrap phases over time and reference to first frame
    for k = 1:bBase
        for l = 1:bBase
            tmp = squeeze(Arg(k,l,:));
            Arg(k,l,:) = unwrap(tmp);
        end
    end
    Arg = Arg - Arg(:,:,1);

    t_offset = 0;  % seconds
    for t = 1:T-1
        Lt = L(t);
        alpha = linspace(0,1,Lt);       % 1 x Lt
        alphaRow = alpha(:).';          % row for broadcasting

        % Use all bands up to bBase
        idxBands = 1:bBase;

        % Lower-triangle voices over the active bands
        [Kgrid,Lgrid] = ndgrid(idxBands, idxBands);
        mask = (Kgrid >= Lgrid);
        kIdx = Kgrid(mask);
        lIdx = Lgrid(mask);
        M = numel(kIdx);

        % Magnitudes & phases at t and t+1
        m0_full = Abs(:,:,t);
        m1_full = Abs(:,:,t+1);
        p0_full = Arg(:,:,t);
        p1_full = Arg(:,:,t+1);

        linIdx = sub2ind([bBase, bBase], kIdx, lIdx);
        m0_pairs = m0_full(linIdx);      % (M x 1)
        m1_pairs = m1_full(linIdx);
        p0_pairs = p0_full(linIdx);
        p1_pairs = p1_full(linIdx);

        % Interpolate magnitudes and phases
        mags = m0_pairs.*(1 - alphaRow) + m1_pairs.*alphaRow;   % (M x Lt)
        phis = p0_pairs.*(1 - alphaRow) + p1_pairs.*alphaRow;   % (M x Lt)

        % Frequencies for left/right indices
        fL0 = Ffull(kIdx,t);   fL1 = Ffull(kIdx,t+1);
        fR0 = Ffull(lIdx,t);   fR1 = Ffull(lIdx,t+1);
        fLs = fL0.*(1 - alphaRow) + fL1.*alphaRow;             % (M x Lt)
        fRs = fR0.*(1 - alphaRow) + fR1.*alphaRow;             % (M x Lt)

        % Absolute time vector for this interval
        t_frame = (0:Lt-1)/fs + t_offset;   % 1 x Lt
        tM = repmat(t_frame, M, 1);         % (M x Lt)

        % Binaural additive synthesis
        frameL = mags .* sin(2*pi.*fLs.*tM + phis);
        frameR = mags .* sin(2*pi.*fRs.*tM - phis);

        % Write into output
        i0 = starts(t); i1 = i0 + Lt - 1;
        audio_stereo(i0:i1,1) = sum(frameL,1).';
        audio_stereo(i0:i1,2) = sum(frameR,1).';

        t_offset = t_offset + Lt/fs;
    end

    % Normalize
    mx = max(abs(audio_stereo(:)));
    if mx > 0
        audio_stereo = 0.9 * audio_stereo / mx;
    end
end
