%% Quantum Inflation Simulation via NMQSD
% This script computes a single stochastic trajectory of the inflaton
% field using the Non-Markovian Quantum State Diffusion (NMQSD) approach,
% and generates a Wigner‐function video of the evolution.

clc;
clearvars;
close all;
rng('shuffle');   % seed RNG based on current time

%% Physical & Numerical Parameters
hbar      = 1;        % Reduced Planck constant
mu    = 0.5;      % Potential parameter: height scale
beta3=0.025;        % Potential parameter: separation between minima
beta4=0.13;
Hb        =5;     % Hubble scale parameter
mChi=50;
omegaChi  = mChi/Hb;        % Environmental frequency
lambda    =100;      % Coupling strength

% Time‐integration grid for SSE
N0        =-2;                       % Initial e-fold
Nsteps    =500000;                    % Number of N grid points
% Nsteps    =2000;                    % Number of N grid points

Nend  = 1;

% TSpan = linspace(N0, Nend, Nsteps+1);

% reference uniform grid just to define weights
t_ref = linspace(N0, Nend, Nsteps+1);

% weights for each interval (length Nsteps)
% w = 1./exp(abs(3*(t_ref(1:end-1) + 0.5)));
w = 1./exp(abs(3*(0.5+t_ref(1:end-1) )));

% normalise so the steps sum to Nend-N0
w = w / sum(w);                     % sum(w) = 1
dN_vec = (Nend - N0) * w;           % length Nsteps, sum(dN_vec)=Nend-N0

% build non-uniform TSpan
plotSpan =linspace(N0, Nend, 1001);
TSpan = [N0, N0 + cumsum(dN_vec)];  % length Nsteps+1


% Basis sizes
bSize     =400;       % Fock basis dimension
Nx        = 200;      % Grid dimension for x/p representation

% Wigner video parameters
Xview     =15;        % Plot limits in phase space


%% Precompute Fock Operators
% Creation/annihilation operators
a     = zeros(bSize);
for n = 1 : bSize-1
    a(n,n+1) = sqrt(n);
end
adag  = a';
Xhat  = sqrt(hbar/2) * (adag + a);
Phat  = 1i * sqrt(hbar/2) * (adag - a);
Id    = eye(bSize);

[V,D] = eig(Xhat);
xvals  = diag(D);

%--- Numerical tolerance for "zero" ---
tol = 10*eps(max(abs(xvals)));   % loose-ish tolerance

%--- Indices of eigenvalues strictly > 0 (exclude ~0) ---
idx_pos = xvals > tol;

%--- Projector onto x>0 ---
ProjR = V(:,idx_pos) * V(:,idx_pos)';   % (bSize x bSize) Hermitian idempotent

%--- (Optional) enforce Hermiticity/idempotency tiny cleanup ---
ProjR = (ProjR + ProjR')/2;


%% Initial Instantaneous Hamiltonian & State
H0     = @(Ne) (0.5/Hb) * (Phat^2 * exp(-3*Ne) ...
    + (-(mu^2/2)*Xhat^2 + (2*mu*beta3/3)*Xhat^3 + (beta4^2-beta3^2)*Xhat^4/4) * exp(3*Ne));

% Build the Hamiltonian for the chosen e-fold N0
Hinit = H0(N0);

% --- small/medium matrix: use eig and sort -------------------------------
[Vec, E] = eig(full(Hinit));                 % E is diagonal
[~, idx] = min(diag(E));                 % position of lowest eigen-value
PsiIn = Vec(:, idx);
PsiIn = PsiIn / norm(PsiIn);             % normalise





%% Construct Hermite‐function Basis for x‐space

dX=sqrt(2*pi/(Nx*hbar));
% dX=3/100;

dP=2*pi/(dX*Nx/hbar);
for n=1:Nx
    x(n)=(-Nx/2+(n-1/2))*dX;
    p(n)=(-Nx/2+(n-1/2))*dP; %shift
end

HVector=zeros(Nx,Nx);
for n = 0:Nx-1
    HVector(:, n+1) = sqrt(dX / (2^n * factorial(n))) * (pi * hbar)^(-0.25).* exp(-x.^2 / (2 * hbar)) .* hermite_poly(n, x / sqrt(hbar));
end


%
trace(ProjR*(PsiIn*PsiIn'))





%% SixPoint Renormalized

% coefficients matching
% C(n_i - n_j) = (3*(9 cos(ωΔ) + cos(3ωΔ))) / (128 Hb^3 ω^3)
a1 = 27/(128*Hb^3*omegaChi^3);
a3 =  3/(128*Hb^3*omegaChi^3);

% grid NList = TSpan (1×(Nsteps+1))
LRank = 4;   % no constant piece any more

% allocate Lfac: (Nsteps+1) × 4
Lfac = zeros(numel(TSpan), LRank);

% fill columns (basis functions of N)
Lfac(:,1) = sqrt(a1) * cos(omegaChi * TSpan);    % cos(ωN)
Lfac(:,2) = sqrt(a1) * sin(omegaChi * TSpan);    % sin(ωN)
Lfac(:,3) = sqrt(a3) * cos(3*omegaChi * TSpan);  % cos(3ωN)
Lfac(:,4) = sqrt(a3) * sin(3*omegaChi * TSpan);  % sin(3ωN)

% check: Cmat ≈ Lfac2 * Lfac2.'
% [Ni,Nj] = ndgrid(TSpan, TSpan);
% Cmat = (3*(9*cos(omegaChi*(Ni - Nj)) + ...
%            cos(3*omegaChi*(Ni - Nj)))) ./ (128*Hb^3*omegaChi^3);
% 
% norm(Cmat - (Lfac2*Lfac2'),'fro')


%% Single NMQSD
% standard complex Gaussian in R^LRank, CN(0, I_LRank)
zeta = (randn(LRank,1) + 1i*randn(LRank,1))/sqrt(2);

% correlated noise with covariance C = Lfac2*Lfac2'
etaC = Lfac * zeta;     % size (Nsteps+1) x 1
% zeta = (randn(Lrank,Nsteps+1) + 1i*randn(Lrank,Nsteps+1))/sqrt(2);     % √2 variance → unit variance
% etaC = Lfac * zeta;                       % coloured noise column vector



[plotSpan,PsiSSE, RhoSSE] = NMQSD_SingleTrajectory_Hybrid(  bSize,etaC, Lfac, TSpan,plotSpan, hbar, mu, beta3, beta4, lambda, Hb);
nFramesSSE=length(plotSpan);
filenameWorkspace=sprintf('WorkspaceNMSSE_lambda_%.2g_WChi_%.2g.mat',lambda,omegaChi);



%% SSE Observables
nFrames=length(plotSpan);


expectHSSE = zeros(1,nFrames);
PuritySSE  = zeros(1,nFrames);
NormSSE = zeros(1,nFrames);
CtSSE      = zeros(1,nFrames);
expXSSE    = zeros(1,nFrames);
expPSSE    = zeros(1,nFrames);

PsiSSEPos  = zeros(Nx,nFrames);
RhoSSEPos  = zeros(Nx,Nx,nFrames);

for k = 1 : nFrames
    NeCurr   = plotSpan(k);
    Hcurr    = H0(NeCurr);
    Rho        = RhoSSE(:,:,k);

    expectHSSE(k)  = real(trace(Hcurr * Rho));
    % Rho=Rho/trace(Rho);
    NormSSE(k)  = real(trace(Rho));
    PuritySSE(k)   = trace(Rho^2)/trace(Rho)^2;
    CtSSE(k)       = real(trace(ProjR * Rho)) / NormSSE(k);
    expXSSE(k)     = real(trace(Xhat * Rho))/ NormSSE(k);
    expPSSE(k)     = real(trace(Phat * Rho))/ NormSSE(k);
    PsiSSEPos(:,k) = HVector * PsiSSE(1:Nx,k)/ norm(PsiSSE(1:Nx,k));

end
dotCtSSE = gradient(CtSSE, plotSpan);
close all
save(filenameWorkspace);



%% NMMaster Equation Trajectory
%
% [plotSpan, RhoMaster] = NMQSD_MasterEquation_lowrank(  PsiIn*PsiIn',   Lfac, Xhat, Phat, TSpan, hbar,mu, beta3, beta4, lambda, Hb);
% close all
% clear flipbook contPlot expline f figureHandle timeText writerObj expLine CdatCont WigSSE eta etaC Cmat

%% Master Observables

% expectHMaster = zeros(1,nFrames);
% PurityMaster  = zeros(1,nFrames);
% NormMaster = zeros(1,nFrames);
% CtMaster      = zeros(1,nFrames);
% expXMaster    = zeros(1,nFrames);
% expPMaster    = zeros(1,nFrames);
%
% PsiMasterPos  = zeros(Nx,nFrames);
% RhoMasterPos  = zeros(Nx,Nx,nFrames);
%
% parfor k = 1 : nFrames
%     NeCurr   = plotSpan(k);
%     Hcurr    = H0(NeCurr);
%     Rho        = RhoMaster(:,:,k);
%
%     expectHMaster(k)  = real(trace(Hcurr * Rho));
%     NormMaster(k)  = real(trace(Rho));
%     PurityMaster(k)   = trace(Rho^2)/trace(Rho)^2;
%     CtMaster(k)       = real(trace(ProjR * Rho)) / NormMaster(k);
%     expXMaster(k)     = real(trace(Xhat * Rho));
%     expPMaster(k)     = real(trace(Phat * Rho));
%     RhoMasterPos(:,:,k) = HVector * RhoMaster(:,:,k) * HVector';
%
% end
% dotCtMaster = gradient(CtMaster, plotSpan);
%


%% Create Wigner‐Function Video
WigSSE = PsiWigner(PsiSSEPos, x, p, hbar);
% WigSSE = RhoWigner(RhoSSEPos, x, p, hbar);


Xc=linspace(-Xview, Xview, Nx);
Pc=linspace(-Xview, Xview, Nx);
Hinit = @(Z,n) (exp(3*plotSpan(n))*(-(mu^2/2)*Z(1)^2+(2*mu*beta3/3)*Z(1)^3+((beta4^2-beta3^2)/4)*Z(1)^4)+0.5*Z(2)^2*exp(-3*plotSpan(n)))/Hb;

% Define your meshgrid
[xMesh, pMesh] = meshgrid(x, p); % Adjust the limits and resolution as needed
[xMesh1, pMesh1] = meshgrid(Xc, Pc); % Adjust the limits and resolution as needed

% Compute initial contour data
Z_initial = arrayfun(@(x, y) Hinit([x; y], 1), xMesh, pMesh);
Z_final = arrayfun(@(x, y) Hinit([x; y], 1001), xMesh, pMesh);
Ztotal=[Z_initial;Z_final];
cmin0 = min(Ztotal(:), [], 'omitnan');
cmax0 = max(Ztotal(:), [], 'omitnan');


u = linspace(0, 1, 50);   % parameter
u = u.^5;                 % cluster toward cmin0

levels = cmin0 + (cmax0 - cmin0) * u;



% Compute initial contour data
Z_initial = arrayfun(@(x, y) Hinit([x; y], 1), xMesh, pMesh);


% Define aspect ratio for the figure
aspectRatio = [6, 6];


%%
targets=[N0,0,Nend];
[~, idxOrig] = min(abs(plotSpan- targets(1)), [], 2);   % idx(i) indexes p nearest to t(i)
[~, idxZero] = min(abs(plotSpan- targets(2)), [], 2);   % idx(i) indexes p nearest to t(i)
WigImagesSSE(:,:,1)=WigSSE(:,:, idxOrig);
WigImagesSSE(:,:,2)=WigSSE(:,:, idxZero);
WigImagesSSE(:,:,3)=WigSSE(:,:, end);

% Helper for Hamiltonian contour at a given frame index
Hgrid_at = @(k) ( exp(3*k) .* (-(mu^2/2).*xMesh1.^2 +(2*mu*beta3/3).*xMesh1.^3 +((beta4^2-beta3^2)/4).*xMesh1.^4) ...
                + 0.5 .* pMesh1.^2 .* exp(-3*k) ) / Hb;

% 
% % Common plotting settings
cax = [-0.05 0.15];
fmtText = @(Nval,Cval) sprintf('$N=%.2f,\\;\\langle\\hat\\theta_{\\phi^{+}}\\rangle=%.2f$', Nval, Cval);
% 
% ---------- (a) N_e = -0.75 ----------
figA = figure('Position', [100 100 800 600]);  % wider figure for visibility
axA = axes(figA); hold(axA,'on');
hA = pcolor(axA, x, p, WigImagesSSE(:,:,1));
shading('interp')
colormap(axA,'parula');
contour(axA, xMesh1, pMesh1, Hgrid_at(targets(1)), levels, 'LineColor', 'k');
xlim(axA,[-Xview Xview]); ylim(axA,[-Xview Xview]); axis(axA,'square'); box(axA,'on'); caxis(axA,cax);
xticks(linspace(-Xview,Xview,7))
colorbar(axA,'FontSize',18);
expLine = plot(expXSSE(1:idxOrig), expPSSE(1:idxOrig), 'w-', 'LineWidth', 1); % White line with specified width
text(axA, 0.98, 0.98, fmtText(targets(1), CtSSE(idxOrig)), ...
    'Units','normalized','HorizontalAlignment','right','VerticalAlignment','top', ...
    'Interpreter','latex','FontSize',24,'FontWeight','bold','Color','w');
set(axA, 'FontSize', 20, 'LineWidth', 1.5, 'Box', 'on', 'TickLabelInterpreter', 'latex');
xlabel(axA,'$\phi$','Interpreter','latex','FontSize',25);
ylabel(axA,'$\pi_\phi$','Interpreter','latex','FontSize',25);
title(axA, '(a)','Interpreter','latex','FontSize',30);
box on
exportgraphics(figA,'NMSSEwigner_panel_a_Ne_-0p75.pdf','ContentType', 'image', 'Resolution', 600);

% ---------- (b) N_e = 0 ----------
figB = figure('Position', [100 100 800 600]);  % wider figure for visibility
axB = axes(figB); hold(axB,'on');
hB = pcolor(axB, x, p, WigImagesSSE(:,:,2));
shading('interp')
colormap(axB,'parula');
contour(axB, xMesh1, pMesh1, Hgrid_at(targets(2)), levels, 'LineColor', 'k');
xlim(axB,[-Xview Xview]); ylim(axB,[-Xview Xview]); axis(axB,'square'); box(axB,'on'); caxis(axB,cax);
xticks(linspace(-Xview,Xview,7))
colorbar(axB,'FontSize',18);
expLine = plot(expXSSE(1:idxZero), expPSSE(1:idxZero), 'w-', 'LineWidth', 1); % White line with specified width
text(axB, 0.98, 0.98, fmtText(targets(2), CtSSE(idxZero)), ...
    'Units','normalized','HorizontalAlignment','right','VerticalAlignment','top', ...
    'Interpreter','latex','FontSize',24,'FontWeight','bold','Color','w');
set(axB, 'FontSize', 20, 'LineWidth', 1.5, 'Box', 'on', 'TickLabelInterpreter', 'latex');
xlabel(axB,'$\phi$','Interpreter','latex','FontSize',25);
ylabel(axB,'$\pi_\phi$','Interpreter','latex','FontSize',25);
title(axB, '(b)','Interpreter','latex','FontSize',30);
box on
exportgraphics(figB,'NMSSEwigner_panel_b_Ne_0.pdf','ContentType', 'image', 'Resolution', 600);

% ---------- (c) N_e = +1.5 ----------
figA = figure('Position', [100 100 800 600]);  % wider figure for visibility
axA = axes(figA); hold(axA,'on');
hC = pcolor(axA, x, p, WigImagesSSE(:,:,3));
shading('interp')
colormap(axA,'parula');
contour(axA, xMesh1, pMesh1, Hgrid_at(targets(3)), levels, 'LineColor', 'k');
xlim(axA,[-Xview Xview]); ylim(axA,[-Xview Xview]); axis(axA,'square'); box(axA,'on'); caxis(axA,cax);
xticks(linspace(-Xview,Xview,7))
colorbar(axA,'FontSize',18);
expLine = plot(expXSSE(1:end), expPSSE(1:end), 'w-', 'LineWidth', 1); % White line with specified width
text(axA, 0.98, 0.98, fmtText(targets(3),CtSSE(end)), ...
    'Units','normalized','HorizontalAlignment','right','VerticalAlignment','top', ...
    'Interpreter','latex','FontSize',24,'FontWeight','bold','Color','w');
set(axA, 'FontSize', 20, 'LineWidth', 1.5, 'Box', 'on', 'TickLabelInterpreter', 'latex');
xlabel(axA,'$\phi$','Interpreter','latex','FontSize',25);
ylabel(axA,'$\pi_\phi$','Interpreter','latex','FontSize',25);
title(axA, '(c)','Interpreter','latex','FontSize',30);
box on
exportgraphics(figA,'NMSSEwigner_panel_c_Ne_+0p75.pdf','ContentType', 'image', 'Resolution', 600);
%% Video
clear flipbook

WigSSE = PsiWigner(PsiSSEPos, x, p, hbar);
% WigSSE = RhoWigner(RhoSSEPos, x, p, hbar);

Xc = linspace(-Xview, Xview, Nx);
Pc = linspace(-Xview, Xview, Nx);

Hinit = @(Z,n) ( ...
    exp(3*plotSpan(n)) * ( ...
        -(mu^2/2)*Z(1)^2 + (2*mu*beta3/3)*Z(1)^3 + ((beta4^2-beta3^2)/4)*Z(1)^4 ) ...
    + 0.5*Z(2)^2 * exp(-3*plotSpan(n)) ) / Hb;

% Meshgrids
[xMesh,  pMesh]  = meshgrid(x,  p);      % for Wigner
[xMesh1, pMesh1] = meshgrid(Xc, Pc);     % for contours

% Compute initial contour data on contour grid
Z_initial = arrayfun(@(xx, yy) Hinit([xx; yy], 1), xMesh1, pMesh1);

% Define aspect ratio for the figure
aspectRatio = [6, 6];

% Create a figure with specified position and aspect ratio
figureHandle = figure('Position', [100, 100, 800, 800 * aspectRatio(2) / aspectRatio(1)]);

% Initialize pcolor plot with the first set of data
f = pcolor(xMesh, pMesh, WigSSE(:,:,1));
xlim([-Xview Xview]);
ylim([-Xview Xview]);
xticks(linspace(-Xview,Xview,7))
shading interp
colormap("parula")

% Set color axis limits
clim([-0.05 0.15])

% Add a colorbar with increased font size
colorbar('FontSize', 14);

title( sprintf(...
    'Non Markov SSE, $(\\hat L_{\\phi})$, $\\omega_{\\chi}=%.2f$, $\\lambda = %.2f$, $\\tilde \\mu = %.2f$', ...
    omegaChi,lambda, mu/Hb), ...
    'FontSize', 20, 'Interpreter', 'latex' );

xlabel('$\phi$',      'FontSize', 25, 'Interpreter', 'latex');
ylabel('$\pi_{\phi}$','FontSize', 25, 'Interpreter', 'latex');

set(gca, 'FontSize', 16, 'LineWidth', 1.5, 'Box', 'on', ...
    'TickLabelInterpreter', 'latex');
axis square

hold on;

% Initialize an empty line plot for experimental data
expLine = plot(NaN, NaN, 'w-', 'LineWidth', 1);


% Initialize contour plot with initial Z data and nonlinear levels
[~, contPlot] = contour(xMesh1, pMesh1, Z_initial, levels, 'LineColor', "k");

% Initialize the time display
timeText = text(0.95, 0.95, '', ...
    'Units', 'normalized', ...
    'HorizontalAlignment', 'right', ...
    'VerticalAlignment', 'top', ...
    'FontSize', 25, ...
    'FontWeight', 'bold', ...
    'Color', 'w');
uistack(timeText, 'top');

% Number of frames
nFramesSSE = size(WigSSE, 3);

% Main simulation loop
for n = 1:nFramesSSE
    % Current time
    t = plotSpan(n);

    % Update the pcolor plot with new data for the current time step
    set(f, 'CData', WigSSE(:,:,n));

    % Compute contour data based on Hinit
    CdatCont = arrayfun(@(xx, yy) Hinit([xx; yy], n), xMesh1, pMesh1);

    % Delete existing contour plot to prevent overlap
    if ~isempty(contPlot) && isgraphics(contPlot)
        delete(contPlot);
    end

    % Recompute and plot new contours with fixed nonlinear levels
    [~, contPlot] = contour(xMesh1, pMesh1, CdatCont, levels, 'LineColor', "k");

    % Update the experimental line plot with new data point
    xData = get(expLine, 'XData');
    yData = get(expLine, 'YData');
    set(expLine, 'XData', [xData, expXSSE(n)], ...
                 'YData', [yData, expPSSE(n)]);

    % Update the text object with the current time / observable
    set(timeText, 'Interpreter', 'latex', ...
        'String', sprintf('$N = $ %.2f, $\\langle \\hat\\theta_{\\phi^{+}}\\rangle = $ %.2f', ...
                          t, CtSSE(n)));
    uistack(timeText, 'top');

    % Refresh the plot
    drawnow;

    % Capture the frame
    flipbook(n) = getframe(gcf);
end

% Save video
SSEVideoPath = sprintf('Wigner_NonMarkov_SSEVideo_w_chi=%.2g_lambda_%.2g_Hubble_%.2g.avi', omegaChi,lambda, Hb);
writerObj    = VideoWriter(SSEVideoPath);

writerObj.FrameRate = nFramesSSE/30;
open(writerObj);
writeVideo(writerObj, flipbook);
close(writerObj);

close all;


%%


%% Sonification
% ===================== Tunables =====================
f0     = 60;       % base note (Hz), C
fs     = 44100;        % audio sample rate


HamStart=H0(N0);
HamMid=H0(0);
HamFinish=H0(Nend);

evalsStart  = sort(real(eig(HamStart)),  'ascend');
fsStart=(evalsStart-min(evalsStart)+1)*f0;
evalsMid  = sort(real(eig(HamMid)),  'ascend');
fsMid=(evalsStart-min(evalsMid)+1)*f0;
evalsFinish = sort(real(eig(HamFinish)), 'ascend');
fsFinish=(evalsFinish-min(evalsFinish)+1)*f0;

BSound=min([sum(fsStart  < fs/2), sum(fsMid  < fs/2), sum(fsFinish < fs/2) ])
nFrames=length(plotSpan);
for n=1:nFrames
    RhoSSE(:,:,n)=PsiSSE(:,n)*PsiSSE(:,n)';
end

% ===================== FFmpeg check =====================
[ff_ok, ~] = system('ffmpeg -version');
if ff_ok ~= 0
    error('FFmpeg is not installed or not on PATH.');
end

% ===================== Render 30s audio for each trajectory =====================
[SSEAudioPath,  SSEAudioDur]  = write_simple_sonification('SSE',  RhoSSE,         plotSpan,  Xhat, Phat, Hb, mu, beta3,  beta4,BSound, f0, fs);
% (Each *_AudioDur will be 30.000 s)

SSEMP4Out  =sprintf( 'Non_Markov_SSE_EigenbasisSimple_lambda_%.2g_Hubble_%.2g.mp4', lambda, Hb);

% Mux without forcing -t; videos are 30 s and WAVs are 30 s → perfect sync

merge_audio_video_ffmpeg(SSEVideoPath,  SSEAudioPath,  SSEMP4Out,  SSEAudioDur);

disp('All sonifications rendered and merged with videos (30 s each).')

%% Spectrograms
for n=1:nFrames
    RhoSSE(:,:,n)=PsiSSE(:,n)*PsiSSE(:,n)';
end
% --- compute energies & occupations ---
[~, EnergiesSSE, SSEOcc]  = HamiltonianEigenrep2(RhoSSE,         plotSpan,  Xhat, Phat, Hb, mu, beta3,beta4);

%% plotting

% --- make the three figures in order: Ham, SSE, Lind ---
makePanel('EnergyLines_panelNon_Markov_SSE.pdf',  plotSpan,  EnergiesSSE,  SSEOcc,  '(a)');
%% Exit
% run('MarkovCoarseGrainedLindbladScript.m')

exit


%% Local Functions

function Hn = hermite_poly(n, x)
% Recursively compute physicists' Hermite polynomials H_n(x)
if n == 0
    Hn = ones(size(x));
elseif n == 1
    Hn = 2*x;
else
    H0 = ones(size(x));
    H1 = 2*x;
    for k = 2 : n
        Hn = 2*x .* H1 - 2*(k-1)*H0;
        H0 = H1;
        H1 = Hn;
    end
end
end

function [plotSpan, PsiStore, RhoStore] = SchrodingerSingleTrajectory( ...
    PsiIn, Xhat, Phat, TSpan, plotSpan, hbar, ...
    mu, beta3, beta4, Hb)
% NMQSD_SingleTrajectory  -- single normalised NMQSD path
%
%   * 2-stage RK2 (Heun / explicit trapezoidal) in "time" N
%   * explicit √〈ψ|ψ〉 renormalisation after predictor and corrector
%   * snapshots taken at TSpan indices closest to requested plotSpan
% ----------------------------------------------------------------------

% ---------- time grid -------------------------------------------------
Nsteps = numel(TSpan) - 1;
N0     = TSpan(1); %#ok<NASGU> % keep if you need N0 elsewhere

% ---------- map plotSpan -> nearest indices in TSpan ------------------
Nplot    = numel(plotSpan);
tIdxReal = interp1(TSpan, 1:numel(TSpan), plotSpan, 'nearest', 'extrap');
plotIdx  = round(tIdxReal);                 % integer indices into TSpan
plotIdx  = max(1, min(numel(TSpan), plotIdx));  % clamp just in case

% snap the output times to the actual grid we will use
plotSpan = TSpan(plotIdx);

% ---------- storage ---------------------------------------------------
Ndim     = size(PsiIn,1);
PsiStore = zeros(Ndim, Nplot);
RhoStore = zeros(Ndim, Ndim, Nplot);

% initial state (assume first plot time corresponds to start)
psi = PsiIn / sqrt(PsiIn' * PsiIn);   % guarantee unit norm
PsiStore(:,1)   = psi;
RhoStore(:,:,1) = psi*psi';

nextPlot = 2;  % next plot index to fill (1 already done)

Xhat = sparse(Xhat);
Phat = sparse(Phat);
P2   = (Phat * Phat');

% ---------- static potential operator ---------------------------------
Vchi = -(mu^2/2)     * (Xhat*Xhat') ...
    + (2*beta3*mu/3) * Xhat^3   ...
    + ((beta4^2 - beta3^2)/4) * (Xhat*Xhat')^2;

% =====================================================================
%                             MAIN LOOP (Heun)
% =====================================================================
for n = 1:Nsteps

    % ---- grid points and local step ---------------------------------
    Nn   = TSpan(n);
    Nn1  = TSpan(n+1);
    dN   = Nn1 - Nn;

    % ---- H(Nn) -------------------------------------------------------
    H_n  = (0.5*P2*exp(-3*Nn ) + Vchi*exp(3*Nn )) / Hb;

    % ---- predictor: forward Euler from Nn to Nn1 --------------------
    D_n      = -1i/hbar * H_n * psi;
    psi_pred = psi + dN * D_n;
    psi_pred = psi_pred /norm(psi_pred);  % renormalise

    % ---- H(Nn1) and corrector (Heun) --------------------------------
    H_n1  = (0.5*P2*exp(-3*Nn1) + Vchi*exp(3*Nn1)) / Hb;
    D_n1  = -1i/hbar * H_n1 * psi_pred;

    psi = psi + 0.5 * dN * (D_n + D_n1);
    psi = psi / norm(psi);                           % renormalise

    % ---- snapshots: store at requested indices ----------------------
    % we just advanced to time index (n+1)
    while nextPlot <= Nplot && plotIdx(nextPlot) == (n+1)
        PsiStore(:, nextPlot)   = psi;        % already normalised
        RhoStore(:,:, nextPlot) = psi*psi';

        % optional diagnostics every 10th snapshot
        phi    = real(psi' * Xhat * psi);
        phiVar = real(psi' * (Xhat^2) * psi) - phi^2;
        fprintf('Schrodinger: φ = %+6.3f   Var(φ) = %.3f   N = %.3f   snap %d\n', ...
            phi, phiVar, Nn1, nextPlot);

        nextPlot = nextPlot + 1;
    end
end

end




% function [plotSpan, PsiStore, RhoStore] = NMQSD_SingleTrajectory( ...
%     PsiIn, etaC, Lfac, Xhat, Phat, TSpan, plotSpan, hbar, ...
%     mu, beta3, beta4, lambda, Hb)
% % NMQSD_SingleTrajectory  -- low-rank version (needs Lfac)
% %
% %   * 2-stage RK2 (Heun / explicit trapezoidal) in "time" N
% %   * explicit renormalisation of ψ after predictor and corrector
% %   * snapshots at TSpan indices closest to requested plotSpan
% % ----------------------------------------------------------------------
% 
% % ---------- time grid -------------------------------------------------
% Nsteps = numel(TSpan) - 1;
% 
% % ---------- map plotSpan -> nearest indices in TSpan ------------------
% Nplot    = numel(plotSpan);
% tIdxReal = interp1(TSpan, 1:numel(TSpan), plotSpan, 'nearest', 'extrap');
% plotIdx  = round(tIdxReal);                         % integer indices
% plotIdx  = max(1, min(numel(TSpan), plotIdx));      % clamp to [1, end]
% 
% % snap the output times to the actual grid we will use
% plotSpan = TSpan(plotIdx);
% 
% % ---------- storage ---------------------------------------------------
% Ndim     = size(PsiIn,1);
% Id       = eye(Ndim);
% 
% PsiStore = zeros(Ndim, Nplot);
% RhoStore = zeros(Ndim, Ndim, Nplot);
% 
% psi = PsiIn / norm(PsiIn);          % unit-norm initial state
% 
% % assume first plot time corresponds to initial time
% PsiStore(:,1)   = psi;
% RhoStore(:,:,1) = psi*psi';
% nextPlot        = 2;
% 
% Xhat = sparse(Xhat);
% Phat = sparse(Phat);
% P2   = (Phat * Phat');
% 
% % ---------- static potential -----------------------------------------
% Vchi = -(mu^2/2)       * (Xhat*Xhat') ...
%     + (2*beta3*mu/3)  * Xhat^3       ...
%     + ((beta4^2-beta3^2)/4) * (Xhat*Xhat')^2;
% 
% % ---------- auxiliary G-matrix (Ndim × r) ----------------------------
% r = size(Lfac,2);
% G = zeros(Ndim, r);                 % columns G_k
% 
% % initial functional derivative F₁ at TSpan(1) with first step size
% if Nsteps >= 1
%     dN0   = TSpan(2) - TSpan(1);
% else
%     dN0   = 0;
% end
% L0_op  = (lambda/Hb)*exp(3*TSpan(1))*Xhat;
% phiExp = real(psi' * L0_op * psi);
% Fnew   = (L0_op - phiExp*Id) * psi * dN0;
% 
% % add L₁,k F₁ to every G_k  (row 1 of Lfac)
% G = Fnew * Lfac(1,:);               % Ndim×1 times 1×r -> Ndim×r
% 
% % =====================================================================
% %                           MAIN LOOP (Heun)
% % =====================================================================
% for n = 1:Nsteps
% 
%     % ---- grid points and local step ---------------------------------
%     Nn   = TSpan(n);
%     Nn1  = TSpan(n+1);
%     dN   = Nn1 - Nn;
% 
%     % ---- operators at Nn --------------------------------------------
%     H_n = (0.5*P2*exp(-3*Nn ) + Vchi*exp(3*Nn ))  / Hb;
%     L_n = (lambda/Hb)*exp(3*Nn) * Xhat;
% 
%     % ---- coloured noise at N_{n+1} ----------------------------------
%     eta_n1 = etaC(n+1);
% 
%     % ---- memory term at Nn (rank-r contraction) ---------------------
%     Lrow  = Lfac(n+1,:).';          % r×1, row n+1 of Lfac
%     Mem_n = G * Lrow;               % Ndim×1
% 
%     % ---- drift operator at Nn ---------------------------------------
%     D_n    = -1i/hbar*H_n + (L_n - phiExp*Id)*conj(eta_n1);
%     drift_n = D_n*psi - (L_n - phiExp*Id)*Mem_n;
% 
%     % ---- predictor (Heun) -------------------------------------------
%     psi_pred = psi + dN * drift_n;
%     psi_pred = psi_pred / norm(psi_pred);
% 
%     % propagate G homogeneously with the same D_n
%     G_pred = G + dN * (D_n * G);
% 
%     % ---- operators at Nn1 (for corrector) ---------------------------
%     H_n1 = (0.5*P2*exp(-3*Nn1) + Vchi*exp(3*Nn1)) / Hb;
%     L_n1 = (lambda/Hb)*exp(3*Nn1) * Xhat;
% 
%     % ---- predictor expectations and memory at Nn1 -------------------
%     phi_pred = real(psi_pred' * L_n1 * psi_pred);
%     Mem_pred = G_pred * Lrow;       % still uses row n+1 of Lfac
% 
%     % ---- drift at Nn1 (using predictor state) -----------------------
%     D_n1    = -1i/hbar*H_n1 + (L_n1 - phi_pred*Id)*conj(eta_n1);
%     drift_n1 = D_n1*psi_pred - (L_n1 - phi_pred*Id)*Mem_pred;
% 
%     % ---- corrector step for psi -------------------------------------
%     psi = psi + 0.5*dN*(drift_n + drift_n1);
%     psi = psi / norm(psi);
% 
%     % ---- corrector step for homogeneous G ---------------------------
%     G_hom = G + 0.5*dN*(D_n*G + D_n1*G_pred);
% 
%     % ---- new functional derivative F_{n+1} --------------------------
%     L_np1_op = L_n1;                        % same matrix
%     phiExp   = real(psi' * L_np1_op * psi); % this φ is for next step
%     Fnew     = (L_np1_op - phiExp*Id) * psi * dN;
% 
%     % ---- update G with source term (add L_{n+1,k} F_{n+1}) ----------
%     G = G_hom + Fnew * Lfac(n+1,:);        % outer product
% 
%     % ---- snapshots: store at requested indices ----------------------
%     % we have just advanced to time index (n+1)
%     while nextPlot <= Nplot && plotIdx(nextPlot) == (n+1)
%         PsiStore(:,nextPlot)   = psi;
%         RhoStore(:,:,nextPlot) = psi*psi';
% 
%         phi    = real(psi' * Xhat * psi);
%         phiVar = real(psi' * (Xhat^2) * psi) - phi^2;
%         fprintf('NM-SSE (low-rank): φ = %+6.3f   Var(φ) = %.3f   N = %.3f   snap %d\n', ...
%             phi, phiVar, Nn1, nextPlot);
% 
%         nextPlot = nextPlot + 1;
%     end
% end
% end

function makePanel(figName, plotSpan, Energies, Occ, panelLabel)
    fig = figure('Position',[100 100 2400 600]);  %#ok<NASGU>
    ax  = axes; hold(ax,'on'); set(gcf,'Renderer','opengl');

    % draw colored lines into ax
    [~, cb] = plotEnergyLinesByOccupation( ...
        plotSpan, Energies, Occ, ...
        'CutoffFinal',    1e-20, ...
        'MinOccForColor', 1e-10, ...
        'LineWidth',      2, ...
        'Colormap',       'jet', ...
        'Parent',         ax);

    % your layout/limits
    set(ax,'YLim',[1e0 1e3],'YLimMode','manual');
    set(ax,'XLim',[min(plotSpan) max(plotSpan)],'XLimMode','manual');
    axis(ax,'normal');
    set(ax,'PlotBoxAspectRatioMode','auto','DataAspectRatioMode','auto');
    ax.XTick = linspace(-2,1, 7);
    % place axes & colorbar nicely
    drawnow;
    ax.Units = 'normalized'; cb.Units = 'normalized';
    ax.Position = [0.08 0.15 0.78 0.78];
    cb.Position(1) = 0.88;

    % styling
    set(ax, 'FontSize',20, 'LineWidth',1.5, 'Box','on', 'TickLabelInterpreter','latex');
    xlabel(ax,'$N$','Interpreter','latex','FontSize',25);
    ylabel(ax,'$K_s$','Interpreter','latex','FontSize',25);
    title(ax, panelLabel,'Interpreter','latex','FontSize',30);
    set(cb,'FontSize',18);

    % export vector PDF
    exportgraphics(gcf, figName, 'ContentType','image');
end


%% ============================== HELPER FUNCTIONS (BOTTOM) ==============================
function [ax, cb, cLimits] = plotEnergyLinesByOccupation(plotSpan, EnergiesOut, EnergyOcc, varargin)
% Plot energy-vs-time lines with color = ADDITIVE occupation in a log-energy bin.
% Returns:
%   ax       : axes used (your 'Parent' if provided)
%   cb       : colorbar handle
%   cLimits  : [cmin cmax] in log10 units used for CLim
%
% NOTE: This function does NOT set x/y limits, aspect, or axis 'tight/square'.
%       You control layout/limits outside.

% ------------ options ------------
p = inputParser;
addParameter(p,'IdxCut',[],@(x)isnumeric(x)&&isscalar(x)&&x>=1);
addParameter(p,'CutoffFinal',1e-3,@(x)isnumeric(x)&&isscalar(x)&&x>0);
addParameter(p,'MinOccForColor',1e-12,@(x)isnumeric(x)&&isscalar(x)&&x>0);
addParameter(p,'LineWidth',1.5,@(x)isnumeric(x)&&isscalar(x)&&x>0);
addParameter(p,'Colormap','turbo');
addParameter(p,'NumColorBins',500,@(x)isnumeric(x)&&isscalar(x)&&x>=2);
addParameter(p,'ColorERange',[],@(x)isnumeric(x)&&numel(x)==2);
addParameter(p,'Parent',[],@(x) isempty(x) || ishghandle(x,'axes'));
parse(p,varargin{:});
idxCut       = p.Results.IdxCut;
cutFinal     = p.Results.CutoffFinal;
minOcc       = p.Results.MinOccForColor;
lw           = p.Results.LineWidth;
cmapName     = p.Results.Colormap;
NbColor      = p.Results.NumColorBins;
colorERange  = p.Results.ColorERange;
ax           = p.Results.Parent;

% ------------ sizes & sanity ------------
[b, N] = size(EnergiesOut);
assert(isequal(size(EnergyOcc), [b N]), 'EnergyOcc must be size(EnergiesOut).');
t = plotSpan(:).';

% target axes
if isempty(ax)
    fig = figure('Position',[100 100 800 600]); %#ok<NASGU>
    ax  = axes; 
end
hold(ax,'on'); set(gcf,'Renderer','opengl');

% auto idxCut
if isempty(idxCut)
    idxCut = find(EnergyOcc(:,end) >= cutFinal, 1, 'last');
    if isempty(idxCut), idxCut = min(b,1); end
end
idxCut = min(max(1, round(idxCut)), b);

% ------------ log-energy binning for COLOR aggregation ------------
Eall = EnergiesOut(1:idxCut,:);
if isempty(colorERange)
    EminPos = min(Eall(Eall>0),[],'all');
    if isempty(EminPos), error('All energies are non-positive; need positive energies for log bins.'); end
    EmaxVal = max(Eall,[],'all');
    if ~(EmaxVal > EminPos), EmaxVal = EminPos*(1+1e-9); end
    Emin = 0.999*EminPos;
    Emax = EmaxVal;
else
    Emin = colorERange(1);
    Emax = colorERange(2);
    if Emin <= 0, error('ColorERange(1) must be > 0 for log-spaced bins.'); end
end
Eedges = logspace(log10(Emin), log10(Emax), NbColor+1).';

% ------------ additive color values per level/time ------------
Csum = zeros(idxCut, N);   % summed occupation for each level/time via its bin
for n = 1:N
    E_n   = Eall(:,n);
    occ_n = EnergyOcc(1:idxCut, n);
    idx_n = discretize(E_n, Eedges);              % 1..NbColor or NaN

    S = accumarray(idx_n(~isnan(idx_n)), occ_n(~isnan(idx_n)), [NbColor,1], @sum, 0);
    valid = ~isnan(idx_n);
    Csum(valid,n) = S(idx_n(valid));
end
Csum = max(Csum, minOcc);  % floor to avoid -Inf in log10

% global color limits in log10
cmin = log10(min(Csum(:)));
cmax = log10(max(Csum(:)));
cLimits = [cmin cmax];

% ------------ draw colored "ribbons" (edges only) ------------
for k = 1:idxCut
    Ek = EnergiesOut(k,:);            % 1 x N
    Ck = log10(Csum(k,:));            % log-color per time

    X2 = [t; t];
    Y2 = [Ek; Ek];
    Z2 = zeros(2, N);
    C2 = [Ck; Ck];

    surf(ax, X2, Y2, Z2, C2, ...
        'FaceColor','none', ...
        'EdgeColor','interp', ...
        'LineWidth', lw, ...
        'EdgeAlpha', 1); 
end

% axes scale only (no limits/aspect here)
set(ax,'YScale','log');

% colormap + colorbar
try, colormap(ax, cmapName); catch, colormap(ax,'parula'); end
caxis(ax, cLimits);
cb = colorbar(ax);
pmin = floor(cmin); pmax = ceil(cmax); if pmax < pmin, pmax = pmin; end
ticks = pmin:pmax;
if numel(ticks) < 2
    ticks = [cmin cmax];
    cb.Ticks = ticks;
    cb.TickLabels = arrayfun(@(v) sprintf('10^{%.2g}', v), ticks, 'uni', 0);
else
    cb.Ticks = ticks;
    cb.TickLabels = arrayfun(@(p) sprintf('10^{%d}', p), ticks, 'uni', 0);
end
ylabel(cb, '$P(K_s)$','Interpreter','latex');

end



function [outRho, EnergiesOut, EnergyOcc] = HamiltonianEigenrep2(RhoIn, plotSpan, Xhat, Phat, Hb, mu, beta3, beta4)
% HamiltonianEigenrep2
% Inputs:
%   RhoIn (bSize x bSize x N): density matrices in the fixed basis at each time
%   plotSpan (1 x N): times (or e-folds Ne) to evaluate
%   Xhat, Phat: operators in the fixed basis
%   Hb, Height, Separation: scalars used in the Hamiltonian
%
% Outputs:
%   outRho   (bSize x bSize x N): density matrices in the instantaneous energy basis (sorted by energy)
%   EnergiesOut (bSize x N): energies at each time (shifted so min=1, matching your original behavior)
%   EnergyOcc   (bSize x N): vector of energy occupations (populations) at each time

bSize = size(RhoIn, 1);
N     = length(plotSpan);

EnergiesOut = zeros(bSize, N);
EnergyOcc   = zeros(bSize, N);
outRho      = zeros(bSize, bSize, N);

Id  = eye(bSize);
H0     = @(Ne) (0.5/Hb) * (Phat^2 * exp(-3*Ne) ...
    + (-(mu^2/2)*Xhat^2 + (2*mu*beta3/3)*Xhat^3 + (beta4^2-beta3^2)*Xhat^4/4) * exp(3*Ne));

for n = 1:N
    % Diagonalize instantaneous Hamiltonian
    [V, D] = eig(H0(plotSpan(n)));
    evals  = real(diag(D));

    % Sort energies ascending, and reorder eigenvectors consistently
    [evals_sorted, idx] = sort(evals, 'ascend');
    V = V(:, idx);

    % Store (normalized) energies like in your original code
    EnergiesOut(:, n) = evals_sorted - min(evals_sorted) + 1;

    % Transform rho into the energy eigenbasis
    TempRho = V' * RhoIn(:, :, n) * V;
    TempRho = 0.5 * (TempRho + TempRho');         % enforce Hermiticity
    TempRho = TempRho / trace(TempRho);           % normalize

    outRho(:, :, n) = TempRho;

    % Occupations = diagonal elements in energy basis
    occ = real(diag(TempRho));

    % Numerical cleanup: clip tiny negative values and renormalize
    occ(occ < 0 & occ > -1e-12) = 0;
    s = sum(occ);
    if s ~= 0
        occ = occ / s;
    end

    EnergyOcc(:, n) = occ;
end
end




function merge_audio_video_ffmpeg(videoPath, audioPath, outPath, ~)
    if ~isfile(videoPath), error('Video file not found: %s', videoPath); end
    if ~isfile(audioPath), error('Audio file not found: %s', audioPath); end

    cmd = sprintf(['ffmpeg -y -i "%s" -i "%s" ' ...
                   '-map 0:v:0 -map 1:a:0 ' ...
                   '-c:v libx264 -c:a aac -b:a 192k -shortest "%s"'], ...
                   videoPath, audioPath, outPath);
    disp(['Executing: ' cmd]);
    [st, out] = system(cmd);
    if st ~= 0
        fprintf(2, 'FFmpeg error output:\n%s\n', out);
        error('Failed to merge audio and video to %s', outPath);
    else
        fprintf('Merged audio+video -> %s\n', outPath);
    end
end



