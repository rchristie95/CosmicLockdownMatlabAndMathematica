



%% 

clc;
clearvars;
close all;
tStart = cputime;

%% Physical & Numerical Parameters
hbar      = 1;        % Reduced Planck constant

mu    = 0.5;      % Potential parameter: height scale
beta3=0.025;        % Potential parameter: separation between minima
beta4=0.13;



Hb        =5;     % Hubble scale parameter
lambda    =0.05;      % Coupling strength
% Time‐integration grid for SSE
N0        =-2;                       % Initial e-fold
Nend  = 1;

TSpan = linspace(N0, Nend,2);

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

%% instanteneous trajectory
plotSpanInst=linspace(-2,1,1001);
Hinit = @(Z,n) (exp(3*plotSpanInst(n))*(-(mu^2/2)*Z(1)^2+(2*mu*beta3/3)*Z(1)^3+((beta4^2-beta3^2)/4)*Z(1)^4)+0.5*Z(2)^2*exp(-3*plotSpanInst(n)))/Hb;

[PsiInst,RhoInst]= AdiabaticGroundStates(bSize, hbar, mu, beta3, beta4, Hb, plotSpanInst);
targets = [-2, 0.0,1];

for k = 1:1001
    % Build the Hamiltonian for the chosen e-fold N0
    NeCurr   = plotSpanInst(k);
    Hcurr    = H0(NeCurr);

    Rho=RhoInst(:,:,k);

    expectHInst(k)  = real(trace(Hcurr * Rho));
    % Rho=Rho/trace(Rho);
    NormInst(k)  = real(trace(Rho));
    PurityInst(k)   = trace(Rho^2)/trace(Rho)^2;
    CtInst(k)       = abs(trace(ProjR * Rho*ProjR )) / NormInst(k);
    expXInst(k)     = real(trace(Xhat * Rho));
    expPInst(k)     = real(trace(Phat * Rho));
    varXInst(k)=real(trace(Rho*Xhat^2))-expXInst(k)^2;
    varPInst(k)=real(trace(Rho*Phat^2))-expPInst(k)^2;
    varXPInst(k)=0.5*real(trace(Rho*(Phat*Xhat+Xhat*Phat)))-expPInst(k)*expXInst(k);


    PsiInstPos(:,k) = HVector * PsiInst(1:Nx,k)/norm(PsiInst(1:Nx,k));
    WigInst(:,:,k)= PsiWigner( PsiInstPos(:,k), x, p, hbar);

end


%%
% % Phase-space grid
[xMesh, pMesh] = meshgrid(x, p);
Xc=linspace(-Xview, Xview, Nx);
Pc=linspace(-Xview, Xview, Nx);
[xMesh1, pMesh1] = meshgrid(Xc, Pc);
% 
% Helper for Hamiltonian contour at a given frame index
Hgrid_at = @(k) ( exp(3*k) .* (-(mu^2/2).*xMesh1.^2 +(2*mu*beta3/3).*xMesh1.^3 +((beta4^2-beta3^2)/4).*xMesh1.^4) ...
                + 0.5 .* pMesh1.^2 .* exp(-3*k) ) / Hb;

% Compute initial contour data
Z_initial = arrayfun(@(x, y) Hinit([x; y], 1), xMesh, pMesh);
Z_final = arrayfun(@(x, y) Hinit([x; y], 1001), xMesh, pMesh);
Ztotal=[Z_initial;Z_final];
cmin0 = min(Ztotal(:), [], 'omitnan');
cmax0 = max(Ztotal(:), [], 'omitnan');


u = linspace(0, 1, 50);   % parameter
u = u.^5;                 % cluster toward cmin0

levels = cmin0 + (cmax0 - cmin0) * u;

[~, idxZero] = min(abs(plotSpanInst), [], 2);   % idx(i) indexes p nearest to t(i)



% % Common plotting settings
cax = [-0.05 0.15];
fmtText = @(Nval,Cval) sprintf('$N=%.2f,\\;\\langle\\hat\\theta_{\\phi^{+}}\\rangle=%.2f$', Nval, Cval);
% 
% % ---------- (a) N_e = -1.5 ----------
fig = figure('Position', [100 100 800 600]);  % wider figure for visibility
ax = axes(fig); hold(ax,'on');
hA = pcolor(ax, x, p, WigInst(:,:,1));
shading('interp')
% set(hA,'Interpolation','bilinear');   % or 'bicubic'
colormap(ax,'parula');
contour(ax, xMesh1, pMesh1, Hgrid_at(targets(1)), levels, 'LineColor', 'k');
xlim(ax,[-Xview Xview]); ylim(ax,[-Xview Xview]); axis(ax,'square'); box(ax,'on'); caxis(ax,cax);
xticks(linspace(-Xview,Xview,7))
colorbar(ax,'FontSize',18);
text(ax, 0.98, 0.98, fmtText(targets(1), CtInst(1)), ...
    'Units','normalized','HorizontalAlignment','right','VerticalAlignment','top', ...
    'Interpreter','latex','FontSize',24,'FontWeight','bold','Color','w');
set(ax, 'FontSize', 20, 'LineWidth', 1.5, 'Box', 'on', 'TickLabelInterpreter', 'latex');
xlabel(ax,'$\phi$','Interpreter','latex','FontSize',25);
ylabel(ax,'$\pi_\phi$','Interpreter','latex','FontSize',25);
title(ax, '(a)','Interpreter','latex','FontSize',30);
box on
exportgraphics(fig,'wigner_panel_a_Ne_-1p5.pdf','ContentType', 'image', 'Resolution', 600);

% ---------- (b) N_e = 0 ----------
fig = figure('Position', [100 100 800 600]);  % wider figure for visibility
ax = axes(fig); hold(ax,'on');
hB = pcolor(ax, x, p, WigInst(:,:,idxZero));
shading('interp')
% set(hB,'Interpolation','bilinear');   % or 'bicubic'
colormap(ax,'parula');
contour(ax, xMesh1, pMesh1, Hgrid_at(targets(2)), levels, 'LineColor', 'k');
xlim(ax,[-Xview Xview]); ylim(ax,[-Xview Xview]); axis(ax,'square'); box(ax,'on'); caxis(ax,cax);
expLine = plot(expXInst(1:idxZero), expPInst(1:idxZero), 'w-', 'LineWidth', 1); % White line with specified width
xticks(linspace(-Xview,Xview,7))

colorbar(ax,'FontSize',18);
text(ax, 0.98, 0.98, fmtText(targets(2), CtInst(idxZero)), ...
    'Units','normalized','HorizontalAlignment','right','VerticalAlignment','top', ...
    'Interpreter','latex','FontSize',24,'FontWeight','bold','Color','w');
set(ax, 'FontSize', 20, 'LineWidth', 1.5, 'Box', 'on', 'TickLabelInterpreter', 'latex');
xlabel(ax,'$\phi$','Interpreter','latex','FontSize',25);
ylabel(ax,'$\pi_\phi$','Interpreter','latex','FontSize',25);
title(ax, '(b)','Interpreter','latex','FontSize',30);
box on
exportgraphics(fig,'wigner_panel_b_Ne_0.pdf','ContentType', 'image', 'Resolution', 600);

% ---------- (c) N_e = +1.5 ----------
fig = figure('Position', [100 100 800 600]);  % wider figure for visibility
ax = axes(fig); hold(ax,'on');
hC = pcolor(ax, x, p, WigInst(:,:,end));
shading('interp')
% set(hC,'Interpolation','bilinear');   % or 'bicubic'
colormap(ax,'parula');
contour(ax, xMesh1, pMesh1, Hgrid_at(targets(3)), levels, 'LineColor', 'k');
xlim(ax,[-Xview Xview]); ylim(ax,[-Xview Xview]); axis(ax,'square'); box(ax,'on'); caxis(ax,cax);
expLine = plot(expXInst(1:end), expPInst(1:end), 'w-', 'LineWidth', 1); % White line with specified width

xticks(linspace(-Xview,Xview,7))

colorbar(ax,'FontSize',18);
text(ax, 0.98, 0.98, fmtText(targets(3), CtInst(end)), ...
    'Units','normalized','HorizontalAlignment','right','VerticalAlignment','top', ...
    'Interpreter','latex','FontSize',24,'FontWeight','bold','Color','w');
set(ax, 'FontSize', 20, 'LineWidth', 1.5, 'Box', 'on', 'TickLabelInterpreter', 'latex');
xlabel(ax,'$\phi$','Interpreter','latex','FontSize',25);
ylabel(ax,'$\pi_\phi$','Interpreter','latex','FontSize',25);
title(ax, '(c)','Interpreter','latex','FontSize',30);
box on
exportgraphics(fig,'wigner_panel_c_Ne_+0p75.pdf','ContentType', 'image', 'Resolution', 600);



 %% InstVideo

clear flipbook


Xc=linspace(-Xview, Xview, Nx);
Pc=linspace(-Xview, Xview, Nx);
% H = @(Z,n) (6*exp(3*plotSpan(n))*sqrt((0.5*c2*Z(1)^2+0.5*Z(2)^2*exp(-6*plotSpan(n)))/3));
Hinit = @(Z,n) (exp(3*plotSpanInst(n))*(-(mu^2/2)*Z(1)^2+(2*mu*beta3/3)*Z(1)^3+((beta4^2-beta3^2)/4)*Z(1)^4)+0.5*Z(2)^2*exp(-3*plotSpanInst(n)))/Hb;

% Define your meshgrid
[xMesh, pMesh] = meshgrid(x, p); % Adjust the limits and resolution as needed
[xMesh1, pMesh1] = meshgrid(Xc, Pc); % Adjust the limits and resolution as needed



% Define aspect ratio for the figure
aspectRatio = [6, 6];

% Create a figure with specified position and aspect ratio
figureHandle = figure('Position', [100, 100, 800, 800 * aspectRatio(2) / aspectRatio(1)]);

% Initialize pcolor plot with the first set of data
f = pcolor(xMesh, pMesh, WigInst(:,:,1));
xlim([-Xview Xview]);
ylim([-Xview Xview]);
xticks(linspace(-Xview,Xview,7))



shading interp
colormap("parula")  % Changed colormap to 'parula' for perceptually uniform colors

% Set color axis limits
clim([-0.05 0.15])  % Ensure this range covers both oscillator and inverted oscillator regimes

% Add a colorbar with increased font size
colorbar('FontSize', 14);

title( sprintf(...
    'Instantaneous Ground State'), ...
    'FontSize', 20, 'Interpreter', 'latex' );

xlabel('$\phi$', 'FontSize', 25, 'Interpreter', 'latex');
ylabel('$\pi_{\phi}$', 'FontSize', 25, 'Interpreter', 'latex');

% Customize axis properties for better aesthetics
set(gca, 'FontSize', 16, 'LineWidth', 1.5, 'Box', 'on', 'TickLabelInterpreter', 'latex');
axis square  % Use square axes for equal aspect ratio

% Hold the current plot to allow multiple graphics objects
hold on;

% Initialize an empty line plot for experimental data
expLine = plot(NaN, NaN, 'w-', 'LineWidth', 1); % White line with specified width

% Initialize contour plot with initial Z data
[~, contPlot] = contour(xMesh1, pMesh1, Z_initial, levels, 'LineColor', "k");

% Initialize the time display
timeText = text(0.95, 0.95, '', ...
    'Units', 'normalized', ...          % Position relative to axes (0 to 1)
    'HorizontalAlignment', 'right', ... % Align text to the right
    'VerticalAlignment', 'top', ...     % Align text to the top
    'FontSize', 25, ...                  % Set desired font size
    'FontWeight', 'bold', ...            % Make text bold for visibility
    'Color', 'w');                       % Set text color to black
uistack(timeText, 'top');                 % Ensure text is on top

% Main simulation loop
for n = 1:length(WigInst(1,1,:))
    % Retrieve the current simulation time from plotSpanInst
    t = plotSpanInst(n);

    % Update the pcolor plot with new data for the current time step
    set(f, 'CData', WigInst(:,:,n));


    % ---------------------- Dynamic Contour Update ----------------------

    % Compute contour data based on Hinit
    CdatCont = arrayfun(@(xx, yy) Hinit([xx; yy], n), xMesh1, pMesh1);

    % Delete existing contour plot to prevent overlap
    if ~isempty(contPlot) && isgraphics(contPlot)
        delete(contPlot);
    end

    % Recompute and plot new contours with fixed nonlinear levels
    [~, contPlot] = contour(xMesh1, pMesh1, CdatCont, levels, 'LineColor', "k");
    % ---------------------- End of Dynamic Contour Update -------------------

    % Update the experimental line plot with new data point
    xData = get(expLine, 'XData');
    yData = get(expLine, 'YData');
    set(expLine, 'XData', [xData, expXInst(n)], 'YData', [yData, expPInst(n)]);

    % ---------------------- Update Time Display ----------------------

    % Update the text object with the current time, formatted to two decimal places
    set(timeText, 'Interpreter', 'latex', ...
        'String', sprintf('$N = $ %.2f, $\\langle \\hat\\theta_{\\phi^{+}}\\rangle = $ %.2f', t, CtInst(n)));


    % Ensure the time text stays on top of other plot elements
    uistack(timeText, 'top');

    % ---------------------- End of Time Display Update -------------------

    % Refresh the plot to reflect updates
    drawnow;

    % Optional: Capture the frame for a flipbook or video
    flipbook(n) = getframe(gcf);
end

% Save video
InstVideoPath= sprintf('WignerGroundStateVideo.avi');
writerObj = VideoWriter(InstVideoPath);
writerObj.FrameRate = 1001/30;
open(writerObj);
writeVideo(writerObj, flipbook);
close(writerObj);

close all;

%% Single Reference Schrodinger

targets = [-1, 0.0, 1];

tic
[plotSpanHam,PsiSchrodinger , RhoSchrodinger]=SchrodingerSingleTrajectory_ExpStep_1000( TSpan,bSize, hbar,mu,beta3, beta4, Hb);
timeHam=toc

nFramesHam=length(plotSpanHam);


expectHSchrodinger = zeros(1,nFramesHam);
PuritySchrodinger  = zeros(1,nFramesHam);
NormSchrodinger = zeros(1,nFramesHam);
CtSchrodinger      = zeros(1,nFramesHam);
expXSchrodinger    = zeros(1,nFramesHam);
expPSchrodinger    = zeros(1,nFramesHam);
varXSchrodinger    = zeros(1,nFramesHam);
varPSchrodinger    = zeros(1,nFramesHam);
varXPSchrodinger    = zeros(1,nFramesHam);

PsiSchrodingerPos  = zeros(Nx,nFramesHam);
% RhoSchrodingerPos  = zeros(Nx,Nx,nFramesHam);

for k = 1 : nFramesHam
    NeCurr   = plotSpanHam(k);
    Hcurr    = H0(NeCurr);
    Rho        = RhoSchrodinger(:,:,k);

    expectHSchrodinger(k)  = real(trace(Hcurr * Rho));
    % Rho=Rho/trace(Rho);
    NormSchrodinger(k)  = real(trace(Rho));
    PuritySchrodinger(k)   = trace(Rho^2)/trace(Rho)^2;
    CtSchrodinger(k)       = abs(trace(ProjR * Rho*ProjR )) / NormSchrodinger(k);
    expXSchrodinger(k)     = real(trace(Xhat * Rho));
    expPSchrodinger(k)     = real(trace(Phat * Rho));
    varXSchrodinger(k)=real(trace(Rho*Xhat^2))-expXSchrodinger(k)^2;
    varPSchrodinger(k)=real(trace(Rho*Phat^2))-expPSchrodinger(k)^2;
    varXPSchrodinger(k)=0.5*real(trace(Rho*(Phat*Xhat+Xhat*Phat)))-expPSchrodinger(k)*expXSchrodinger(k);


    PsiSchrodingerPos(:,k) = HVector * PsiSchrodinger(1:Nx,k);
%     RhoSchrodingerPos(:,:,k) = HVector * RhoSchrodinger(:,:,k) * HVector';
end
% Schrodinger Video
clear flipbook
WigSchrodinger = PsiWigner(PsiSchrodingerPos, x, p, hbar);
% WigSSE = RhoWigner(RhoSchrodingerPos, x, p, hbar);

%% Panels Hamilton

[~, idxOrig] = min(abs(plotSpanHam- targets(1)), [], 2);   % idx(i) indexes p nearest to t(i)
[~, idxZero] = min(abs(plotSpanHam- targets(2)), [], 2);   % idx(i) indexes p nearest to t(i)
WigImagesHam(:,:,1)=WigSchrodinger(:,:, idxOrig);
WigImagesHam(:,:,2)=WigSchrodinger(:,:, idxZero);
WigImagesHam(:,:,3)=WigSchrodinger(:,:, end);


% ---------- (a) N_e = -0.75 ----------
fig = figure('Position', [100 100 800 600]);  % wider figure for visibility
ax = axes(fig); hold(ax,'on');
hA = pcolor(ax, x, p, WigImagesHam(:,:,1));
shading('interp')

% set(hA,'Interpolation','bilinear');   % or 'bicubic'
colormap(ax,'parula');
contour(ax, xMesh1, pMesh1, Hgrid_at(targets(1)), levels, 'LineColor', 'k');
xlim(ax,[-Xview Xview]); ylim(ax,[-Xview Xview]); axis(ax,'square'); box(ax,'on'); caxis(ax,cax);
xticks(linspace(-Xview,Xview,7))
colorbar(ax,'FontSize',18);
expLine = plot(expXSchrodinger(1:idxOrig), expPSchrodinger(1:idxOrig), 'w-', 'LineWidth', 1); % White line with specified width
text(ax, 0.98, 0.98, fmtText(targets(1), CtSchrodinger(idxOrig)), ...
    'Units','normalized','HorizontalAlignment','right','VerticalAlignment','top', ...
    'Interpreter','latex','FontSize',24,'FontWeight','bold','Color','w');
set(ax, 'FontSize', 20, 'LineWidth', 1.5, 'Box', 'on', 'TickLabelInterpreter', 'latex');
xlabel(ax,'$\phi$','Interpreter','latex','FontSize',25);
ylabel(ax,'$\pi_\phi$','Interpreter','latex','FontSize',25);
title(ax, '(a)','Interpreter','latex','FontSize',30);
box on
exportgraphics(fig,'Hamwigner_panel_a_Ne_-0p75.pdf','ContentType', 'image', 'Resolution', 600);

% ---------- (b) N_e = 0 ----------
fig = figure('Position', [100 100 800 600]);  % wider figure for visibility
ax = axes(fig); hold(ax,'on');
hB = pcolor(ax, x, p, WigImagesHam(:,:,2));
% set(hB,'Interpolation','bilinear');   % or 'bicubic'
shading('interp')

colormap(ax,'parula');
contour(ax, xMesh1, pMesh1, Hgrid_at(targets(2)), levels, 'LineColor', 'k');
xlim(ax,[-Xview Xview]); ylim(ax,[-Xview Xview]); axis(ax,'square'); box(ax,'on'); caxis(ax,cax);
xticks(linspace(-Xview,Xview,7))
colorbar(ax,'FontSize',18);
expLine = plot(expXSchrodinger(1:idxZero), expPSchrodinger(1:idxZero), 'w-', 'LineWidth', 1); % White line with specified width
text(ax, 0.98, 0.98, fmtText(targets(2), CtSchrodinger(idxZero)), ...
    'Units','normalized','HorizontalAlignment','right','VerticalAlignment','top', ...
    'Interpreter','latex','FontSize',24,'FontWeight','bold','Color','w');
set(ax, 'FontSize', 20, 'LineWidth', 1.5, 'Box', 'on', 'TickLabelInterpreter', 'latex');
xlabel(ax,'$\phi$','Interpreter','latex','FontSize',25);
ylabel(ax,'$\pi_\phi$','Interpreter','latex','FontSize',25);
title(ax, '(b)','Interpreter','latex','FontSize',30);
box on
exportgraphics(fig,'Hamwigner_panel_b_Ne_0.pdf','ContentType', 'image', 'Resolution', 600);

% ---------- (c) N_e = +1.5 ----------
fig = figure('Position', [100 100 800 600]);  % wider figure for visibility
ax = axes(fig); hold(ax,'on');
hC = pcolor(ax, x, p, WigImagesHam(:,:,3));
% set(hC,'Interpolation','bilinear');   % or 'bicubic'
shading('interp')

colormap(ax,'parula');
contour(ax, xMesh1, pMesh1, Hgrid_at(targets(3)), levels, 'LineColor', 'k');
xlim(ax,[-Xview Xview]); ylim(ax,[-Xview Xview]); axis(ax,'square'); box(ax,'on'); caxis(ax,cax);
xticks(linspace(-Xview,Xview,7))
colorbar(ax,'FontSize',18);
expLine = plot(expXSchrodinger(1:end), expPSchrodinger(1:end), 'w-', 'LineWidth', 1); % White line with specified width
text(ax, 0.98, 0.98, fmtText(targets(3),CtSchrodinger(end)), ...
    'Units','normalized','HorizontalAlignment','right','VerticalAlignment','top', ...
    'Interpreter','latex','FontSize',24,'FontWeight','bold','Color','w');
set(ax, 'FontSize', 20, 'LineWidth', 1.5, 'Box', 'on', 'TickLabelInterpreter', 'latex');
xlabel(ax,'$\phi$','Interpreter','latex','FontSize',25);
ylabel(ax,'$\pi_\phi$','Interpreter','latex','FontSize',25);
title(ax, '(c)','Interpreter','latex','FontSize',30);
box on
exportgraphics(fig,'Hamwigner_panel_c_Ne_+0p75.pdf','ContentType', 'image', 'Resolution', 600);


%% Hamiltonian Video
Xc=linspace(-Xview, Xview, Nx);
Pc=linspace(-Xview, Xview, Nx);
% H = @(Z,n) (6*exp(3*plotSpanHam(n))*sqrt((0.5*c2*Z(1)^2+0.5*Z(2)^2*exp(-6*plotSpanHam(n)))/3));
Hinit = @(Z,n) (exp(3*plotSpanHam(n))*(-(mu^2/2)*Z(1)^2+(2*mu*beta3/3)*Z(1)^3+((beta4^2-beta3^2)/4)*Z(1)^4)+0.5*Z(2)^2*exp(-3*plotSpanHam(n)))/Hb;

% Define your meshgrid
[xMesh, pMesh] = meshgrid(x, p); % Adjust the limits and resolution as needed
[xMesh1, pMesh1] = meshgrid(Xc, Pc); % Adjust the limits and resolution as needed

% Compute initial contour data
Z_initial = arrayfun(@(x, y) Hinit([x; y], 1), xMesh, pMesh);

% Define aspect ratio for the figure
aspectRatio = [6, 6];

% Create a figure with specified position and aspect ratio
figureHandle = figure('Position', [100, 100, 800, 800 * aspectRatio(2) / aspectRatio(1)]);

% Initialize pcolor plot with the first set of data
f = pcolor(xMesh, pMesh, WigSchrodinger(:,:,1));
xlim([-Xview Xview]);
ylim([-Xview Xview]);
shading interp
colormap("parula")  % Changed colormap to 'parula' for perceptually uniform colors

% Set color axis limits
clim([-0.05 0.15])  % Ensure this range covers both oscillator and inverted oscillator regimes

% Add a colorbar with increased font size
colorbar('FontSize', 14);

title(sprintf(...
    'Hamiltonian , $\\tilde \\mu = %.2f$',mu/Hb), ...
    'FontSize', 20, 'Interpreter', 'latex' );

xlabel('$\phi$', 'FontSize', 25, 'Interpreter', 'latex');
ylabel('$\pi_{\phi}$', 'FontSize', 25, 'Interpreter', 'latex');
xticks(linspace(-Xview,Xview,7))

% Customize axis properties for better aesthetics
set(gca, 'FontSize', 16, 'LineWidth', 1.5, 'Box', 'on', 'TickLabelInterpreter', 'latex');
axis square  % Use square axes for equal aspect ratio

% Hold the current plot to allow multiple graphics objects
hold on;

% Initialize an empty line plot for experimental data
expLine = plot(NaN, NaN, 'w-', 'LineWidth', 1); % White line with specified width

% Initialize contour plot with initial Z data
[~, contPlot] = contour(xMesh1, pMesh1, Z_initial, levels, 'LineColor', "k");

% Initialize the time display
timeText = text(0.95, 0.95, '', ...
    'Units', 'normalized', ...          % Position relative to axes (0 to 1)
    'HorizontalAlignment', 'right', ... % Align text to the right
    'VerticalAlignment', 'top', ...     % Align text to the top
    'FontSize', 25, ...                  % Set desired font size
    'FontWeight', 'bold', ...            % Make text bold for visibility
    'Color', 'w');                       % Set text color to black
uistack(timeText, 'top');                 % Ensure text is on top

% Main simulation loop
for n = 1:length(WigSchrodinger(1,1,:))
    % Retrieve the current simulation time from plotSpanHam
    t = plotSpanHam(n);

    % Update the pcolor plot with new data for the current time step
    set(f, 'CData', WigSchrodinger(:,:,n));

    % ---------------------- Dynamic Contour Update ----------------------

    % Compute contour data based on Hinit
    CdatCont = arrayfun(@(xx, yy) Hinit([xx; yy], n), xMesh1, pMesh1);

    % Delete existing contour plot to prevent overlap
    if ~isempty(contPlot) && isgraphics(contPlot)
        delete(contPlot);
    end

    % Recompute and plot new contours with fixed nonlinear levels
    [~, contPlot] = contour(xMesh1, pMesh1, CdatCont, levels, 'LineColor', "k");

    % ---------------------- End of Dynamic Contour Update -------------------

    % Update the experimental line plot with new data point
    xData = get(expLine, 'XData');
    yData = get(expLine, 'YData');
    set(expLine, 'XData', [xData, expXSchrodinger(n)], 'YData', [yData, expPSchrodinger(n)]);

    % ---------------------- Update Time Display ----------------------

    % Update the text object with the current time, formatted to two decimal places
    set(timeText, 'Interpreter', 'latex', ...
        'String', sprintf('$N = $ %.2f, $\\langle \\hat\\theta_{\\phi^{+}}\\rangle = $ %.2f', t, CtSchrodinger(n)));


    % Ensure the time text stays on top of other plot elements
    uistack(timeText, 'top');

    % ---------------------- End of Time Display Update -------------------

    % Refresh the plot to reflect updates
    drawnow;

    % Optional: Capture the frame for a flipbook or video
    flipbook(n) = getframe(gcf);
end

% Save video
HamVideoPath  = sprintf('WignerSchrodingerVideo_Hubble_%.2g.avi', Hb);
writerObj = VideoWriter(HamVideoPath);
writerObj.FrameRate = nFramesHam/30;
open(writerObj);
writeVideo(writerObj, flipbook);
close(writerObj);

close all;

% Common plotting settings
cax = [-0.05 0.15];
fmtText = @(Nval,Cval) sprintf('$N=%.2f,\\;\\langle\\hat\\theta_{\\phi^{+}}\\rangle=%.2f$', Nval, Cval);






%% Single SSE

rng('shuffle');   % seed RNG based on current time
[dN_plot, plotSpanSSE] = dnPPlot(N0, Nend,1e-7,1001);
% eta=(randn(Nsteps+1,1));
tic
% [~,PsiSSE, RhoSSE] = SSEDynamics_X_Sparse(   bSize, hbar, mu, beta3, beta4, lambda, Hb,plotSpanSSE,dN_plot);
% [~,PsiSSE, RhoSSE] = SSEDynamics_XP_ExpIntSparse(   bSize, hbar, mu, beta3, beta4, lambda, Hb, Xhat, Phat,plotSpanSSE,dN_plot);


timeSSE=toc
nFramesSSE=length(plotSpanSSE);


expectHSSE = zeros(1,nFramesSSE);
PuritySSE  = zeros(1,nFramesSSE);
NormSSE = zeros(1,nFramesSSE);
CtSSE      = zeros(1,nFramesSSE);
expXSSE    = zeros(1,nFramesSSE);
expPSSE    = zeros(1,nFramesSSE);


PsiSSEPos  = zeros(Nx,nFramesSSE);

for k = 1 : nFramesSSE


    NeCurr   = plotSpanSSE(k);
    Hcurr    = H0(NeCurr);
    % RhoSSE(:,:,k)        = PsiSSE(:,k)*PsiSSE(:,k)';
    Rho        = RhoSSE(:,:,k);

    expectHSSE(k)  = real(trace(Hcurr * Rho));
    % Rho=Rho/trace(Rho);
    NormSSE(k)  = real(trace(Rho));
    PuritySSE(k)   = trace(Rho^2)/trace(Rho)^2;
    CtSSE(k)       = abs(trace(ProjR * Rho*ProjR )) / NormSSE(k);
    expXSSE(k)     = real(trace(Xhat * Rho));
    expPSSE(k)     = real(trace(Phat * Rho));


    PsiSSEPos(:,k) = HVector * PsiSSE(1:Nx,k);


end

%% Create Wigner‐Function Video
clear flipbook

WigSSE = PsiWigner(PsiSSEPos, x, p, hbar);
% WigSSE = RhoWigner(RhoSSEPos, x, p, hbar);

Xc = linspace(-Xview, Xview, Nx);
Pc = linspace(-Xview, Xview, Nx);

Hinit = @(Z,n) ( ...
    exp(3*plotSpanSSE(n)) * ( ...
        -(mu^2/2)*Z(1)^2 + (2*mu*beta3/3)*Z(1)^3 + ((beta4^2-beta3^2)/4)*Z(1)^4 ) ...
    + 0.5*Z(2)^2 * exp(-3*plotSpanSSE(n)) ) / Hb;

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
f = pcolor(xMesh, pMesh,WigSSE(:,:,1));
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
    'SSE $(\\hat L_{\\phi})$ , True Vacuum, $\\lambda = %.2f$, $\\tilde \\mu = %.2f$', ...
    lambda, mu/Hb), ...
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
    t = plotSpanSSE(n);

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
SSEVideoPath = sprintf('Wigner_X_SSEVideo_lambda_%.2g_Hubble_%.2g.avi', lambda, Hb);
writerObj    = VideoWriter(SSEVideoPath);

writerObj.FrameRate = nFramesSSE/30;
open(writerObj);
writeVideo(writerObj, flipbook);
close(writerObj);

close all;


%%
rng('shuffle');   % seed RNG based on current time

% eta=(randn(Nsteps+1,1));
tic
% [~,PsiSSE2, RhoSSE2] = SSEDynamics_X_Sparse(   bSize, hbar, mu, beta3, beta4, lambda, Hb,plotSpanSSE,dN_plot);


timeSSE=toc
nFramesSSE=length(plotSpanSSE);


PuritySSE2  = zeros(1,nFramesSSE);
NormSSE2 = zeros(1,nFramesSSE);
CtSSE2      = zeros(1,nFramesSSE);
expXSSE2    = zeros(1,nFramesSSE);
expPSSE2    = zeros(1,nFramesSSE);


PsiSSEPos2  = zeros(Nx,nFramesSSE);

for k = 1 : nFramesSSE


    NeCurr   = plotSpanSSE(k);
    Hcurr    = H0(NeCurr);
    % RhoSSE(:,:,k)        = PsiSSE(:,k)*PsiSSE(:,k)';
    Rho        = RhoSSE2(:,:,k);

    expectHSSE2(k)  = real(trace(Hcurr * Rho));
    % Rho=Rho/trace(Rho);
    NormSSE2(k)  = real(trace(Rho));
    PuritySSE2(k)   = trace(Rho^2)/trace(Rho)^2;
    CtSSE2(k)       = abs(trace(ProjR * Rho*ProjR )) / NormSSE(k);
    expXSSE2(k)     = real(trace(Xhat * Rho));
    expPSSE2(k)     = real(trace(Phat * Rho));

    PsiSSEPos2(:,k) = HVector * PsiSSE2(1:Nx,k);


end
close all
filenameWorkspace = sprintf('Workspace_X_Individuals_lambda%.2g_Hubble_%.2g.mat', lambda, Hb);
save(filenameWorkspace)

%% Create Wigner‐Function Video
clear flipbook


WigSSE2 = PsiWigner(PsiSSEPos2, x, p, hbar);
% WigSSE = RhoWigner(RhoSSEPos, x, p, hbar);


% Compute initial contour data on contour grid
Z_initial = arrayfun(@(xx, yy) Hinit([xx; yy], 1), xMesh1, pMesh1);

% Define aspect ratio for the figure
aspectRatio = [6, 6];

% Create a figure with specified position and aspect ratio
figureHandle = figure('Position', [100, 100, 800, 800 * aspectRatio(2) / aspectRatio(1)]);

% Initialize pcolor plot with the first set of data
f = pcolor(xMesh, pMesh, WigSSE2(:,:,1));
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
    'SSE $(\\hat L_{\\phi})$, False Vacuum,  $\\lambda = %.2f$, $\\tilde \\mu = %.2f$', ...
    lambda, mu/Hb), ...
    'FontSize', 20, 'Interpreter', 'latex' );

xlabel('$\phi$',      'FontSize', 25, 'Interpreter', 'latex');
ylabel('$\pi_{\phi}$','FontSize', 25, 'Interpreter', 'latex');

set(gca, 'FontSize', 16, 'LineWidth', 1.5, 'Box', 'on', ...
    'TickLabelInterpreter', 'latex');
axis square

hold on;

% Initialize an empty line plot for experimental data
expLine = plot(NaN, NaN, 'w-', 'LineWidth', 1);

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
    t = plotSpanSSE(n);

    % Update the pcolor plot with new data for the current time step
    set(f, 'CData', WigSSE2(:,:,n));

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
    set(expLine, 'XData', [xData, expXSSE2(n)], ...
                 'YData', [yData, expPSSE2(n)]);

    % Update the text object with the current time / observable
    set(timeText, 'Interpreter', 'latex', ...
        'String', sprintf('$N = $ %.2f, $\\langle \\hat\\theta_{\\phi^{+}}\\rangle = $ %.2f', ...
                          t, CtSSE2(n)));
    uistack(timeText, 'top');

    % Refresh the plot
    drawnow;

    % Capture the frame
    flipbook(n) = getframe(gcf);
end

% Save video
SSEVideoPath2 = sprintf('Wigner_X_SSEVideo2_lambda_%.2g_Hubble_%.2g.avi', lambda, Hb);
writerObj    = VideoWriter(SSEVideoPath);

writerObj.FrameRate = nFramesSSE/30;
open(writerObj);
writeVideo(writerObj, flipbook);
close(writerObj);

close all;


%% Single Lindblad

tic
% [plotSpanLind, RhoLind,dNsLind] = Markov2Lindblads_Adaptive_Sparse_Dopri (PsiIn*PsiIn', Xhat, Phat, TSpan, hbar, mu,beta3, beta4, lambda, Hb,true);
% [plotSpanLind, RhoLind] = Markov_2Lindblads_ExpStep_1000 ( TSpan,bSize, hbar, mu,beta3, beta4, lambda, Hb,true);
[plotSpanLind, RhoLind] = Markov_LindbladX_ExpStep_1000 ( TSpan,bSize, hbar, mu,beta3, beta4, lambda, Hb,true);

timeLind=toc

nFramesLind=length(plotSpanLind);


expectHLind = zeros(1,nFramesLind);
PurityLind  = zeros(1,nFramesLind);
NormLind = zeros(1,nFramesLind);
CtLind      = zeros(1,nFramesLind);
expXLind    = zeros(1,nFramesLind);
expPLind    = zeros(1,nFramesLind);
varXLind    = zeros(1,nFramesLind);
varPLind    = zeros(1,nFramesLind);
varXPLind    = zeros(1,nFramesLind);

PsiLindPos  = zeros(Nx,nFramesLind);
RhoLindPos  = zeros(Nx,Nx,nFramesLind);

for k = 1 : nFramesLind
    NeCurr   = plotSpanLind(k);
    Hcurr    = H0(NeCurr);
    Rho        = RhoLind(:,:,k);

    expectHLind(k)  = real(trace(Hcurr * Rho));
    % Rho=Rho/trace(Rho);
    NormLind(k)  = real(trace(Rho));
    PurityLind(k)   = trace(Rho^2)/trace(Rho)^2;
    CtLind(k)       = abs(trace(ProjR * Rho*ProjR )) / NormLind(k);
    expXLind(k)     = real(trace(Xhat * Rho));
    expPLind(k)     = real(trace(Phat * Rho));
    varXLind(k)=real(trace(Rho*Xhat^2))-expXLind(k)^2;
    varPLind(k)=real(trace(Rho*Phat^2))-expPLind(k)^2;
    varXPLind(k)=0.5*real(trace(Rho*(Phat*Xhat+Xhat*Phat)))-expPLind(k)*expXLind(k);

    RhoLindPos(:,:,k) = HVector *Rho(1:Nx,1:Nx)* HVector';

end
dotCtLind = gradient(CtLind, plotSpanLind);



% expXLind=expXHam;
% expPLind=expPHam;
save(filenameWorkspace)


%%  ⟨X⟩ SSE subplot
% Export 4 separate panels with LaTeX labels and no titles

lw = 1.5; fs = 25;

% 1) <X>
f1 = figure('Position',[100 100 800 600]); ax1 = axes(f1); hold(ax1,'on');
plot(plotSpanSSE,  expXSSE,         'b-', 'LineWidth', lw);
plot(plotSpanSSE,  expXSSE2,         'm-', 'LineWidth', lw);
plot(plotSpanHam,  expXSchrodinger, 'r-', 'LineWidth', lw);
plot(plotSpanLind, expXLind,        'k--','LineWidth', lw);
xlabel('$N$','Interpreter','latex','FontSize',fs);
ylabel('$\langle\hat  \phi \rangle$','Interpreter','latex','FontSize',fs);
title( '(a)','Interpreter','latex','FontSize',30);
grid on;    % add grid lines
% legend({'SSE','Schr\"odinger','Lindblad'},'Interpreter','latex','Location','best','Box','off');
set(ax1,'TickLabelInterpreter','latex','FontSize',fs,'LineWidth',1.2); axis(ax1,'tight'); box(ax1,'on');
exportgraphics(f1, sprintf('expX_vsNe_lambda_%.2g_Hubble_%.2g.pdf',lambda,Hb), 'ContentType','vector');

% 2) <P>
f2 = figure('Position',[100 100 800 600]); ax2 = axes(f2); hold(ax2,'on');
plot(plotSpanSSE,  expPSSE,         'b-', 'LineWidth', lw);
plot(plotSpanSSE,  expPSSE2,         'm-', 'LineWidth', lw);
plot(plotSpanHam,  expPSchrodinger, 'r-', 'LineWidth', lw);
plot(plotSpanLind, expPLind,        'k--','LineWidth', lw);
xlabel(' $N$','Interpreter','latex','FontSize',fs);
ylabel('$\langle \hat \pi_{\phi} \rangle$','Interpreter','latex','FontSize',fs);
title( '(b)','Interpreter','latex','FontSize',30);
grid on;    % add grid lines
% legend({'SSE','Schr\"odinger','Lindblad'},'Interpreter','latex','Location','best','Box','off');
set(ax2,'TickLabelInterpreter','latex','FontSize',fs,'LineWidth',1.2); axis(ax2,'tight'); box(ax2,'on');
exportgraphics(f2, sprintf('expP_vsNe_lambda_%.2g_Hubble_%.2g.pdf',lambda,Hb), 'ContentType','vector');

% 3) <H>
f3 = figure('Position',[100 100 800 600]); ax3 = axes(f3); hold(ax3,'on');
plot(plotSpanSSE,  expectHSSE,         'b-', 'LineWidth', lw);
plot(plotSpanSSE,  expectHSSE2,         'm-', 'LineWidth', lw);
plot(plotSpanHam,  expectHSchrodinger, 'r-', 'LineWidth', lw);
plot(plotSpanLind, expectHLind,        'k--','LineWidth', lw);
xlabel(' $N$','Interpreter','latex','FontSize',fs);
ylabel('$\langle \hat K_S \rangle$','Interpreter','latex','FontSize',fs);
title( '(c)','Interpreter','latex','FontSize',30);
grid on;    % add grid lines
legend({'SSE (False Vacuum)','SSE (True Vacuum)','Schr\"odinger','Lindblad'},'Interpreter','latex','Location','best','Box','on');
% legend({'SSE','Schr\"odinger','Lindblad'},'Interpreter','latex','Location','best','Box','on');

set(ax3,'TickLabelInterpreter','latex','FontSize',fs,'LineWidth',1.2); axis(ax3,'tight'); box(ax3,'on');
exportgraphics(f3, sprintf('expH_vsNe_lambda_%.2g_Hubble_%.2g.pdf',lambda,Hb), 'ContentType','vector');






%% Panels SSE


[~, idxOrig] = min(abs(plotSpanSSE- targets(1)), [], 2);   % idx(i) indexes p nearest to t(i)
[~, idxZero] = min(abs(plotSpanSSE- targets(2)), [], 2);   % idx(i) indexes p nearest to t(i)
WigImagesSSE(:,:,1)=WigSSE(:,:, idxOrig);
WigImagesSSE(:,:,2)=WigSSE(:,:, idxZero);
WigImagesSSE(:,:,3)=WigSSE(:,:, end);
WigImagesSSE(:,:,4)=WigSSE2(:,:, idxOrig);
WigImagesSSE(:,:,5)=WigSSE2(:,:, idxZero);
WigImagesSSE(:,:,6)=WigSSE2(:,:, end);

% ---------- (a) N_e = -0.75 ----------
fig = figure('Position', [100 100 800 600]);  % wider figure for visibility
ax = axes(fig); hold(ax,'on');
hA = pcolor(ax, x, p, WigImagesSSE(:,:,1));
shading('interp')

colormap(ax,'parula');
contour(ax, xMesh1, pMesh1, Hgrid_at(targets(1)), levels, 'LineColor', 'k');
xlim(ax,[-Xview Xview]); ylim(ax,[-Xview Xview]); axis(ax,'square'); box(ax,'on'); caxis(ax,cax);
xticks(linspace(-Xview,Xview,7))
colorbar(ax,'FontSize',18);
expLine = plot(expXSSE(1:idxOrig), expPSSE(1:idxOrig), 'w-', 'LineWidth', 1); % White line with specified width
text(ax, 0.98, 0.98, fmtText(targets(1), CtSSE(idxOrig)), ...
    'Units','normalized','HorizontalAlignment','right','VerticalAlignment','top', ...
    'Interpreter','latex','FontSize',24,'FontWeight','bold','Color','w');
set(ax, 'FontSize', 20, 'LineWidth', 1.5, 'Box', 'on', 'TickLabelInterpreter', 'latex');
xlabel(ax,'$\phi$','Interpreter','latex','FontSize',25);
ylabel(ax,'$\pi_\phi$','Interpreter','latex','FontSize',25);
title(ax, '(a)','Interpreter','latex','FontSize',30);
box on
exportgraphics(fig,'SSE_X_wigner_panel_a_Ne_-0p75.pdf','ContentType', 'image', 'Resolution', 600);

% ---------- (b) N_e = 0 ----------
fig = figure('Position', [100 100 800 600]);  % wider figure for visibility
ax = axes(fig); hold(ax,'on');
hB = pcolor(ax, x, p, WigImagesSSE(:,:,2));
shading('interp')
colormap(ax,'parula');
contour(ax, xMesh1, pMesh1, Hgrid_at(targets(2)), levels, 'LineColor', 'k');
xlim(ax,[-Xview Xview]); ylim(ax,[-Xview Xview]); axis(ax,'square'); box(ax,'on'); caxis(ax,cax);
xticks(linspace(-Xview,Xview,7))
colorbar(ax,'FontSize',18);
expLine = plot(expXSSE(1:idxZero), expPSSE(1:idxZero), 'w-', 'LineWidth', 1); % White line with specified width
text(ax, 0.98, 0.98, fmtText(targets(2), CtSSE(idxZero)), ...
    'Units','normalized','HorizontalAlignment','right','VerticalAlignment','top', ...
    'Interpreter','latex','FontSize',24,'FontWeight','bold','Color','w');
set(ax, 'FontSize', 20, 'LineWidth', 1.5, 'Box', 'on', 'TickLabelInterpreter', 'latex');
xlabel(ax,'$\phi$','Interpreter','latex','FontSize',25);
ylabel(ax,'$\pi_\phi$','Interpreter','latex','FontSize',25);
title(ax, '(b)','Interpreter','latex','FontSize',30);
box on
exportgraphics(fig,'SSE_X_wigner_panel_b_Ne_0.pdf','ContentType', 'image', 'Resolution', 600);

% ---------- (c) N_e = +1.5 ----------
fig = figure('Position', [100 100 800 600]);  % wider figure for visibility
ax = axes(fig); hold(ax,'on');
hC = pcolor(ax, x, p, WigImagesSSE(:,:,3));
shading('interp')
colormap(ax,'parula');
contour(ax, xMesh1, pMesh1, Hgrid_at(targets(3)), levels, 'LineColor', 'k');
xlim(ax,[-Xview Xview]); ylim(ax,[-Xview Xview]); axis(ax,'square'); box(ax,'on'); caxis(ax,cax);
xticks(linspace(-Xview,Xview,7))
colorbar(ax,'FontSize',18);
expLine = plot(expXSSE(1:end), expPSSE(1:end), 'w-', 'LineWidth', 1); % White line with specified width
text(ax, 0.98, 0.98, fmtText(targets(3),CtSSE(end)), ...
    'Units','normalized','HorizontalAlignment','right','VerticalAlignment','top', ...
    'Interpreter','latex','FontSize',24,'FontWeight','bold','Color','w');
set(ax, 'FontSize', 20, 'LineWidth', 1.5, 'Box', 'on', 'TickLabelInterpreter', 'latex');
xlabel(ax,'$\phi$','Interpreter','latex','FontSize',25);
ylabel(ax,'$\pi_\phi$','Interpreter','latex','FontSize',25);
title(ax, '(c)','Interpreter','latex','FontSize',30);
box on
exportgraphics(fig,'SSE_X_wigner_panel_c_Ne_+0p75.pdf','ContentType', 'image', 'Resolution', 600);

% ---------- (a) N_e = -0.75 ----------
fig = figure('Position', [100 100 800 600]);  % wider figure for visibility
ax = axes(fig); hold(ax,'on');
hA = pcolor(ax, x, p, WigImagesSSE(:,:,4));
shading('interp')
colormap(ax,'parula');
contour(ax, xMesh1, pMesh1, Hgrid_at(targets(1)), levels, 'LineColor', 'k');
xlim(ax,[-Xview Xview]); ylim(ax,[-Xview Xview]); axis(ax,'square'); box(ax,'on'); caxis(ax,cax);
xticks(linspace(-Xview,Xview,7))
colorbar(ax,'FontSize',18);
expLine = plot(expXSSE2(1:idxOrig), expPSSE2(1:idxOrig), 'w-', 'LineWidth', 1); % White line with specified width
text(ax, 0.98, 0.98, fmtText(targets(1), CtSSE2(idxOrig)), ...
    'Units','normalized','HorizontalAlignment','right','VerticalAlignment','top', ...
    'Interpreter','latex','FontSize',24,'FontWeight','bold','Color','w');
set(ax, 'FontSize', 20, 'LineWidth', 1.5, 'Box', 'on', 'TickLabelInterpreter', 'latex');
xlabel(ax,'$\phi$','Interpreter','latex','FontSize',25);
ylabel(ax,'$\pi_\phi$','Interpreter','latex','FontSize',25);
title(ax, '(d)','Interpreter','latex','FontSize',30);
box on
exportgraphics(fig,'SSE_X_wigner_panel2_a_Ne_-0p75.pdf','ContentType', 'image', 'Resolution', 600);

% ---------- (b) N_e = 0 ----------
fig = figure('Position', [100 100 800 600]);  % wider figure for visibility
ax = axes(fig); hold(ax,'on');
hB = pcolor(ax, x, p, WigImagesSSE(:,:,5));
shading('interp')
colormap(ax,'parula');
contour(ax, xMesh1, pMesh1, Hgrid_at(targets(2)), levels, 'LineColor', 'k');
xlim(ax,[-Xview Xview]); ylim(ax,[-Xview Xview]); axis(ax,'square'); box(ax,'on'); caxis(ax,cax);
xticks(linspace(-Xview,Xview,7))
colorbar(ax,'FontSize',18);
expLine = plot(expXSSE2(1:idxZero), expPSSE2(1:idxZero), 'w-', 'LineWidth', 1); % White line with specified width
text(ax, 0.98, 0.98, fmtText(targets(2), CtSSE2(idxZero)), ...
    'Units','normalized','HorizontalAlignment','right','VerticalAlignment','top', ...
    'Interpreter','latex','FontSize',24,'FontWeight','bold','Color','w');
set(ax, 'FontSize', 20, 'LineWidth', 1.5, 'Box', 'on', 'TickLabelInterpreter', 'latex');
xlabel(ax,'$\phi$','Interpreter','latex','FontSize',25);
ylabel(ax,'$\pi_\phi$','Interpreter','latex','FontSize',25);
title(ax, '(e)','Interpreter','latex','FontSize',30);
box on
exportgraphics(fig,'SSE_X_wigner_panel2_b_Ne_0.pdf','ContentType', 'image', 'Resolution', 600);

% ---------- (c) N_e = +1.5 ----------
fig = figure('Position', [100 100 800 600]);  % wider figure for visibility
ax = axes(fig); hold(ax,'on');
hC = pcolor(ax, x, p, WigImagesSSE(:,:,6));
shading('interp')
colormap(ax,'parula');
contour(ax, xMesh1, pMesh1, Hgrid_at(targets(3)), levels, 'LineColor', 'k');
xlim(ax,[-Xview Xview]); ylim(ax,[-Xview Xview]); axis(ax,'square'); box(ax,'on'); caxis(ax,cax);
xticks(linspace(-Xview,Xview,7))
colorbar(ax,'FontSize',18);
expLine = plot(expXSSE2(1:end), expPSSE2(1:end), 'w-', 'LineWidth', 1); % White line with specified width
text(ax, 0.98, 0.98, fmtText(targets(3),CtSSE2(end)), ...
    'Units','normalized','HorizontalAlignment','right','VerticalAlignment','top', ...
    'Interpreter','latex','FontSize',24,'FontWeight','bold','Color','w');
set(ax, 'FontSize', 20, 'LineWidth', 1.5, 'Box', 'on', 'TickLabelInterpreter', 'latex');
xlabel(ax,'$\phi$','Interpreter','latex','FontSize',25);
ylabel(ax,'$\pi_\phi$','Interpreter','latex','FontSize',25);
title(ax, '(f)','Interpreter','latex','FontSize',30);
box on
exportgraphics(fig,'SSE_X_wigner_panel2_c_Ne_+0p75.pdf','ContentType', 'image', 'Resolution', 600);


%% Create a figure with specified position and aspect ratio
close all;

clear flipbook
WigLind = RhoWigner(RhoLindPos, x, p, hbar);

figureHandle = figure('Position', [100, 100, 800, 800 * aspectRatio(2) / aspectRatio(1)]);
Hinit = @(Z,n) (exp(3*plotSpanLind(n))*(-(mu^2/2)*Z(1)^2+(2*mu*beta3/3)*Z(1)^3+((beta4^2-beta3^2)/4)*Z(1)^4)+0.5*Z(2)^2*exp(-3*plotSpanLind(n)))/Hb;

% Initialize pcolor plot with the first set of data
f = pcolor(xMesh, pMesh, WigLind(:,:,1));
xlim([-Xview Xview]);
ylim([-Xview Xview]);
shading interp
colormap("parula")  % Changed colormap to 'parula' for perceptually uniform colors

% Set color axis limits
clim([-0.05 0.15])  % Ensure this range covers both oscillator and inverted oscillator regimes
xticks(linspace(-Xview,Xview,7))
% Add a colorbar with increased font size
colorbar('FontSize', 14);

title( sprintf(...
    'Lindblad  $(\\hat L_{\\phi})$, $\\lambda = %.2f$, $\\tilde \\mu = %.2f$', ...
    lambda, mu/Hb), ...
    'FontSize', 20, 'Interpreter', 'latex' );
% title( sprintf(...
%     'Lindblad , $H = %.2f$', Hb), ...
%     'FontSize', 20, 'Interpreter', 'latex' );
% xlabel('$\phi$', 'FontSize', 25, 'Interpreter', 'latex');
% ylabel('$\pi_{\phi}$', 'FontSize', 25, 'Interpreter', 'latex');

% Customize axis properties for better aesthetics
set(gca, 'FontSize', 16, 'LineWidth', 1.5, 'Box', 'on', 'TickLabelInterpreter', 'latex');
axis square  % Use square axes for equal aspect ratio

% Hold the current plot to allow multiple graphics objects
hold on;

% Initialize an empty line plot for experimental data
expLine = plot(NaN, NaN, 'w-', 'LineWidth', 1); % White line with specified width

% Initialize contour plot with initial Z data
[~, contPlot] = contour(xMesh1, pMesh1, Z_initial, levels, 'LineColor', "k"); % 100 contour levels in black

% Initialize the time display
timeText = text(0.95, 0.95, '', ...
    'Units', 'normalized', ...          % Position relative to axes (0 to 1)
    'HorizontalAlignment', 'right', ... % Align text to the right
    'VerticalAlignment', 'top', ...     % Align text to the top
    'FontSize', 25, ...                  % Set desired font size
    'FontWeight', 'bold', ...            % Make text bold for visibility
    'Color', 'w');                       % Set text color to black
uistack(timeText, 'top');                 % Ensure text is on top

% Main simulation loop
for n = 1:length(WigLind(1,1,:))
    % Retrieve the current simulation time from plotSpanLind
    t = plotSpanLind(n);

    % Update the pcolor plot with new data for the current time step
    set(f, 'CData', WigLind(:,:,n));


    % ---------------------- Dynamic Contour Update ----------------------

    % Compute contour data based on Hinit
    CdatCont = arrayfun(@(xx, yy) Hinit([xx; yy], n), xMesh1, pMesh1);

    % Delete existing contour plot to prevent overlap
    if ~isempty(contPlot) && isgraphics(contPlot)
        delete(contPlot);
    end

    % Recompute and plot new contours with fixed nonlinear levels
    [~, contPlot] = contour(xMesh1, pMesh1, CdatCont, levels, 'LineColor', "k");
    % ---------------------- End of Dynamic Contour Update -------------------

    % Update the experimental line plot with new data point
    xData = get(expLine, 'XData');
    yData = get(expLine, 'YData');
    set(expLine, 'XData', [xData, expXLind(n)], 'YData', [yData, expPLind(n)]);

    % ---------------------- Update Time Display ----------------------

    % Update the text object with the current time, formatted to two decimal places
    set(timeText, 'Interpreter', 'latex', ...
        'String', sprintf('$N = $ %.2f, $\\langle \\hat\\theta_{\\phi^{+}}\\rangle = $ %.2f', t, CtLind(n)));


    % Ensure the time text stays on top of other plot elements
    uistack(timeText, 'top');

    % ---------------------- End of Time Display Update -------------------

    % Refresh the plot to reflect updates
    drawnow;

    % Optional: Capture the frame for a flipbook or video
    flipbook(n) = getframe(gcf);
end

% Save video
LindVideoPath = sprintf('Wigner_X_LindVideo_lambda_%.2g_Hubble_%.2g.avi', lambda, Hb);
writerObj = VideoWriter(LindVideoPath);

writerObj.FrameRate = nFramesLind/30;
open(writerObj);
writeVideo(writerObj, flipbook);
close(writerObj);


%% Panels Lindblad



[~, idxOrig] = min(abs(plotSpanLind- targets(1)), [], 2);   % idx(i) indexes p nearest to t(i)
[~, idxZero] = min(abs(plotSpanLind- targets(2)), [], 2);   % idx(i) indexes p nearest to t(i)
WigImagesLind(:,:,1)=WigLind(:,:, idxOrig);
WigImagesLind(:,:,2)=WigLind(:,:, idxZero);
WigImagesLind(:,:,3)=WigLind(:,:, end);


% ---------- (a) N_e = -0.75 ----------
fig = figure('Position', [100 100 800 600]);  % wider figure for visibility
ax = axes(fig); hold(ax,'on');
hA = pcolor(ax, x, p, WigImagesLind(:,:,1));
shading('interp')
colormap(ax,'parula');
contour(ax, xMesh1, pMesh1, Hgrid_at(targets(1)), levels, 'LineColor', 'k');
xlim(ax,[-Xview Xview]); ylim(ax,[-Xview Xview]); axis(ax,'square'); box(ax,'on'); caxis(ax,cax);
xticks(linspace(-Xview,Xview,7))
colorbar(ax,'FontSize',18);
expLine = plot(expXLind(1:idxOrig), expPLind(1:idxOrig), 'w-', 'LineWidth', 1); % White line with specified width
text(ax, 0.98, 0.98, fmtText(targets(1), CtLind(idxOrig)), ...
    'Units','normalized','HorizontalAlignment','right','VerticalAlignment','top', ...
    'Interpreter','latex','FontSize',24,'FontWeight','bold','Color','w');
set(ax, 'FontSize', 20, 'LineWidth', 1.5, 'Box', 'on', 'TickLabelInterpreter', 'latex');
xlabel(ax,'$\phi$','Interpreter','latex','FontSize',25);
ylabel(ax,'$\pi_\phi$','Interpreter','latex','FontSize',25);
title(ax, '(a)','Interpreter','latex','FontSize',30);
box on
exportgraphics(fig,'Lindwigner_panel_a_Ne_-0p75.pdf','ContentType', 'image', 'Resolution', 600);

% ---------- (b) N_e = 0 ----------
fig = figure('Position', [100 100 800 600]);  % wider figure for visibility
ax = axes(fig); hold(ax,'on');
hB = pcolor(ax, x, p, WigImagesLind(:,:,2));
shading('interp')
colormap(ax,'parula');
contour(ax, xMesh1, pMesh1, Hgrid_at(targets(2)), levels, 'LineColor', 'k');
xlim(ax,[-Xview Xview]); ylim(ax,[-Xview Xview]); axis(ax,'square'); box(ax,'on'); caxis(ax,cax);
xticks(linspace(-Xview,Xview,7))
colorbar(ax,'FontSize',18);
expLine = plot(expXLind(1:idxZero), expPLind(1:idxZero), 'w-', 'LineWidth', 1); % White line with specified width
text(ax, 0.98, 0.98, fmtText(targets(2), CtLind(idxZero)), ...
    'Units','normalized','HorizontalAlignment','right','VerticalAlignment','top', ...
    'Interpreter','latex','FontSize',24,'FontWeight','bold','Color','w');
set(ax, 'FontSize', 20, 'LineWidth', 1.5, 'Box', 'on', 'TickLabelInterpreter', 'latex');
xlabel(ax,'$\phi$','Interpreter','latex','FontSize',25);
ylabel(ax,'$\pi_\phi$','Interpreter','latex','FontSize',25);
title(ax, '(b)','Interpreter','latex','FontSize',30);
box on
exportgraphics(fig,'Lindwigner_panel_b_Ne_0.pdf','ContentType', 'image', 'Resolution', 600);

% ---------- (c) N_e = +1.5 ----------
fig = figure('Position', [100 100 800 600]);  % wider figure for visibility
ax = axes(fig); hold(ax,'on');
hC = pcolor(ax, x, p, WigImagesLind(:,:,3));
shading('interp')
colormap(ax,'parula');
contour(ax, xMesh1, pMesh1, Hgrid_at(targets(3)), levels, 'LineColor', 'k');
xlim(ax,[-Xview Xview]); ylim(ax,[-Xview Xview]); axis(ax,'square'); box(ax,'on'); caxis(ax,cax);
xticks(linspace(-Xview,Xview,7))
colorbar(ax,'FontSize',18);
expLine = plot(expXLind(1:end), expPLind(1:end), 'w-', 'LineWidth', 1); % White line with specified width
text(ax, 0.98, 0.98, fmtText(targets(3),CtLind(end)), ...
    'Units','normalized','HorizontalAlignment','right','VerticalAlignment','top', ...
    'Interpreter','latex','FontSize',24,'FontWeight','bold','Color','w');
set(ax, 'FontSize', 20, 'LineWidth', 1.5, 'Box', 'on', 'TickLabelInterpreter', 'latex');
xlabel(ax,'$\phi$','Interpreter','latex','FontSize',25);
ylabel(ax,'$\pi_\phi$','Interpreter','latex','FontSize',25);
title(ax, '(c)','Interpreter','latex','FontSize',30);
box on
exportgraphics(fig,'Lindwigner_panel_c_Ne_+0p75.pdf','ContentType', 'image', 'Resolution', 600);
% 
% 
% %
% 
close all;
% 
tEnd = cputime - tStart
%% close all
varsToKeep = { ...
    'filenameWorkspace', ...
    'PsiInst','RhoInst', ...
    'PsiSchrodinger','RhoSchrodinger', ...
    'PsiSSE','RhoSSE', ...
    'PsiSSE2','RhoSSE2', ...
    'RhoLind', ...
    'plotSpanInst','plotSpanHam','plotSpanSSE','plotSpanLind', ...
    'N0','Nend','H0', ...
    'Xhat','Phat', ...
    'Hb','mu','beta3','beta4','lambda', ...
    'InstVideoPath','HamVideoPath','SSEVideoPath','SSEVideoPath2','LindVideoPath'};

clearvars('-except', varsToKeep{:});   % nukes x/p grids, HVector, Wigner arrays, etc.

save(filenameWorkspace,'-v7.3');      % big-variable MAT-file

%% Sonification
% ===================== Tunables =====================
% f0     = 65.406;       % base note (Hz), C
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

% Number of frequencies strictly below Nyquist
BSound=min([sum(fsStart  < fs/2), sum(fsMid  < fs/2), sum(fsFinish < fs/2) ])
% ===================== FFmpeg check =====================
[ff_ok, ~] = system('ffmpeg -version');
if ff_ok ~= 0
    error('FFmpeg is not installed or not on PATH.');
end


for n=1:length(plotSpanSSE)
    RhoSSE(:,:,n)=PsiSSE(:,n)*PsiSSE(:,n)';
    RhoSSE2(:,:,n)=PsiSSE2(:,n)*PsiSSE2(:,n)';
end
for n=1:length(plotSpanInst)
    RhoInst(:,:,n)=PsiInst(:,n)*PsiInst(:,n)';
    RhoSchrodinger(:,:,n)=PsiSchrodinger(:,n)*PsiSchrodinger(:,n)';
end



% ===================== Render 30s audio for each trajectory =====================
% [InstAudioPath,  InstAudioDur]  = write_simple_sonification('Inst',  RhoInst, plotSpanInst,  Xhat, Phat, Hb, mu, beta3, beta4,BSound, f0, fs);
% [HamAudioPath,  HamAudioDur]  = write_simple_sonification('Ham',  RhoSchrodinger, plotSpanHam,  Xhat, Phat, Hb, mu, beta3, beta4, BSound, f0, fs);
[SSEAudioPath,  SSEAudioDur]  = write_simple_sonification('SSE',  RhoSSE,         plotSpanSSE,  Xhat, Phat, Hb, mu, beta3,  beta4,BSound, f0, fs);
[SSEAudioPath2,  SSEAudioDur2]  = write_simple_sonification('SSE2',  RhoSSE2,         plotSpanSSE,  Xhat, Phat, Hb, mu, beta3,  beta4,BSound, f0, fs);
% [LindAudioPath, LindAudioDur] = write_simple_sonification('Lind', RhoLind,        plotSpanLind, Xhat, Phat, Hb, mu, beta3, beta4,BSound, f0, fs);
% % (Each *_AudioDur will be 30.000 s)

% ===================== Merge each with its 30s Wigner video =====================

% HamMP4Out  =sprintf( 'Ham_EigenbasisSimple_synced_Hubble_%.2g.mp4', Hb);
SSEMP4Out  =sprintf( 'SSE_X_EigenbasisSimple_lambda_%.2g_Hubble_%.2g.mp4', lambda, Hb);
SSEMP4Out2  =sprintf( 'SSE2_X_EigenbasisSimple_lambda_%.2g_Hubble_%.2g.mp4', lambda, Hb);
% LindMP4Out = sprintf('Lind_X_EigenbasisSimple_lambda_%.2g_Hubble_%.2g.mp4', lambda, Hb);

% Mux without forcing -t; videos are 30 s and WAVs are 30 s → perfect sync
% merge_audio_video_ffmpeg(HamVideoPath,  HamAudioPath,  HamMP4Out,  HamAudioDur);
merge_audio_video_ffmpeg(SSEVideoPath,  SSEAudioPath,  SSEMP4Out,  SSEAudioDur);
merge_audio_video_ffmpeg(SSEVideoPath2,  SSEAudioPath2,  SSEMP4Out2,  SSEAudioDur2);
% merge_audio_video_ffmpeg(LindVideoPath, LindAudioPath, LindMP4Out, LindAudioDur);

disp('All sonifications rendered and merged with videos (30 s each).')


%%
InstVideoPath= sprintf('WignerGroundStateVideo.avi');
InstAudioPath='Inst_EigenbasisSimple.wav';
InstMP4Out  = 'Inst_EigenbasisSimple_synced.mp4';
merge_audio_video_ffmpeg(InstVideoPath,  InstAudioPath,  InstMP4Out,30);

%% Spectrograms


% % --- compute energies & occupations ---
% [~, EnergiesHam, HamOcc]  = HamiltonianEigenrep2(RhoSchrodinger, plotSpanHam,  Xhat, Phat, Hb, mu, beta3, beta4);
% [~, EnergiesSSE, SSEOcc]  = HamiltonianEigenrep2(RhoSSE,         plotSpanSSE,  Xhat, Phat, Hb, mu, beta3,beta4);
% [~, EnergiesSSE2, SSEOcc2]  = HamiltonianEigenrep2(RhoSSE2,         plotSpanSSE,  Xhat, Phat, Hb, mu, beta3,beta4);
% [~, EnergiesLind, LindOcc]= HamiltonianEigenrep2(RhoLind,        plotSpanLind, Xhat, Phat, Hb, mu, beta3, beta4);

%% plotting

% % --- make the three figures in order: Ham, SSE, Lind ---
% makePanel('EnergyLines_panelHam.pdf',  plotSpanHam,  EnergiesHam,  HamOcc,  '(a)');
% makePanel('EnergyLines_panelSSE.pdf',  plotSpanSSE,  EnergiesSSE,  SSEOcc,  '(a)');
% makePanel('EnergyLines_panelSSE2.pdf',  plotSpanSSE,  EnergiesSSE2,  SSEOcc2,  '(a)');
% makePanel('EnergyLines_panelLind.pdf', plotSpanLind, EnergiesLind, LindOcc, '(c)');

%% Exit
% run('MarkovCoarseGrainedLindbladScript.m')
% close all
% save(filenameWorkspace)
exit


%% --- panel maker (local function) ---
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




function Hn = hermite_poly(n, x)
% Recursively compute physicists' Hermite polynomials H_n(x)
if n == 0
    Hn = ones(size(x));
elseif n == 1
    Hn = 2*x;
else
    H0nes = ones(size(x));
    H1 = 2*x;
    for k = 2 : n
        Hn = 2*x .* H1 - 2*(k-1)*H0nes;
        H0nes = H1;
        H1 = Hn;
    end
end
end


function [dN_plot, plotSpan] = dnPPlot(N0, Nend, ScaledN0, nFrames)


    % ----- plotting grid (small) -----
    plotSpan = linspace(N0, Nend, nFrames);

    
    weights=exp(abs(3*(plotSpan+0.5)));
    dN_plot=(weights(1)./weights)*ScaledN0;
end