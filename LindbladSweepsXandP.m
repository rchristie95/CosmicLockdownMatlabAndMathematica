%% Quantum Inflation Simulations
% and generates a Wigner‐function video of the evolution.

clc;
clearvars;
close all;
tStart = cputime;

%% Physical & Numerical Parameters
hbar      = 1;        % Reduced Planck constant


mu    = 0.5;      % Potential parameter: height scale
beta3=0.025;        % Potential parameter: separation between minima
beta4=0.13;
lambda=0.1;
Hb        =5;     % Hubble scale parameter

Vpot=@(x) (-0.5*mu^2*x.^2 +2*mu*beta3*x.^3/3+(beta4^2-beta3^2)*x.^4/4);

% Time‐integration grid for SSE
N0        =-2;                       % Initial e-fold
Nend  = 1;
% N0        =-1;                       % Initial e-fold
% Nend  = -0.95;
TSpan = linspace(N0, Nend,2);

% Basis sizes
bSize     = 300;       % Fock basis dimension
Nx        = 200;      % Grid dimension for x/p representation

% Wigner video parameters
Xview     = 6;        % Plot limits in phase space

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
H = H0(N0);

% --- small/medium matrix: use eig and sort -------------------------------
[Vec, E] = eig(full(H));                 % E is diagonal
[~, idx] = min(diag(E));                 % position of lowest eigen-value
PsiIn = Vec(:, idx);
PsiIn = PsiIn / norm(PsiIn);             % normalise


%% Zeno-Sweep Hb Lind
% ---------- pre-allocate output cells ----------
HbRange=[5,25];
NZenoHb=length(HbRange);
mutildeRange=mu./HbRange;

CtSweepLind      = cell(NZenoHb,1);
PuritySweepLind  = cell(NZenoHb,1);      % ← fixed spelling
PlotSpanSweepLindHb  = cell(NZenoHb,1);


for n = 1:NZenoHb
    [plotSpanTemp, RhoTemp] = Markov_2Lindblads_ExpStep_1000_InitSchrodinger( TSpan,bSize, hbar,mu, beta3, beta4, lambda, HbRange(n),true);
    nFrames      = numel(plotSpanTemp);    % or however you define it earlier

    % ––– 3. pre-allocate slice-variables –––
    CtSweep    = zeros(1,nFrames);
    PurityLind = zeros(1,nFrames);

    % ––– 4. inner serial loop –––
    for k = 1:nFrames
        RhoCurrent = RhoTemp(:,:,k);

        trRho      = trace(RhoCurrent);                % scalar
        CtSweep(k)  = real(trace(ProjR * RhoCurrent*ProjR )) / real(trRho);
        PurityLind(k) = trace(RhoCurrent*RhoCurrent) / (trRho^2);
    end

    % ––– 5. assign once per worker (sliced outputs) –––
    CtSweepLind{n}    = CtSweep;
    PuritySweepLind{n} = PurityLind;
    PlotSpanSweepLindHb{n} = plotSpanTemp;

end

save('workspaceSweepsXandP')
% delete(gcp('nocreate'))

%%
% ==== Figure (a): Ct ====
figure('Position',[100 100 800 600]); hold on

hLines  = gobjects(NZenoHb,1);
labels  = strings(NZenoHb,1);

for n = 1:NZenoHb
    hLines(n) = plot(PlotSpanSweepLindHb{n}, CtSweepLind{n}, 'LineWidth', 1.5);

    labels(n) = sprintf('$\\tilde{\\mu}= %.3f$', mutildeRange(n));

    endSweepTheta(n) = CtSweepLind{n}(end);
end

xlim([TSpan(1),TSpan(end)]);
xlabel('$N$','FontSize',25,'Interpreter','latex');
ylabel('$P_{\rm false}$','FontSize',25,'Interpreter','latex');
set(gca,'FontSize',20,'LineWidth',1.5,'Box','on','TickLabelInterpreter','latex');

legend(hLines, labels, 'Interpreter','latex','Location','southeast','FontSize',25);

grid on; box on; title('(a)','FontSize',30,'Interpreter','latex');

filename = 'LindbladCt_XandP_HubbleSweepX.pdf';
exportgraphics(gcf, filename, 'ContentType', 'vector');

% ==== Figure (b): Purity ====
figure('Position',[100 100 800 600]); hold on

hLines  = gobjects(NZenoHb,1);
labels  = strings(NZenoHb,1);

for n = 1:NZenoHb
    hLines(n) = plot(PlotSpanSweepLindHb{n}, PuritySweepLind{n}, 'LineWidth', 1.5);

    labels(n) = sprintf('$\\tilde{\\mu}= %.3f$', mutildeRange(n));


    endSweepTheta(n) = PuritySweepLind{n}(end);

end

xlim([TSpan(1),TSpan(end)]);
xlabel('$N$','FontSize',25,'Interpreter','latex');
ylabel('Purity','FontSize',25,'Interpreter','latex');
set(gca,'FontSize',20,'LineWidth',1.5,'Box','on','TickLabelInterpreter','latex');

legend(hLines, labels, 'Interpreter','latex','Location','southwest','FontSize',20);

grid on; box on; title('(b)','FontSize',30,'Interpreter','latex');

filename = 'LindbladPurity_XandP_HubbleSweepX.pdf';
exportgraphics(gcf, filename, 'ContentType', 'vector');


%% ZenoSweepLambda
NZenoLambda        = 5;
LambdaRange  = linspace(0,0.2,NZenoLambda);
% c = parcluster('local');  % sometimes supports NumThreads
% c.NumWorkers = 5;
% c.NumThreads = 5;         % if available in your version
% parpool(c,5);

CtSweepLambdaCell      = cell(NZenoLambda,1);
PuritySweepLambdaCell = cell(NZenoLambda,1);      % ← fixed spelling
PlotSpanSweepLambda = cell(NZenoLambda,1);

% parpool(NZenoLambda);


for n = 1:NZenoLambda
    
    [plotSpanTemp, RhoTemp,~] =Markov_2Lindblads_ExpStep_1000_InitSchrodinger(  PsiIn*PsiIn', TSpan, bSize, hbar,mu, beta3, beta4, LambdaRange(n), Hb,true);

    
    nFrames      = numel(plotSpanTemp);    % or however you define it earlier

    % ––– 3. pre-allocate slice-variables –––
    CtSweep    = zeros(1,nFrames);
    PurityLind = zeros(1,nFrames);

    % ––– 4. inner serial loop –––
    for k = 1:nFrames
        RhoCurrent = RhoTemp(:,:,k);

        trRho      = trace(RhoCurrent);                % scalar
        CtSweep(k)  = real(trace(ProjR * RhoCurrent*ProjR )) / real(trRho);
        PurityLind(k) = trace(RhoCurrent*RhoCurrent) / (trRho^2);
    end

    % ––– 5. assign once per worker (sliced outputs) –––
    CtSweepLambdaCell{n}    = CtSweep;
    PuritySweepLambdaCell{n} = PurityLind;
    PlotSpanSweepLind{n} = plotSpanTemp;

end
tEnd = cputime - tStart

save('workspaceSweepsXandP')


% delete(gcp('nocreate'))




%%

figure('Position', [100 100 800 600]);  % wider figure for visibility
hold on;
for n = 1:NZenoLambda
    % Plot each simulation's Norm vs. PlotSpan, labeling by Gamma value and MFPTSweep value.
    plot(PlotSpanSweepLind{n}, CtSweepLambdaCell{n}, 'LineWidth', 1.5, ...
        'DisplayName', sprintf('$\\lambda= %.3f$', LambdaRange(n)));
    endSweepTheta(n) = CtSweepLambdaCell{n}(end);
end
title('(b)', 'FontSize', 30, 'Interpreter', 'latex');

xlim([TSpan(1),TSpan(end)])
xlabel('$N$', 'FontSize', 25, 'Interpreter', 'latex');
ylabel('$P_{\rm false}$', 'FontSize', 25, 'Interpreter', 'latex');

% Customize axis properties for better aesthetics
set(gca, 'FontSize', 20, 'LineWidth', 1.5, 'Box', 'on', 'TickLabelInterpreter', 'latex');

% Set legend properties: move to bottom right and set font size to 14
hLeg = legend('show');
% set(hLeg, 'Interpreter', 'latex', 'Location', 'southeast', 'FontSize', 12);
set(hLeg, 'Interpreter', 'latex', 'Location', 'southwest', 'FontSize', 25);

hold off;
grid on;    % add grid lines
box on;     % draw a box around the axes
title('(b)', 'FontSize', 30, 'Interpreter', 'latex');

% Export the figure
filename = sprintf('LindbladCt_XandP_LambdaSweep_Hb_%.2g.pdf',Hb);
exportgraphics(gcf, filename, 'ContentType', 'vector');

figure('Position', [100 100 800 600]);  % wider figure for visibility
hold on;

for n = 1:NZenoLambda
    % Plot each simulation's Norm vs. PlotSpan, labeling by Gamma value and MFPTSweep value.
    plot(PlotSpanSweepLind{n}, real(PuritySweepLambdaCell{n}), 'LineWidth', 1.5, 'DisplayName', sprintf('$\\lambda= %.3f$', LambdaRange(n)));
    %     endSweepTheta(n) = PuritySweepLind{n}(end);
end
xlim([TSpan(1),TSpan(end)])
xlabel('$N$', 'FontSize', 25, 'Interpreter', 'latex');
ylabel('Purity', 'FontSize', 25, 'Interpreter', 'latex');

% Customize axis properties for better aesthetics
set(gca, 'FontSize', 20, 'LineWidth', 1.5, 'Box', 'on', 'TickLabelInterpreter', 'latex');

% Set legend properties: move to bottom right and set font size to 14
hLeg = legend('show');
% set(hLeg, 'Interpreter', 'latex', 'Location', 'southeast', 'FontSize', 12);
set(hLeg, 'Interpreter', 'latex', 'Location', 'southwest', 'FontSize', 20);

hold off;
grid on;    % add grid lines
box on;     % draw a box around the axes
title('(c)', 'FontSize', 30, 'Interpreter', 'latex');

% Export the figure
filename = sprintf('LindbladPurity_XandP_LambdaSweep_Hb_%.2g.pdf',Hb);
exportgraphics(gcf, filename, 'ContentType', 'vector');

%% Run the next script
% delete(gcp('nocreate'));




