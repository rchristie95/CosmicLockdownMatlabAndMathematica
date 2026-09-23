function [plotSpan,RhoStore] = Markov_LindbladX3_ExpStep_1000(TSpan,bSize,hbar,mu,beta3,beta4,lambda,Hb,displayFlag,maxStep,volume)
% 1001 output snapshots; internal steps split exactly at any monitoring switch.
if nargin<10, maxStep=max((TSpan(end)-TSpan(1))/1000,eps); end
if nargin<11, volume=1; end
c=struct('workflow','lindblad','model','x3','basis',bSize,'hbar',hbar,'mu',mu,'beta3',beta3,'beta4',beta4,'H',Hb,'lambda',lambda,...
    'initial',TSpan(1),'final',TSpan(end),'frames',1001,'step',maxStep,'volume',volume);
r=FockWorkflow(c); plotSpan=r.times; RhoStore=r.rho;
if displayFlag, fprintf('X3 GKLS: %d snapshots, final trace %.12g\n',numel(plotSpan),real(trace(RhoStore(:,:,end)))); end
end
