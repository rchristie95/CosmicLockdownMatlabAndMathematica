function [plotSpan,PsiStore,RhoStore] = SchrodingerSingleTrajectory_ExpStep_1000(TSpan,bSize,hbar,mu,beta3,beta4,Hb,volume,maxStep)
% Big basis 3*bSize; projected output is intentionally NOT renormalized.
if nargin<8, volume=1; end
if nargin<9, maxStep=max((TSpan(end)-TSpan(1))/1000,eps); end
c=struct('workflow','closed','model','x','basis',bSize,'hbar',hbar,'mu',mu,'beta3',beta3,'beta4',beta4,'H',Hb,'lambda',0,...
    'initial',TSpan(1),'final',TSpan(end),'frames',1001,'step',maxStep,'volume',volume);
r=FockWorkflow(c); plotSpan=r.times; PsiStore=r.psi; RhoStore=r.rho;
end
