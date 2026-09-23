function [tvals,PsiOut,RhoOut] = SSEDynamics_X2_Sparse(bSize,hbar,mu,beta3,beta4,lambda,Hb,PlotSpan,dNs,noise,volume)
% Callable X2 workflow; supplied noise consists of standard normal samples.
if nargin<10, noise=[]; end
if nargin<11, volume=1; end
if isscalar(dNs), dNs=repmat(dNs,1,max(1,numel(PlotSpan)-1)); end
c=struct('workflow','sse','model','x2','basis',bSize,'hbar',hbar,'mu',mu,'beta3',beta3,'beta4',beta4,'H',Hb,'lambda',lambda,...
    'initial',PlotSpan(1),'final',PlotSpan(end),'times',PlotSpan,...
    'steps',dNs,'volume',volume,'noise',noise);
r=FockWorkflow(c); tvals=r.times; PsiOut=r.psi; RhoOut=r.rho;
end
