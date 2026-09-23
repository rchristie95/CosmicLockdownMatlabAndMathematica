function [tvals,PsiOut,RhoOut] = SSEDynamics_X_Sparse(bSize,hbar,mu,beta3,beta4,lambda,Hb,PlotSpan,dNs,volume,noise)
% Callable X workflow; supplied noise consists of standard normal samples.
if nargin<10, volume=4*sqrt(2); end
if nargin<11, noise=[]; end
if isscalar(dNs), dNs=repmat(dNs,1,max(1,numel(PlotSpan)-1)); end
c=struct('workflow','sse','model','x','basis',bSize,'hbar',hbar,'mu',mu,'beta3',beta3,'beta4',beta4,'H',Hb,'lambda',lambda,...
    'initial',PlotSpan(1),'final',PlotSpan(end),'times',PlotSpan,...
    'steps',dNs,'volume',volume,'noise',noise);
r=FockWorkflow(c); tvals=r.times; PsiOut=r.psi; RhoOut=r.rho;
end
