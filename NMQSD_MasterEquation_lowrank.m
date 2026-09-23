function [plotSpan,RhoStore]=NMQSD_MasterEquation_lowrank(rho,Lfac,Xhat,Phat,TSpan,hbar,mu,beta3,beta4,lambda,Hb)
% Adaptive augmented memory ODE; supplied real factors are linearly interpolated.
assert(all(diff(TSpan)>0)&&isreal(Lfac)&&size(Lfac,1)==numel(TSpan));
indices=unique([1: max(1,floor((numel(TSpan)-1)/999)):numel(TSpan),numel(TSpan)]);
plotSpan=TSpan(indices);
c=struct('workflow','nm-density','basis',size(rho,1),'hbar',hbar,'mu',mu,'beta3',beta3,...
    'beta4',beta4,'lambda',lambda,'H',Hb,'initial',TSpan(1),'final',TSpan(end),...
    'times',plotSpan,'step',max(diff(TSpan)),'rho',rho,'X',Xhat,'P',Phat,...
    'bathTimes',TSpan(:),'Lfac',Lfac);
r=FockWorkflow(c); RhoStore=r.rho;
end
