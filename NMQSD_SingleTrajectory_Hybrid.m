function [plotSpan,PsiStore,RhoStore]=NMQSD_SingleTrajectory_Hybrid(bSize,etaC,Lfac,TSpan,plotSpan,hbar,mu,beta3,beta4,lambda,Hb)
% Compatibility entry point: adaptive rank-factorized coloured-noise dynamics.
% The supplied bath and noise are linearly interpolated; outputs are exact times.
assert(all(diff(TSpan)>0)&&isreal(Lfac)&&size(Lfac,1)==numel(TSpan));
c=struct('workflow','nm-sse','basis',bSize,'hbar',hbar,'mu',mu,'beta3',beta3,...
    'beta4',beta4,'lambda',lambda,'H',Hb,'initial',TSpan(1),'final',TSpan(end),...
    'times',unique([TSpan(1),plotSpan(:).',TSpan(end)]),'step',max(diff(TSpan)),...
    'bathTimes',TSpan(:),'Lfac',Lfac,'etaTimes',TSpan(:),'eta',etaC(:));
r=FockWorkflow(c); [present,index]=ismember(plotSpan,r.times); assert(all(present));
PsiStore=r.psi(:,index); RhoStore=r.rho(:,:,index);
end
