function [PsiSmallInst,RhoSmallInst] = AdiabaticGroundStates(bSize,hbar,mu,beta3,beta4,Hb,plotSpanInst,volume)
% Ground states in 2*bSize, projected and normalized in bSize.
if nargin<8, volume=1; end
c=struct('workflow','adiabatic','model','x','basis',bSize,'hbar',hbar,'mu',mu,'beta3',beta3,'beta4',beta4,'H',Hb,'lambda',0,...
    'initial',plotSpanInst(1),'final',plotSpanInst(end),'times',plotSpanInst,'volume',volume);
r=FockWorkflow(c); PsiSmallInst=r.psi; RhoSmallInst=r.rho;
end
