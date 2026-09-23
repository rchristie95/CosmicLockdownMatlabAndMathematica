function run_position_monitoring_tests
% Small-matrix regression tests; no figure, audio or large simulation is run.
root = fileparts(fileparts(mfilename('fullpath')));
addpath(root);
mu=.5; b3=.025; b4=.13; Hb=5; volume=4*sqrt(2); lambda=.05;

% Factors of volume and two, and the N=0 crossover convention.
[k,u,g,Nstar] = PositionMonitoringCoefficients(0,mu,lambda,Hb,volume);
assert(abs(Nstar)<1e-14);
assert(abs(k-1/(2*Hb*volume))<1e-14);
assert(abs(u-volume/Hb)<1e-14);
assert(abs(g-131*pi*lambda^2/(256*mu^5*volume))<1e-14);
[~,~,gf] = PositionMonitoringCoefficients(1,mu,lambda,Hb);
assert(abs(gf-9.1719798570921913)<1e-12);
shift=log(volume)/3;
legacyLambda=sqrt(2)*lambda/volume^(3/2);
for N=[-2,0,1]
    [k,u,g] = PositionMonitoringCoefficients(N,mu,lambda,Hb,volume);
    assert(abs(exp(-3*(N+shift))/(2*Hb)/k-1)<1e-13);
    assert(abs(exp(3*(N+shift))/Hb/u-1)<1e-13);
    oldGamma=131*pi*legacyLambda^2*exp(6*(N+shift))/(512*mu^5);
    assert(abs(oldGamma/g-1)<1e-13);
end

% New-state normalization, including a deliberately unnormalized input.
psi=[sqrt(.4);sqrt(.6)]; H=[.2,.3;.3,-.1]; L=diag([-1,1]);
dt=.02; dW=.17;
meanL=real(psi'*L*psi); centered=L-meanL*eye(2);
expected=psi+(-1i*H*psi-.5*centered^2*psi)*dt+centered*psi*dW;
expected=expected/norm(expected);
actual=PositionMonitoringEulerStep(7*psi,dt,dW,H,L);
assert(norm(actual-expected)<1e-13);
assert(abs(norm(actual)-1)<1e-14);

% Deterministic Gaussian quadrature of the stochastic update checks that
% its ensemble has the required Lindblad rate, not twice/half that rate.
nq=24; J=diag(sqrt(1:nq-1),1)+diag(sqrt(1:nq-1),-1);
[Q,D]=eig(J); weights=Q(1,:).^2; nodes=diag(D);
gamma=.7; L=sqrt(gamma)*diag([-1,1]); dt=1e-3; average=zeros(2);
for j=1:nq
    p=PositionMonitoringEulerStep(psi,dt,sqrt(dt)*nodes(j),zeros(2),L);
    average=average+weights(j)*(p*p');
end
exact=psi*psi'; exact(1,2)=exact(1,2)*exp(-2*gamma*dt); exact(2,1)=exact(1,2);
assert(norm(average-exact,'fro')<5e-6);

% Actual GKLS solver: monitoring must be active BEFORE N=-1.
span=[-1.501,-1.499]; bSize=4;
[~,closed]=Markov_LindbladX_ExpStep_1000(span,bSize,1,mu,b3,b4,0,Hb,false,volume);
[~,open]=Markov_LindbladX_ExpStep_1000(span,bSize,1,mu,b3,b4,50,Hb,false,volume);
r0=closed(:,:,end); r1=open(:,:,end);
assert(abs(trace(r1)-1)<1e-12);
assert(norm(r1-r1','fro')<1e-11);
assert(min(eig((r1+r1')/2))>-1e-10);
assert(abs(trace(r0*r0)-1)<1e-10);
assert(real(trace(r1*r1)) < real(trace(r0*r0))-1e-5);

% Actual SSE solver: early-time noise changes the state and each output is normalized.
rng(17); [~,p0]=SSEDynamics_X_Sparse(bSize,1,mu,b3,b4,0,Hb,span,2e-5,volume);
rng(17); [~,p1]=SSEDynamics_X_Sparse(bSize,1,mu,b3,b4,50,Hb,span,2e-5,volume);
assert(max(abs(sum(abs(p1).^2,1)-1))<1e-12);
assert(norm(p1(:,end)*p1(:,end)'-p0(:,end)*p0(:,end)','fro')>1e-6);

% Ground-state preparation must include BOTH volume factors, in both helpers.
[~,p]=SchrodingerSingleTrajectory_ExpStep_1000([-2,-2],bSize,1,mu,b3,b4,Hb,volume);
expected=projected_ground(3*bSize,bSize,-2,mu,b3,b4,Hb,volume,false);
assert(norm(p(:,1)*p(:,1)'-expected*expected','fro')<1e-11);
[p,~]=AdiabaticGroundStates(bSize,1,mu,b3,b4,Hb,-2,volume);
expected=projected_ground(2*bSize,bSize,-2,mu,b3,b4,Hb,volume,true);
assert(norm(p(:,1)*p(:,1)'-expected*expected','fro')<1e-9);
fprintf('PASS: volume, crossover, SSE normalization/ensemble, early monitoring, GKLS positivity and ground-state preparation.\n');
end

function p=projected_ground(n,b,N,mu,b3,b4,Hb,volume,normalize)
a=diag(sqrt(1:n-1),1); X=(a+a')/sqrt(2); P=1i*(a'-a)/sqrt(2);
V=-mu^2*X^2/2+2*b3*mu*X^3/3+(b4^2-b3^2)*X^4/4;
K=exp(-3*N)*P^2/(2*Hb*volume)+exp(3*N)*volume*V/Hb;
[U,E]=eig(K); [~,j]=min(diag(E)); p=U(1:b,j);
if normalize, p=p/norm(p); end
end
