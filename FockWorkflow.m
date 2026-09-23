function result = FockWorkflow(c)
%FOCKWORKFLOW Common callable numerical workflows, independent of graphics.
% Matching C++ entry point: cosmic_fock. All matrices use the Fock basis.
defaults=struct('workflow','sse','model','x','basis',24,'frames',21,'hbar',1,...
    'mu',.5,'beta3',.025,'beta4',.13,'lambda',.05,'H',5,'initial',-2,...
    'final',1,'step',.001,'volume',0,'omega',1,'times',[],'noise',[],'zeta',[], 'steps',[], 'rtol',1e-9,'atol',1e-11);
fields=fieldnames(defaults);
for j=1:numel(fields), if ~isfield(c,fields{j}), c.(fields{j})=defaults.(fields{j}); end, end
nm=startsWith(c.workflow,'nm-'); power=find(strcmp(c.model,{'x','x2','x3'}));
assert(~isempty(power),'Unknown model');
if c.volume==0, c.volume=1; if power==1 && ~nm, c.volume=4*sqrt(2); end, end
assert(c.rtol>0&&c.atol>0&&isfinite(c.rtol)&&isfinite(c.atol));
assert(c.basis>=2&&c.H>0&&c.step>0&&c.final>=c.initial&&c.hbar>0&&c.mu>0&&c.lambda>=0&&c.omega>0&&c.volume>0);
assert(~any(strcmp(c.workflow,{'sse','lindblad'}))||power~=1||c.hbar==1,'Paper X model requires hbar=1.');
sse=strcmp(c.workflow,'sse'); lind=strcmp(c.workflow,'lindblad'); closed=strcmp(c.workflow,'closed');
adi=strcmp(c.workflow,'adiabatic'); nms=strcmp(c.workflow,'nm-sse');
nmd=strcmp(c.workflow,'nm-density');
assert(sse||lind||closed||adi||nms||nmd,'Unknown workflow');
assert(~nm||power==1,'Non-Markovian workflow uses X.');
mult=3; if sse||adi||nm, mult=2; end
small=operators(c.basis,c,power); big=operators(mult*c.basis,c,power);
psi=ground(H(big,c.initial,c)); rho=project(psi,c.basis,true); rho=rho*rho';
if isfield(c,'rho'), rho=c.rho; end
if isfield(c,'X'), small.X=c.X; small.P2=c.P*c.P; small.V=-c.mu^2*c.X^2/2+2*c.beta3*c.mu*c.X^3/3+(c.beta4^2-c.beta3^2)*c.X^4/4; end
times=c.times;
if isempty(times), times=linspace(c.initial,c.final,c.frames); if c.initial==c.final, times=c.initial; end, end
assert(abs(times(1)-c.initial)<1e-12&&abs(times(end)-c.final)<1e-12&&all(diff(times)>0));
if isempty(c.zeta), c.zeta=(randn(4,1)+1i*randn(4,1))/sqrt(2); end
assert(numel(c.zeta)==4);
result=struct('times',times,'psi',[],'rho',zeros(c.basis,c.basis,numel(times)),...
    'noise',[],'zeta',c.zeta,'config',c,'adaptiveStats',[0,0]);
if ~lind&&~nmd, result.psi=zeros(c.basis,numel(times)); end
rank=4; if isfield(c,'Lfac'), rank=size(c.Lfac,2); end
aux=zeros(c.basis,rank); memory=zeros(c.basis,c.basis*(1+rank)); started=false; ni=0;
if nms&&c.initial>=-1, psi=project(psi,c.basis,true); end
t=c.initial; record(1);
for frame=2:numel(times)
    maxstep=c.step; if ~isempty(c.steps), maxstep=c.steps(frame-1); end
    assert(maxstep>0);
    while ~adi && t<times(frame)-1e-13
        dt=min(maxstep,times(frame)-t);
        if nmd || nms&&t>=-1, dt=times(frame)-t; end
        if ((sse||lind)&&power~=1 || nms) && t<-1 && t+dt>-1, dt=-1-t; end
        next=t+dt;
        assert(next>t,'Step is too small to advance floating-point time.');
        if closed
            psi=closed_step(psi,big,t,dt,c);
        elseif sse
            ni=ni+1;
            if isempty(c.noise), z=randn; else, assert(ni<=numel(c.noise),'Noise file too short'); z=c.noise(ni); end
            result.noise(ni)=z;
            q=rate(t,c,power)*expm1(6*dt)/(6*c.hbar);
            if power~=1 && t<-1, q=0; end
            psi=split_state(psi,big,t,dt,c,q,z);
        elseif lind
            if power~=1 && t<-1
                psi=closed_step(psi,big,t,dt,c); p=project(psi,c.basis,true); rho=p*p';
            else
                q=rate(t,c,power)*expm1(6*dt)/(6*c.hbar);
                rho=split_density(rho,small,t,dt,c,q);
            end
        elseif nms
            if t<-1
                psi=closed_step(psi,big,t,dt,c);
            else
                if ~started, psi=project(psi,c.basis,true); started=true; end
                state=[psi,aux];
                [state,stats]=AdaptiveMemoryStep(@trajectory_rhs,state,t,next,maxstep,c.rtol,c.atol);
                psi=state(:,1); aux=state(:,2:end); result.adaptiveStats=result.adaptiveStats+stats;
            end
        elseif nmd
            memory(:,1:c.basis)=rho;
            [memory,stats]=AdaptiveMemoryStep(@density_rhs,memory,t,next,maxstep,c.rtol,c.atol);
            rho=memory(:,1:c.basis); result.adaptiveStats=result.adaptiveStats+stats;
        end
        t=next;
    end
    t=times(frame); record(frame);
end
if sse&&~isempty(c.noise), assert(ni==numel(c.noise),'Unused noise samples'); end
    function row=bath_at(u)
        row=bath(u,c);
        if isfield(c,'Lfac'), row=interp1(c.bathTimes,c.Lfac,u,'linear').'; end
    end
    function dy=trajectory_rhs(u,y)
        p=y(:,1); row=bath_at(u);
        if isfield(c,'etaTimes'), eta=interp1(c.etaTimes,c.eta,u,'linear'); else, eta=row.'*c.zeta(:); end
        ll=(c.lambda/c.H)*exp(3*u)*small.X;
        centered=ll-real(p'*ll*p)/real(p'*p)*eye(c.basis);
        d=(-1i/c.hbar)*H(small,u,c)+conj(eta)*centered;
        f=d*p-centered*(y(:,2:end)*row); f=f-p*real(p'*f)/real(p'*p);
        dy=[f,d*y(:,2:end)+(centered*p)*row.'];
    end
    function dy=density_rhs(u,y)
        ll=(c.lambda/c.H)*exp(3*u)*small.X; row=bath_at(u);
        r=y(:,1:c.basis); k=ll*r-r*ll; memorySum=zeros(c.basis); dy=zeros(size(y));
        for j=1:rank
            cols=j*c.basis+(1:c.basis); memorySum=memorySum+row(j)*y(:,cols); dy(:,cols)=row(j)*k;
        end
        hh=H(small,u,c); dy(:,1:c.basis)=(-1i/c.hbar)*(hh*r-r*hh)-(ll*memorySum-memorySum*ll);
    end
    function record(k)
        if adi, p=project(ground(H(big,t,c)),c.basis,true); r=p*p';
        elseif lind||nmd, p=[]; r=rho;
        else, p=project(psi,c.basis,~closed); r=p*p'; end
        assert(all(isfinite(r(:))),'Non-finite density');
        result.rho(:,:,k)=r; if ~isempty(p), result.psi(:,k)=p; end
    end
end
function o=operators(n,c,power)
a=diag(sqrt(1:n-1),1); o.X=sparse(sqrt(c.hbar/2)*(a+a')); o.P=sparse(1i*sqrt(c.hbar/2)*(a'-a));
o.P2=o.P*o.P; x2=o.X*o.X; x3=x2*o.X;
o.V=-c.mu^2*x2/2+2*c.beta3*c.mu*x3/3+(c.beta4^2-c.beta3^2)*(x2*x2)/4;
o.L=o.X^power;
[o.ux,ex]=eig(full(o.X)); o.x=real(diag(ex));
[o.up,ep]=eig(full(o.P2)); o.p2=real(diag(ep));
o.potential=-c.mu^2*o.x.^2/2+2*c.beta3*c.mu*o.x.^3/3+(c.beta4^2-c.beta3^2)*o.x.^4/4;
o.a=o.x.^power;
end
function h=H(o,t,c), h=exp(-3*t)*o.P2/(2*c.H*c.volume)+exp(3*t)*c.volume*o.V/c.H; end
function p=ground(h)
[u,e]=eig(full(h)); [~,j]=min(real(diag(e))); p=u(:,j); [~,k]=max(abs(p)); p=p*conj(p(k))/abs(p(k));
end
function p=normalize(p), assert(all(isfinite(p))&&norm(p)>0); p=p/norm(p); end
function p=project(p,b,doNorm), p=p(1:b); if doNorm, p=normalize(p); end, end
function g=rate(t,c,power)
if power==1, a=131*pi/(256*c.mu^5*c.volume); elseif power==2, a=pi/(64*c.mu^4); else, a=pi/(8*c.mu^3); end
g=a*c.lambda^2*exp(6*t);
end
function row=bath(t,c)
d=128*(c.H*c.omega)^3;
row=[sqrt(27/d)*cos(c.omega*t);sqrt(27/d)*sin(c.omega*t);sqrt(3/d)*cos(3*c.omega*t);sqrt(3/d)*sin(3*c.omega*t)];
end
function [kt,vt]=clocks(t,dt,c)
kt=exp(-3*t)*(-expm1(-3*dt))/(6*c.H*c.volume*c.hbar);
vt=exp(3*t)*expm1(3*dt)*c.volume/(3*c.H*c.hbar);
end
function p=kinetic(p,o,k), p=o.up*(exp(-1i*k*o.p2).*(o.up'*p)); end
function p=split_state(p,o,t,dt,c,q,z)
[kt,vt]=clocks(t,dt,c); p=o.ux'*kinetic(p,o,.5*kt);
if q>0, p=GaussianMeasurementStep(p,o.a,q,z); end
p=kinetic(o.ux*(exp(-1i*vt*o.potential).*p),o,.5*kt);
end
function p=closed_step(p,o,t,dt,c)
w=1/(2-2^(1/3));
for coefficient=[w,1-2*w,w]
    h=coefficient*dt; p=split_state(p,o,t,h,c,0,0); t=t+h;
end
end
function r=kinetic_density(r,o,k)
d=exp(-1i*k*o.p2); r=o.up'*r*o.up; r=o.up*((d.*r).*d')*o.up';
end
function r=split_density(r,o,t,dt,c,q)
[kt,vt]=clocks(t,dt,c); r=o.ux'*kinetic_density(r,o,.5*kt)*o.ux;
r=r.*exp(-1i*vt*(o.potential-o.potential.')-.5*q*(o.a-o.a.').^2);
r=kinetic_density(o.ux*r*o.ux',o,.5*kt);
end
