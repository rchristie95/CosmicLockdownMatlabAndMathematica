function result = FockWorkflow(c)
%FOCKWORKFLOW Common callable numerical workflows, independent of graphics.
% Matching C++ entry point: cosmic_fock. All matrices use the Fock basis.
defaults=struct('workflow','sse','model','x','basis',24,'frames',21,'hbar',1,...
    'mu',.5,'beta3',.025,'beta4',.13,'lambda',.05,'H',5,'initial',-2,...
    'final',1,'step',.001,'volume',0,'omega',1,'times',[],'noise',[],'zeta',[], 'steps',[]);
fields=fieldnames(defaults);
for j=1:numel(fields), if ~isfield(c,fields{j}), c.(fields{j})=defaults.(fields{j}); end, end
nm=startsWith(c.workflow,'nm-'); power=find(strcmp(c.model,{'x','x2','x3'}));
assert(~isempty(power),'Unknown model');
if c.volume==0, c.volume=1; if power==1 && ~nm, c.volume=4*sqrt(2); end, end
assert(c.basis>=2&&c.H>0&&c.step>0&&c.final>=c.initial&&c.hbar>0&&c.mu>0&&c.lambda>=0&&c.omega>0&&c.volume>0);
assert(~any(strcmp(c.workflow,{'sse','lindblad'}))||power~=1||c.hbar==1,'Paper X model requires hbar=1.');
sse=strcmp(c.workflow,'sse'); lind=strcmp(c.workflow,'lindblad'); closed=strcmp(c.workflow,'closed');
adi=strcmp(c.workflow,'adiabatic'); nms=strcmp(c.workflow,'nm-sse');
nmd=any(strcmp(c.workflow,{'nm-density','nm-density-full'}));
assert(sse||lind||closed||adi||nms||nmd,'Unknown workflow');
assert(~nm||power==1,'Non-Markovian workflow uses X.');
mult=3; if sse||adi||nm, mult=2; end
small=operators(c.basis,c,power); big=operators(mult*c.basis,c,power);
psi=ground(H(big,c.initial,c)); rho=project(psi,c.basis,true); rho=rho*rho';
if isfield(c,'rho'), rho=c.rho; end
times=c.times;
if isempty(times), times=linspace(c.initial,c.final,c.frames); if c.initial==c.final, times=c.initial; end, end
assert(abs(times(1)-c.initial)<1e-12&&abs(times(end)-c.final)<1e-12&&all(diff(times)>0));
if isempty(c.zeta), c.zeta=(randn(4,1)+1i*randn(4,1))/sqrt(2); end
assert(numel(c.zeta)==4);
result=struct('times',times,'psi',[],'rho',zeros(c.basis,c.basis,numel(times)),...
    'noise',[],'zeta',c.zeta,'config',c);
if ~lind&&~nmd, result.psi=zeros(c.basis,numel(times)); end
aux=zeros(c.basis,4); accum=zeros(c.basis,c.basis,4); hist={}; historyWeights={}; started=false; ni=0;
if nms&&c.initial>=-1, psi=project(psi,c.basis,true); end
t=c.initial; record(1);
for frame=2:numel(times)
    maxstep=c.step; if ~isempty(c.steps), maxstep=c.steps(frame-1); end
    assert(maxstep>0);
    while ~adi && t<times(frame)-1e-13
        dt=min(maxstep,times(frame)-t);
        if ((sse||lind)&&power~=1 || nms) && t<-1 && t+dt>-1, dt=-1-t; end
        next=t+dt;
        assert(next>t,'Step is too small to advance floating-point time.');
        if closed
            psi=unitary(psi,H(big,t+.5*dt,c),dt,c.hbar);
        elseif sse
            ni=ni+1;
            if isempty(c.noise), z=randn; else, assert(ni<=numel(c.noise),'Noise file too short'); z=c.noise(ni); end
            result.noise(ni)=z;
            if c.lambda==0 || power~=1 && t<-1
                psi=unitary(psi,H(big,t,c),dt,c.hbar);
            else
                psi=stochastic(psi,H(big,t,c),H(big,next,c),sqrt(rate(t,c,power))*big.L,...
                    sqrt(rate(next,c,power))*big.L,dt,sqrt(dt)*z,c.hbar,power);
            end
        elseif lind
            if power~=1 && t<-1
                psi=unitary(psi,H(big,t+.5*dt,c),dt,c.hbar); p=project(psi,c.basis,true); rho=p*p';
            else
                hh=H(small,t+.5*dt,c); ll=sqrt(rate(t+.5*dt,c,power))*small.L; l2=ll'*ll;
                action=@(r) (-1i/c.hbar)*(hh*r-r*hh)+(ll*r*ll'-.5*(l2*r+r*l2))/c.hbar;
                rho=expmv(action,rho,dt,[],(2*norm(hh,1)+2*norm(ll,1)^2)/c.hbar);
                rho=rho/real(trace(rho));
            end
        elseif nms
            if t<-1
                psi=unitary(psi,H(big,t,c),dt,c.hbar);
            else
                if ~started
                    psi=project(psi,c.basis,true); ll=(c.lambda/c.H)*exp(3*t)*small.X;
                    f=(ll*psi-real(psi'*ll*psi)*psi)*dt; aux=f*bath(t,c).'; started=true;
                end
                ln=(c.lambda/c.H)*exp(3*t)*small.X; le=(c.lambda/c.H)*exp(3*next)*small.X;
                row=bath(next,c); eta=row.'*c.zeta(:);
                if isfield(c,'etaTimes'), eta=interp1(c.etaTimes,c.eta,next,'linear'); end
                centered=ln-real(psi'*ln*psi)*eye(c.basis);
                dn=(-1i/c.hbar)*H(small,t,c)+conj(eta)*centered;
                d0=dn*psi-centered*(aux*row); pred=normalize(psi+dt*d0); auxp=aux+dt*dn*aux;
                ce=le-real(pred'*le*pred)*eye(c.basis); de=(-1i/c.hbar)*H(small,next,c)+conj(eta)*ce;
                d1=de*pred-ce*(auxp*row); psi=normalize(psi+.5*dt*(d0+d1));
                homogeneous=aux+.5*dt*(dn*aux+de*auxp);
                f=(le*psi-real(psi'*le*psi)*psi)*dt; aux=homogeneous+f*row.';
            end
        elseif nmd
            ln=(c.lambda/c.H)*exp(3*t)*small.X; lm=(c.lambda/c.H)*exp(3*(t+.5*dt))*small.X;
            kk=ln*rho-rho*ln; past=bath(t,c); row=bath(next,c); sumMemory=zeros(c.basis);
            if strcmp(c.workflow,'nm-density-full')
                hist{end+1}=dt*kk; historyWeights{end+1}=past; %#ok<AGROW>
                for j=1:numel(hist), sumMemory=sumMemory+(row.'*historyWeights{j})*hist{j}; end
            else
                for j=1:4, accum(:,:,j)=accum(:,:,j)+dt*past(j)*kk; sumMemory=sumMemory+row(j)*accum(:,:,j); end
            end
            hn=H(small,t,c); hm=H(small,t+.5*dt,c);
            d0=(-1i/c.hbar)*(hn*rho-rho*hn)-(ln*sumMemory-sumMemory*ln);
            half=rho+.5*dt*d0;
            rho=rho+dt*((-1i/c.hbar)*(hm*half-half*hm)-(lm*sumMemory-sumMemory*lm));
            rho=.5*(rho+rho'); rho=rho/trace(rho);
        end
        t=next;
    end
    t=times(frame); record(frame);
end
if sse&&~isempty(c.noise), assert(ni==numel(c.noise),'Unused noise samples'); end
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
end
function h=H(o,t,c), h=exp(-3*t)*o.P2/(2*c.H*c.volume)+exp(3*t)*c.volume*o.V/c.H; end
function p=ground(h)
[u,e]=eig(full(h)); [~,j]=min(real(diag(e))); p=u(:,j); [~,k]=max(abs(p)); p=p*conj(p(k))/abs(p(k));
end
function p=normalize(p), assert(all(isfinite(p))&&norm(p)>0); p=p/norm(p); end
function p=project(p,b,doNorm), p=p(1:b); if doNorm, p=normalize(p); end, end
function p=unitary(p,h,dt,hbar), p=normalize(expmv((-1i/hbar)*h,p,dt)); end
function g=rate(t,c,power)
if power==1, a=131*pi/(256*c.mu^5*c.volume); elseif power==2, a=pi/(64*c.mu^4); else, a=pi/(8*c.mu^3); end
g=a*c.lambda^2*exp(6*t);
end
function row=bath(t,c)
d=128*(c.H*c.omega)^3;
row=[sqrt(27/d)*cos(c.omega*t);sqrt(27/d)*sin(c.omega*t);sqrt(3/d)*cos(3*c.omega*t);sqrt(3/d)*sin(3*c.omega*t)];
end
function g=diffusion(p,l,hbar)
mean=real(p'*l*p)/real(p'*p); g=(l*p-mean*p)/sqrt(hbar);
end
function f=drift(p,h,l,hbar)
mean=real(p'*l*p)/real(p'*p); w=l*p-mean*p; f=(-1i/hbar)*h*p-.5/hbar*(l*w-mean*w);
end
function p=stochastic(p,h0,h1,l0,l1,dt,dw,hbar,power)
p=normalize(p); f=drift(p,h0,l0,hbar); g=diffusion(p,l0,hbar);
if power~=2, p=normalize(p+dt*f+dw*g); return; end
sq=sqrt(dt); shift=g*(.5*(dw^2-dt)/sq); pred=p+dt*f;
p=normalize(p+.5*dt*(f+drift(pred,h1,l1,hbar))+dw*g+...
    .5*sq*(diffusion(p+shift,l1,hbar)-diffusion(p-shift,l1,hbar)));
end
