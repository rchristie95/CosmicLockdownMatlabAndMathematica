function run_fock_port_tests(folder)
% Cross-language tests against corrected, independently written MATLAB code.
root=fileparts(fileparts(mfilename('fullpath'))); addpath(root);
if nargin<1, folder=fullfile(root,'build','fock-verification'); end
names=jsondecode(fileread(fullfile(folder,'fixtures.json')));
report=cell(numel(names),4);
for k=1:numel(names)
    name=names{k}; dir=fullfile(folder,name);
    meta=jsondecode(fileread(fullfile(dir,'metadata.json')));
    c=meta; c.times=readmatrix(fullfile(dir,'observables.csv')); c.times=c.times(:,1).';
    c.steps=meta.interval_steps;
    noiseFile=fullfile(dir,'noise.txt'); if isfile(noiseFile), c.noise=readmatrix(noiseFile); end
    noise=readmatrix(fullfile(dir,'bath_noise.txt')); c.zeta=noise(:,1)+1i*noise(:,2);
    result=FockWorkflow(c); expected=readcomplex(fullfile(dir,'rho.f64'),[c.basis,c.basis,numel(c.times)]);
    delta=result.rho-expected;
    err=norm(delta(:))/max(1,norm(expected(:)));
    assert(err<1e-7,'%s trajectory mismatch %.12g',name,err);
    initialError=norm(result.rho(:,:,1)-expected(:,:,1),'fro'); assert(initialError<1e-9);
    op=readcomplex(fullfile(dir,'operators.f64'),[c.basis,c.basis,4]);
    a=diag(sqrt(1:c.basis-1),1); x=sqrt(c.hbar/2)*(a+a'); p=1i*sqrt(c.hbar/2)*(a'-a);
    v=-c.mu^2*x^2/2+2*c.beta3*c.mu*x^3/3+(c.beta4^2-c.beta3^2)*x^4/4;
    hh=exp(-3*c.initial)*p^2/(2*c.H*c.volume)+exp(3*c.initial)*c.volume*v/c.H;
    power=find(strcmp(c.model,{'x','x2','x3'})); want=cat(3,x,p,hh,x^power);
    operatorError=norm(op(:)-want(:))/max(1,norm(want(:))); assert(operatorError<1e-9);
    if c.has_psi
        psi=readcomplex(fullfile(dir,'psi.f64'),[c.basis,numel(c.times)]);
        for j=1:numel(c.times)
            overlap=result.psi(:,j)'*psi(:,j); phase=exp(1i*angle(overlap));
            assert(norm(phase*result.psi(:,j)-psi(:,j))<1e-7);
        end
    end
    if strcmp(name,'variable-step')
        [tt,pp,rr]=SSEDynamics_X3_Sparse(c.basis,c.hbar,c.mu,c.beta3,c.beta4,c.lambda,c.H,c.times,c.steps,c.noise,c.volume);
        assert(norm(rr(:)-expected(:))<1e-7&&max(abs(tt-c.times))<1e-13&&all(isfinite(pp(:))));
    end
    if startsWith(name,'sse-')
        if power==1
            [~,~,rr]=SSEDynamics_X_Sparse(c.basis,c.hbar,c.mu,c.beta3,c.beta4,c.lambda,c.H,c.times,c.step,c.volume,c.noise);
        else
            fun=str2func(sprintf('SSEDynamics_X%d_Sparse',power));
            [~,~,rr]=fun(c.basis,c.hbar,c.mu,c.beta3,c.beta4,c.lambda,c.H,c.times,c.step,c.noise,c.volume);
        end
        assert(norm(rr(:)-expected(:))<1e-7);
    end
    if startsWith(name,'adiabatic-')
        [~,rr]=AdiabaticGroundStates(c.basis,c.hbar,c.mu,c.beta3,c.beta4,c.H,c.times,c.volume);
        assert(norm(rr(:)-expected(:))<1e-7);
    end
    report(k,:)={name,err,initialError,operatorError};
    fprintf('PASS MATLAB/C++ %s: relative trajectory error %.3g\n',name,err);
end
writecell([{'workflow','trajectory_error','initial_error','operator_error'};report],fullfile(folder,'matlab-parity.csv'));

% Actual legacy public GKLS/closed signatures (1000 integration intervals).
c=struct('workflow','closed','model','x','basis',4,'hbar',1,'mu',.5,'beta3',.025,'beta4',.13,...
    'H',5,'volume',4*sqrt(2),'lambda',.2,'initial',-.201,'final',-.199,'frames',1001,'step',.002/1000);
r=FockWorkflow(c);
[tt,pp,rr]=SchrodingerSingleTrajectory_ExpStep_1000([c.initial,c.final],4,1,.5,.025,.13,5,c.volume);
assert(norm(rr(:)-r.rho(:))<1e-10&&max(abs(tt-r.times))<1e-13&&norm(pp-r.psi,'fro')<1e-10);
for model={'x','x2','x3'}
    c.model=model{1}; c.workflow='lindblad'; c.volume=1;
    if strcmp(c.model,'x'), c.volume=4*sqrt(2); end
    r=FockWorkflow(c);
    if strcmp(c.model,'x')
        [tt,rr]=Markov_LindbladX_ExpStep_1000([c.initial,c.final],4,1,.5,.025,.13,.2,5,false,c.volume);
    else
        fun=str2func(['Markov_Lindblad',upper(c.model),'_ExpStep_1000']);
        [tt,rr]=fun([c.initial,c.final],4,1,.5,.025,.13,.2,5,false,c.step,c.volume);
    end
    assert(norm(rr(:)-r.rho(:))<1e-10&&max(abs(tt-r.times))<1e-13);
end

% Independently retained NM functions: uniform AND nonuniform memory grids.
for nonuniform=[false,true]
    c=struct('workflow','nm-density','model','x','basis',4,'initial',-.2,'final',-.18,...
        'frames',21,'step',.001,'lambda',2,'H',5,'omega',1,'volume',1);
    times=linspace(c.initial,c.final,21);
    if nonuniform, times=c.initial+.02*linspace(0,1,21).^1.3; end
    c.times=times; c.steps=diff(times); r=FockWorkflow(c);
    a=diag(sqrt(1:3),1); x=(a+a')/sqrt(2); p=1i*(a'-a)/sqrt(2);
    lf=[sqrt(27/(128*5^3))*cos(times(:)),sqrt(27/(128*5^3))*sin(times(:)),...
        sqrt(3/(128*5^3))*cos(3*times(:)),sqrt(3/(128*5^3))*sin(3*times(:))];
    [ta,ra]=NMQSD_MasterEquation_lowrank(r.rho(:,:,1),lf,x,p,times,1,.5,.025,.13,2,5);
    assert(norm(ra(:)-r.rho(:))<1e-8); % sampled bath interpolation error
    assert(max(abs(ta-times))<1e-13);
    c.workflow='nm-sse'; c.zeta=[.1+.2i;-.3+.1i;.2-.1i;.1-.2i]; r=FockWorkflow(c);
    [tn,~,rn]=NMQSD_SingleTrajectory_Hybrid(4,lf*c.zeta,lf,times,times,1,.5,.025,.13,2,5);
    assert(norm(rn(:)-r.rho(:))<1e-8&&max(abs(tn-times))<1e-13);
end
fprintf('PASS: public wrappers, adaptive non-Markovian wrappers.\n');

% Existing FFT Wigner helpers against an independent oscillator Gaussian and
% a pure-state density, with sampled coefficients including sqrt(dx).
n=64; dx=sqrt(2*pi/n); x=(-n/2:n/2-1)*dx; p=x;
psi=exp(-x(:).^2/2)*sqrt(dx)/pi^.25;
wp=PsiWigner(psi,x,p,1); wr=RhoWigner(psi*psi',x,p,1);
exact=exp(-p(:).^2-x.^2)/pi;
assert(max(abs(wp(:)-exact(:)))<1e-10&&norm(wp(:)-wr(:))<1e-10);
assert(abs(sum(wp(:))*dx^2-1)<1e-10);
fprintf('PASS: legacy pure/density Wigner helpers, Gaussian normalization.\n');
end
function result=readcomplex(file,shape)
fid=fopen(file,'rb'); assert(fid>=0); close=onCleanup(@()fclose(fid)); %#ok<NASGU>
data=fread(fid,Inf,'double=>double',0,'ieee-le'); result=reshape(data(1:2:end)+1i*data(2:2:end),shape);
end
