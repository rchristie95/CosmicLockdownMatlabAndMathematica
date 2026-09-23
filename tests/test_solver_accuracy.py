"""Independent references and statistical refinement for the production solvers."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from scipy.integrate import solve_ivp, simpson
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'cpp'/'cosmic_lockdown'))
from fock_media import read, operators

def main(binary,output):
    binary=Path(binary).resolve(); output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=True)
    report={}
    def run(name,wf='sse',model='x3',step=.001,basis=6,initial=-.2,final=-.18,coupling=.2,extra=()):
        folder=output/name
        subprocess.run([str(binary),'--workflow',wf,'--model',model,'--basis',str(basis),
            '--initial',str(initial),'--final',str(final),'--step',str(step),'--frames','2','--lambda',str(coupling),
            '--no-wigner','--output',str(folder),*extra],check=True,stdout=subprocess.PIPE)
        return folder,read(folder)
    # Compare GKLS to an independently integrated full, time-dependent generator.
    for model in ['x','x2','x3']:
        folder,(c,rr,oo)=run('gkls-'+model,wf='lindblad',model=model,step=.01,final=.0,coupling=2)
        x,p,v=operators(c); a=np.linalg.matrix_power(x,{'x':1,'x2':2,'x3':3}[model]);a2=a@a
        pref={'x':131*np.pi/(256*c['mu']**5*c['volume']),'x2':np.pi/(64*c['mu']**4),'x3':np.pi/(8*c['mu']**3)}[model]
        def rhs(t,y):
            r=y.reshape(x.shape);h=np.exp(-3*t)*p@p/(2*c['H']*c['volume'])+np.exp(3*t)*c['volume']*v/c['H']
            return (-1j*(h@r-r@h)+pref*c['lambda']**2*np.exp(6*t)*(a@r@a-.5*(a2@r+r@a2))).ravel()/c['hbar']
        exact=solve_ivp(rhs,(-.2,0),rr[0].ravel(),method='DOP853',rtol=2e-12,atol=2e-14)
        assert exact.success
        errors=[]
        for step in ([.001,.0005,.00025] if model=='x3' else [.01,.005,.0025]):
            _,(_,r,o)=run(f'gkls-{model}-{step}',wf='lindblad',model=model,step=step,final=0,coupling=2)
            errors.append(float(np.linalg.norm(r[-1].ravel()-exact.y[:,-1])))
            assert abs(o['trace'][-1]-1)<1e-10 and o['min_eigenvalue'].min()>-1e-11
        assert errors[2]<.3*errors[1] and errors[1]<.3*errors[0],(model,errors)
        report['gkls-'+model]={'reference':'DOP853 full generator','errors':errors}
    # Closed propagation: full 3b Hamiltonian reference, retaining projected norm.
    _,(c,rr,oo)=run('closed',wf='closed',model='x',step=.02,final=.2)
    big=dict(c,basis=3*c['basis']);x,p,v=operators(big)
    def h(t):return np.exp(-3*t)*p@p/(2*c['H']*c['volume'])+np.exp(3*t)*c['volume']*v/c['H']
    _,eig=np.linalg.eigh(h(-.2));psi=eig[:,0]
    ref=solve_ivp(lambda t,y:-1j*h(t)@y,(-.2,.2),psi,method='DOP853',rtol=2e-13,atol=2e-15)
    assert ref.success
    pfinal=ref.y[:c['basis'],-1];target=np.outer(pfinal,pfinal.conj());errors=[]
    for step in [.02,.01,.005]:
        _,(_,r,o)=run(f'closed-{step}',wf='closed',model='x',step=step,final=.2)
        errors.append(float(np.linalg.norm(r[-1]-target)))
    assert errors[1]<.1*errors[0] and errors[2]<max(.1*errors[1],2e-12),errors
    report['closed']={'reference':'DOP853 full 3b Hamiltonian','errors':errors}
    # Shared Brownian paths in the integrated measurement clock. A statistical
    # test replaces the invalid requirement that each path improve monotonically.
    for model in ['x','x2','x3']:
        differences=[]
        for seed in range(24):
            z=np.random.default_rng(300+seed).standard_normal(160)
            dt=.000125;t=-.2+np.arange(160)*dt; q=np.exp(6*t)*np.expm1(6*dt)/6
            increments=np.sqrt(q)*z;states=[]
            for stride in [8,4,2,1]:
                zc=increments.reshape(-1,stride).sum(1)/np.sqrt(q.reshape(-1,stride).sum(1))
                file=output/'noise.txt';np.savetxt(file,zc,fmt='%.17g')
                _,(_,r,o)=run(f'path-{model}-{seed}-{stride}',model=model,step=dt*stride,extra=('--noise',str(file)))
                states.append(r[-1])
            differences.append([np.linalg.norm(s-states[-1])**2 for s in states[:-1]])
        rms=np.sqrt(np.mean(differences,axis=0));assert rms[2]<rms[1]<rms[0],(model,rms)
        report['sse-'+model]={'paths':24,'reference_step':.000125,'rms_density_errors':rms.tolist()}
    # NM density: independent DOP853 augmented equations and direct full-history
    # Simpson quadrature of the converged trajectory, kept only as a test.
    folder,(c,r,o)=run('nm-density',wf='nm-density',model='x',step=.1,final=.1,coupling=20)
    x,p,v=operators(c);n=c['basis'];scale=np.sqrt(np.array([27,27,3,3])/(128*c['H']**3))
    def factors(t):return scale*np.array([np.cos(t),np.sin(t),np.cos(3*t),np.sin(3*t)])
    def hh(t):return np.exp(-3*t)*p@p/(2*c['H'])+np.exp(3*t)*v/c['H']
    def ll(t):return c['lambda']/c['H']*np.exp(3*t)*x
    def comm(a,b):return a@b-b@a
    def rhs(t,y):
        y=y.reshape(5,n,n);dy=np.empty_like(y);f=factors(t);l=ll(t)
        dy[0]=-1j*comm(hh(t),y[0])-comm(l,np.einsum('k,kij->ij',f,y[1:]))
        dy[1:]=f[:,None,None]*comm(l,y[0]);return dy.ravel()
    y0=np.zeros((5,n,n),complex);y0[0]=r[0]
    ref=solve_ivp(rhs,(-.2,.1),y0.ravel(),method='DOP853',rtol=2e-12,atol=2e-14,dense_output=True)
    assert ref.success; final=ref.y[:,-1].reshape(5,n,n)
    error=np.linalg.norm(r[-1]-final[0]);assert error<1e-8,error
    tt=np.linspace(-.2,.1,1001);rr=ref.sol(tt).T.reshape(-1,5,n,n)[:,0]
    integrand=np.array([np.dot(factors(.1),factors(t))*comm(ll(t),a) for t,a in zip(tt,rr)])
    direct=simpson(integrand,x=tt,axis=0);factored=np.einsum('k,kij->ij',factors(.1),final[1:])
    err=np.linalg.norm(direct-factored);assert err<1e-10,err
    assert c['accepted_steps']>0
    report['nm-density']={'reference_error':float(error),'full_history_quadrature_error':float(err),'accepted':c['accepted_steps'],'rejected':c['rejected_steps']}
    folder,(c,r,o)=run('nm-sse',wf='nm-sse',model='x',step=.1,final=.1,coupling=20)
    noise=np.loadtxt(folder/'bath_noise.txt');zeta=noise[:,0]+1j*noise[:,1]
    data=np.fromfile(folder/'psi.f64','<f8');states=(data[::2]+1j*data[1::2]).reshape(2,n)
    y0=np.zeros((n,5),complex);y0[:,0]=states[0]
    def sse_rhs(t,y):
        y=y.reshape(n,5);psi=y[:,0];g=y[:,1:];l=ll(t);row=factors(t)
        mean=np.vdot(psi,l@psi).real/np.vdot(psi,psi).real;centered=l-mean*np.eye(n)
        d=-1j*hh(t)+np.conj(row@zeta)*centered;f=d@psi-centered@g@row
        f-=psi*np.vdot(psi,f).real/np.vdot(psi,psi).real
        return np.column_stack([f,d@g+np.outer(centered@psi,row)]).ravel()
    ref=solve_ivp(sse_rhs,(-.2,.1),y0.ravel(),method='DOP853',rtol=2e-12,atol=2e-14)
    assert ref.success;pf=ref.y[:,-1].reshape(n,5)[:,0]
    error=np.linalg.norm(r[-1]-np.outer(pf,pf.conj()));assert error<1e-8,error
    report['nm-sse']={'reference_error':float(error),'accepted':c['accepted_steps'],'rejected':c['rejected_steps']}
    for wf in ['nm-sse','nm-density']:
        _,(c,r,o)=run('zero-'+wf,wf=wf,model='x',step=.1,final=.1,coupling=0)
        ref=solve_ivp(lambda t,y:(-1j*comm(hh(t),y.reshape(n,n))).ravel(),(-.2,.1),r[0].ravel(),method='DOP853',rtol=2e-12,atol=2e-14)
        assert ref.success
        error=np.linalg.norm(r[-1].ravel()-ref.y[:,-1]);assert error<1e-8,error
        report['zero-'+wf]={'unitary_reference_error':float(error)}
    # Stiff measurement stress, including late-time X^3. No continuum-convergence
    # claim is made for this intentionally coarse basis/time grid.
    for workflow in ['sse','lindblad']:
        _,(c,r,o)=run('stress-'+workflow,wf=workflow,basis=24,initial=-.2,final=1,step=.005,coupling=.5)
        assert np.isfinite(r).all() and max(abs(o['trace']-1))<1e-10 and min(o['min_eigenvalue'])>-1e-10
    (output/'solver-accuracy.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('binary');p.add_argument('--output',default='build/solver-accuracy');a=p.parse_args();main(a.binary,a.output)
