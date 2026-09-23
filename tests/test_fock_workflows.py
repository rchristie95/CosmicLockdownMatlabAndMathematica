"""End-to-end, analytic, refinement and media checks; produces MATLAB fixtures."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from scipy.linalg import expm
from scipy.integrate import trapezoid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'cpp' / 'cosmic_lockdown'))
from fock_media import read, operators, analytic_wigner, render


def main(binary, output, media=False):
    binary = Path(binary).resolve(); output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    report = {'cases': [], 'refinement': [], 'analytic': {}, 'media': []}
    large=np.zeros((400,400),complex);large[0,0]=1;wide=np.array([0.,10.,28.])
    expected=np.exp(-wide[:,None]**2-wide[None,:]**2)/np.pi
    assert np.max(abs(analytic_wigner(large,wide)-expected))<1e-14
    large[:]=0;large[-1,-1]=1
    large_wigner=analytic_wigner(large,wide)
    assert np.isfinite(large_wigner).all() and abs(large_wigner[0,0]+1/np.pi)<1e-12
    report['analytic']['large_basis_wigner']=400
    def run(name, workflow='sse', model='x', initial=-.2, final=-.18, step=.001,
            basis=6, frames=3, coupling=.2, extra=(), wigner=False):
        folder = output / name
        command = [str(binary), '--output', str(folder), '--workflow', workflow, '--model', model,
                   '--initial', str(initial), '--final', str(final), '--step', str(step), '--basis', str(basis),
                   '--frames', str(frames), '--lambda', str(coupling), '--extent', '8', '--grid', '65']
        if not wigner: command += ['--no-wigner']
        subprocess.run(command + list(extra), check=True, stdout=subprocess.PIPE, text=True)
        if workflow == 'sweep':
            return folder
        meta, rho, obs = read(folder)
        assert np.isfinite(rho).all() and np.isfinite(obs.tolist()).all()
        assert np.max(abs(rho - rho.conj().transpose(0, 2, 1))) < 1e-10
        if workflow != 'closed': assert np.max(abs(obs['trace'] - 1)) < 1e-10
        if not workflow.startswith('nm-density'):
            assert obs['min_eigenvalue'].min() > -1e-10
        assert meta['ground_relative_residual'] < 1e-12
        report['cases'].append({'name': name, 'workflow': workflow, 'model': model,
                                'min_eigenvalue': float(obs['min_eigenvalue'].min())})
        return folder

    fixtures = []
    for model in ['x', 'x2', 'x3']:
        for workflow in ['adiabatic', 'closed', 'sse', 'lindblad']:
            folder = run(f'{workflow}-{model}', workflow, model, wigner=True)
            fixtures.append(folder.name)
            meta, rho, obs = read(folder)
            exported = np.fromfile(folder / 'wigner.f64', '<f8').reshape(3, 65, 65)
            axis = np.fromfile(folder / 'phase_axis.f64', '<f8')
            ref = analytic_wigner(rho[-1], axis)
            assert np.max(abs(exported[-1] - ref)) < 1e-12
            mass = trapezoid(trapezoid(exported[-1], axis), axis)
            assert abs(mass - obs['trace'][-1]) < 1e-7
    for workflow in ['nm-sse', 'nm-density', 'nm-density-full']:
        folder = run(workflow, workflow, coupling=2, wigner=True)
        fixtures.append(folder.name)
    a = read(output / 'nm-density')[1]; b = read(output / 'nm-density-full')[1]
    assert np.max(abs(a - b)) < 1e-12
    report['analytic']['memory_lowrank_full_error'] = float(np.max(abs(a - b)))
    # Exact switch splitting, supplied nonuniform output times and interval steps.
    for workflow in ['sse', 'lindblad', 'nm-sse']:
        folder = run('switch-' + workflow, workflow, 'x2' if workflow != 'nm-sse' else 'x',
                     initial=-1.003, final=-.996, step=.002, extra=('--times', '-1.003,-1.001,-.996'), coupling=2)
        fixtures.append(folder.name)
    folder = run('variable-step', 'sse', 'x3', extra=('--steps', '.002,.001'))
    fixtures.append(folder.name)
    # Exactly supplied random samples must reproduce the same state.
    replay = run('noise-replay', extra=('--noise', str(output / 'sse-x' / 'noise.txt')))
    assert np.array_equal(read(replay)[1], read(output / 'sse-x')[1])
    # Independent one-step midpoint exponential for a lambda=0 GKLS state.
    folder = run('zero-channel', 'lindblad', final=-.199, coupling=0, frames=2)
    meta, rho, obs = read(folder); x, p, v = operators(meta)
    mid = (meta['initial'] + meta['final']) / 2
    h = np.exp(-3*mid)*p@p/(2*meta['H']*meta['volume']) + np.exp(3*mid)*meta['volume']*v/meta['H']
    u = expm(-1j*.001*h)
    error = np.linalg.norm(rho[-1] - u@rho[0]@u.conj().T)
    assert error < 1e-12; report['analytic']['unitary_density_error'] = float(error)
    fixtures.append(folder.name)
    # Time and basis refinements are measured separately; no claim that small
    # CI bases reproduce the full paper's continuum physics.
    for workflow, model in [('closed','x'),('sse','x'),('sse','x2'),('sse','x3'),
                            ('lindblad','x'),('lindblad','x2'),('lindblad','x3'),
                            ('nm-sse','x'),('nm-density','x')]:
        sizes = []
        fine_noise = np.random.default_rng(1729).normal(size=80) * np.sqrt(.00025)
        for j, step in enumerate([.001,.0005,.00025]):
            extra = []
            if workflow == 'sse':
                stride = [4,2,1][j]
                z = fine_noise.reshape(-1,stride).sum(axis=1)/np.sqrt(step)
                noise_file = output / f'refine-noise-{j}.txt'; np.savetxt(noise_file,z,fmt='%.17g')
                extra = ['--noise',str(noise_file)]
            folder = run(f'refine-{workflow}-{model}-{j}',workflow,model,step=step,extra=extra)
            sizes.append(read(folder)[1][-1])
        d1=np.linalg.norm(sizes[0]-sizes[1]); d2=np.linalg.norm(sizes[1]-sizes[2])
        assert d2 <= max(1.2*d1,1e-11), (workflow,model,d1,d2)
        bases=[]
        for basis in [8,12,16]:
            folder=run(f'basis-{workflow}-{model}-{basis}',workflow,model,basis=basis,step=.00025,
                       extra=extra if workflow=='sse' else ())
            _,rr,oo=read(folder)
            bases.append(np.array([oo['mean_phi'][-1],oo['variance_phi'][-1],oo['Pfalse'][-1]]))
        report['refinement'].append({'workflow':workflow,'model':model,'dt_errors':[float(d1),float(d2)],
            'basis_8_12_observable_change':float(np.linalg.norm(bases[0]-bases[1])),
            'basis_12_16_observable_change':float(np.linalg.norm(bases[1]-bases[2])),
            'interpretation':'finite-basis short-run checks; basis convergence is measured, not assumed'})
    # Colour-noise covariance independently equals the stated cosine kernel.
    tt=np.linspace(-1,1,7); om=1; hb=5
    factors=np.stack([np.sqrt(27/(128*hb**3))*np.cos(tt),np.sqrt(27/(128*hb**3))*np.sin(tt),
                      np.sqrt(3/(128*hb**3))*np.cos(3*tt),np.sqrt(3/(128*hb**3))*np.sin(3*tt)],axis=1)
    delta=tt[:,None]-tt[None,:]
    kernel=3*(9*np.cos(om*delta)+np.cos(3*om*delta))/(128*hb**3*om**3)
    assert np.max(abs(factors@factors.T-kernel)) < 1e-16
    report['analytic']['bath_covariance_factorization_error']=float(np.max(abs(factors@factors.T-kernel)))
    for sweep,values in [('lambda','0,.05,.1'),('hubble','0,.25,.5,5')]:
        folder=run('sweep-'+sweep,'sweep',extra=('--sweep',sweep,'--values',values))
        render(folder)
    # Every retained top-level numerical driver has a headless rendering route.
    for name in ['sse-x','sse-x2','sse-x3','lindblad-x','lindblad-x2','lindblad-x3','nm-sse']:
        render(output/name,media=media,duration=.3,fs=16000)
        if media:
            from scipy.io import wavfile
            sample_rate,wave=wavfile.read(output/name/'sonification.wav')
            assert wave.shape==(4800,2) and sample_rate==16000
            assert (output/name/'sonified.mp4').stat().st_size>1000
            report['media'].append(name)
    # Bad options/noise fail explicitly, not via a silent fallback.
    bad=subprocess.run([str(binary),'--hubble','0','--no-wigner','--output',str(output/'invalid')],capture_output=True)
    assert bad.returncode != 0
    (output/'fixtures.json').write_text(json.dumps(fixtures,indent=2))
    (output/'verification.json').write_text(json.dumps(report,indent=2))
    print(f'PASS: {len(report["cases"])} runs, analytic checks, refinements and {len(report["media"])} media exports',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('binary');p.add_argument('--output',default='build/fock-verification');p.add_argument('--media',action='store_true')
    args=p.parse_args();main(args.binary,args.output,args.media)
