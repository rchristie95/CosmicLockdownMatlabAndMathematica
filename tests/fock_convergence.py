"""Optional larger-basis study; separate from fast cross-language CI fixtures."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'cpp'/'cosmic_lockdown'))
from fock_media import read


def main(binary,output):
    binary=Path(binary).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(712)
    fine=rng.standard_normal(400)*np.sqrt(.000005)
    cases=[]
    for workflow,model in [('adiabatic','x'),('closed','x'),('sse','x'),('sse','x2'),('sse','x3'),
                           ('lindblad','x'),('lindblad','x2'),('lindblad','x3'),('nm-sse','x'),('nm-density','x')]:
        values=[]
        for basis in [24,48,72]:
            for step in [.00001,.000005]:
                name=f'{workflow}-{model}-{basis}-{step:g}';folder=output/name
                command=[str(binary),'--workflow',workflow,'--model',model,'--basis',str(basis),
                         '--initial','-.2','--final','-.198','--step',str(step),'--lambda','.05','--frames','2',
                         '--no-wigner','--output',str(folder)]
                if workflow=='sse':
                    stride=2 if step==.00001 else 1
                    noise=fine.reshape(-1,stride).sum(axis=1)/np.sqrt(step)
                    file=output/f'noise-{step:g}.txt';np.savetxt(file,noise,fmt='%.17g')
                    command+=['--noise',str(file)]
                subprocess.run(command,check=True,stdout=subprocess.PIPE)
                meta,rho,obs=read(folder)
                vector=[float(obs[key][-1]) for key in ['mean_phi','variance_phi','Pfalse']]
                values.append({'basis':basis,'step':step,'observables':vector,'min_eigenvalue':float(obs['min_eigenvalue'].min())})
        fine_values=[np.array(v['observables']) for v in values if v['step']==.000005]
        changes=[float(np.linalg.norm(fine_values[k+1]-fine_values[k])) for k in range(2)]
        dt_changes=[float(np.linalg.norm(np.array(values[k]['observables'])-values[k+1]['observables'])) for k in [0,2,4]]
        cases.append({'workflow':workflow,'model':model,'values':values,'basis_changes':changes,'timestep_changes':dt_changes})
        print(workflow,model,'basis changes',changes,'step changes',dt_changes,flush=True)
        (output/'larger-basis-report.json').write_text(json.dumps(cases,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('binary');p.add_argument('--output',default='build/fock-convergence')
    args=p.parse_args();main(args.binary,args.output)
