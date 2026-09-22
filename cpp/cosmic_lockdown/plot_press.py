"""Minimal press artwork from C++ Wigner arrays, without changing their values."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

def main(folder):
    folder=Path(folder)
    meta=json.loads((folder/'metadata.json').read_text())
    x=np.fromfile(folder/'phi.f64',dtype='<f8')
    p=np.fromfile(folder/'momentum.f64',dtype='<f8')
    v=np.fromfile(folder/'potential.f64',dtype='<f8')
    # Crop the view only. Preserve the same axes and linear colour scale in both panels.
    xi=np.flatnonzero(abs(x)<=12)
    pi=np.flatnonzero(abs(p)<=15)
    arrays=[]
    for kind in ['off','on']:
        raw=np.memmap(folder/f'wigner_{kind}.f64',dtype='<f8',mode='r',shape=(len(p),len(x)))
        arrays.append(np.array(raw[np.ix_(pi,xi)]))
    bg='#faf8f3';ink='#203b4b';blue='#006784';amber='#a64d12'
    plt.rcParams.update({'font.family':'DejaVu Sans','text.color':ink,
        'figure.facecolor':bg,'axes.facecolor':bg,'savefig.facecolor':bg,'svg.fonttype':'none'})
    cmap=LinearSegmentedColormap.from_list('signed',['#064e68','#2596ac',bg,'#f3b44e','#b25212'],N=2048)
    bound=max(float(abs(z).max()) for z in arrays)
    norm=Normalize(-bound,bound)
    fig=plt.figure(figsize=(14,8.6))
    fig.text(.5,.925,'Cosmic lockdown',ha='center',fontsize=29,weight='bold')
    pot=fig.add_axes([.25,.682,.50,.18])
    pot.plot(x,v,linewidth=2.2,color=ink)
    pot.fill_between(x,v,-2.1,color='#e7eceb')
    pot.set(xlim=(-7.4,6.4),ylim=(-2.1,2.3));pot.set_axis_off()
    xt=-meta['mu']/(meta['beta4']-meta['beta3'])
    xf=meta['mu']/(meta['beta4']+meta['beta3'])
    pot.text(xt,-2.45,'True vacuum',ha='center',va='top',fontsize=12,clip_on=False)
    pot.text(xf,-2.45,'False vacuum',ha='center',va='top',fontsize=12,color=amber,clip_on=False)
    for j,(left,label,sub,color) in enumerate([
        (.055,'Environment off','Quantum interference',blue),
        (.535,'Environment on','Localised in one well',amber)]):
        fig.text(left+.205,.586,label,ha='center',fontsize=21,weight='bold',color=color)
        ax=fig.add_axes([left,.139,.410,.405])
        dx=x[1]-x[0];dp=p[1]-p[0]
        ax.imshow(arrays[j],origin='lower',extent=[x[xi[0]]-dx/2,x[xi[-1]]+dx/2,p[pi[0]]-dp/2,p[pi[-1]]+dp/2],
            cmap=cmap,norm=norm,aspect='auto',interpolation='antialiased',rasterized=True)
        ax.set(xlim=(-12,12),ylim=(-15,15));ax.set_axis_off()
        fig.text(left+.205,.086,sub,ha='center',fontsize=15,color=color)
    for ext in ['png','svg','pdf']:
        fig.savefig(folder/f'cosmic-lockdown-press.{ext}',dpi=300)
    if meta.get('model') == 'paper':
        model_description='the position-monitoring equations of arXiv:2512.14204v2 (2.7) and (3.15)'
    else:
        model_description='the historical repository generator at commit d382d97 (not the corrected paper coefficients)'
    caption=(
        '# Suggested press caption\n\n'
        'Cosmic lockdown. A quantum field evolves in the same double-well potential with its environment switched off (left) or on (right). '
        'The alternating colours on the left reveal quantum interference. On the right, environmental monitoring has localised one selected realisation on the false-vacuum side. '
        'The study explains how such monitoring suppresses coherent tunnelling; it does not rule out rare crossings over the barrier.\n\n'
        f'Technical provenance: these are new numerical Wigner functions computed by the C++ solver using {model_description}, '
        'not AI-generated patterns or the original published frames. '
        f"H={meta['H']}, mu={meta['mu']}, beta3={meta['beta3']}, beta4={meta['beta4']}, volume={meta['volume']:.12g}, "
        f"lambda=0 / {meta['lambda_on']}, N={meta['N_final']}, monitoring from N={meta['monitoring_start']}. "
        'The top curve is the exact shared potential. The lower panels are phase-space views: horizontal position is field amplitude and vertical position is its canonical momentum. '
        'Both use the same viewport (field -12 to 12, momentum -15 to 15) and the same linear signed colour scale: teal negative, amber positive, ivory zero. '
        'No values were altered to add or remove fringes; the display crops a larger computed domain. '
        'The right panel is a selected stochastic trajectory, not the ensemble average; other trajectories can select the true-vacuum side. '
        f"The displayed seed is {meta['seed'] + meta['selected_index']}; for the supplied final image it was chosen from a 16-seed coarse-grid pilot, then recomputed at full spatial resolution using the same time step. "
        'A snapshot by itself is not a tunnelling-rate measurement.\n'
    )
    (folder/'press-caption.md').write_text(caption,encoding='utf-8')
    print(folder/'cosmic-lockdown-press.png')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder');main(parser.parse_args().folder)
