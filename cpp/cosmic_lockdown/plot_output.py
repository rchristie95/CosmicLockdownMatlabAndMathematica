"""Plot arrays computed by cosmic_lockdown.exe. No simulation or Wigner calculation here."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

def main(folder):
    folder=Path(folder)
    meta=json.loads((folder/'metadata.json').read_text())
    x=np.fromfile(folder/'phi.f64',dtype='<f8')
    p=np.fromfile(folder/'momentum.f64',dtype='<f8')
    v=np.fromfile(folder/'potential.f64',dtype='<f8')
    arrays=[np.fromfile(folder/f'wigner_{kind}.f64',dtype='<f8').reshape(len(p),len(x)) for kind in ['off','on']]
    bg='#f9f8f4'; ink='#203547'; muted='#596975'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':ink,'axes.labelcolor':ink,'xtick.color':muted,'ytick.color':muted,'axes.edgecolor':'#b9c4cb','figure.facecolor':bg,'axes.facecolor':bg,'savefig.facecolor':bg,'svg.fonttype':'none'})
    cmap=LinearSegmentedColormap.from_list('signed',['#075476','#50a1b4',bg,'#f3b353','#b15116'],N=1024)
    scale=max(abs(z).max() for z in arrays); norm=Normalize(-scale,scale)
    fig=plt.figure(figsize=(14,9.4))
    fig.text(.06,.94,'Quantum interference and cosmic lockdown',fontsize=24,weight='bold')
    fig.text(.06,.900,'Same double-well potential. Same initial state. Environmental monitoring switched off or on.',fontsize=11,color=muted)
    pot=fig.add_axes([.27,.687,.46,.145])
    pot.plot(x,v/meta['mu']**4,color=ink,lw=2)
    pot.fill_between(x,v/meta['mu']**4,-33,color='#e5eaed')
    xt=-meta['mu']/(meta['beta4']-meta['beta3']); xf=meta['mu']/(meta['beta4']+meta['beta3'])
    def V(q):return (-.5*meta['mu']**2*q*q+2*meta['beta3']*meta['mu']*q**3/3+(meta['beta4']**2-meta['beta3']**2)*q**4/4)/meta['mu']**4
    pot.scatter([xt,xf,0],[V(xt),V(xf),0],s=22,color=[ink,'#b15116',ink],zorder=5)
    pot.text(xt,-37,'True vacuum',ha='center',va='top',fontsize=10)
    pot.text(xf,-37,'False vacuum',ha='center',va='top',fontsize=10,color='#a65117')
    pot.annotate('Barrier',(0,0),(0,19),ha='center',fontsize=9,arrowprops={'arrowstyle':'-','lw':.7,'color':muted})
    pot.set(xlim=(-7.8,6.7),ylim=(-33,52),ylabel=r'$V(\phi)/\mu^4$')
    pot.set_xticks([]);pot.set_yticks([-25,0,50]);pot.tick_params(length=0,labelsize=8)
    for s in ['top','right','bottom']:pot.spines[s].set_visible(False)
    for j,(a,title,subtitle,description) in enumerate([
        (.065,'ENVIRONMENT OFF','Coherent quantum evolution','Interference fringes between the wells'),
        (.545,'ENVIRONMENT ON','Position monitoring: selected false-vacuum trajectory','A narrow, localised quantum state')]):
        fig.text(a,.607,title,fontsize=16,weight='bold',color='#075476' if j==0 else '#a65117')
        fig.text(a,.579,subtitle,fontsize=10,color=muted)
        ax=fig.add_axes([a,.229,.39,.32])
        dx=x[1]-x[0];dp=p[1]-p[0]
        ax.imshow(arrays[j],origin='lower',extent=[x[0]-dx/2,x[-1]+dx/2,p[0]-dp/2,p[-1]+dp/2],cmap=cmap,norm=norm,aspect='auto',interpolation='bilinear',rasterized=True)
        ax.set(xlim=(-15,15),ylim=(-15,15),xlabel=r'Field amplitude $\phi$')
        ax.set_xticks([-10,0,10]);ax.set_yticks([-10,0,10]);ax.tick_params(length=3,labelsize=9)
        if j==0:ax.set_ylabel(r'Momentum $\pi_\phi$')
        else:ax.set_yticklabels([])
        for s in ['top','right']:ax.spines[s].set_visible(False)
        fig.text(a,.158,description,fontsize=12)
    ca=fig.add_axes([.365,.082,.27,.013]); cb=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=ca,orientation='horizontal')
    cb.set_ticks([-0.3,0,.3]);cb.ax.tick_params(labelsize=8,length=2);cb.outline.set_visible(False)
    fig.text(.5,.106,'SIGNED WIGNER FUNCTION  /  IDENTICAL LINEAR COLOUR SCALE',ha='center',fontsize=8,color=muted)
    model=meta.get('model','repository').capitalize()
    fig.text(.065,.029,f"C++ numerical data • {model} equations: H = {meta['H']:g}, λ = 0 / {meta['lambda_on']:g}, N = {meta['N_final']:g}. Right: one selected stochastic realisation, not an ensemble average.",fontsize=8,color=muted)
    for ext in ['png','svg']:
        fig.savefig(folder/f'cosmic-lockdown-cpp.{ext}',dpi=250)
    print(folder/'cosmic-lockdown-cpp.png')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('folder');main(a.parse_args().folder)
