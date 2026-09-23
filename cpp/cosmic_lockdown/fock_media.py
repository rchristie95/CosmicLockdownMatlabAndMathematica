"""Plots, signed Wigner movies and eigenbasis sonification of cosmic_fock data."""
import argparse
import json
import subprocess
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from scipy.special import eval_genlaguerre, gammaln
from scipy.io import wavfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.ticker import MaxNLocator


def read(folder):
    folder = Path(folder)
    meta = json.loads((folder / 'metadata.json').read_text())
    b, frames = meta['basis'], meta['frames']
    raw = np.fromfile(folder / 'rho.f64', dtype='<c16')
    rho = raw.reshape(frames, b, b).transpose(0, 2, 1)
    obs = np.atleast_1d(np.genfromtxt(folder / 'observables.csv', delimiter=',', names=True))
    return meta, rho, obs


def operators(meta):
    b, h = meta['basis'], meta['hbar']
    a = np.diag(np.sqrt(np.arange(1, b)), 1)
    x = np.sqrt(h / 2) * (a + a.T)
    p = 1j * np.sqrt(h / 2) * (a.T - a)
    x2 = x @ x
    v = -meta['mu'] ** 2 * x2 / 2 + 2 * meta['beta3'] * meta['mu'] * x2 @ x / 3
    v += (meta['beta4'] ** 2 - meta['beta3'] ** 2) * x2 @ x2 / 4
    return x, p, v


def analytic_wigner(rho, axis, hbar=1):
    """Independent vectorized Fock formula; array order is momentum, field."""
    xx, pp = np.meshgrid(axis, axis)
    radius = (xx ** 2 + pp ** 2) / hbar
    z = np.sqrt(2 / hbar) * (xx - 1j * pp)
    value = np.zeros_like(radius)
    if len(rho) > 64:
        # Scaled recurrence avoids overflow of z**k and factorial ratios.
        # The small-basis path below remains an independent SciPy cross-check.
        angle = np.arctan2(-pp, xx)
        for d in range(len(rho)):
            with np.errstate(divide='ignore'):
                logscale = -radius + (d * .5 * np.log(2 * radius) if d else 0) - .5 * gammaln(d + 1)
            phase = np.exp(1j * d * angle)
            previous = np.zeros_like(radius); current = np.ones_like(radius)
            for n in range(len(rho) - d):
                with np.errstate(divide='ignore'):
                    amplitude = np.sign(current) * np.exp(logscale + np.log(abs(current)))
                value += (2 * (rho[n + d, n] * phase).real if d else rho[n, n].real) * amplitude
                denominator = np.sqrt((n + 1) * (n + d + 1))
                following = -(2 * n + 1 + d - 2 * radius) * current / denominator - np.sqrt(n * (n + d)) * previous / denominator
                previous, current = current, following
                scale = np.maximum(abs(previous), abs(current))
                high = scale > 1e100; low = (scale > 0) & (scale < 1e-100)
                previous[high] *= 1e-100; current[high] *= 1e-100; logscale[high] += 100 * np.log(10.)
                previous[low] *= 1e100; current[low] *= 1e100; logscale[low] -= 100 * np.log(10.)
        if not np.isfinite(value).all():
            raise ArithmeticError('Wigner recurrence failed')
        return value / (np.pi * hbar)
    for n in range(len(rho)):
        value += rho[n, n].real * (-1) ** n * eval_genlaguerre(n, 0, 2 * radius)
        for m in range(n + 1, len(rho)):
            coefficient = np.exp((gammaln(n + 1) - gammaln(m + 1)) / 2)
            value += 2 * (rho[m, n] * z ** (m - n)).real * coefficient * (-1) ** n * eval_genlaguerre(n, m - n, 2 * radius)
    return value * np.exp(-radius) / (np.pi * hbar)


def eigenbasis(meta, rho, times):
    """Deterministic row-greedy overlap assignment, matching the MATLAB helper."""
    _, p, v = operators(meta)
    energies, transformed, previous = [], [], None
    for r, t in zip(rho, times):
        h = np.exp(-3 * t) * p @ p / (2 * meta['H'] * meta['volume'])
        h += np.exp(3 * t) * meta['volume'] * v / meta['H']
        e, u = eigh(h)
        if previous is None:
            phases = u[np.argmax(abs(u), axis=0), np.arange(len(e))]
        else:
            overlap = abs(previous.conj().T @ u)
            perm, used = [], set()
            for row in overlap:
                j = max((k for k in range(len(e)) if k not in used), key=lambda k: row[k])
                perm.append(j); used.add(j)
            e, u = e[perm], u[:, perm]
            phases = np.diag(previous.conj().T @ u)
        u *= np.exp(-1j * np.angle(phases))
        transformed.append(u.conj().T @ r @ u / np.trace(r).real)
        energies.append(e - e.min() + 1)
        previous = u
    return np.array(energies), np.array(transformed)


def sonify(energies, rho, times, duration=30, fs=44100, f0=60, bands=12):
    """MATLAB's lower-triangle binaural synthesis, with Nyquist filtering."""
    count = int(round(duration * fs)); output = np.zeros((count, 2))
    if len(times) < 2:
        return output
    bands = min(bands, rho.shape[1])
    k, l = np.tril_indices(bands)
    mags = abs(rho[:, k, l]); phases = np.unwrap(np.angle(rho[:, k, l]), axis=0)
    phases -= phases[0]
    borders = np.rint((times - times[0]) / (times[-1] - times[0]) * count).astype(int)
    for j in range(len(times) - 1):
        start, stop = borders[j:j + 2]; size = stop - start
        if size == 0:
            continue
        fraction = np.linspace(0, 1, size)[:, None]
        mag = mags[j] * (1 - fraction) + mags[j + 1] * fraction
        phase = phases[j] * (1 - fraction) + phases[j + 1] * fraction
        time = np.arange(start, stop)[:, None] / fs
        for channel, indices in enumerate((k, l)):
            freq = f0 * (energies[j, indices] * (1 - fraction) + energies[j + 1, indices] * fraction)
            valid = (freq > 0) & (freq < fs / 2)
            output[start:stop, channel] = np.sum(mag * valid * np.sin(2 * np.pi * freq * time + (1 if channel == 0 else -1) * phase), axis=1)
    peak = abs(output).max()
    if peak:
        output *= .9 / peak
    return output


def render(folder, media=False, duration=30, fs=44100):
    folder = Path(folder)
    if (folder / 'sweep.csv').exists():
        import csv
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7, 7))
        with (folder / 'sweep.csv').open() as file:
            for row in csv.DictReader(file):
                _, _, obs = read(folder / row['directory'])
                label = f"{row['parameter']}={row['value']}" + (' (adiabatic)' if row['workflow'] == 'adiabatic' else '')
                axes[0].plot(obs['N'], obs['Pfalse'], label=label)
                axes[1].plot(obs['N'], obs['purity'])
        axes[0].set(ylabel='False-side probability'); axes[0].legend()
        axes[1].set(xlabel='N', ylabel='Purity'); fig.tight_layout()
        fig.savefig(folder / 'sweep.png', dpi=180); plt.close(fig)
        return
    meta, rho, obs = read(folder)
    times = obs['N']; axis = np.linspace(-meta['extent'], meta['extent'], meta['grid'])
    if meta['wigner_exported']:
        wig = np.fromfile(folder / 'wigner.f64', dtype='<f8').reshape(len(times), len(axis), len(axis))
    else:
        wig = np.array([analytic_wigner(r, axis, meta['hbar']) for r in rho])
    np.savez_compressed(folder / 'plot_data.npz', times=times, axis=axis, wigner=wig)
    energies, transformed = eigenbasis(meta, rho, times)
    np.savez_compressed(folder / 'energy_data.npz', times=times, energies=energies, density=transformed)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    bound = max(abs(wig).max(), 1e-8)
    im = axes[0].imshow(wig[-1], origin='lower', extent=[axis[0], axis[-1]] * 2,
                        cmap='RdBu_r', vmin=-bound, vmax=bound, aspect='auto')
    axes[0].plot(obs['mean_phi'], obs['mean_momentum'], 'k-', lw=.7)
    axes[0].set(xlabel='Field', ylabel='Canonical momentum', title=f"{meta['workflow']} / {meta['model']}, N={times[-1]:.3g}")
    fig.colorbar(im, ax=axes[0], label='Signed Wigner function')
    axes[1].plot(times, obs['Pfalse'], label='False-side probability')
    axes[1].plot(times, obs['purity'], label='Purity')
    axes[1].set(xlabel='N'); axes[1].xaxis.set_major_locator(MaxNLocator(5))
    axes[1].legend(); fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(folder / f'diagnostics.{ext}', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5))
    for j in range(meta['basis']):
        ax.scatter(times, energies[:, j], c=transformed[:, j, j].real, vmin=0, vmax=1, s=8, cmap='viridis')
    ax.set(xlabel='N', ylabel='Tracked energy above ground + 1', yscale='log')
    fig.tight_layout(); fig.savefig(folder / 'energy_occupations.png', dpi=180); plt.close(fig)
    if media:
        if not duration > 0:
            raise ValueError('duration must be positive')
        audio = sonify(energies, transformed, times, duration, fs)
        wavfile.write(folder / 'sonification.wav', fs, (audio * 32767).astype(np.int16))
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(wig[0], origin='lower', extent=[axis[0], axis[-1]] * 2, cmap='RdBu_r', vmin=-bound, vmax=bound, aspect='auto')
        line, = ax.plot([], [], 'k-', lw=.8)
        ax.set(xlabel='Field', ylabel='Canonical momentum')
        writer = FFMpegWriter(fps=20, codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])
        with writer.saving(fig, str(folder / 'wigner.mp4'), dpi=100):
            for t in np.linspace(times[0], times[-1], max(1, int(round(duration * 20)))):
                j = int(np.argmin(abs(times - t)))
                im.set_data(wig[j]); line.set_data(obs['mean_phi'][:j+1], obs['mean_momentum'][:j+1])
                ax.set_title(f"{meta['workflow']} / {meta['model']}, N={times[j]:.3f}"); writer.grab_frame()
        plt.close(fig)
        subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(folder / 'wigner.mp4'), '-i', str(folder / 'sonification.wav'),
                        '-c:v', 'copy', '-c:a', 'aac', '-shortest', str(folder / 'sonified.mp4')], check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder'); parser.add_argument('--media', action='store_true')
    parser.add_argument('--duration', type=float, default=30)
    args = parser.parse_args(); render(args.folder, args.media, args.duration)
