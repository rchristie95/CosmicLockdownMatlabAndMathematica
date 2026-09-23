# Cosmic Lockdown: MATLAB, Mathematica and C++

Code for [Cosmic Lockdown: When Decoherence Saves the Universe from Tunneling](https://arxiv.org/abs/2512.14204).

Run the retained numerical workflows without MATLAB using **[cpp/cosmic_lockdown](cpp/cosmic_lockdown/README.md)**:

- `cosmic_fock`: C++17 Fock-basis counterparts for ground states, closed evolution, X/X²/X³ stochastic and Lindblad dynamics, sweeps, and non-Markovian trajectories/density matrices. Eigen 3.4.0 is fetched with a verified SHA-256 during CMake configuration.
- `cosmic_lockdown`: the existing dependency-free Fourier-grid position-monitoring solver for high-resolution press illustrations. Its command line and method are unchanged.
- `fock_media.py`: Python figures, Wigner animations, energy occupations and stereo sonification; ffmpeg assembles videos.

Momentum-only and mixed X/P monitoring programs have been removed. Momentum in the Hamiltonian, phase-space coordinates and diagnostics remains. Mathematica notebooks remain as derivation material. See the [complete coverage and verification guide](docs/fock-port-coverage.md).

## Corrected position-monitoring workflow

`IndividualTrajectories_X.m`, `LindbladSweepsX.m`, `SSEDynamics_X_Sparse.m` and `Markov_LindbladX_ExpStep_1000.m` now use the position-monitoring equations of arXiv:2512.14204v2, Eqs. (2.7), (2.8), (3.14), (3.15), with hbar=1:

```text
volume = 4 sqrt(2)
K(N) = exp(-3N) P^2/(2 H volume) + exp(3N) volume V(X)/H
Gamma(N) = 131 pi lambda^2 exp(6N)/(256 mu^5 volume)
L(N) = sqrt(Gamma(N)) X
d rho/dN = -i[K,rho] - Gamma(N)/2 [X,[X,rho]]
```

For mu=0.5, the characteristic crossover `log(1/(2 volume^2 mu^6))/6` is zero. No additional shift of N is applied. Monitoring is active from the initial N=-2. Shared Hamiltonian helpers take an optional volume argument; the corrected X drivers pass it explicitly. Their default remains 1 for compatibility with the other historical workflows. Sonification now uses the same Hamiltonian potential prefactor as propagation.

These corrections change numerical results relative to commit `d382d97`. See [the correction note](docs/position-monitoring-corrections.md). The higher-power and non-Markovian ports preserve their separate model conventions; software parity is not a rederivation of those models from the paper.

## MATLAB prerequisites and tests

The repository includes a sparse/matrix-free exponential-action implementation with convention `expmv(A,v,t) = exp(t*A)*v`. Numerical MATLAB workflows share `FockWorkflow.m`; the original function signatures remain available. Figure/video/audio drivers retain their graphics and ffmpeg requirements. Fresh stochastic runs may end in either vacuum.

Run the small numerical regression tests from the repository root:

```matlab
addpath('tests');
run_position_monitoring_tests;
run_fock_port_tests; % after the Python verification suite generates C++ fixtures
```

GitHub Actions builds and tests C++ on Linux and Windows, generates numerical reference data, and compares it to MATLAB R2025b. Tests cover every numerical workflow, common supplied noise, operators, ground states, exponential action, model coefficients, crossover, Wigner normalization, switching, and nonuniform memory integration. Linux also renders short videos with audio. Small-basis parity and short-run refinements do not establish convergence of full paper-scale figures; inspect the generated refinement report before choosing production settings.
