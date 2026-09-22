# Cosmic Lockdown: MATLAB, Mathematica and C++

Code for [Cosmic Lockdown: When Decoherence Saves the Universe from Tunneling](https://arxiv.org/abs/2512.14204).

The independent, dependency-free C++17 implementation is in **[cpp/cosmic_lockdown](cpp/cosmic_lockdown/README.md)**. It can generate environment-off/on wavefunctions and signed Wigner functions without MATLAB. Its Fourier-grid splitting method differs from the MATLAB Fock-basis implementation.

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

These corrections change numerical results relative to commit `d382d97`. See [the correction note](docs/position-monitoring-corrections.md) for the precise differences and limits of validation. The momentum, higher-power coupling and non-Markovian workflows have not received this equation audit.

## MATLAB prerequisites and tests

Production exponential propagation requires an external `expmv(A,v,t)` implementation satisfying `expmv(A,v,t) = exp(t*A)*v`; this dependency was also required by the original code and is not bundled here. Check your implementation's argument order. Figure/video/audio drivers require their plotting, video and ffmpeg dependencies and can take substantial time and memory. The two restored SSE calls generate new random trajectories; they do not guarantee opposite final vacua or reproduce historical frames.

Run the small numerical regression tests from the repository root:

```matlab
addpath('tests');
run_position_monitoring_tests;
```

The test runner temporarily adds a **test-only**, size-limited dense exponential implementation, so the tests do not require an external `expmv`. Do not add `tests/support` to the production MATLAB path. Tests cover model coefficients, crossover/rescaling, normalized stochastic updates and their ensemble channel, monitoring before N=-1, density-matrix trace/positivity, and volume-dependent ground-state preparation. GitHub Actions runs these tests in MATLAB and the C++ self-tests on Linux and Windows. These checks do not replace production basis/time-step convergence or a full revalidation of the paper's figures.
