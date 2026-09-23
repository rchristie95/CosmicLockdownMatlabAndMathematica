# Recorded verification

The implementation adds missing ports rather than claiming the previous Fourier solver covered all MATLAB workflows. Momentum-only and mixed X/P monitoring are removed; both Mathematica notebooks remain.

## Cross-language and execution checks

The first complete validation batch passed on Windows, Linux and MATLAB R2025b in [GitHub run 35819013683](https://github.com/rchristie95/CosmicLockdownMatlabAndMathematica/actions/runs/35819013683). The subsequent large-basis Wigner fix is also covered by the same CI suite. Consult the latest branch/main Actions run for the final commit status.

- 75 numerical runs cover all retained workflow families, switch-crossing cases, variable steps, replay, and refinements. Both parameter sweeps execute and render.
- 20 C++ datasets are compared against corrected MATLAB. Maximum relative trajectory error: **3.72e-12**, below the required 1e-7. Maximum relative operator error: **8.39e-18**, below 1e-9. See [the recorded parity table](verification/matlab-parity.csv).
- Actual MATLAB legacy function signatures, extracted NM helpers and nonuniform-grid memory contractions are tested. The bundled production exponential is checked against MATLAB's independent dense `expm`, including sparse/nonnormal matrices and negative times.
- Signed Wigner data passes Gaussian, integral, momentum-sign and pure/density checks. The new scaled recurrence also tests basis 400, including points where the Gaussian prefactor alone underflows; no negative values are clipped.
- Six ensemble comparisons (three coupling powers, two timesteps) check coherence and population against exact dephasing using Monte Carlo standard errors and an explicit step-error tolerance.
- Seven short animations, stereo WAV files and sonified MP4s are generated and checked. These verify media functionality; their small bases and short durations are not final scientific illustrations.

## Larger-basis study

An additional Windows/GCC 15.2 study uses bases 24, 48 and 72; steps 1e-5 and 5e-6; N=-0.2 to -0.198; H=5; lambda=.05; and the documented model presets. SSE comparisons use shared Brownian increments. The entries below are Euclidean changes in the vector `(mean field, field variance, false-side probability)`; this combines differently dimensioned observables and is a numerical comparison metric, not a physical distance.

| Workflow | Basis 48 → 72, finer step | Step 1e-5 → 5e-6, basis 72 |
|---|---:|---:|
| Adiabatic X | 2.31e-11 | 0 |
| Closed X | 2.44e-11 | 1.08e-14 |
| SSE X | 2.62e-11 | 7.67e-8 |
| SSE X² | 6.44e-6 | 4.00e-7 |
| SSE X³ | 2.55e-6 | **5.18e-3** |
| Lindblad X | 2.44e-11 | 3.93e-14 |
| Lindblad X² | 6.00e-6 | 3.97e-15 |
| Lindblad X³ | 6.00e-6 | 2.05e-13 |
| NM trajectory | 6.00e-6 | 1.48e-12 |
| NM density | 6.00e-6 | 2.15e-14 |

Reproduce with `python tests/fock_convergence.py build/cosmic_fock`. Full observations are in [larger-basis-report.json](verification/larger-basis-report.json).

**Remaining limits:** X³ stochastic dynamics is still timestep-sensitive in this benchmark; use further refinement before extracting quantitative conclusions. The fast CI bases 8/12/16 are demonstrably insufficient for continuum double-well predictions. The study above covers a short interval, not the full original N=-2 to 1 calculations. Neither passing parity nor normalization establishes convergence of long trajectories, tunnelling rates, or the validity/positivity of the non-Markovian approximation. NM minimum eigenvalues are exported without clipping. No full rederivation of higher-power or NM equations, and no port of the symbolic Mathematica notebooks, is claimed.
