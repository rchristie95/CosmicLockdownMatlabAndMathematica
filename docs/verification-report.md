# Recorded verification

This report accompanies the production solver replacement described in [solver choices](solver-choices.md). Earlier Euler/SRK results remain in git history at `b93c5cc`; they are not the current implementation.

## X³ correction

The unchanged larger-basis benchmark uses N=-0.2 to -0.198, H=5, lambda=.05, bases 24/48/72, and steps 1e-5 / 5e-6. Coupled standard-normal inputs are aggregated from the same fine Brownian path for comparison with the earlier study.

At basis 72, the timestep change in `(mean field, field variance, false-side probability)` fell from **0.00517727 to 0.0000386789**, a **134-fold reduction**. The finer-step basis-48 to basis-72 change is **0.00000254574**. This vector mixes differently dimensioned observables and is a comparison metric, not a physical distance. It measures step differences, not absolute error against an exact trajectory.

The exact Gaussian measurement update removes the stiff explicit measurement drift. It does not remove kinetic/measurement splitting error, stochastic sampling error or basis truncation. In the stronger-coupling small-basis test (lambda=.2, b=6), X³ RMS density differences against the finest reference remain 0.1034, 0.0855 and 0.0244 for steps .001, .0005 and .00025. Those coarse settings are **not quantitatively converged**. Stable normalized trajectories alone must not be used as a convergence criterion.

## Independent numerical references

`tests/test_solver_accuracy.py` uses SciPy DOP853 to integrate the full time-dependent equations, independently of the production splitting/tableau:

- Closed evolution shows fourth-order refinement: density errors 2.41e-11, 1.50e-12, 9.73e-14 for steps .02, .01, .005 in the tested interval.
- Lindblad X/X²/X³ show second-order refinement. X³ uses .001/.0005/.00025 to reach the refinement regime at lambda=2; errors are 3.57e-5, 1.02e-5, 2.67e-6. Coarser stable steps do not yet show the asymptotic order.
- 24 coupled stochastic paths per model give decreasing RMS error against a finer reference. Exact dephasing and Born populations are also tested with Monte Carlo standard errors, including strong measurement clocks up to q=10.
- Adaptive non-Markovian density and trajectory agree with independent integration to approximately 1.6e-11 and 7.6e-12. Direct full-history Simpson quadrature agrees with the rank-four contraction to 3.8e-16. Both zero-coupling limits agree with unitary integration to 2.1e-10. Adaptive rejection is exercised explicitly.
- Late-time X³ stress tests reach N=1 at lambda=.5, preserving finite states and Lindblad positivity/trace. This is a stability test, not a continuum-convergence result.

See [machine-readable solver checks](verification/solver-accuracy.json) and [larger-basis data](verification/larger-basis-report.json). Reproduce with `python tests/test_solver_accuracy.py build/cosmic_fock` and `python tests/fock_convergence.py build/cosmic_fock`.

## Cross-language and workflow coverage

The CI pipeline builds and tests C++ on Windows and Linux, runs the independent reference/statistical suite, renders seven short media exports, then compares supplied-noise datasets against MATLAB R2025b. Required relative tolerances remain 1e-9 for operators/initial projectors and 1e-7 for short trajectories, allowing a global state phase. Strong X³ and adaptive-rejection cases are included. The [coverage map](fock-port-coverage.md) identifies the retained programs and helpers.

Final cross-language results are recorded after the validation branch passes CI. The previous implementation's successful CI does not validate these replacements.

## Scope

These are finite-basis and short-interval verification studies. They do not establish convergence of the original full N=-2 to 1, basis-400 figures or rare tunnelling rates. No positivity guarantee is asserted for the retained non-Markovian approximation, and its density eigenvalues are exported without clipping. No rederivation of the X²/X³ or non-Markovian models is claimed. Strongly stiff memory regimes may need a separately validated implicit integrator.
