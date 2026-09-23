# Position-monitoring corrections

This note compares the X-monitoring workflow at commit `d382d97dca93768d08187b90ca30d97747bbc862` with the written equations of [arXiv:2512.14204v2](https://arxiv.org/html/2512.14204v2). It records the corrections now implemented; it is not a validation of the other environmental channels or every original figure.

## Coefficients and time convention

The paper fixes `volume=4 sqrt(2)` and includes inverse volume in the kinetic Hamiltonian term, volume in its potential term, and inverse volume in the decoherence rate. The historical X propagators omitted volume. They also used `Gamma=131 pi lambda^2 exp(6N)/(512 mu^5)` with the conventional dissipator `D[X]rho=-[X,[X,rho]]/2`. The paper instead requires denominator `256 mu^5 volume` in Gamma. Its noise amplitude is sqrt(Gamma), not Gamma. The corrected SSE and master equation now share `FockWorkflow.m`; `PositionMonitoringCoefficients.m` remains an independently tested coefficient helper.

At the same numerical N and lambda, the old rate is `volume/2=2 sqrt(2)` times the paper's rate. At N=1, mu=.5, lambda=.05, the corrected Gamma is 9.1719798571 instead of 25.9422766154. Setting volume=1 alone does not recover the historical coefficient: the factor of two is a separate issue.

The characteristic crossover is `Nstar=log(1/(2 volume^2 mu^6))/6`, which equals zero for the paper's volume and mu=.5. The corrected code uses this convention directly for the initial state and every time-dependent coefficient. It applies no second shift. Historical volume=1 instead gives Nstar≈.577623. Algebraically, mapping paper coefficients to the old generator requires both `N_old=N_paper+log(volume)/3` and `lambda_old=sqrt(2)*lambda_paper/volume^(3/2)`, plus mapped preparation/end times and monitoring schedules. Relabelling the time axis alone is insufficient.

## MATLAB implementation changes

- Both X solvers now include monitoring from the supplied initial time; the undocumented N=-1 cutoff is removed.
- The SSE now uses exact Gaussian measurement and spectral unitary substeps. This supersedes the earlier Euler normalization repair; the Euler helper has been removed.
- The GKLS solver propagates its density matrix in the chosen storage basis throughout, including early times. Its initial state is still prepared in the larger basis and projected. This replaces the old early-time unitary propagation in a different basis; production basis convergence must therefore be rechecked.
- X drivers pass the paper volume through ground-state preparation, closed evolution, SSE/GKLS evolution, energy diagnostics, contours and sonification. The diagnostic/sonification Hamiltonian's erroneous factor of one half on V is corrected.
- Both commented-out SSE calls are restored. Fresh random runs can finish on either side; they are not guaranteed to reproduce the old true/false examples.
- Shared adiabatic/Schrodinger helpers retain volume=1 when the new optional argument is omitted, preserving other workflows' historical Hamiltonian convention. Other environment models remain unaudited.

## Verification and limits

The MATLAB tests exercise the actual small-matrix solver paths, coefficient identities, Gaussian-quadrature averaging of an SSE step, normalization, monitoring before N=-1, GKLS trace/positivity, and volume-dependent initial ground states. The spectral propagators are checked against independent integration of the full time-dependent generator. The temporary production `expmv` implementation and unused Hamiltonian helper have been removed. The public CI workflow runs the tests in MATLAB R2025b.

The separate C++ solver has numerical self-tests and documented spatial/time refinements, but uses Fourier splitting instead of the MATLAB finite Fock basis. Changing the basis, integrator, RNG and coefficients means it does not reproduce old individual trajectories. Neither a localized example nor a pair of Wigner snapshots is a precision tunnelling-rate measurement. Original numerical figures and sweep convergence should be reassessed before attributing their quantitative values to the corrected equations.

The later [Fock-port coverage guide](fock-port-coverage.md) documents the additional C++ Fock solver, retained higher-power/NM conventions, cross-language comparisons and remaining convergence limits.
