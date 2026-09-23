# Production solver choices

There is one production method per numerical family, in both MATLAB and C++. These are choices for this repository's polynomial Hamiltonian and Hermitian position monitoring, not a claim that one integrator is optimal for every quantum model.

| Dynamics | Retained method | Reason |
|---|---|---|
| Adiabatic ground states | Hermitian eigensolver | Direct residual checks; no fictitious-time integration error. |
| Closed Fock evolution | Fourth-order symmetric spectral composition | Unitary factors, cached X and P² eigensystems, no repeated full Hamiltonian exponential. |
| X, X², X³ stochastic trajectories | Exact Gaussian measurement + spectral Strang splitting | Integrates the stiff measurement channel exactly; samples its Born law; no unstable Euler drift or SRK stage. |
| X, X², X³ Lindblad evolution and sweeps | Exact dephasing + spectral Strang splitting | Completely positive, trace-preserving factors; no n²-by-n² Liouvillian or positivity repair. |
| Non-Markovian trajectories/densities | Adaptive Dormand–Prince 5(4) on augmented memory equations | The cosine kernel has exactly four factors; memory storage does not grow with the number of time steps. Local error is controlled. |
| Large uniform Fourier grids | Existing FFT split propagation and exact measurement | A different spatial representation, appropriate for the high-resolution X illustration; not a legacy Euler alternative. |

The spectral matrices are cached once per basis. Dense Fock transforms still cost O(b²) per state step and O(b³) per density step. Large spatial grids therefore remain a good reason to keep the Fourier executable. There is no global speed ranking implied by these choices.

## Exact measurement, and what remains approximate

For A=X^k and q=integral[Gamma(N)/hbar] dN, expand the state as c_j in the X eigenbasis. The measurement outcome Y has distribution

    sum_j |c_j|² Normal(2 q a_j, q),  a_j=x_j^k.

The update is c_j <- c_j exp(a_j Y - q a_j²), followed by normalization. The code evaluates this in logarithmic form. It samples the mixture by inverse CDF using one standard normal quantile, including a survival-function calculation for the upper tail. Averaging this exact instrument gives

    rho_ij <- rho_ij exp[-q (a_i-a_j)²/2].

This is an exact finite measurement channel and approaches the same normalized, real-noise SSE as the step tends to zero. It does not make the combined kinetic/measurement dynamics exact. Potential and measurement commute in this discretization; kinetic energy does not. Splitting errors and basis truncation must still be checked, particularly for X³.

The exponential rate is integrated analytically: q=Gamma(t)*expm1(6 dt)/(6 hbar). Model-specific rates, volumes and activation times have not been changed. All powers use the same instrument, with their own A and Gamma. Negative-time composition coefficients are used only for closed unitary propagation, never for dephasing or measurement.

Noise files now mean Gaussian-mixture quantiles, not Euler dW inputs. They still contain one standard normal per integration step, including inactive steps. Old random streams cannot reproduce old Euler trajectories. Cross-language replay is exact up to numerical tolerance. Strong-refinement tests aggregate Gaussian increments in the integrated measurement clock. A single noisy path need not have monotonically decreasing differences; the statistical test uses 24 independent coupled paths.

## Non-Markovian memory equations

Write the retained real bath factor vector as f(N), with covariance f(N)^T f(M), and eta=f(N)^T zeta. With L=(lambda/H) exp(3N) X, the density equations are

    rho' = -i[H,rho]/hbar - [L, sum_k f_k G_k]
    G_k' = f_k [L,rho],    G_k(initial)=0.

For trajectories, C=L-<L>, D=-iH/hbar+conj(eta) C:

    raw = D psi - C G f
    psi' = raw - psi Re(psi† raw)/(psi†psi)
    G' = D G + (C psi) f^T,    G(activation)=0.

These are the continuous limits of the previous retained memory approximations. The old extra initial source proportional to the first timestep has been removed. The normalized gauge corresponds to continuous normalization, rather than repairing the norm after each Heun update. This is a numerical correction, not a new derivation of the underlying non-Markovian approximation. Density trace and Hermiticity follow from the equations; the integrator does not symmetrize, renormalize or clip eigenvalues.

Defaults are relative tolerance 1e-9 and absolute tolerance 1e-11, configurable with `--rtol` and `--atol` / MATLAB fields `rtol` and `atol`. `--step` caps internal steps. Metadata records accepted/rejected steps. White-noise measurement steps are never rejected adaptively. Standalone MATLAB compatibility functions interpolate supplied real bath factors and coloured noise linearly; refining the integrator does not remove sampled-input interpolation error.

Dormand–Prince is appropriate for the smooth, nonstiff augmented equations tested here. Very stiff non-Markovian parameter choices may require an implicit BDF/Radau implementation and separate validation; no claim of uniformly efficient handling of those regimes is made. The memory approximation also has no guaranteed positivity, and the trajectory and density approximations are not asserted to be exact ensemble equivalents.

## Removed production methods

Normalized Euler and the X² scalar SRK, explicit Heun memory propagation, quadratic-cost full-history density propagation, the unused local Heun Schrödinger function, and the temporary scaled-Taylor exponential implementation have been removed. The redundant files `PositionMonitoringEulerStep.m`, `Ham_Step_Big.m`, `NMQSD_MasterEquation_pagemat.m`, and `expmv.m` are deleted. Public trajectory/density wrappers retain their names for compatibility; their old integrators do not remain behind an option. `nm-density-full` is no longer a supported command. Direct full-history quadrature is retained only as an independent verification calculation.

## Basis for the choices

- Rouchon, [quantum stochastic master equations and Kraus formulations](https://arxiv.org/abs/2208.07416), motivates physical-state-preserving instruments. The Gaussian formula above follows directly by completing the square in its outcome probability; the implemented quantile sampler is not claimed to be Rouchon's specific first-order algorithm.
- Yoshida, [Construction of higher order symplectic integrators](https://doi.org/10.1016/0375-9601(90)90092-3), provides the symmetric composition coefficients. Here they compose time-advancing, self-adjoint unitary split steps with analytically integrated scalar coefficients.
- MathWorks' [ODE solver guidance](https://www.mathworks.com/help/matlab/math/choose-an-ode-solver.html) distinguishes nonstiff explicit Runge–Kutta from stiff implicit solvers. Both languages use the same independently implemented Dormand–Prince tableau and error norm for reproducible comparisons.

See [verification report](verification-report.md) for measured errors and the limits of those checks.
