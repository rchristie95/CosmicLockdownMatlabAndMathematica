# Fock port: coverage, conventions and verification

## Build and run

```sh
cmake -S cpp/cosmic_lockdown -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
./build/cosmic_fock --workflow sse --model x --basis 24 --initial -.2 --final -.18 --step .0001 --output results/x
python cpp/cosmic_lockdown/fock_media.py results/x --media --duration 30
python tests/test_fock_workflows.py build/cosmic_fock --media
```

On MSVC, executables are under `build/Release/` and end in `.exe`. Python needs numpy, scipy and matplotlib; media needs ffmpeg on PATH. CMake downloads the pinned Eigen 3.4.0 archive and verifies its SHA-256. For offline builds supply `FETCHCONTENT_SOURCE_DIR_EIGEN` pointing to that version's extracted source. The legacy Fourier executable remains available unchanged.

`cosmic_fock --help` lists all options. Numerical controls include `--basis`, `--step`, `--frames`, `--times` (comma-separated output times including both endpoints), and `--steps` (one maximum internal step per output interval). Physics options include `--hbar`, `--mu`, `--beta3`, `--beta4`, `--lambda`, `--hubble`, `--volume`, `--omega`, `--initial`, and `--final`. `--noise` supplies one standard normal value per SSE integration step, including inactive intervals; `--bath-noise` supplies four real/imaginary pairs. Both are exported for replay. RNG streams themselves need not agree between platforms or languages. Unused or missing supplied SSE samples are errors.

The default Fock basis of 24 is a demonstration setting, **not** a converged replacement for the original basis-400 calculations. The original high-power SSE algorithms are explicit and can become unstable for stiff parameters. Refine the timestep and basis together; normalization alone does not demonstrate accuracy.

## Coverage map

All commands below use `cosmic_fock`; model names are `x`, `x2`, and `x3`. Shared plotting blocks map to `fock_media.py`, not duplicated C++ graphics code. The two `.nb` notebooks are retained, not ported.

| Retained MATLAB program/helper | Replacement | Verification |
|---|---|---|
| `IndividualTrajectories_X.m` | `--workflow sse/closed/adiabatic/lindblad --model x`, then media renderer | All four numerical paths; X media export |
| `IndividualTrajectories_Z2.m` | `--workflow sse/lindblad --model x2`, then renderer | X² SRK, density and media |
| `IndividualTrajectories_X3.m` | `--workflow sse/lindblad --model x3`, then renderer | X³ Euler, density and media |
| `LindbladSweepsX.m` | `--workflow sweep --sweep lambda/hubble --values ...` | Both sweeps; H=0 produces an adiabatic reference |
| `AdiabaticGroundStates.m` | `--workflow adiabatic` | Ground-state projectors and actual wrapper |
| `SchrodingerSingleTrajectory_ExpStep_1000.m` | `--workflow closed` | Midpoint exponential; unnormalized projected output; wrapper |
| `SSEDynamics_X_Sparse.m` | `--workflow sse --model x` | Supplied-noise MATLAB comparison and ensemble tests |
| `SSEDynamics_X2_Sparse.m` | `--workflow sse --model x2` | SRK supplied-noise comparison, switch and refinement |
| `SSEDynamics_X3_Sparse.m` | `--workflow sse --model x3` | Euler supplied-noise comparison, variable steps and refinement |
| `Markov_LindbladX_ExpStep_1000.m` | `--workflow lindblad --model x` | MATLAB parity, trace, Hermiticity, positivity |
| `Markov_LindbladX2_ExpStep_1000.m` | `--workflow lindblad --model x2` | MATLAB parity, big/small-basis switch and positivity |
| `Markov_LindbladX3_ExpStep_1000.m` | `--workflow lindblad --model x3` | MATLAB parity and positivity |
| `NonMarkovianScript.m` | `--workflow nm-sse/nm-density/nm-density-full`, then renderer | All three numerical paths and NM media |
| Extracted `NMQSD_SingleTrajectory_Hybrid.m` | `--workflow nm-sse` | Standalone MATLAB helper versus common solver |
| Extracted `NMQSD_MasterEquation_lowrank.m` | `--workflow nm-density` | Low-rank/full-history and nonuniform-grid comparisons |
| Extracted `NMQSD_MasterEquation_pagemat.m` | `--workflow nm-density-full` | Direct memory sum and corrected timestamps |
| Extracted `Ham_Step_Big.m` | Fock exponential action | Dense `expm` reference and closed evolution |
| `PositionMonitoringCoefficients.m` | X preset in Fock/Fourier solvers | Volume, crossover and rate regression |
| `PositionMonitoringEulerStep.m` | Normalized Hermitian-channel Euler update | Independent one-step test and ensemble channel |
| `PsiWigner.m` | Analytic Fock Wigner export, pure-state case | Gaussian, sign/momentum marginal, FFT helper parity |
| `RhoWigner.m` | Same analytic Wigner export for a density matrix | Pure/density equivalence and normalization |
| `write_simple_sonification.m` | `fock_media.py`: tracked eigenbasis and binaural synthesis | Duration, sample rate, finite samples and media mux |
| `merge_audio_video_ffmpeg.m` | Checked subprocess call to ffmpeg | Seven short sonified video exports |
| `parfor_wait.m` | Console progress | Headless workflow execution; no GUI dependency |
| New `FockWorkflow.m` | Shared C++ Fock engine | Every model/workflow compared against MATLAB |
| New `expmv.m` | Scaled Taylor action in C++ | Independent dense Padé exponential, including nonnormal matrices |

Original MATLAB media drivers still produce their historical layouts. The Python equivalents provide the same kinds of scientific outputs, not pixel-identical styling or identical encoded media bytes. Embedded Hermite transforms, moments, energy occupations and phase tracking are covered by the common Fock/Wigner/media implementations.

## Model and algorithm conventions

- X monitoring: paper volume `4 sqrt(2)`, rate `131*pi*lambda^2*exp(6N)/(256*mu^5*volume)`, active from the initial time; no additional time shift. hbar=1.
- X²/X³: retain volume 1 and the respective rates `pi*lambda^2*exp(6N)/(64*mu^4)` and `pi*lambda^2*exp(6N)/(8*mu^3)`. Monitoring starts at N=-1. Explicit volume overrides alter the Hamiltonian; they do not constitute a derived new high-power decoherence model.
- SSE evolves in 2b and normalizes projected b-dimensional snapshots. X/X³ use normalized Euler; X² uses the original scalar-noise SRK stages, with normalized expectation values at intermediate stages. At zero coupling, unitary exponential steps replace the explicit stochastic integrator.
- Closed evolution uses midpoint Hamiltonians in 3b and deliberately retains the norm lost in projection. Adiabatic states use 2b and normalize the projection.
- X GKLS prepares its initial state in 3b and evolves the projected density in b from the start. X²/X³ retain big-basis unitary evolution before -1 and small-basis GKLS afterwards. Steps split at the switch.
- For nonunit hbar in X²/X³, both SSE and GKLS now use the same dissipator divided by hbar, matching the stochastic equation's noise normalization. The previous GKLS implementation omitted this factor; default hbar=1 results are unaffected.
- NM trajectory uses the retained hybrid scheme: big-basis unitary propagation before -1, then the small-basis Heun/memory approximation. NM density implementations retain their own equation active from the initial time; they are not asserted to be an exact ensemble equivalent of the trajectory approximation.
- The rank-four bath has covariance `3*(9*cos(omega*dN)+cos(3*omega*dN))/(128*H^3*omega^3)`. Both memory algorithms weight each history sample by its own timestep. In the standalone MATLAB hybrid helper, an inserted switch sample is interpolated from the supplied discrete bath/noise data.

## Recorded corrections

1. X³ Euler normalized by the old state norm. It now normalizes the updated vector. Intermediate stochastic means use normalized expectations.
2. Higher-power SSE activation depended on the *next output time*. Integration now splits at -1, so an output interval crossing the switch cannot activate monitoring early.
3. NM density outputs were labelled N before a completed step. Labels now use the advanced time and always include the final frame.
4. NM density memory sums used a single constant timestep. They now integrate nonuniform histories correctly; full-history and low-rank forms are tested against one another.
5. The external production exponential dependency is replaced with a checked sparse/matrix-free implementation. Numerical failures raise errors rather than returning partial success.

These are software/numerical corrections. The higher-power rates and non-Markovian approximation have not been rederived from the paper.

## Output and acceptance

`metadata.json` records the model and conventions. `observables.csv` contains actual snapshot times, projected trace, normalized purity, field/momentum moments, false-side probability, energy and minimum density eigenvalue. `psi.f64` (when applicable), `rho.f64`, and `operators.f64` contain float64 real/imaginary pairs, column-major within each frame. `operators.f64` stores X, P, initial small-basis Hamiltonian and the unscaled coupling operator. The supported CI platforms are little-endian. `wigner.f64` is frame/momentum/field order, with `phase_axis.f64` defining both axes. Signed values are retained, including any negativity from a non-positive approximate NM density. `--no-wigner` skips the transform; Python can compute it later.

Verification emits `verification.json`, per-workflow data and `matlab-parity.csv` as CI artifacts. Required cross-language tolerances are 1e-9 for operators/initial projectors and 1e-7 for short trajectories. Exponential action is checked against independent dense `expm`; Wigner data against analytic Gaussian and an independent Laguerre implementation. C++ ensemble checks compare populations/coherences to exact dephasing with Monte Carlo error bars and timestep refinement. Negative NM eigenvalues are reported rather than clipped or hidden.

The CI basis-8/12/16 study deliberately reports observable changes; those small bases are not converged for the double well. Passing software parity does not certify original basis-400/long-time figures, rare-transition rates, or the physical validity of the non-Markovian approximation. Use the refinement diagnostics when selecting production parameters.
