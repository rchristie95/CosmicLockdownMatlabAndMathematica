# C++ solvers

There are now two executables. `cosmic_fock` covers every retained numerical MATLAB workflow with the corresponding Fock-basis methods; see [coverage, commands and tests](../../docs/fock-port-coverage.md). CMake fetches its pinned Eigen dependency. Rendering/media uses `fock_media.py`.

The following documents the unchanged Fourier solver, `cosmic_lockdown`.

# Standalone C++ position-monitoring solver

This folder contains all C++ source and its optional plotting scripts. The Fourier executable itself requires no MATLAB, Eigen or FFTW. `main.cpp` computes the ground state, dynamics and Wigner transforms. Python only renders the resulting arrays.

## Build and test

From the repository root:

```sh
cmake -S cpp/cosmic_lockdown -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

Alternatively:

```sh
g++ -O3 -std=c++17 -Wall -Wextra -Wpedantic cpp/cosmic_lockdown/main.cpp -o cosmic_lockdown
./cosmic_lockdown --self-test
```

Windows: use `cosmic_lockdown.exe`; add `-static` with MinGW if a self-contained executable is wanted. Visual Studio CMake builds place it under `build/Release/`.

## Generate data and figures

```sh
./build/cosmic_lockdown --model paper --output results
python cpp/cosmic_lockdown/plot_press.py results
```

Plotting requires numpy and matplotlib. `plot_output.py` creates the more technical comparison. `plot_press.py` creates PNG/SVG/PDF artwork and a separate explanatory caption. The simulation may take several minutes and its binary Wigner arrays occupy approximately 0.5 GB. Build products and generated simulation files are not committed.

Defaults are mu=.5, beta3=.025, beta4=.13, H=5, lambda_on=.05, lambda_off=0, N=-2 to 1, 32,768 points on [-48,48), dN=.001, and seed 20260928. This seed is an **illustrative selection**, found in a 16-seed pilot at grid 4096, dN=.001, seeds 20260922–20260937, then recomputed at full resolution. It is not an ensemble estimate. The default requires Pfalse>.999, mean field within 1 of the false minimum, variance<.5 and |mean momentum|<10. Among eligible trajectories, choose the closest mean to the false minimum. If none qualifies, the run reports failure. `--selection first-false` instead takes the first with Pfalse>.999. Use `--trajectories`, `--seed`, `--grid`, `--step`, `--hubble`, `--lambda`, `--final` and `--extent` to change a run; `--help` lists options.

`--model paper` uses the corrected equations documented in the root README, volume 4 sqrt(2), crossover N=0 and monitoring from N=-2. `--model repository` preserves the **historical generator at commit d382d97**, with volume=1, denominator 512 in Gamma and monitoring from N=-1. It does not refer to the now-corrected MATLAB code. At unchanged N and lambda those generators differ. The legacy option is not a reproduction of MATLAB's basis, random stream or projection.

## Method and outputs

The Fourier-grid ground state is obtained by inverse iteration. Symmetric splitting combines Hamiltonian exponentials and Gaussian measurement updates sampled from their Born law. This approaches the paper's real-noise normalized SSE in the continuous-step limit; it is an alternative to the MATLAB Euler/Fock method. `std::normal_distribution` streams can differ across C++ standard libraries.

- `states.csv`: field grid and complex off/on state coefficients, including sqrt(dx).
- `history.csv`: active Gamma and false-side populations.
- `trajectory_outcomes.csv`: endpoint moments for every candidate.
- `metadata.json`: model, volume, crossover, grid, step, seed, selection and numerical checks. `source_commit` identifies the historical reference repository version.
- `phi.f64`, `momentum.f64`, `potential.f64`: little-endian float64 vectors.
- `wigner_off.f64`, `wigner_on.f64`: row-major arrays of shape `[momentum_grid, grid]`. At default resolution, 16,384 spatial columns are retained. Wigner diagnostics integrate the full momentum domain before viewport cropping.

`--no-wigner` skips the expensive transform for dynamics checks. A quick zero-coupling software check is `--grid 512 --lambda 0 --trajectories 1 --no-wigner`; that grid is insufficient for final interference artwork.

Self-tests check FFT inversion, analytic Gaussian Wigner data, unitary normalization, the zero-coupling update, a nonzero measurement ensemble against exact dephasing, and volume/time/coupling rescaling. Production runs check Wigner integrals and position marginals. For the documented press example, grid doubling 16,384 -> 32,768 gives closed-state infidelity 2.10e-4; halving dN=.001 -> .0005 gives 4.63e-11. No full stochastic time-step convergence or precision tunnelling rate is claimed. The selected narrow packet does not imply that every trajectory selects the false vacuum or that rare over-barrier events are impossible.
