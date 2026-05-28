# VGA1028c GaAsBi Dielectric Tensor Results

This folder contains curated dielectric-function and critical-point results for sample `VGA1028c`.

## Contents

- `dielectric_functions_index.csv` - index for the full-spectrum wavelength-by-wavelength dielectric-function CSV parts.
- `dielectric_functions/dielectric_*nm.csv` - fitted complex dielectric functions with `epsilon_x = epsilon_y = epsilon_perp` and `epsilon_z = epsilon_parallel`.
- `critical_point_candidates.csv` - derivative peak candidates from `|d2 epsilon / dE2|`.
- `critical_point_fits.csv` - local complex second-derivative critical-point fits.
- `critical_points_summary.md` - compact list of detected critical-point energy groups.
- `valence_band_splitting_projection_summary.md` - projection-dependent near-gap splitting estimate.
- `found_values.csv` - compact table of main extracted values.
- `images/wavelength_fit_dielectric_functions_ev.png` - fitted complex dielectric functions plotted versus energy in eV.
- `images/wavelength_fit_dielectric_axis_difference_ev.png` - anisotropy plot showing axis differences versus energy in eV.
- `images/critical_point_markers_on_eps2.png` - critical-point markers over the imaginary dielectric response.
- `completeease/VGA1028c_completeease_dielectric_tensor.mat` - MATLAB export for CompleteEASE-style fitting and reuse.

## Main Values

- Thickness used for wavelength-by-wavelength refinement: `470.8 nm`.
- Azimuth offset: `phi0 = -2.93 deg`.
- Near-gap projection-dependent splitting estimate: `35-45 meV`.
- Strong anisotropy-related derivative peaks: `0.948`, `2.472`, `2.623`, `2.796`, `2.959`, `3.087 eV`.

The wavelength-by-wavelength dielectric extraction froze geometry/thickness and fitted only `epsilon_perp` and `epsilon_parallel` at each wavelength.

## Dielectric Function Plots

### Complex Dielectric Functions

![Fitted complex dielectric functions versus energy](images/wavelength_fit_dielectric_functions_ev.png)

### Dielectric Axis Difference

![Dielectric axis difference versus energy](images/wavelength_fit_dielectric_axis_difference_ev.png)

### Critical Point Markers

![Critical point markers over epsilon 2](images/critical_point_markers_on_eps2.png)

## CompleteEASE Export

The MATLAB export contains the uniaxial dielectric tensor arrays for reuse:

- `epsilon_x = epsilon_y = epsilon_perp`
- `epsilon_z = epsilon_parallel`
- `wavelength_nm`, `energy_eV`, `epsilon_tensor_diag`, `n_perp`, `k_perp`, `n_parallel`, `k_parallel`, `thickness_nm`, and `phi0_deg`
