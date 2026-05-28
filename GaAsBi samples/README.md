# GaAsBi Sample Outputs

This folder contains curated sample-level outputs derived from Mueller-matrix
ellipsometry analysis. Raw measurement `.dat` files are intentionally excluded.

## Samples

| Sample | Contents |
| --- | --- |
| [`VGA1028c`](VGA1028c/) | Uniaxial dielectric tensor fit, dielectric plots, critical-point candidates, valence-band splitting estimate, and CompleteEASE/MATLAB export. |

## Data Convention

The dielectric tensor is stored using the uniaxial model:

- `epsilon_x = epsilon_y = epsilon_perp`
- `epsilon_z = epsilon_parallel`

Each sample folder should include a `README.md`, compact extracted values,
CSV dielectric functions, and plots needed to inspect the fit quality.
