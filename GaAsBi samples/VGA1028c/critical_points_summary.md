# Critical Point Derivative Summary

Source dielectric data:

`wavelength_fit_dielectric_functions_ev.csv`

Method:

- interpolated to a uniform photon-energy grid
- smoothed with Savitzky-Golay filtering
- calculated `d epsilon / dE` and `d2 epsilon / dE2`
- found peaks in `|d2 epsilon / dE2|`
- locally fitted a complex second-derivative CP line shape

The energies below are best treated as initial CP estimates, not final physical
assignments.

## Strong derivative peaks

### epsilon_perp

```text
0.754 eV
1.001 eV
1.125 eV
1.312 eV
1.486 eV
3.006 eV
3.105 eV
4.875 eV
```

### epsilon_parallel

```text
0.754 eV
1.005 eV
1.168 eV
1.310 eV
1.393 eV
3.025 eV
3.115 eV
4.858 eV
```

### epsilon_parallel - epsilon_perp

These are the strongest anisotropy-related derivative peaks:

```text
0.948 eV
2.472 eV
2.623 eV
2.796 eV
2.959 eV
3.087 eV
3.266 eV
3.383 eV
```

## Local CP-fit energies

The local complex derivative fit shifts some peak positions. The most useful
clusters are:

```text
~0.85 eV
~1.08-1.14 eV
~1.32 eV
~1.39 eV
~2.55-2.86 eV anisotropy region
~3.07-3.10 eV strong high-energy feature
~4.89 eV UV feature
```

## Notes

- The low-energy features below about 0.9 eV are close to the spectral edge and
  should be treated cautiously.
- The 1.1-1.4 eV group is the most relevant near-gap/ordering-related region.
- The 2.5-3.4 eV anisotropy peaks are consistent with the slide note about a
  higher-energy feature, likely connected to band-folding/order effects.
- Several local CP fits hit the chosen broadening/window bounds, so these
  should seed a constrained global model rather than be used as final values.
