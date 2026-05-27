# Projection-Dependent Valence-Band Splitting Check

Using derivative CP candidates from `epsilon_perp` and `epsilon_parallel`:

```text
perp peak      parallel peak    difference
1.000654 eV    1.004540 eV      +3.9 meV
1.125010 eV    1.167757 eV      +42.7 meV
1.311543 eV    1.309600 eV      -1.9 meV
1.486418 eV    1.484475 eV      -1.9 meV
```

The clearest projection-dependent splitting is therefore near the near-gap feature:

```text
Delta E_parallel-perp ~= 43 meV
```

The 1.31 eV and 1.49 eV structures are essentially not split between projections in this derivative analysis.

A local complex CP derivative fit gives a less stable but comparable estimate if the prominent second feature is paired as:

```text
perp 1.106855 eV vs parallel 1.141181 eV -> about 34 meV
```

So the robust statement is that the near-gap projection splitting is on the order of 35-45 meV in the current wavelength-by-wavelength extraction.
