"""Quantify how sensitive the extracted GaAsBi Eg and valence-band splitting are
to analysis choices, and compare the anisotropy-derived Eg against an
absorption-edge Eg from the absolute transmission channel.

This is a read-only diagnostic. It does NOT change pipeline behavior. It exists
to measure the bias discussed in the audit:

  * current pipeline: Eg = lower split-transition energy of the strongest
    LD/LB pair, fit on the real part of the anisotropy difference spectrum;
  * alternative pairing: choose the lowest-energy admissible LD/LB pair instead
    of the strongest;
  * alternative fit component: fit the imaginary part instead of the real part;
  * absorptive reference: Eg from A(E) = -ln(T) on the absolute M11 transmission,
    which is what transmission / PL / PR effectively probe.

Usage:
    python3 eg_bias_diagnostic.py [path/to/file.dat]

With no argument it picks the first rotation-series ``*_X_*.dat`` it finds.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

import differential_decomposition as dd
from decomposition_math import _odd_window_points, _running_nanmedian


def absorption_edge_eV(energy_eV: np.ndarray, m11_transmission: np.ndarray) -> dict:
    """Estimate the band edge from the absorptive channel.

    Uses A(E) = -ln(T) with T averaged over rotations, lightly smoothed, and
    reports the energy of steepest absorption rise (max dA/dE) and the
    half-rise energy between the low- and high-energy plateaus. Read-only
    reference for comparison; the pipeline does not use this for Eg.
    """
    energy = np.asarray(energy_eV, dtype=np.float64)
    transmission = np.asarray(m11_transmission, dtype=np.float64)
    if transmission.ndim == 2:
        transmission = np.nanmean(transmission, axis=0)

    order = np.argsort(energy)
    e = energy[order]
    t = transmission[order]
    finite = np.isfinite(e) & np.isfinite(t)
    e = e[finite]
    t = np.clip(t[finite], 1.0e-6, None)
    if e.size < 8:
        return {
            "edge_max_slope_eV": np.nan,
            "edge_half_rise_eV": np.nan,
            "edge_onset_eV": np.nan,
        }

    absorbance = -np.log(t)
    window = _odd_window_points(e, 0.02, minimum=5)
    absorbance_smooth = _running_nanmedian(absorbance, window)
    slope = np.gradient(absorbance_smooth, e)
    edge_max_slope = float(e[int(np.nanargmax(slope))])

    tail = max(3, e.size // 10)
    low_plateau = float(np.nanmedian(absorbance_smooth[:tail]))
    high_plateau = float(np.nanmedian(absorbance_smooth[-tail:]))
    half_level = 0.5 * (low_plateau + high_plateau)
    crossings = np.where(absorbance_smooth >= half_level)[0]
    edge_half = float(e[crossings[0]]) if crossings.size else np.nan

    mad = float(np.nanmedian(np.abs(absorbance_smooth[:tail] - low_plateau)))
    scale = 1.4826 * mad if mad > 0 else float(np.nanstd(absorbance_smooth[:tail]))
    threshold = low_plateau + 3.0 * (scale if np.isfinite(scale) and scale > 0 else 1.0)
    above = np.where(absorbance_smooth >= threshold)[0]
    edge_onset = float(e[above[0]]) if above.size else np.nan

    return {
        "edge_max_slope_eV": edge_max_slope,
        "edge_half_rise_eV": edge_half,
        "edge_onset_eV": edge_onset,
    }


def lowest_energy_pair(estimates: list[dict], tolerance_meV: float = 7.5) -> dict | None:
    """Re-pair LD/LB estimates choosing the lowest-energy admissible pair.

    Same admissibility as the library consensus (overlapping energy windows OR
    splitting agreement within tolerance), but ranked by transition energy
    rather than by feature strength / smallest spread.
    """
    usable = [
        estimate
        for estimate in estimates
        if estimate.get("success")
        and np.isfinite(float(estimate.get("splitting_meV", np.nan)))
    ]
    lds = [e for e in usable if e.get("term_prefix") == "linear_dichroism"]
    lbs = [e for e in usable if e.get("term_prefix") == "linear_birefringence"]

    candidates = []
    for ld in lds:
        for lb in lbs:
            overlap = dd._estimate_energy_windows_overlap(ld, lb)
            difference = abs(float(ld["splitting_meV"]) - float(lb["splitting_meV"]))
            if not (overlap or difference <= tolerance_meV):
                continue
            lowers = [
                float(ld.get("lower_transition_eV", np.nan)),
                float(lb.get("lower_transition_eV", np.nan)),
            ]
            lowers = [x for x in lowers if np.isfinite(x)]
            bandgap = float(np.mean(lowers)) if (overlap and lowers) else np.nan
            candidates.append(
                {
                    "bandgap_eV": bandgap,
                    "splitting_meV": float(
                        np.mean([ld["splitting_meV"], lb["splitting_meV"]])
                    ),
                    "spread_meV": difference,
                    "energy_windows_overlap": overlap,
                }
            )

    if not candidates:
        return None
    candidates.sort(
        key=lambda c: (
            not np.isfinite(c["bandgap_eV"]),
            c["bandgap_eV"] if np.isfinite(c["bandgap_eV"]) else 1.0e9,
        )
    )
    return candidates[0]


def _fmt(value: float, digits: int = 4) -> str:
    if value is None or not np.isfinite(value):
        return "   n/a"
    return f"{value:.{digits}f}"


def run(path: Path) -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="logm result may be inaccurate*")
        result = dd.decompose_woollam_dat(str(path))
    energy = np.asarray(result["energy_eV"], dtype=np.float64)
    print(f"file: {path.name}")
    print(
        f"  rotations={result['rotations_deg'].size}  "
        f"energy points={energy.size}  "
        f"range={np.nanmin(energy):.3f}-{np.nanmax(energy):.3f} eV"
    )

    warning_messages = result.get("diagnostics", {}).get("warnings", [])
    if warning_messages:
        print("  decomposition warnings:")
        for warning in warning_messages:
            print(f"    - {warning}")

    edge = absorption_edge_eV(energy, result["m11_transmission"])
    print()
    print("absorptive-channel band edge (reference, from -ln T):")
    print(f"  max-slope edge   Eg = {_fmt(edge['edge_max_slope_eV'])} eV  (pipeline default)")
    print(f"  half-rise edge   Eg = {_fmt(edge['edge_half_rise_eV'])} eV")
    print(f"  onset edge       Eg = {_fmt(edge['edge_onset_eV'])} eV")

    print()
    header = (
        f"{'fit component':>14} | {'pairing':>14} | "
        f"{'Eg (eV)':>9} | {'split (meV)':>11} | {'spread (meV)':>12}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for component in ("real", "imag"):
        estimate = dd.estimate_valence_band_splittings(result, fit_component=component)
        estimates = estimate["estimates"]
        primary = estimate["consensus"].get("primary")

        if primary is not None:
            rows.append(
                (
                    component,
                    "strongest",
                    primary.get("bandgap_eV", np.nan),
                    primary.get("splitting_meV", np.nan),
                    primary.get("spread_meV", np.nan),
                )
            )
        else:
            rows.append((component, "strongest", np.nan, np.nan, np.nan))

        lowest = lowest_energy_pair(estimates)
        if lowest is not None:
            rows.append(
                (
                    component,
                    "lowest-energy",
                    lowest["bandgap_eV"],
                    lowest["splitting_meV"],
                    lowest["spread_meV"],
                )
            )
        else:
            rows.append((component, "lowest-energy", np.nan, np.nan, np.nan))

    for component, pairing, eg, split, spread in rows:
        print(
            f"{component:>14} | {pairing:>14} | "
            f"{_fmt(eg):>9} | {_fmt(split, 1):>11} | {_fmt(spread, 1):>12}"
        )

    eg_values = [row[2] for row in rows if np.isfinite(row[2])]
    if eg_values:
        print()
        print(
            f"anisotropy Eg spread across choices: "
            f"{min(eg_values):.4f}-{max(eg_values):.4f} eV "
            f"(range {1000.0 * (max(eg_values) - min(eg_values)):.0f} meV)"
        )
        if np.isfinite(edge["edge_max_slope_eV"]):
            offsets = [1000.0 * (eg - edge["edge_max_slope_eV"]) for eg in eg_values]
            print(
                "offset of anisotropy Eg vs absorptive max-slope edge: "
                f"{min(offsets):+.0f} to {max(offsets):+.0f} meV"
            )


def _default_dat() -> Path | None:
    here = Path(__file__).resolve().parent
    rotation_series = sorted(here.glob("*_X_*.dat"))
    if rotation_series:
        return rotation_series[0]
    any_dat = sorted(here.glob("*.dat"))
    return any_dat[0] if any_dat else None


def main(argv: list[str]) -> int:
    if len(argv) > 1:
        path = Path(argv[1])
    else:
        path = _default_dat()
        if path is None:
            print("No .dat file found. Pass a path explicitly.", file=sys.stderr)
            return 2
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return 2
    run(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
