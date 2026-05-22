"""Fit a split critical-point transition model to decomposed anisotropy spectra."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import warnings

import numpy as np

import differential_decomposition as dd


def _json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_fit_csv(path: Path, fit: dict) -> None:
    collapsed = fit["collapsed"]
    spectrum = np.asarray(collapsed["spectrum"])
    scatter = np.asarray(collapsed["scatter"])
    energy = np.asarray(fit["energy_eV"], dtype=np.float64)
    values = np.asarray(fit["values"])
    fitted = np.asarray(fit["fitted_values"])
    residuals = np.asarray(fit["residuals"])

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "energy_eV",
                "fit_data_real",
                "fit_data_imag",
                "fit_real",
                "fit_imag",
                "residual_real",
                "residual_imag",
                "collapsed_real",
                "collapsed_imag",
                "rotation_scatter_abs",
            ]
        )
        full_energy = np.asarray(fit.get("source_energy_eV", energy), dtype=np.float64)
        collapsed_by_energy = {
            float(e): (complex(s), float(abs(sc)))
            for e, s, sc in zip(full_energy, spectrum, scatter)
        }
        for e, y, yhat, resid in zip(energy, values, fitted, residuals):
            collapsed_value, scatter_abs = collapsed_by_energy.get(
                float(e), (complex(np.nan, np.nan), np.nan)
            )
            writer.writerow(
                [
                    f"{float(e):.10g}",
                    f"{float(np.real(y)):.10g}",
                    f"{float(np.imag(y)):.10g}",
                    f"{float(np.real(yhat)):.10g}",
                    f"{float(np.imag(yhat)):.10g}",
                    f"{float(np.real(resid)):.10g}",
                    f"{float(np.imag(resid)):.10g}",
                    f"{float(np.real(collapsed_value)):.10g}",
                    f"{float(np.imag(collapsed_value)):.10g}",
                    f"{scatter_abs:.10g}",
                ]
            )


def _write_feature_csv(path: Path, feature_scan: dict) -> None:
    fields = [
        "rank",
        "term_prefix",
        "energy_eV",
        "kind",
        "component",
        "component_value",
        "baseline_value",
        "detrended_value",
        "amplitude_abs",
        "prominence_abs",
        "local_noise",
        "z_score",
        "prominence_z",
        "rotation_scatter_abs",
        "scatter_ratio",
        "baseline_width_eV",
        "score",
        "inside_split_fit_window",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for feature in feature_scan.get("features", []):
            writer.writerow({field: feature.get(field, "") for field in fields})


def _write_splitting_csv(path: Path, splitting_estimates: dict) -> None:
    fields = [
        "rank",
        "term_prefix",
        "success",
        "splitting_meV",
        "splitting_eV",
        "lower_transition_eV",
        "upper_transition_eV",
        "center_eV",
        "broadening_meV",
        "energy_window_min_eV",
        "energy_window_max_eV",
        "feature_energies_eV",
        "feature_ranks",
        "assignment_quality",
        "normalization_scale",
        "n_initial_guesses",
        "candidate_delta_meV",
        "near_best_delta_meV",
        "delta_std_meV",
        "fit_stability",
        "rmse",
        "mae",
        "n_points",
        "message",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, estimate in enumerate(splitting_estimates.get("estimates", []), start=1):
            window = estimate.get("energy_window_eV", ("", ""))
            row = {field: estimate.get(field, "") for field in fields}
            row["rank"] = rank
            row["energy_window_min_eV"] = window[0] if len(window) > 0 else ""
            row["energy_window_max_eV"] = window[1] if len(window) > 1 else ""
            row["feature_energies_eV"] = ";".join(
                f"{float(energy):.10g}" for energy in estimate.get("feature_energies_eV", [])
            )
            row["feature_ranks"] = ";".join(
                str(rank_value) for rank_value in estimate.get("feature_ranks", [])
            )
            row["candidate_delta_meV"] = ";".join(
                f"{float(delta):.10g}" for delta in estimate.get("candidate_delta_meV", [])
            )
            row["near_best_delta_meV"] = ";".join(
                f"{float(delta):.10g}" for delta in estimate.get("near_best_delta_meV", [])
            )
            writer.writerow(row)


def _write_consensus_csv(path: Path, splitting_estimates: dict) -> None:
    fields = [
        "rank",
        "bandgap_eV",
        "recommended_delta_vb_meV",
        "recommended_delta_source",
        "kk_splitting_meV",
        "kk_component_max_difference_meV",
        "kk_fit_success",
        "kk_fit_message",
        "bandgap_spread_eV",
        "upper_transition_eV",
        "upper_transition_spread_eV",
        "center_eV",
        "splitting_meV",
        "spread_meV",
        "std_meV",
        "agreement_tolerance_meV",
        "within_agreement_tolerance",
        "confidence",
        "basis",
        "energy_windows_overlap",
        "requires_manual_delta_vb",
        "math_warnings",
        "component_estimate_ranks",
        "component_terms",
        "component_splittings_meV",
        "component_lower_transition_eV",
        "component_upper_transition_eV",
        "component_center_eV",
        "component_assignment_quality",
        "component_fit_stability",
        "component_energy_windows_eV",
    ]
    consensus = splitting_estimates.get("results", splitting_estimates.get("consensus", {}))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for match in consensus.get("matches", []):
            row = {}
            for field in fields:
                value = match.get(field, "")
                if isinstance(value, float) and not np.isfinite(value):
                    value = ""
                row[field] = value
            row["component_estimate_ranks"] = ";".join(
                str(rank) for rank in match.get("component_estimate_ranks", [])
            )
            row["component_terms"] = ";".join(match.get("component_terms", []))
            row["component_splittings_meV"] = ";".join(
                f"{float(value):.10g}"
                for value in match.get("component_splittings_meV", [])
            )
            row["component_lower_transition_eV"] = ";".join(
                f"{float(value):.10g}"
                for value in match.get("component_lower_transition_eV", [])
            )
            row["component_upper_transition_eV"] = ";".join(
                f"{float(value):.10g}"
                for value in match.get("component_upper_transition_eV", [])
            )
            row["component_center_eV"] = ";".join(
                f"{float(value):.10g}"
                for value in match.get("component_center_eV", [])
            )
            row["component_assignment_quality"] = ";".join(
                match.get("component_assignment_quality", [])
            )
            row["component_fit_stability"] = ";".join(
                match.get("component_fit_stability", [])
            )
            row["math_warnings"] = ";".join(
                str(warning) for warning in match.get("math_warnings", [])
            )
            row["component_energy_windows_eV"] = ";".join(
                f"{float(window[0]):.10g}-{float(window[1]):.10g}"
                for window in match.get("component_energy_windows_eV", [])
                if len(window) >= 2
            )
            writer.writerow(row)


def _write_short_report_csv(path: Path, rows: list[dict]) -> None:
    fields = ["section", "metric", "value", "unit", "note"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _mark_features_inside_fit_window(feature_scan: dict, fit: dict) -> None:
    window = fit.get("energy_window_eV", (None, None))
    if window[0] is None or window[1] is None:
        return
    lower = float(window[0])
    upper = float(window[1])
    for feature in feature_scan.get("features", []):
        energy = float(feature["energy_eV"])
        feature["inside_split_fit_window"] = lower <= energy <= upper


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose a Woollam Mueller .dat file and fit the rotation-collapsed "
            "anisotropy spectrum with two split critical-point oscillators."
        )
    )
    parser.add_argument(
        "dat_path",
        nargs="?",
        default="VGA1044a_MMt_X_0-135-5.dat",
        help="Woollam .dat export to analyze.",
    )
    parser.add_argument(
        "--term-prefix",
        default="linear_dichroism",
        choices=["linear_dichroism", "linear_birefringence"],
        help="Anisotropy vector to collapse and fit.",
    )
    parser.add_argument("--energy-min", type=float, default=1.00)
    parser.add_argument("--energy-max", type=float, default=1.40)
    parser.add_argument("--axis-offset-deg", type=float, default=0.0)
    parser.add_argument(
        "--vector-part",
        default="real",
        choices=["real", "imag", "abs"],
        help="Part of the decomposed x/y anisotropy terms used for rotation collapse.",
    )
    parser.add_argument(
        "--fit-component",
        default="real",
        choices=["real", "imag", "abs", "complex"],
        help="Component of the collapsed complex spectrum used by the fit.",
    )
    parser.add_argument("--max-delta-eV", type=float, default=0.20)
    parser.add_argument("--exponent", type=float, default=-0.5)
    parser.add_argument(
        "--no-feature-scan",
        action="store_true",
        help=(
            "Skip automatic feature-candidate scanning and derived valence-band "
            "splitting estimates."
        ),
    )
    parser.add_argument("--output-dir", default="split_transition_fit_output")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matplotlib_config_dir = output_dir / ".matplotlib"
    matplotlib_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config_dir))

    warnings.filterwarnings("ignore", message="logm result may be inaccurate*")
    result = dd.decompose_woollam_dat(args.dat_path, output_dir=output_dir)
    fit = dd.fit_split_transition_to_decomposition(
        result,
        term_prefix=args.term_prefix,
        energy_min=args.energy_min,
        energy_max=args.energy_max,
        vector_part=args.vector_part,
        fit_component=args.fit_component,
        axis_offset_deg=args.axis_offset_deg,
        exponent=args.exponent,
        max_delta_eV=args.max_delta_eV,
    )
    fit["source_energy_eV"] = result["energy_eV"]
    dd.plot_split_transition_fit(fit)

    feature_scan = None
    splitting_estimates = None
    if not args.no_feature_scan:
        feature_scan = dd.detect_decomposition_features(result)
        _mark_features_inside_fit_window(feature_scan, fit)
        dd.plot_feature_scan(result, feature_scan)
        _write_feature_csv(output_dir / "feature_candidates.csv", feature_scan)
        splitting_estimates = dd.estimate_valence_band_splittings(
            result,
            feature_scan,
            exponent=args.exponent,
            max_delta_eV=args.max_delta_eV,
        )
        _write_splitting_csv(
            output_dir / "valence_band_splitting.csv",
            splitting_estimates,
        )
        _write_consensus_csv(
            output_dir / "valence_band_results.csv",
            splitting_estimates,
        )

    summary = {
        "dat_path": str(args.dat_path),
        "term_prefix": args.term_prefix,
        "energy_window_eV": [args.energy_min, args.energy_max],
        "axis_offset_deg": args.axis_offset_deg,
        "vector_part": args.vector_part,
        "fit_component": args.fit_component,
        "exponent": args.exponent,
        "max_delta_eV": args.max_delta_eV,
        "success": fit["success"],
        "message": fit["message"],
        "parameters": fit["parameters"],
        "rmse": fit["rmse"],
        "mae": fit["mae"],
        "n_points": fit["n_points"],
        "decomposition_warnings": result["diagnostics"]["warnings"],
        "dat_warnings": result["dat_diagnostics"]["warnings"],
    }
    if feature_scan is not None:
        summary["feature_scan"] = feature_scan
    if splitting_estimates is not None:
        summary["valence_band_splitting"] = splitting_estimates
    report_rows = dd.build_short_report_rows(
        result,
        fit=fit,
        feature_scan=feature_scan,
        splitting_estimates=splitting_estimates,
        dat_path=args.dat_path,
    )
    _write_short_report_csv(output_dir / "short_report.csv", report_rows)
    summary_path = output_dir / "split_transition_fit_summary.json"
    summary_path.write_text(json.dumps(_json_ready(summary), indent=2), encoding="utf-8")
    _write_fit_csv(output_dir / "split_transition_fit_curve.csv", fit)

    params = fit["parameters"]
    print(f"Fit success: {fit['success']} ({fit['message']})")
    print(
        "Split: "
        f"{params['delta_meV']:.2f} meV "
        f"({params['lower_transition_eV']:.5f} -> {params['upper_transition_eV']:.5f} eV)"
    )
    print(
        "Center/Gamma: "
        f"{params['center_eV']:.5f} eV / {1000.0 * params['broadening_eV']:.2f} meV"
    )
    print(f"RMSE: {fit['rmse']:.6g} over {fit['n_points']} points")
    if feature_scan is not None and feature_scan.get("features"):
        print("Candidate features:")
        for feature in feature_scan["features"][:8]:
            scatter_ratio = feature.get("scatter_ratio")
            scatter_text = "n/a" if scatter_ratio is None else f"{float(scatter_ratio):.1f}"
            window_text = (
                "inside fit window"
                if feature.get("inside_split_fit_window")
                else "outside fit window"
            )
            print(
                f"  {feature['term_prefix']}: {float(feature['energy_eV']):.5f} eV "
                f"{feature['kind']}; score={float(feature['score']):.1f}; "
                f"z={float(feature['z_score']):.1f}; "
                f"scatter_ratio={scatter_text}; {window_text}"
            )
    if splitting_estimates is not None and splitting_estimates.get("estimates"):
        consensus = splitting_estimates.get(
            "results",
            splitting_estimates.get("consensus", {}),
        )
        primary = consensus.get("primary")
        if primary:
            values = ", ".join(
                f"{float(value):.2f}"
                for value in primary.get("component_splittings_meV", [])
            )
            bandgap = float(primary.get("bandgap_eV", np.nan))
            bandgap_text = f"Eg={bandgap:.5f} eV, " if np.isfinite(bandgap) else ""
            try:
                recommended = float(primary.get("recommended_delta_vb_meV", np.nan))
            except (TypeError, ValueError):
                recommended = np.nan
            recommended_text = (
                f"recommended_delta_vb={recommended:.2f} meV, "
                if np.isfinite(recommended)
                else "recommended_delta_vb=manual review, "
            )
            try:
                kk_value = float(primary.get("kk_splitting_meV", np.nan))
            except (TypeError, ValueError):
                kk_value = np.nan
            kk_text = (
                f"kk_split={kk_value:.2f} meV, "
                if np.isfinite(kk_value)
                else ""
            )
            print(
                "Results: "
                f"{bandgap_text}"
                f"{recommended_text}"
                f"{kk_text}"
                f"raw_ld_lb_mean={float(primary['splitting_meV']):.2f} meV "
                f"(components: {values} meV; "
                f"spread={float(primary['spread_meV']):.2f} meV; "
                f"{primary['confidence']}; {primary['basis']})"
            )
            warnings_text = "; ".join(primary.get("math_warnings", []))
            if warnings_text:
                print(f"  Math warning: {warnings_text}")
            elif primary.get("within_agreement_tolerance") is False:
                print(
                    "  Note: energy windows overlap, but the LD/LB splitting "
                    "spread is above the agreement tolerance."
                )
        print("Valence-band splitting estimates:")
        print(f"  {splitting_estimates.get('note', '')}")
        for estimate in splitting_estimates["estimates"][:6]:
            if not estimate.get("success"):
                print(
                    f"  {estimate['term_prefix']}: fit failed in "
                    f"{estimate['energy_window_eV'][0]:.4f}-"
                    f"{estimate['energy_window_eV'][1]:.4f} eV"
                )
                continue
            print(
                f"  {estimate['term_prefix']}: "
                f"{float(estimate['splitting_meV']):.2f} meV "
                f"({float(estimate['lower_transition_eV']):.5f} -> "
                f"{float(estimate['upper_transition_eV']):.5f} eV), "
                f"window {float(estimate['energy_window_eV'][0]):.4f}-"
                f"{float(estimate['energy_window_eV'][1]):.4f} eV, "
                f"{estimate.get('assignment_quality', 'unknown')}, "
                f"{estimate.get('fit_stability', 'unknown')}"
            )
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
