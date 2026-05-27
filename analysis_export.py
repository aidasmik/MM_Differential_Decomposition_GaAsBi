"""Shared export helpers for decomposition analysis outputs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


FIT_CURVE_FIELDS = [
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

FEATURE_FIELDS = [
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

SPLITTING_FIELDS = [
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

CONSENSUS_FIELDS = [
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
    "transition_tolerance_meV",
    "lower_transition_spread_meV",
    "upper_transition_spread_meV",
    "center_spread_meV",
    "within_agreement_tolerance",
    "transitions_within_tolerance",
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

SHORT_REPORT_FIELDS = ["section", "metric", "value", "unit", "note"]


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_fit_csv(
    path: Path,
    fit: dict[str, Any],
    source_energy_eV: np.ndarray | None = None,
) -> None:
    collapsed = fit["collapsed"]
    spectrum = np.asarray(collapsed["spectrum"])
    scatter = np.asarray(collapsed["scatter"])
    energy = np.asarray(fit["energy_eV"], dtype=np.float64)
    if source_energy_eV is None:
        source_energy_eV = fit.get("source_energy_eV", energy)
    full_energy = np.asarray(source_energy_eV, dtype=np.float64)
    collapsed_by_energy = {
        float(e): (complex(s), float(abs(sc)))
        for e, s, sc in zip(full_energy, spectrum, scatter)
    }

    values = np.asarray(fit["values"])
    fitted = np.asarray(fit["fitted_values"])
    residuals = np.asarray(fit["residuals"])

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIT_CURVE_FIELDS)
        for e_v, value, fitted_value, residual in zip(energy, values, fitted, residuals):
            collapsed_value, scatter_abs = collapsed_by_energy.get(
                float(e_v), (complex(np.nan, np.nan), np.nan)
            )
            writer.writerow(
                [
                    f"{float(e_v):.10g}",
                    f"{float(np.real(value)):.10g}",
                    f"{float(np.imag(value)):.10g}",
                    f"{float(np.real(fitted_value)):.10g}",
                    f"{float(np.imag(fitted_value)):.10g}",
                    f"{float(np.real(residual)):.10g}",
                    f"{float(np.imag(residual)):.10g}",
                    f"{float(np.real(collapsed_value)):.10g}",
                    f"{float(np.imag(collapsed_value)):.10g}",
                    f"{scatter_abs:.10g}",
                ]
            )


def write_feature_csv(path: Path, feature_scan: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FEATURE_FIELDS)
        writer.writeheader()
        for feature in feature_scan.get("features", []):
            writer.writerow({field: feature.get(field, "") for field in FEATURE_FIELDS})


def write_splitting_csv(path: Path, splitting_estimates: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SPLITTING_FIELDS)
        writer.writeheader()
        for rank, estimate in enumerate(splitting_estimates.get("estimates", []), start=1):
            window = estimate.get("energy_window_eV", ("", ""))
            row = {field: estimate.get(field, "") for field in SPLITTING_FIELDS}
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


def write_consensus_csv(path: Path, splitting_estimates: dict[str, Any]) -> None:
    consensus = splitting_estimates.get("results", splitting_estimates.get("consensus", {}))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CONSENSUS_FIELDS)
        writer.writeheader()
        for match in consensus.get("matches", []):
            row = {}
            for field in CONSENSUS_FIELDS:
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


def write_short_report_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHORT_REPORT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SHORT_REPORT_FIELDS})


def mark_features_inside_fit_window(
    feature_scan: dict[str, Any],
    fit: dict[str, Any] | None,
) -> None:
    if fit is None:
        return
    window = fit.get("energy_window_eV", (None, None))
    if window[0] is None or window[1] is None:
        return
    lower = float(window[0])
    upper = float(window[1])
    for feature in feature_scan.get("features", []):
        energy = float(feature["energy_eV"])
        feature["inside_split_fit_window"] = lower <= energy <= upper
