"""Tkinter app for Mueller-log decomposition and split-transition fitting."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import traceback
import warnings

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

import differential_decomposition as dd
import shared_results_database as sdb


TERM_EXPORTS = [
    "linear_dichroism_x",
    "linear_dichroism_y",
    "linear_dichroism_magnitude",
    "linear_birefringence_x",
    "linear_birefringence_y",
    "linear_birefringence_magnitude",
    "circular_dichroism",
    "circular_birefringence",
    "dichroism_axis_angle_deg",
    "birefringence_axis_angle_deg",
]


def json_ready(value):
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


def finite_summary(values: np.ndarray) -> dict[str, float | int]:
    arr = np.asarray(values)
    real = np.real(arr)
    finite = np.isfinite(real)
    if not np.any(finite):
        return {"finite_count": 0, "min": np.nan, "max": np.nan, "mean": np.nan}
    return {
        "finite_count": int(np.count_nonzero(finite)),
        "min": float(np.nanmin(real)),
        "max": float(np.nanmax(real)),
        "mean": float(np.nanmean(real)),
    }


def write_terms_csv(path: Path, result: dict) -> None:
    energy = np.asarray(result["energy_eV"], dtype=np.float64)
    rotations = np.asarray(result["rotations_deg"], dtype=np.float64)
    terms = result["terms"]
    available_terms = [term for term in TERM_EXPORTS if term in terms]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        header = ["rotation_deg", "energy_eV"]
        for term_name in available_terms:
            header.extend([f"{term_name}_real", f"{term_name}_imag"])
        writer.writerow(header)
        for rotation_index, rotation in enumerate(rotations):
            for energy_index, e_v in enumerate(energy):
                row = [f"{rotation:.10g}", f"{e_v:.10g}"]
                for term_name in available_terms:
                    value = np.asarray(terms[term_name])[rotation_index, energy_index]
                    row.extend(
                        [
                            f"{float(np.real(value)):.10g}",
                            f"{float(np.imag(value)):.10g}",
                        ]
                    )
                writer.writerow(row)


def write_fit_csv(path: Path, fit: dict, source_energy_eV: np.ndarray) -> None:
    collapsed = fit["collapsed"]
    spectrum = np.asarray(collapsed["spectrum"])
    scatter = np.asarray(collapsed["scatter"])
    collapsed_by_energy = {
        float(e): (complex(s), float(abs(sc)))
        for e, s, sc in zip(np.asarray(source_energy_eV), spectrum, scatter)
    }

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


def write_feature_csv(path: Path, feature_scan: dict) -> None:
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


def write_splitting_csv(path: Path, splitting_estimates: dict) -> None:
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


def write_consensus_csv(path: Path, splitting_estimates: dict) -> None:
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


def write_short_report_csv(path: Path, rows: list[dict]) -> None:
    fields = ["section", "metric", "value", "unit", "note"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _mark_features_inside_fit_window(feature_scan: dict, fit: dict | None) -> None:
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


def build_summary(
    result: dict,
    fit: dict | None,
    feature_scan: dict | None,
    splitting_estimates: dict | None,
    dat_path: Path,
    output_dir: Path,
) -> dict:
    energy = np.asarray(result["energy_eV"], dtype=np.float64)
    rotations = np.asarray(result["rotations_deg"], dtype=np.float64)
    summary = {
        "dat_path": str(dat_path),
        "output_dir": str(output_dir),
        "thickness": result["thickness"],
        "is_differential": result["is_differential"],
        "n_energy": int(energy.size),
        "n_rotations": int(rotations.size),
        "energy_min_eV": float(np.nanmin(energy)),
        "energy_max_eV": float(np.nanmax(energy)),
        "rotations_deg": rotations.tolist(),
        "term_summaries": {
            term: finite_summary(result["terms"][term])
            for term in TERM_EXPORTS
            if term in result["terms"]
        },
        "decomposition_warnings": result["diagnostics"]["warnings"],
        "dat_warnings": result["dat_diagnostics"]["warnings"],
    }
    if feature_scan is not None:
        _mark_features_inside_fit_window(feature_scan, fit)
        summary["feature_scan"] = feature_scan
    if splitting_estimates is not None:
        summary["valence_band_splitting"] = splitting_estimates
    if fit is not None:
        summary["split_transition_fit"] = {
            "success": fit["success"],
            "message": fit["message"],
            "term_prefix": fit["term_prefix"],
            "energy_window_eV": list(fit["energy_window_eV"]),
            "vector_part": fit["vector_part"],
            "component": fit["component"],
            "axis_offset_deg": fit["axis_offset_deg"],
            "exponent": fit["exponent"],
            "parameters": fit["parameters"],
            "rmse": fit["rmse"],
            "mae": fit["mae"],
            "n_points": fit["n_points"],
        }
    return summary


def run_analysis(config: dict, progress: queue.Queue) -> dict:
    dat_path = Path(config["dat_path"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    matplotlib_config_dir = output_dir / ".matplotlib"
    matplotlib_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config_dir))

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    warnings.filterwarnings("ignore", message="logm result may be inaccurate*")

    progress.put(("log", "Reading Woollam data and running Mueller-log decomposition..."))
    result = dd.decompose_woollam_dat(
        dat_path,
        thickness=config["thickness"],
        output_dir=output_dir,
    )

    progress.put(("log", "Saving decomposition overview and term CSV..."))
    overview_fig = dd.plot_decomposition_overview(result)
    plt.close(overview_fig)
    terms_csv = output_dir / "decomposition_terms.csv"
    write_terms_csv(terms_csv, result)

    progress.put(("log", "Scanning collapsed spectra for candidate features..."))
    feature_scan = dd.detect_decomposition_features(result)
    feature_csv = output_dir / "feature_candidates.csv"
    feature_fig = dd.plot_feature_scan(result, feature_scan)
    plt.close(feature_fig)
    feature_plot_path = output_dir / "feature_candidates.png"

    progress.put(("log", "Estimating valence-band splitting from local split fits..."))
    splitting_estimates = dd.estimate_valence_band_splittings(
        result,
        feature_scan,
        exponent=config.get("exponent", -0.5),
        max_delta_eV=config.get("max_delta_eV") or 0.20,
    )
    splitting_csv = output_dir / "valence_band_splitting.csv"
    results_csv = output_dir / "valence_band_results.csv"
    short_report_csv = output_dir / "short_report.csv"

    fit = None
    fit_plot_path = None
    fit_csv = None
    if config["run_split_fit"]:
        progress.put(("log", "Fitting split-transition model..."))
        fit = dd.fit_split_transition_to_decomposition(
            result,
            term_prefix=config["term_prefix"],
            energy_min=config["energy_min"],
            energy_max=config["energy_max"],
            vector_part=config["vector_part"],
            fit_component=config["fit_component"],
            axis_offset_deg=config["axis_offset_deg"],
            exponent=config["exponent"],
            max_delta_eV=config["max_delta_eV"],
        )
        fit["source_energy_eV"] = result["energy_eV"]
        fit_fig = dd.plot_split_transition_fit(fit)
        plt.close(fit_fig)
        window = fit["energy_window_eV"]
        fit_plot_path = (
            output_dir
            / f"{fit['term_prefix']}_split_fit_{float(window[0]):.3f}_{float(window[1]):.3f}.png"
        )
        fit_csv = output_dir / "split_transition_fit_curve.csv"
        write_fit_csv(fit_csv, fit, result["energy_eV"])

    summary = build_summary(
        result,
        fit,
        feature_scan,
        splitting_estimates,
        dat_path,
        output_dir,
    )
    write_feature_csv(feature_csv, feature_scan)
    write_splitting_csv(splitting_csv, splitting_estimates)
    write_consensus_csv(results_csv, splitting_estimates)
    report_rows = dd.build_short_report_rows(
        result,
        fit=fit,
        feature_scan=feature_scan,
        splitting_estimates=splitting_estimates,
        dat_path=dat_path,
    )
    write_short_report_csv(short_report_csv, report_rows)
    summary_path = output_dir / "analysis_summary.json"
    summary_path.write_text(json.dumps(json_ready(summary), indent=2), encoding="utf-8")

    progress.put(("log", "Done."))
    return {
        "result": result,
        "fit": fit,
        "summary": summary,
        "summary_path": summary_path,
        "terms_csv": terms_csv,
        "overview_plot": output_dir / "decomposition_overview.png",
        "feature_csv": feature_csv,
        "feature_plot": feature_plot_path,
        "splitting_csv": splitting_csv,
        "results_csv": results_csv,
        "short_report_csv": short_report_csv,
        "fit_plot": fit_plot_path,
        "fit_csv": fit_csv,
    }


class MuellerDecompositionApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Mueller Decomposition")
        self.geometry("920x720")
        self.minsize(820, 620)

        self.messages: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None
        self.last_output_dir: Path | None = None
        self.last_analysis_output: dict | None = None
        self.current_summary: dict | None = None
        self.current_consensus: dict | None = None
        self.plot_images: list[tk.PhotoImage] = []
        self.dynamic_output_tabs: list[ttk.Frame] = []
        self.database_plot_tabs: list[ttk.Frame] = []
        self.feature_tree: ttk.Treeview | None = None
        self.splitting_tree: ttk.Treeview | None = None
        self.results_tree: ttk.Treeview | None = None
        self.database_tree: ttk.Treeview | None = None
        self.result_match_by_iid: dict[str, dict] = {}
        self.database_record_by_iid: dict[str, dict] = {}

        self.dat_path = tk.StringVar(value="")
        self.output_dir = tk.StringVar(value="")
        self.thickness = tk.StringVar(value="")
        self.run_split_fit = tk.BooleanVar(value=True)
        self.term_prefix = tk.StringVar(value="linear_dichroism")
        self.energy_min = tk.StringVar(value="1.00")
        self.energy_max = tk.StringVar(value="1.40")
        self.axis_offset_deg = tk.StringVar(value="0.0")
        self.vector_part = tk.StringVar(value="real")
        self.fit_component = tk.StringVar(value="real")
        self.max_delta_eV = tk.StringVar(value="0.20")
        self.exponent = tk.StringVar(value="-0.5")
        self.status = tk.StringVar(value="Choose a data file to start.")
        detected_database = sdb.find_database_clone(Path.cwd())
        self.database_dir = tk.StringVar(
            value=str(detected_database or sdb.DEFAULT_DATABASE_DIR)
        )
        self.db_sample_id = tk.StringVar(value="")
        self.db_bi_percent = tk.StringVar(value="")
        self.db_temperature_c = tk.StringVar(value="")
        self.db_result_rank = tk.StringVar(value="")
        self.db_eg_eV = tk.StringVar(value="")
        self.db_splitting_meV = tk.StringVar(value="")
        self.db_status = tk.StringVar(value="accepted")
        self.db_analyst = tk.StringVar(value="")
        self.db_message = tk.StringVar(value="")
        self.db_git_message = tk.StringVar(value="Update GaAsBi results database")
        self.db_notes_text: tk.Text | None = None
        self.db_rank_combo: ttk.Combobox | None = None

        self._build_ui()
        self.after(150, self._poll_messages)

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        settings = ttk.LabelFrame(self, text="Inputs", padding=12)
        settings.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 6))
        settings.columnconfigure(1, weight=1)

        ttk.Label(settings, text="Data file").grid(row=0, column=0, sticky="w")
        ttk.Entry(settings, textvariable=self.dat_path).grid(
            row=0, column=1, sticky="ew", padx=8
        )
        ttk.Button(settings, text="Browse", command=self._browse_dat).grid(
            row=0, column=2, sticky="ew"
        )

        ttk.Label(settings, text="Thickness").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(settings, textvariable=self.thickness, width=16).grid(
            row=1, column=1, sticky="w", padx=8, pady=(8, 0)
        )
        ttk.Label(settings, text="blank = integrated generator").grid(
            row=1, column=1, sticky="w", padx=(170, 8), pady=(8, 0)
        )

        ttk.Label(settings, text="Output folder").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(settings, textvariable=self.output_dir).grid(
            row=2, column=1, sticky="ew", padx=8, pady=(8, 0)
        )
        ttk.Button(settings, text="Browse", command=self._browse_output).grid(
            row=2, column=2, sticky="ew", pady=(8, 0)
        )

        fit = ttk.LabelFrame(self, text="Split-Transition Fit", padding=12)
        fit.grid(row=1, column=0, sticky="ew", padx=12, pady=6)
        for column in range(8):
            fit.columnconfigure(column, weight=1)

        ttk.Checkbutton(
            fit,
            text="Run split fit",
            variable=self.run_split_fit,
            command=self._toggle_fit_controls,
        ).grid(row=0, column=0, sticky="w")

        ttk.Label(fit, text="Term").grid(row=0, column=1, sticky="e")
        self.term_combo = ttk.Combobox(
            fit,
            textvariable=self.term_prefix,
            values=("linear_dichroism", "linear_birefringence"),
            state="readonly",
            width=20,
        )
        self.term_combo.grid(row=0, column=2, sticky="ew", padx=(6, 12))

        ttk.Label(fit, text="Energy min").grid(row=0, column=3, sticky="e")
        self.energy_min_entry = ttk.Entry(fit, textvariable=self.energy_min, width=10)
        self.energy_min_entry.grid(row=0, column=4, sticky="ew", padx=(6, 12))

        ttk.Label(fit, text="Energy max").grid(row=0, column=5, sticky="e")
        self.energy_max_entry = ttk.Entry(fit, textvariable=self.energy_max, width=10)
        self.energy_max_entry.grid(row=0, column=6, sticky="ew", padx=(6, 12))

        ttk.Label(fit, text="Axis offset deg").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.axis_entry = ttk.Entry(fit, textvariable=self.axis_offset_deg, width=10)
        self.axis_entry.grid(row=1, column=1, sticky="ew", padx=(6, 12), pady=(8, 0))

        ttk.Label(fit, text="Vector part").grid(row=1, column=2, sticky="e", pady=(8, 0))
        self.vector_combo = ttk.Combobox(
            fit,
            textvariable=self.vector_part,
            values=("real", "imag", "abs"),
            state="readonly",
            width=8,
        )
        self.vector_combo.grid(row=1, column=3, sticky="ew", padx=(6, 12), pady=(8, 0))

        ttk.Label(fit, text="Fit component").grid(row=1, column=4, sticky="e", pady=(8, 0))
        self.component_combo = ttk.Combobox(
            fit,
            textvariable=self.fit_component,
            values=("real", "imag", "abs", "complex"),
            state="readonly",
            width=10,
        )
        self.component_combo.grid(row=1, column=5, sticky="ew", padx=(6, 12), pady=(8, 0))

        ttk.Label(fit, text="Max Delta eV").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.delta_entry = ttk.Entry(fit, textvariable=self.max_delta_eV, width=10)
        self.delta_entry.grid(row=2, column=1, sticky="ew", padx=(6, 12), pady=(8, 0))

        ttk.Label(fit, text="CP exponent").grid(row=2, column=2, sticky="e", pady=(8, 0))
        self.exponent_entry = ttk.Entry(fit, textvariable=self.exponent, width=10)
        self.exponent_entry.grid(row=2, column=3, sticky="ew", padx=(6, 12), pady=(8, 0))

        actions = ttk.Frame(self, padding=(12, 0))
        actions.grid(row=3, column=0, sticky="ew", pady=(8, 12))
        actions.columnconfigure(3, weight=1)
        self.run_button = ttk.Button(actions, text="Run Analysis", command=self._start_run)
        self.run_button.grid(row=0, column=0, sticky="w")
        self.open_button = ttk.Button(
            actions,
            text="Open Output Folder",
            command=self._open_output_folder,
            state="disabled",
        )
        self.open_button.grid(row=0, column=1, sticky="w", padx=(8, 0))
        self.progress = ttk.Progressbar(actions, mode="indeterminate", length=180)
        self.progress.grid(row=0, column=2, sticky="w", padx=12)
        ttk.Label(actions, textvariable=self.status).grid(row=0, column=3, sticky="w")

        output = ttk.LabelFrame(self, text="Output", padding=8)
        output.grid(row=2, column=0, sticky="nsew", padx=12, pady=6)
        output.rowconfigure(0, weight=1)
        output.columnconfigure(0, weight=1)

        self.output_notebook = ttk.Notebook(output)
        self.output_notebook.grid(row=0, column=0, sticky="nsew")

        self.results_tab = ttk.Frame(self.output_notebook, padding=4)
        self.results_tab.rowconfigure(0, weight=1)
        self.results_tab.columnconfigure(0, weight=1)
        self.output_notebook.add(self.results_tab, text="Summary")

        self.results_text = scrolledtext.ScrolledText(self.results_tab, wrap="word", height=18)
        self.results_text.grid(row=0, column=0, sticky="nsew")
        self.results_text.configure(state="disabled")

    def _browse_dat(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Woollam .dat file",
            filetypes=(("Data files", "*.dat"), ("All files", "*.*")),
        )
        if not path:
            return
        self.dat_path.set(path)
        if not self.output_dir.get().strip():
            dat_path = Path(path)
            self.output_dir.set(str(dat_path.with_suffix("").parent / f"{dat_path.stem}_results"))

    def _browse_output(self) -> None:
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.output_dir.set(path)

    def _toggle_fit_controls(self) -> None:
        state = "normal" if self.run_split_fit.get() else "disabled"
        readonly_state = "readonly" if self.run_split_fit.get() else "disabled"
        for widget in (
            self.energy_min_entry,
            self.energy_max_entry,
            self.axis_entry,
            self.delta_entry,
            self.exponent_entry,
        ):
            widget.configure(state=state)
        for widget in (self.term_combo, self.vector_combo, self.component_combo):
            widget.configure(state=readonly_state)

    def _parse_optional_float(self, label: str, text: str) -> float | None:
        value = text.strip()
        if value == "":
            return None
        try:
            parsed = float(value)
        except ValueError as exc:
            raise ValueError(f"{label} must be a number.") from exc
        if not np.isfinite(parsed):
            raise ValueError(f"{label} must be finite.")
        return parsed

    def _parse_required_float(self, label: str, text: str) -> float:
        parsed = self._parse_optional_float(label, text)
        if parsed is None:
            raise ValueError(f"{label} is required.")
        return parsed

    def _build_config(self) -> dict:
        dat_path = Path(self.dat_path.get().strip())
        if not dat_path.exists():
            raise ValueError("Choose an existing data file.")
        output_text = self.output_dir.get().strip()
        if not output_text:
            raise ValueError("Choose an output folder.")

        thickness = self._parse_optional_float("Thickness", self.thickness.get())
        if thickness is not None and thickness <= 0.0:
            raise ValueError("Thickness must be positive when provided.")

        run_split_fit = bool(self.run_split_fit.get())
        if run_split_fit:
            energy_min = self._parse_required_float("Energy min", self.energy_min.get())
            energy_max = self._parse_required_float("Energy max", self.energy_max.get())
            if energy_max <= energy_min:
                raise ValueError("Energy max must be greater than energy min.")

            max_delta = self._parse_required_float("Max Delta eV", self.max_delta_eV.get())
            if max_delta <= 0.0:
                raise ValueError("Max Delta eV must be positive.")
            axis_offset = self._parse_required_float(
                "Axis offset deg", self.axis_offset_deg.get()
            )
            exponent = self._parse_required_float("CP exponent", self.exponent.get())
        else:
            energy_min = None
            energy_max = None
            max_delta = None
            axis_offset = 0.0
            exponent = -0.5

        return {
            "dat_path": str(dat_path),
            "output_dir": output_text,
            "thickness": thickness,
            "run_split_fit": run_split_fit,
            "term_prefix": self.term_prefix.get(),
            "energy_min": energy_min,
            "energy_max": energy_max,
            "axis_offset_deg": axis_offset,
            "vector_part": self.vector_part.get(),
            "fit_component": self.fit_component.get(),
            "max_delta_eV": max_delta,
            "exponent": exponent,
        }

    def _start_run(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        try:
            config = self._build_config()
        except Exception as exc:
            messagebox.showerror("Invalid input", str(exc))
            return

        self._clear_results()
        self._append_results("Starting analysis...\n")
        self.status.set("Running...")
        self.run_button.configure(state="disabled")
        self.open_button.configure(state="disabled")
        self.progress.start(10)

        self.worker = threading.Thread(target=self._worker_main, args=(config,), daemon=True)
        self.worker.start()

    def _worker_main(self, config: dict) -> None:
        try:
            output = run_analysis(config, self.messages)
        except Exception:
            self.messages.put(("error", traceback.format_exc()))
        else:
            self.messages.put(("done", output))

    def _poll_messages(self) -> None:
        try:
            while True:
                kind, payload = self.messages.get_nowait()
                if kind == "log":
                    self._append_results(f"{payload}\n")
                elif kind == "error":
                    self._finish_run()
                    self._append_results("\nAnalysis failed:\n")
                    self._append_results(payload)
                    self.status.set("Failed.")
                    messagebox.showerror("Analysis failed", payload.splitlines()[-1])
                elif kind == "done":
                    self._finish_run()
                    self._show_completed(payload)
        except queue.Empty:
            pass
        self.after(150, self._poll_messages)

    def _finish_run(self) -> None:
        self.progress.stop()
        self.run_button.configure(state="normal")

    def _show_completed(self, output: dict) -> None:
        summary = output["summary"]
        self.last_analysis_output = output
        self.current_summary = summary
        self.last_output_dir = Path(summary["output_dir"])
        self.open_button.configure(state="normal")
        self.status.set("Complete.")

        self._append_results("\nSummary\n")
        self._append_results(f"Data file: {summary['dat_path']}\n")
        self._append_results(f"Output folder: {summary['output_dir']}\n")
        self._append_results(
            f"Energy points: {summary['n_energy']} "
            f"({summary['energy_min_eV']:.4f}-{summary['energy_max_eV']:.4f} eV)\n"
        )
        self._append_results(f"Rotations: {summary['n_rotations']}\n")
        if summary["thickness"] is None:
            self._append_results("Thickness: not provided; results are integrated generators.\n")
        else:
            self._append_results(
                f"Thickness: {summary['thickness']}; results are per thickness unit.\n"
            )

        fit_summary = summary.get("split_transition_fit")
        if fit_summary:
            params = fit_summary["parameters"]
            self._append_results("\nSplit-transition fit\n")
            self._append_results(
                f"Delta E: {params['delta_meV']:.2f} meV "
                f"({params['lower_transition_eV']:.5f} -> "
                f"{params['upper_transition_eV']:.5f} eV)\n"
            )
            self._append_results(
                f"Center: {params['center_eV']:.5f} eV; "
                f"Gamma: {1000.0 * params['broadening_eV']:.2f} meV\n"
            )
            self._append_results(
                f"RMSE: {fit_summary['rmse']:.6g}; points: {fit_summary['n_points']}\n"
            )

        feature_scan = summary.get("feature_scan")
        if feature_scan and feature_scan.get("features"):
            self._append_results("\nCandidate features\n")
            for feature in feature_scan["features"][:8]:
                scatter_ratio = feature.get("scatter_ratio")
                scatter_text = (
                    "n/a" if scatter_ratio is None else f"{float(scatter_ratio):.1f}"
                )
                window_text = (
                    "inside fit window"
                    if feature.get("inside_split_fit_window")
                    else "outside fit window"
                )
                self._append_results(
                    f"{feature['term_prefix']}: {float(feature['energy_eV']):.4f} eV "
                    f"{feature['kind']}; score {float(feature['score']):.1f}; "
                    f"z {float(feature['z_score']):.1f}; "
                    f"scatter ratio {scatter_text}; {window_text}\n"
                )

        splitting_estimates = summary.get("valence_band_splitting")
        if splitting_estimates and splitting_estimates.get("estimates"):
            consensus = splitting_estimates.get(
                "results",
                splitting_estimates.get("consensus", {}),
            )
            primary = consensus.get("primary")
            if primary:
                values = primary.get("component_splittings_meV", [])
                values_text = ", ".join(f"{float(value):.2f}" for value in values)
                self._append_results("\nResults\n")
                bandgap = float(primary.get("bandgap_eV", np.nan))
                within_tolerance = primary.get("within_agreement_tolerance")
                if np.isfinite(bandgap):
                    self._append_results(
                        f"Eg: {bandgap:.5f} eV "
                        f"(spread {float(primary['bandgap_spread_eV']):.5f} eV)\n"
                    )
                else:
                    self._append_results(
                        "Eg: not assigned; selected LD/LB result does not use "
                        "overlapping energy windows\n"
                    )
                recommended = self._format_result_value(
                    primary.get("recommended_delta_vb_meV"),
                    2,
                )
                kk_split = self._format_result_value(primary.get("kk_splitting_meV"), 2)
                if recommended:
                    self._append_results(
                        f"Recommended Delta Vb: {recommended} meV "
                        f"({primary.get('recommended_delta_source', '')})\n"
                    )
                else:
                    self._append_results(
                        "Recommended Delta Vb: manual review required; the "
                        "automatic LD/LB split failed one or more math checks.\n"
                    )
                self._append_results(
                    f"Raw LD/LB split mean: {float(primary['splitting_meV']):.2f} meV "
                    f"(components: {values_text} meV; "
                    f"spread {float(primary['spread_meV']):.2f} meV; "
                    f"{primary['confidence']}; {primary['basis']})\n"
                )
                if kk_split:
                    self._append_results(f"Joint KK split: {kk_split} meV\n")
                warnings_text = "; ".join(primary.get("math_warnings", []))
                if warnings_text:
                    self._append_results(f"Math warning: {warnings_text}\n")
                elif within_tolerance is False:
                    self._append_results(
                        "Note: the selected LD/LB energy windows overlap, but "
                        "the splitting spread is above the agreement tolerance.\n"
                    )

            self._append_results("\nValence-band splitting estimates\n")
            self._append_results(f"{splitting_estimates.get('note', '')}\n")
            for estimate in splitting_estimates["estimates"][:6]:
                if not estimate.get("success"):
                    self._append_results(
                        f"{estimate['term_prefix']}: fit failed in "
                        f"{estimate['energy_window_eV'][0]:.4f}-"
                        f"{estimate['energy_window_eV'][1]:.4f} eV; "
                        f"{estimate.get('message', '')}\n"
                    )
                    continue
                self._append_results(
                    f"{estimate['term_prefix']}: "
                    f"{float(estimate['splitting_meV']):.2f} meV "
                    f"({float(estimate['lower_transition_eV']):.5f} -> "
                    f"{float(estimate['upper_transition_eV']):.5f} eV), "
                    f"window {float(estimate['energy_window_eV'][0]):.4f}-"
                    f"{float(estimate['energy_window_eV'][1]):.4f} eV, "
                    f"{estimate.get('assignment_quality', 'unknown')}, "
                    f"{estimate.get('fit_stability', 'unknown')}\n"
                )

        warnings_out = summary["dat_warnings"] + summary["decomposition_warnings"]
        if warnings_out:
            self._append_results("\nWarnings\n")
            for warning in warnings_out:
                self._append_results(f"- {warning}\n")

        self._append_results("\nFiles\n")
        for key in (
            "summary_path",
            "terms_csv",
            "overview_plot",
            "feature_csv",
            "feature_plot",
            "splitting_csv",
            "results_csv",
            "short_report_csv",
            "fit_csv",
            "fit_plot",
        ):
            path = output.get(key)
            if path:
                self._append_results(f"{Path(path)}\n")

        feature_tab = None
        if feature_scan and feature_scan.get("features"):
            feature_tab = self._add_feature_table(feature_scan["features"])
        if splitting_estimates and splitting_estimates.get("estimates"):
            consensus = splitting_estimates.get(
                "results",
                splitting_estimates.get("consensus", {}),
            )
            self.current_consensus = consensus
            self._add_results_table(consensus)
            self._add_results_panel(consensus)
            self._add_splitting_table(splitting_estimates["estimates"])
            self._add_database_panel(summary, consensus)
        else:
            self.current_consensus = None
        self._add_image_tab("Feature Plot", output.get("feature_plot"))
        self._add_image_tab("Split Fit", output.get("fit_plot"))
        self._add_image_tab("Overview", output.get("overview_plot"))
        if feature_tab is not None:
            self.output_notebook.select(feature_tab)

    def _add_feature_table(self, features: list[dict]) -> ttk.Frame:
        frame = ttk.Frame(self.output_notebook, padding=6)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        columns = (
            "rank",
            "term",
            "energy",
            "kind",
            "score",
            "z",
            "scatter",
            "fit_window",
        )
        tree = ttk.Treeview(frame, columns=columns, show="headings", height=12)
        headings = {
            "rank": "#",
            "term": "Term",
            "energy": "Energy eV",
            "kind": "Kind",
            "score": "Score",
            "z": "z",
            "scatter": "Scatter ratio",
            "fit_window": "Fit window",
        }
        widths = {
            "rank": 50,
            "term": 150,
            "energy": 90,
            "kind": 70,
            "score": 80,
            "z": 70,
            "scatter": 105,
            "fit_window": 100,
        }
        for column in columns:
            tree.heading(column, text=headings[column])
            tree.column(column, width=widths[column], anchor="center", stretch=True)

        y_scroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        x_scroll = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")

        for feature in features:
            scatter_ratio = feature.get("scatter_ratio")
            scatter_text = "" if scatter_ratio is None else f"{float(scatter_ratio):.1f}"
            fit_window = "inside" if feature.get("inside_split_fit_window") else "outside"
            tree.insert(
                "",
                "end",
                values=(
                    feature.get("rank", ""),
                    str(feature.get("term_prefix", "")).replace("_", " "),
                    f"{float(feature['energy_eV']):.4f}",
                    feature.get("kind", ""),
                    f"{float(feature['score']):.1f}",
                    f"{float(feature['z_score']):.1f}",
                    scatter_text,
                    fit_window,
                ),
            )

        self.output_notebook.add(frame, text="Features")
        self.dynamic_output_tabs.append(frame)
        self.feature_tree = tree
        return frame

    def _format_result_value(self, value: Any, precision: int = 2) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return ""
        if not np.isfinite(number):
            return ""
        return f"{number:.{precision}f}"

    def _add_results_panel(self, consensus: dict) -> ttk.Frame | None:
        primary = consensus.get("primary")
        if not primary:
            return None

        frame = ttk.Frame(self.output_notebook, padding=14)
        for column in range(3):
            frame.columnconfigure(column, weight=1)

        bandgap_text = self._format_result_value(primary.get("bandgap_eV"), 5)
        bandgap_spread = self._format_result_value(primary.get("bandgap_spread_eV"), 5)
        recommended_delta = self._format_result_value(
            primary.get("recommended_delta_vb_meV"),
            2,
        )
        recommended_source = str(primary.get("recommended_delta_source", ""))
        splitting_text = self._format_result_value(primary.get("splitting_meV"), 2)
        splitting_spread = self._format_result_value(primary.get("spread_meV"), 2)
        kk_split = self._format_result_value(primary.get("kk_splitting_meV"), 2)
        agreement_tolerance = self._format_result_value(
            primary.get("agreement_tolerance_meV"),
            2,
        )
        within_tolerance = primary.get("within_agreement_tolerance")
        agreement_text = ""
        if within_tolerance is not None:
            agreement_text = "yes" if within_tolerance else "no"
            if splitting_spread and agreement_tolerance:
                comparator = "<=" if within_tolerance else ">"
                agreement_text = (
                    f"{agreement_text} ({splitting_spread} {comparator} "
                    f"{agreement_tolerance} meV)"
                )
        components = ", ".join(
            f"{float(value):.2f}"
            for value in primary.get("component_splittings_meV", [])
            if np.isfinite(float(value))
        )
        lower_transitions = ", ".join(
            f"{float(value):.5f}"
            for value in primary.get("component_lower_transition_eV", [])
            if np.isfinite(float(value))
        )
        upper_transitions = ", ".join(
            f"{float(value):.5f}"
            for value in primary.get("component_upper_transition_eV", [])
            if np.isfinite(float(value))
        )

        rows = [
            (
                "Eg",
                f"{bandgap_text} eV" if bandgap_text else "not assigned",
                "Mean lower transition from overlapping LD/LB result.",
            ),
            (
                "Eg spread",
                f"{bandgap_spread} eV" if bandgap_spread else "",
                "Difference between component lower-transition energies.",
            ),
            (
                "Recommended Delta Vb",
                f"{recommended_delta} meV" if recommended_delta else "manual review",
                recommended_source,
            ),
            (
                "Raw LD/LB split mean",
                f"{splitting_text} meV" if splitting_text else "",
                "Average of the selected LD and LB splitting estimates.",
            ),
            (
                "Joint KK split",
                f"{kk_split} meV" if kk_split else "",
                "One shared LD/LB critical-point model.",
            ),
            (
                "Splitting spread",
                f"{splitting_spread} meV" if splitting_spread else "",
                "Difference between the LD and LB splitting values.",
            ),
            (
                "LD/LB agreement",
                agreement_text,
                "Whether the splitting spread is inside the tolerance.",
            ),
            ("Component splittings", components, "LD, LB values in meV."),
            ("Lower transitions", lower_transitions, "Component transition energies."),
            ("Upper transitions", upper_transitions, "Component transition energies."),
            ("Confidence", str(primary.get("confidence", "")), str(primary.get("basis", ""))),
        ]

        for row_index, (label, value, note) in enumerate(rows):
            ttk.Label(frame, text=label).grid(
                row=row_index,
                column=0,
                sticky="w",
                padx=(0, 12),
                pady=4,
            )
            ttk.Label(frame, text=value, font=("TkDefaultFont", 10, "bold")).grid(
                row=row_index,
                column=1,
                sticky="w",
                pady=4,
            )
            if note:
                ttk.Label(frame, text=note, foreground="gray35").grid(
                    row=row_index,
                    column=2,
                    sticky="w",
                    padx=(12, 0),
                    pady=4,
                )

        next_message_row = len(rows)
        if not bandgap_text:
            ttk.Label(
                frame,
                text=(
                    "Eg is left blank because the selected LD/LB pair does not "
                    "come from overlapping energy windows."
                ),
                foreground="firebrick",
                wraplength=760,
            ).grid(row=next_message_row, column=0, columnspan=3, sticky="w", pady=(12, 0))
            next_message_row += 1
        warnings_text = "; ".join(str(value) for value in primary.get("math_warnings", []))
        if warnings_text:
            ttk.Label(
                frame,
                text=(
                    "Recommended Delta Vb is blank because: "
                    f"{warnings_text}"
                ),
                foreground="firebrick",
                wraplength=860,
            ).grid(row=next_message_row, column=0, columnspan=3, sticky="w", pady=(12, 0))
        elif within_tolerance is False:
            ttk.Label(
                frame,
                text=(
                    "Eg is assigned from overlapping LD/LB energy windows, but "
                    "the LD/LB splitting spread is above the agreement tolerance."
                ),
                foreground="darkorange4",
                wraplength=760,
            ).grid(row=next_message_row, column=0, columnspan=3, sticky="w", pady=(12, 0))

        self.output_notebook.add(frame, text="Results Panel")
        self.dynamic_output_tabs.append(frame)
        return frame

    def _add_results_table(self, consensus: dict) -> ttk.Frame | None:
        matches = consensus.get("matches", [])
        if not matches:
            return None

        frame = ttk.Frame(self.output_notebook, padding=6)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        columns = (
            "rank",
            "eg",
            "recommended",
            "raw_splitting",
            "spread",
            "agreement",
            "confidence",
            "basis",
            "components",
            "quality",
            "stability",
            "warnings",
        )
        headings = {
            "rank": "#",
            "eg": "Eg eV",
            "recommended": "Delta Vb meV",
            "raw_splitting": "Raw mean meV",
            "spread": "Spread meV",
            "agreement": "Within tol.",
            "confidence": "Confidence",
            "basis": "Basis",
            "components": "Components meV",
            "quality": "Assignment",
            "stability": "Fit stability",
            "warnings": "Math warnings",
        }
        widths = {
            "rank": 45,
            "eg": 90,
            "recommended": 110,
            "raw_splitting": 105,
            "spread": 90,
            "agreement": 85,
            "confidence": 105,
            "basis": 190,
            "components": 160,
            "quality": 160,
            "stability": 160,
            "warnings": 340,
        }
        tree = ttk.Treeview(frame, columns=columns, show="headings", height=8)
        for column in columns:
            tree.heading(column, text=headings[column])
            tree.column(column, width=widths[column], anchor="center", stretch=True)

        y_scroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        x_scroll = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")

        for match in matches:
            component_values = ", ".join(
                f"{float(value):.2f}" for value in match.get("component_splittings_meV", [])
            )
            recommended = self._format_result_value(
                match.get("recommended_delta_vb_meV"),
                2,
            )
            quality = ", ".join(match.get("component_assignment_quality", []))
            stability = ", ".join(match.get("component_fit_stability", []))
            warnings_text = "; ".join(
                str(warning) for warning in match.get("math_warnings", [])
            )
            iid = tree.insert(
                "",
                "end",
                values=(
                    match.get("rank", ""),
                    (
                        f"{float(match['bandgap_eV']):.5f}"
                        if np.isfinite(float(match.get("bandgap_eV", np.nan)))
                        else ""
                    ),
                    recommended,
                    f"{float(match['splitting_meV']):.2f}",
                    f"{float(match['spread_meV']):.2f}",
                    (
                        "yes"
                        if match.get("within_agreement_tolerance")
                        else "no"
                    ),
                    match.get("confidence", ""),
                    match.get("basis", ""),
                    component_values,
                    quality,
                    stability,
                    warnings_text,
                ),
            )
            self.result_match_by_iid[iid] = match
        tree.bind("<<TreeviewSelect>>", self._on_result_tree_select)

        self.output_notebook.add(frame, text="Results")
        self.dynamic_output_tabs.append(frame)
        self.results_tree = tree
        return frame

    def _match_by_rank(self, rank_text: str) -> dict | None:
        if not self.current_consensus:
            return None
        try:
            rank = int(str(rank_text).strip())
        except ValueError:
            return None
        for match in self.current_consensus.get("matches", []):
            if int(match.get("rank", 0)) == rank:
                return match
        return None

    def _entry_float(self, value: Any, precision: int = 5) -> str:
        text = self._format_result_value(value, precision)
        return text

    def _fill_database_from_match(self, match: dict | None) -> None:
        if not match:
            return
        self.db_result_rank.set(str(match.get("rank", "")))
        self.db_eg_eV.set(self._entry_float(match.get("bandgap_eV"), 5))
        recommended = self._entry_float(match.get("recommended_delta_vb_meV"), 2)
        self.db_splitting_meV.set(recommended)
        basis = str(match.get("basis", ""))
        confidence = str(match.get("confidence", ""))
        raw_split = self._entry_float(match.get("splitting_meV"), 2)
        warnings_text = "; ".join(str(value) for value in match.get("math_warnings", []))
        if basis or confidence:
            if recommended:
                self.db_message.set(
                    f"Selected rank {match.get('rank', '')}: using recommended "
                    f"Delta Vb {recommended} meV; {confidence}; {basis}"
                )
            else:
                message = (
                    f"Selected rank {match.get('rank', '')}: Delta Vb needs manual "
                    f"review; raw LD/LB mean is {raw_split} meV; {confidence}; {basis}"
                )
                if warnings_text:
                    message = f"{message}; {warnings_text}"
                self.db_message.set(message)

    def _initialize_database_form(self, summary: dict, consensus: dict) -> None:
        sample_id = sdb.default_sample_id(summary.get("dat_path"))
        self.db_sample_id.set(sample_id)
        inferred_temp = sdb.infer_temperature_c(sample_id)
        self.db_temperature_c.set("" if inferred_temp is None else f"{inferred_temp:g}")
        self.db_bi_percent.set("")
        self.db_status.set("accepted")
        self.db_analyst.set("")
        if self.db_notes_text is not None:
            self.db_notes_text.delete("1.0", "end")
        matches = consensus.get("matches", [])
        if self.db_rank_combo is not None:
            self.db_rank_combo.configure(
                values=tuple(str(match.get("rank", "")) for match in matches)
            )
        self._fill_database_from_match(consensus.get("primary"))

    def _add_database_panel(self, summary: dict, consensus: dict) -> ttk.Frame:
        frame = ttk.Frame(self.output_notebook, padding=8)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(10, weight=1)

        ttk.Label(frame, text="Database folder").grid(row=0, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.database_dir).grid(
            row=0,
            column=1,
            sticky="ew",
            padx=(8, 8),
        )
        ttk.Button(frame, text="Browse", command=self._browse_database_dir).grid(
            row=0,
            column=2,
            sticky="ew",
        )
        ttk.Button(frame, text="Clone/Setup", command=self._clone_database_repo).grid(
            row=0,
            column=3,
            sticky="ew",
            padx=(8, 0),
        )

        ttk.Label(frame, text="Sample ID").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(frame, textvariable=self.db_sample_id).grid(
            row=1,
            column=1,
            sticky="ew",
            padx=(8, 8),
            pady=(8, 0),
        )

        ttk.Label(frame, text="Result rank").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.db_rank_combo = ttk.Combobox(
            frame,
            textvariable=self.db_result_rank,
            state="readonly",
            width=12,
        )
        self.db_rank_combo.grid(row=2, column=1, sticky="w", padx=(8, 8), pady=(8, 0))
        self.db_rank_combo.bind("<<ComboboxSelected>>", self._on_database_rank_selected)

        ttk.Label(frame, text="Eg eV").grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(frame, textvariable=self.db_eg_eV, width=16).grid(
            row=3,
            column=1,
            sticky="w",
            padx=(8, 8),
            pady=(8, 0),
        )
        ttk.Label(frame, text="Delta Vb meV").grid(
            row=3,
            column=1,
            sticky="w",
            padx=(170, 8),
            pady=(8, 0),
        )
        ttk.Entry(frame, textvariable=self.db_splitting_meV, width=16).grid(
            row=3,
            column=1,
            sticky="w",
            padx=(275, 8),
            pady=(8, 0),
        )

        ttk.Label(frame, text="Bi %").grid(row=4, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(frame, textvariable=self.db_bi_percent, width=16).grid(
            row=4,
            column=1,
            sticky="w",
            padx=(8, 8),
            pady=(8, 0),
        )
        ttk.Label(frame, text="Temperature C").grid(
            row=4,
            column=1,
            sticky="w",
            padx=(170, 8),
            pady=(8, 0),
        )
        ttk.Entry(frame, textvariable=self.db_temperature_c, width=16).grid(
            row=4,
            column=1,
            sticky="w",
            padx=(275, 8),
            pady=(8, 0),
        )

        ttk.Label(frame, text="Status").grid(row=5, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(
            frame,
            textvariable=self.db_status,
            values=("accepted", "provisional", "needs_review", "excluded"),
            state="readonly",
            width=16,
        ).grid(row=5, column=1, sticky="w", padx=(8, 8), pady=(8, 0))
        ttk.Label(frame, text="Analyst").grid(
            row=5,
            column=1,
            sticky="w",
            padx=(170, 8),
            pady=(8, 0),
        )
        ttk.Entry(frame, textvariable=self.db_analyst, width=20).grid(
            row=5,
            column=1,
            sticky="w",
            padx=(275, 8),
            pady=(8, 0),
        )

        ttk.Label(frame, text="Git commit").grid(row=6, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(frame, textvariable=self.db_git_message).grid(
            row=6,
            column=1,
            columnspan=2,
            sticky="ew",
            padx=(8, 0),
            pady=(8, 0),
        )

        ttk.Label(frame, text="Notes").grid(row=7, column=0, sticky="nw", pady=(8, 0))
        self.db_notes_text = tk.Text(frame, height=3, wrap="word")
        self.db_notes_text.grid(
            row=7,
            column=1,
            columnspan=2,
            sticky="ew",
            padx=(8, 0),
            pady=(8, 0),
        )

        actions = ttk.Frame(frame)
        actions.grid(row=8, column=1, columnspan=2, sticky="ew", padx=(8, 0), pady=(8, 0))
        ttk.Button(actions, text="Save Selected Result", command=self._save_database_record).grid(
            row=0,
            column=0,
            sticky="w",
        )
        ttk.Button(
            actions,
            text="Save + Push to GitHub",
            command=lambda: self._save_database_record(push_to_github=True),
        ).grid(
            row=0,
            column=1,
            sticky="w",
            padx=(8, 0),
        )
        ttk.Button(actions, text="Push Database to GitHub", command=self._push_database_to_github).grid(
            row=0,
            column=2,
            sticky="w",
            padx=(8, 0),
        )
        ttk.Button(actions, text="Delete Selected", command=self._delete_database_record).grid(
            row=0,
            column=3,
            sticky="w",
            padx=(8, 0),
        )
        ttk.Button(
            actions,
            text="Delete + Push",
            command=lambda: self._delete_database_record(push_to_github=True),
        ).grid(
            row=0,
            column=4,
            sticky="w",
            padx=(8, 0),
        )
        ttk.Button(actions, text="Refresh Comparisons", command=self._refresh_database_comparisons).grid(
            row=0,
            column=5,
            sticky="w",
            padx=(8, 0),
        )
        ttk.Button(actions, text="Open Database Folder", command=self._open_database_folder).grid(
            row=0,
            column=6,
            sticky="w",
            padx=(8, 0),
        )

        ttk.Label(frame, textvariable=self.db_message, foreground="gray25").grid(
            row=9,
            column=0,
            columnspan=3,
            sticky="w",
            pady=(8, 0),
        )

        columns = ("sample", "bi", "eg", "splitting", "status", "source")
        self.database_tree = ttk.Treeview(frame, columns=columns, show="headings", height=8)
        headings = {
            "sample": "Sample",
            "bi": "Bi %",
            "eg": "Eg eV",
            "splitting": "Splitting meV",
            "status": "Status",
            "source": "Source",
        }
        widths = {
            "sample": 210,
            "bi": 70,
            "eg": 85,
            "splitting": 110,
            "status": 90,
            "source": 260,
        }
        for column in columns:
            self.database_tree.heading(column, text=headings[column])
            self.database_tree.column(column, width=widths[column], anchor="center", stretch=True)
        y_scroll = ttk.Scrollbar(frame, orient="vertical", command=self.database_tree.yview)
        x_scroll = ttk.Scrollbar(frame, orient="horizontal", command=self.database_tree.xview)
        self.database_tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        self.database_tree.grid(row=10, column=0, columnspan=2, sticky="nsew", pady=(10, 0))
        y_scroll.grid(row=10, column=2, sticky="ns", pady=(10, 0))
        x_scroll.grid(row=11, column=0, columnspan=2, sticky="ew")

        self._initialize_database_form(summary, consensus)
        self._refresh_database_table()

        self.output_notebook.add(frame, text="Database")
        self.dynamic_output_tabs.append(frame)
        return frame

    def _on_database_rank_selected(self, _event: tk.Event | None = None) -> None:
        self._fill_database_from_match(self._match_by_rank(self.db_result_rank.get()))

    def _on_result_tree_select(self, _event: tk.Event | None = None) -> None:
        if self.results_tree is None:
            return
        selection = self.results_tree.selection()
        if not selection:
            return
        match = self.result_match_by_iid.get(selection[0])
        self._fill_database_from_match(match)

    def _browse_database_dir(self) -> None:
        path = filedialog.askdirectory(title="Select shared database folder")
        if path:
            self.database_dir.set(path)
            self._refresh_database_table()

    def _clone_database_repo(self) -> None:
        parent = filedialog.askdirectory(title="Select where to clone the GitHub database")
        if not parent:
            return
        try:
            path = sdb.clone_database_repo(parent)
        except Exception as exc:
            messagebox.showerror("Database clone failed", str(exc))
            self.db_message.set(f"Database clone failed: {exc}")
            return
        self.database_dir.set(str(path))
        self._refresh_database_table()
        self.db_message.set(f"Database folder set to Git clone: {path}")

    def _database_path(self) -> Path:
        text = self.database_dir.get().strip()
        return Path(text or "shared_database")

    def _save_database_record(self, push_to_github: bool = False) -> None:
        if self.current_summary is None:
            messagebox.showerror("No analysis", "Run an analysis before saving a result.")
            return
        sample_id = self.db_sample_id.get().strip()
        if not sample_id:
            messagebox.showerror("Missing sample ID", "Sample ID is required.")
            return
        try:
            eg = self._parse_required_float("Eg eV", self.db_eg_eV.get())
            splitting = self._parse_required_float(
                "Delta Vb meV",
                self.db_splitting_meV.get(),
            )
            bi_percent = self._parse_optional_float("Bi %", self.db_bi_percent.get())
            temperature_c = self._parse_optional_float(
                "Temperature C",
                self.db_temperature_c.get(),
            )
        except Exception as exc:
            messagebox.showerror("Invalid database value", str(exc))
            return

        notes = ""
        if self.db_notes_text is not None:
            notes = self.db_notes_text.get("1.0", "end").strip()
        match = self._match_by_rank(self.db_result_rank.get())
        if match and match.get("requires_manual_delta_vb"):
            warnings_text = "; ".join(
                str(value) for value in match.get("math_warnings", [])
            )
            proceed = messagebox.askyesno(
                "Manual Delta Vb",
                "The selected result failed the automatic Delta Vb math checks. "
                "Save your manually entered splitting anyway?\n\n"
                f"{warnings_text}",
            )
            if not proceed:
                return
        summary_path = None
        if self.last_analysis_output is not None:
            summary_path = self.last_analysis_output.get("summary_path")
        record = sdb.build_record(
            summary=self.current_summary,
            selected_match=match,
            sample_id=sample_id,
            eg_eV=eg,
            valence_band_splitting_meV=splitting,
            bi_percent=bi_percent,
            temperature_C=temperature_c,
            status=self.db_status.get(),
            analyst=self.db_analyst.get().strip(),
            notes=notes,
            analysis_summary_path=summary_path,
        )
        try:
            paths = sdb.upsert_record(record, self._database_path())
        except Exception as exc:
            messagebox.showerror("Database save failed", str(exc))
            return
        self.db_message.set(f"Saved {sample_id} to {paths['jsonl']}")
        self._refresh_database_table()
        self._show_database_comparison_plots(paths)
        if push_to_github:
            self._push_database_to_github()

    def _refresh_database_table(self) -> None:
        if self.database_tree is None:
            return
        for item in self.database_tree.get_children():
            self.database_tree.delete(item)
        self.database_record_by_iid.clear()
        try:
            records = sdb.sorted_records_for_display(sdb.load_records(self._database_path()))
        except Exception as exc:
            self.db_message.set(f"Could not load database: {exc}")
            return
        for record in records:
            iid = self.database_tree.insert(
                "",
                "end",
                values=(
                    record.get("sample_id", ""),
                    self._format_result_value(record.get("bi_percent"), 3),
                    self._format_result_value(record.get("eg_eV"), 5),
                    self._format_result_value(
                        record.get("valence_band_splitting_meV"),
                        2,
                    ),
                    record.get("status", ""),
                    record.get("source_file", ""),
                ),
            )
            self.database_record_by_iid[iid] = record
        if records:
            self.db_message.set(f"Loaded {len(records)} database record(s).")
        else:
            self.db_message.set("Database has no records.")

    def _selected_database_record(self) -> dict | None:
        if self.database_tree is None:
            return None
        selection = self.database_tree.selection()
        if not selection:
            return None
        return self.database_record_by_iid.get(selection[0])

    def _delete_database_record(self, push_to_github: bool = False) -> None:
        record = self._selected_database_record()
        if record is None:
            messagebox.showerror("No database row selected", "Select a row in the database table first.")
            return
        sample_id = str(record.get("sample_id", ""))
        record_id = str(record.get("record_id", ""))
        if not messagebox.askyesno(
            "Delete database record",
            f"Delete {sample_id or record_id} from the shared database?",
        ):
            return
        try:
            paths = sdb.delete_record(record_id, self._database_path())
        except Exception as exc:
            messagebox.showerror("Delete failed", str(exc))
            return
        self.db_message.set(f"Deleted {sample_id or record_id} from database.")
        self._refresh_database_table()
        self._show_database_comparison_plots(paths)
        if push_to_github:
            self._push_database_to_github()

    def _refresh_database_comparisons(self) -> None:
        try:
            records = sdb.load_records(self._database_path())
            paths = sdb.build_comparison_outputs(self._database_path(), records=records)
        except Exception as exc:
            messagebox.showerror("Comparison refresh failed", str(exc))
            return
        self._refresh_database_table()
        self._show_database_comparison_plots(paths)
        self.db_message.set(f"Comparison files refreshed in {paths['comparison_dir']}")

    def _push_database_to_github(self) -> None:
        commit_message = self.db_git_message.get().strip()
        if not commit_message:
            commit_message = "Update GaAsBi results database"
        try:
            info = sdb.commit_and_push_database(
                self._database_path(),
                commit_message=commit_message,
            )
        except Exception as exc:
            messagebox.showerror("GitHub push failed", str(exc))
            self.db_message.set(f"GitHub push failed: {exc}")
            return
        action = "committed and pushed" if info.get("committed") else "pushed"
        self.db_message.set(
            f"Database {action} to {info['remote']}/{info['branch']} "
            f"at {info['head']}"
        )

    def _show_database_comparison_plots(self, paths: dict[str, Path]) -> None:
        for tab in list(self.database_plot_tabs):
            try:
                self.output_notebook.forget(tab)
            except tk.TclError:
                pass
            if tab in self.dynamic_output_tabs:
                self.dynamic_output_tabs.remove(tab)
        self.database_plot_tabs.clear()

        summary_tab = self._add_text_file_tab("DB Summary", paths.get("summary"))
        if summary_tab is not None:
            self.database_plot_tabs.append(summary_tab)
        readme_tab = self._add_text_file_tab("DB README", paths.get("readme"))
        if readme_tab is not None:
            self.database_plot_tabs.append(readme_tab)

        for title, key in (
            ("DB Eg/Bi", "eg_vs_bi_plot"),
            ("DB DeltaVb/Bi", "delta_vb_vs_bi_plot"),
            ("DB Eg/DeltaVb", "eg_vs_delta_vb_plot"),
        ):
            tab = self._add_image_tab(title, paths.get(key))
            if tab is not None:
                self.database_plot_tabs.append(tab)

    def _open_database_folder(self) -> None:
        db_path = self._database_path()
        db_path.mkdir(parents=True, exist_ok=True)
        path = str(db_path)
        try:
            if sys.platform.startswith("win"):
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as exc:
            messagebox.showerror("Could not open database folder", str(exc))

    def _add_splitting_table(self, estimates: list[dict]) -> ttk.Frame:
        frame = ttk.Frame(self.output_notebook, padding=6)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        columns = (
            "rank",
            "term",
            "splitting",
            "lower",
            "upper",
            "center",
            "window",
            "quality",
            "stability",
            "rmse",
            "status",
        )
        headings = {
            "rank": "#",
            "term": "Term",
            "splitting": "Splitting meV",
            "lower": "Lower eV",
            "upper": "Upper eV",
            "center": "Center eV",
            "window": "Fit window eV",
            "quality": "Quality",
            "stability": "Stability",
            "rmse": "RMSE",
            "status": "Status",
        }
        widths = {
            "rank": 45,
            "term": 145,
            "splitting": 105,
            "lower": 85,
            "upper": 85,
            "center": 85,
            "window": 130,
            "quality": 115,
            "stability": 110,
            "rmse": 90,
            "status": 95,
        }
        tree = ttk.Treeview(frame, columns=columns, show="headings", height=10)
        for column in columns:
            tree.heading(column, text=headings[column])
            tree.column(column, width=widths[column], anchor="center", stretch=True)

        y_scroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        x_scroll = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")

        for rank, estimate in enumerate(estimates, start=1):
            window = estimate.get("energy_window_eV", ("", ""))
            if estimate.get("success"):
                values = (
                    rank,
                    str(estimate.get("term_prefix", "")).replace("_", " "),
                    f"{float(estimate['splitting_meV']):.2f}",
                    f"{float(estimate['lower_transition_eV']):.5f}",
                    f"{float(estimate['upper_transition_eV']):.5f}",
                    f"{float(estimate['center_eV']):.5f}",
                    f"{float(window[0]):.3f}-{float(window[1]):.3f}",
                    estimate.get("assignment_quality", ""),
                    estimate.get("fit_stability", ""),
                    f"{float(estimate['rmse']):.3g}",
                    "ok",
                )
            else:
                values = (
                    rank,
                    str(estimate.get("term_prefix", "")).replace("_", " "),
                    "",
                    "",
                    "",
                    "",
                    f"{float(window[0]):.3f}-{float(window[1]):.3f}",
                    estimate.get("assignment_quality", ""),
                    estimate.get("fit_stability", ""),
                    "",
                    "failed",
                )
            tree.insert("", "end", values=values)

        self.output_notebook.add(frame, text="Splitting")
        self.dynamic_output_tabs.append(frame)
        self.splitting_tree = tree
        return frame

    def _add_image_tab(self, title: str, path: Path | str | None) -> ttk.Frame | None:
        if not path:
            return None
        image_path = Path(path)
        if not image_path.exists():
            return None

        frame = ttk.Frame(self.output_notebook, padding=4)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        canvas = tk.Canvas(frame, background="white", highlightthickness=0)
        y_scroll = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        x_scroll = ttk.Scrollbar(frame, orient="horizontal", command=canvas.xview)
        canvas.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")

        try:
            image = tk.PhotoImage(file=str(image_path))
        except tk.TclError as exc:
            self._append_results(f"Could not show {image_path}: {exc}\n")
            return None

        self.plot_images.append(image)
        canvas.create_image(0, 0, anchor="nw", image=image)
        canvas.configure(scrollregion=(0, 0, image.width(), image.height()))

        self.output_notebook.add(frame, text=title)
        self.dynamic_output_tabs.append(frame)
        return frame

    def _add_text_file_tab(self, title: str, path: Path | str | None) -> ttk.Frame | None:
        if not path:
            return None
        text_path = Path(path)
        if not text_path.exists():
            return None

        frame = ttk.Frame(self.output_notebook, padding=4)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        text = scrolledtext.ScrolledText(frame, wrap="none", height=18)
        text.grid(row=0, column=0, sticky="nsew")
        try:
            content = text_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = text_path.read_text(encoding="utf-8", errors="replace")
        text.insert("1.0", content)
        text.configure(state="disabled")

        self.output_notebook.add(frame, text=title)
        self.dynamic_output_tabs.append(frame)
        return frame

    def _open_output_folder(self) -> None:
        if self.last_output_dir is None:
            return
        path = str(self.last_output_dir)
        try:
            if sys.platform.startswith("win"):
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as exc:
            messagebox.showerror("Could not open folder", str(exc))

    def _clear_results(self) -> None:
        for tab in self.dynamic_output_tabs:
            self.output_notebook.forget(tab)
        self.dynamic_output_tabs.clear()
        self.database_plot_tabs.clear()
        self.feature_tree = None
        self.splitting_tree = None
        self.results_tree = None
        self.database_tree = None
        self.result_match_by_iid.clear()
        self.database_record_by_iid.clear()
        self.last_analysis_output = None
        self.current_summary = None
        self.current_consensus = None
        self.plot_images.clear()
        self.results_text.configure(state="normal")
        self.results_text.delete("1.0", "end")
        self.results_text.configure(state="disabled")
        self.output_notebook.select(self.results_tab)

    def _append_results(self, text: str) -> None:
        self.results_text.configure(state="normal")
        self.results_text.insert("end", text)
        self.results_text.see("end")
        self.results_text.configure(state="disabled")


if __name__ == "__main__":
    app = MuellerDecompositionApp()
    app.mainloop()
