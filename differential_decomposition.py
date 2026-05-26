"""Logarithmic / differential decomposition of transmission Mueller matrices.

This module implements an effective homogeneous-slab decomposition for measured
transmission Mueller matrices. For each measured Mueller matrix M, the
logarithmic generator is defined by

    M = exp(L)

or, when a sample thickness d is known,

    M = exp(m d),    m = log(M) / d.

Here L is an integrated differential Mueller matrix and m is an effective
differential Mueller matrix per thickness unit. In transmission ellipsometry,
normalizing M by M[0, 0] removes the absolute scalar transmission factor from
the measurement. In logarithmic form, a scalar transmission factor appears as a
term proportional to the identity matrix, so subtracting trace(L) / 4 separates
isotropic attenuation from anisotropic polarization effects.

Important physical limitation: this decomposition treats the sample as an
effective homogeneous slab. For layered, depolarizing, strongly scattering, or
multiple-reflection dominated samples, the recovered generator should be
interpreted as an effective integrated generator of the measured Mueller matrix,
not necessarily as a local material tensor or a unique microscopic model.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import numpy as np
from scipy.linalg import expm

# Keep these names available through differential_decomposition for existing
# GUI, CLI, and notebook callers. Matrix-log math is re-exported through
# decomposition_math.py from mueller_log_math.py.
from decomposition_math import (
    _DEFAULT_FEATURE_BASELINE_WIDTHS_EV,
    _MATRIX_SHAPE,
    _REAL_IF_CLOSE_ABS,
    _REAL_IF_CLOSE_REL,
    _as_numeric_array,
    _local_split_initial_guesses,
    _spectrum_component,
    _windowed_signal_scale,
    collapse_twofold_anisotropy_spectrum,
    critical_point_profile,
    decompose_generator_terms,
    decompose_mueller_log,
    detect_spectral_features,
    estimate_direct_derivative_split_spectrum,
    fit_kk_consistent_split_spectra,
    fit_split_transition_spectrum,
    kk_split_model,
    matrix_log_batch,
    normalize_mueller,
    reconstruction_error,
    remove_isotropic_part,
    split_transition_model,
)

_HC_EV_NM = 1239.8419843320026


_TERM_LABELS = {
    "isotropic_attenuation": "isotropic attenuation",
    "linear_dichroism_x": "linear dichroism x",
    "linear_dichroism_y": "linear dichroism y",
    "circular_dichroism": "circular dichroism",
    "linear_birefringence_x": "linear birefringence x",
    "linear_birefringence_y": "linear birefringence y",
    "circular_birefringence": "circular birefringence",
    "linear_dichroism_magnitude": "linear dichroism magnitude",
    "linear_birefringence_magnitude": "linear birefringence magnitude",
    "dichroism_axis_angle_deg": "dichroism axis angle",
    "birefringence_axis_angle_deg": "birefringence axis angle",
}

_MM_LABEL_TO_INDEX = {
    f"mm{row}{col}": (row - 1, col - 1)
    for row in range(1, 5)
    for col in range(1, 5)
}
_COMPLETEEASE_NON_MUELLER_DATA_LABELS = {"E", "dPolE", "uR", "AnE", "Aps", "Asp"}


def _coerce_for_plot(values: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Return real values and optionally significant imaginary values for plotting."""
    arr = np.asarray(values)
    if not np.iscomplexobj(arr):
        return arr, None

    imag = np.imag(arr)
    real = np.real(arr)
    scale = max(1.0, float(np.nanmax(np.abs(real))) if real.size else 1.0)
    imag_norm = float(np.nanmax(np.abs(imag))) if imag.size else 0.0
    if imag_norm <= _REAL_IF_CLOSE_ABS + _REAL_IF_CLOSE_REL * scale:
        return real, None
    return real, imag


def _term_label(term_name: str) -> str:
    return _TERM_LABELS.get(term_name, term_name.replace("_", " "))


def _term_unit_label(result: dict[str, Any], term_name: str) -> str:
    if term_name.endswith("_angle_deg"):
        return "deg"
    if result.get("is_differential", False):
        return "per thickness unit"
    return "integrated"


def _save_figure_if_requested(fig: Any, result: dict[str, Any], filename: str) -> None:
    output_dir = result.get("output_dir")
    if not output_dir:
        return
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / filename, bbox_inches="tight", dpi=150)


def _safe_filename(text: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return safe.strip("_") or "plot"


def _parse_completeease_float(text: str) -> float:
    normalized = text.strip().replace(",", "")
    if normalized in {"Infinity", "+Infinity", "Inf", "+Inf"}:
        return np.inf
    if normalized in {"-Infinity", "-Inf"}:
        return -np.inf
    return float(normalized)


def _column_index(column: int | str, mapping: dict[str, int], name: str) -> int:
    if isinstance(column, str):
        if column not in mapping:
            choices = ", ".join(sorted(mapping))
            raise ValueError(f"Unknown {name} {column!r}. Use one of: {choices}.")
        return mapping[column]
    index = int(column)
    if index < 0:
        raise ValueError(f"{name} must be a non-negative split-column index.")
    return index


def _has_completeease_rotation_column(path: Path) -> bool:
    """Return True when the header advertises an Aniso_Theta column."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line in handle:
            parts = line.split()
            if parts and parts[0] in _MM_LABEL_TO_INDEX:
                return False
            if "Aniso_Theta" in line:
                return True
    return False


def _rotation_column_index(column: int | str | None, path: Path) -> tuple[int | None, str]:
    """Resolve the optional CompleteEASE sample-rotation split-column."""
    if column is None:
        return None, "single_angle"
    if isinstance(column, str):
        lowered = column.strip().lower()
        if lowered in {"auto", ""}:
            if _has_completeease_rotation_column(path):
                return -1, "aniso_theta_header"
            return None, "single_angle_no_aniso_theta_header"
        if lowered in {"none", "single", "single_angle"}:
            return None, "single_angle"
        if lowered in {"last", "aniso_theta"}:
            return -1, lowered
        try:
            index = int(lowered)
        except ValueError as exc:
            raise ValueError(
                "rotation_column must be 'auto', 'last', 'none', or a split-column index."
            ) from exc
    else:
        index = int(column)
    if index < 0:
        raise ValueError("rotation_column must be a non-negative split-column index.")
    return index, "explicit_column"


def read_woollam_dat(
    path: str | Path,
    matrix_kind: str = "normalized",
    value_column: int | str = "measured",
    mm11_transmission_column: int | str = "transmission",
    rotation_column: int | str | None = "auto",
    sort_energy: bool = True,
) -> dict[str, Any]:
    """Read a J.A. Woollam CompleteEASE/VASE Mueller-matrix ``.dat`` export.

    Parameters
    ----------
    path:
        Path to the text ``.dat`` file.
    matrix_kind:
        ``"normalized"`` returns the normalized Mueller matrix with M[0, 0] set
        to 1 and stores the exported absolute ``mm11`` transmission separately.
        ``"absolute"`` multiplies the normalized matrix by the exported
        transmission to reconstruct an absolute Mueller matrix.
    value_column:
        Split-column used for the normalized ``mmij`` values other than
        ``mm11``. For the observed CompleteEASE export, ``"measured"`` is
        column 3 and ``"model"`` is column 4 after splitting the line on
        whitespace. The two are often identical in measurement-only exports.
    mm11_transmission_column:
        Split-column used for the absolute ``mm11`` transmission. In this
        export, the ``mm11`` line has an ``Infinity`` placeholder in the
        normalized-value position and stores the absolute transmission in the
        next column.
    rotation_column:
        Split-column used for sample rotation. ``"auto"`` uses the final column
        only when the header contains ``Aniso_Theta``. Files without that header
        are treated as single-angle datasets, because their final columns are
        commonly uncertainty/model columns rather than rotation.
    sort_energy:
        If True, return the energy axis in ascending eV order. The source file
        is wavelength ordered, which is descending energy.

    Returns
    -------
    dict
        Contains ``mueller`` with shape ``(n_rotations, n_energy, 4, 4)``,
        ``energy_eV``, ``wavelength_nm``, ``rotations_deg``,
        ``m11_transmission``, ``aoi_deg``, ``metadata_lines``, and
        ``diagnostics``.

    Notes
    -----
    CompleteEASE export layouts vary with settings. This reader is deliberately
    conservative: it only consumes explicit ``mm11`` ... ``mm44`` rows and
    reports missing, duplicate, or non-finite values. For rotation scans, the
    last column is usually the sample rotation angle (``Aniso_Theta``). For
    single-angle exports, that column is absent and the file is loaded as one
    rotation at 0 deg.
    """
    if matrix_kind not in {"normalized", "absolute"}:
        raise ValueError("matrix_kind must be 'normalized' or 'absolute'.")

    value_col = _column_index(
        value_column,
        {"measured": 3, "model": 4, "fit": 4, "first": 3, "second": 4},
        "value_column",
    )
    mm11_col = _column_index(
        mm11_transmission_column,
        {"transmission": 4, "absolute": 4, "measured": 4, "model": 4},
        "mm11_transmission_column",
    )

    dat_path = Path(path)
    rotation_col, rotation_source = _rotation_column_index(rotation_column, dat_path)
    metadata_lines: list[str] = []
    wavelength_order: list[float] = []
    rotation_order: list[float] = []
    wavelength_to_index: dict[float, int] = {}
    rotation_to_index: dict[float, int] = {}
    matrices: dict[tuple[int, int], np.ndarray] = {}
    seen_counts: dict[tuple[int, int], np.ndarray] = {}
    transmissions: dict[tuple[int, int], float] = {}
    aoi_values: dict[tuple[int, int], float] = {}
    bad_lines: list[tuple[int, str, str]] = []
    n_mueller_rows = 0

    def axis_index(value: float, order: list[float], lookup: dict[float, int]) -> int:
        if value not in lookup:
            lookup[value] = len(order)
            order.append(value)
        return lookup[value]

    with dat_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.split()
            if not parts:
                continue

            label = parts[0]
            if label not in _MM_LABEL_TO_INDEX:
                if (
                    not label.startswith("mm")
                    and label not in _COMPLETEEASE_NON_MUELLER_DATA_LABELS
                ):
                    metadata_lines.append(line.rstrip("\r\n"))
                continue

            n_mueller_rows += 1
            try:
                wavelength_nm = _parse_completeease_float(parts[1])
                aoi_deg = _parse_completeease_float(parts[2])
                if rotation_col is None:
                    rotation_deg = 0.0
                else:
                    rotation_deg = _parse_completeease_float(parts[rotation_col])
            except Exception as exc:
                bad_lines.append((line_number, label, f"axis parse failed: {exc}"))
                continue

            wavelength_index = axis_index(
                wavelength_nm, wavelength_order, wavelength_to_index
            )
            rotation_index = axis_index(rotation_deg, rotation_order, rotation_to_index)
            key = (rotation_index, wavelength_index)
            matrix = matrices.setdefault(key, np.full((4, 4), np.nan, dtype=np.float64))
            seen = seen_counts.setdefault(key, np.zeros((4, 4), dtype=np.int16))
            aoi_values[key] = aoi_deg

            row, col = _MM_LABEL_TO_INDEX[label]
            try:
                if label == "mm11":
                    transmission = _parse_completeease_float(parts[mm11_col])
                    transmissions[key] = transmission
                    value = 1.0
                else:
                    value = _parse_completeease_float(parts[value_col])
            except Exception as exc:
                bad_lines.append((line_number, label, f"value parse failed: {exc}"))
                value = np.nan

            matrix[row, col] = value
            seen[row, col] += 1

    n_rotations = len(rotation_order)
    n_wavelengths = len(wavelength_order)
    mueller = np.full((n_rotations, n_wavelengths, 4, 4), np.nan, dtype=np.float64)
    seen_all = np.zeros((n_rotations, n_wavelengths, 4, 4), dtype=np.int16)
    m11_transmission = np.full((n_rotations, n_wavelengths), np.nan, dtype=np.float64)
    aoi_deg = np.full((n_rotations, n_wavelengths), np.nan, dtype=np.float64)

    for (rotation_index, wavelength_index), matrix in matrices.items():
        mueller[rotation_index, wavelength_index] = matrix
        seen_all[rotation_index, wavelength_index] = seen_counts[
            (rotation_index, wavelength_index)
        ]
    for (rotation_index, wavelength_index), value in transmissions.items():
        m11_transmission[rotation_index, wavelength_index] = value
    for (rotation_index, wavelength_index), value in aoi_values.items():
        aoi_deg[rotation_index, wavelength_index] = value

    if matrix_kind == "absolute":
        mueller = mueller * m11_transmission[..., np.newaxis, np.newaxis]

    wavelengths = np.asarray(wavelength_order, dtype=np.float64)
    energy = _HC_EV_NM / wavelengths
    rotations = np.asarray(rotation_order, dtype=np.float64)

    if sort_energy:
        order = np.argsort(energy)
        wavelengths = wavelengths[order]
        energy = energy[order]
        mueller = mueller[:, order]
        seen_all = seen_all[:, order]
        m11_transmission = m11_transmission[:, order]
        aoi_deg = aoi_deg[:, order]

    missing_elements = int(np.count_nonzero(seen_all == 0))
    duplicate_elements = int(np.count_nonzero(seen_all > 1))
    nonfinite_mueller = int(np.count_nonzero(~np.isfinite(mueller)))
    nonfinite_transmission = int(np.count_nonzero(~np.isfinite(m11_transmission)))

    warnings_out: list[str] = []
    if bad_lines:
        warnings_out.append(f"{len(bad_lines)} Mueller rows could not be parsed.")
    if missing_elements:
        warnings_out.append(f"{missing_elements} Mueller elements are missing.")
    if duplicate_elements:
        warnings_out.append(f"{duplicate_elements} Mueller elements were duplicated.")
    if nonfinite_mueller:
        warnings_out.append(f"{nonfinite_mueller} Mueller array entries are non-finite.")
    if nonfinite_transmission:
        warnings_out.append(
            f"{nonfinite_transmission} mm11 transmission entries are non-finite."
        )

    diagnostics = {
        "warnings": warnings_out,
        "path": str(dat_path),
        "matrix_kind": matrix_kind,
        "value_column": value_col,
        "mm11_transmission_column": mm11_col,
        "rotation_column": rotation_col,
        "rotation_source": rotation_source,
        "sort_energy": sort_energy,
        "n_mueller_rows": n_mueller_rows,
        "n_rotations": n_rotations,
        "n_wavelengths": n_wavelengths,
        "missing_mueller_elements": missing_elements,
        "duplicate_mueller_elements": duplicate_elements,
        "nonfinite_mueller_entries": nonfinite_mueller,
        "nonfinite_m11_transmission_entries": nonfinite_transmission,
        "bad_lines": bad_lines[:20],
        "bad_line_count": len(bad_lines),
    }

    return {
        "mueller": mueller,
        "energy_eV": energy,
        "wavelength_nm": wavelengths,
        "rotations_deg": rotations,
        "m11_transmission": m11_transmission,
        "aoi_deg": aoi_deg,
        "metadata_lines": metadata_lines,
        "diagnostics": diagnostics,
    }


def _orient_dataset(
    mueller: np.ndarray, energy_eV: np.ndarray, rotations_deg: np.ndarray
) -> tuple[np.ndarray, str]:
    if mueller.ndim != 4 or mueller.shape[-2:] != _MATRIX_SHAPE:
        raise ValueError(
            "mueller must have shape (n_rotations, n_energy, 4, 4) "
            "or (n_energy, n_rotations, 4, 4); "
            f"got {mueller.shape}."
        )

    n_energy = len(energy_eV)
    n_rotations = len(rotations_deg)

    if mueller.shape[:2] == (n_rotations, n_energy):
        return mueller, "rotation_energy"
    if mueller.shape[:2] == (n_energy, n_rotations):
        return np.swapaxes(mueller, 0, 1), "energy_rotation"

    raise ValueError(
        "Could not match mueller leading axes to energy_eV and rotations_deg. "
        f"Expected ({n_rotations}, {n_energy}, 4, 4) or "
        f"({n_energy}, {n_rotations}, 4, 4); got {mueller.shape}."
    )


def decompose_dataset(
    mueller: np.ndarray,
    energy_eV: np.ndarray,
    rotations_deg: np.ndarray,
    thickness: float | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Decompose an energy- and rotation-resolved Mueller matrix dataset.

    Parameters
    ----------
    mueller:
        Shape (n_rotations, n_energy, 4, 4) or
        (n_energy, n_rotations, 4, 4). The returned arrays are always oriented
        as (n_rotations, n_energy, 4, 4).
    energy_eV:
        Photon energy axis in electron-volts.
    rotations_deg:
        Sample rotation angles in degrees.
    thickness:
        Optional sample thickness. If provided, returned generators are per
        thickness unit; otherwise they are integrated logarithmic generators.
    output_dir:
        Optional directory used by the plotting helpers to save figures.

    Returns
    -------
    dict
        Dictionary with energy_eV, rotations_deg, L_full, L_aniso, terms,
        diagnostics, output_dir, thickness, and is_differential.
    """
    mueller_arr = _as_numeric_array(mueller)
    energy_arr = np.asarray(energy_eV, dtype=np.float64)
    rotations_arr = np.asarray(rotations_deg, dtype=np.float64)

    if energy_arr.ndim != 1:
        raise ValueError(f"energy_eV must be one-dimensional; got shape {energy_arr.shape}.")
    if rotations_arr.ndim != 1:
        raise ValueError(
            f"rotations_deg must be one-dimensional; got shape {rotations_arr.shape}."
        )

    oriented_mueller, orientation = _orient_dataset(mueller_arr, energy_arr, rotations_arr)
    output_path = None if output_dir is None else str(Path(output_dir))
    if output_path is not None:
        Path(output_path).mkdir(parents=True, exist_ok=True)

    decomposition = decompose_mueller_log(oriented_mueller, thickness=thickness)

    return {
        "energy_eV": energy_arr,
        "rotations_deg": rotations_arr,
        "L_full": decomposition["L_full"],
        "L_aniso": decomposition["L_aniso"],
        "terms": decomposition["terms"],
        "diagnostics": decomposition["diagnostics"],
        "normalized_mueller": decomposition["normalized_mueller"],
        "input_orientation": orientation,
        "output_dir": output_path,
        "thickness": decomposition["thickness"],
        "is_differential": decomposition["is_differential"],
    }


def decompose_woollam_dat(
    path: str | Path,
    thickness: float | None = None,
    output_dir: str | Path | None = None,
    matrix_kind: str = "normalized",
    value_column: int | str = "measured",
    mm11_transmission_column: int | str = "transmission",
    rotation_column: int | str | None = "auto",
    sort_energy: bool = True,
) -> dict[str, Any]:
    """Read a Woollam ``.dat`` file and run logarithmic decomposition.

    This is a convenience wrapper around :func:`read_woollam_dat` and
    :func:`decompose_dataset`. The returned decomposition includes the parsed
    wavelength axis and absolute ``mm11`` transmission so notebooks can inspect
    the original transmission scale even when decomposing normalized Mueller
    matrices.
    """
    loaded = read_woollam_dat(
        path,
        matrix_kind=matrix_kind,
        value_column=value_column,
        mm11_transmission_column=mm11_transmission_column,
        rotation_column=rotation_column,
        sort_energy=sort_energy,
    )
    result = decompose_dataset(
        loaded["mueller"],
        loaded["energy_eV"],
        loaded["rotations_deg"],
        thickness=thickness,
        output_dir=output_dir,
    )
    result["wavelength_nm"] = loaded["wavelength_nm"]
    result["m11_transmission"] = loaded["m11_transmission"]
    result["aoi_deg"] = loaded["aoi_deg"]
    result["dat_metadata_lines"] = loaded["metadata_lines"]
    result["dat_diagnostics"] = loaded["diagnostics"]
    return result


def estimate_direct_derivative_split_to_decomposition(
    result: dict[str, Any],
    *,
    term_prefix: str = "linear_dichroism",
    energy_min: float | None = None,
    energy_max: float | None = None,
    vector_part: str = "real",
    component: str = "real",
    axis_offset_deg: float = 0.0,
    max_delta_eV: float | None = 0.20,
    smooth_width_eV: float = 0.025,
) -> dict[str, Any]:
    """Direct derivative split estimate from a collapsed LD/LB spectrum."""
    terms = result["terms"]
    x_name = f"{term_prefix}_x"
    y_name = f"{term_prefix}_y"
    if x_name not in terms or y_name not in terms:
        raise KeyError(
            f"Could not find {x_name!r} and {y_name!r}. Available terms: {sorted(terms)}"
        )

    collapsed = collapse_twofold_anisotropy_spectrum(
        terms[x_name],
        terms[y_name],
        result["rotations_deg"],
        value_part=vector_part,
        axis_offset_deg=axis_offset_deg,
    )
    estimate = estimate_direct_derivative_split_spectrum(
        result["energy_eV"],
        collapsed["spectrum"],
        energy_min=energy_min,
        energy_max=energy_max,
        component=component,
        max_delta_eV=max_delta_eV,
        smooth_width_eV=smooth_width_eV,
    )
    estimate["term_prefix"] = term_prefix
    estimate["vector_part"] = vector_part
    estimate["axis_offset_deg"] = float(axis_offset_deg)
    return estimate


def detect_decomposition_features(
    result: dict[str, Any],
    *,
    term_prefixes: tuple[str, ...] = ("linear_dichroism", "linear_birefringence"),
    vector_part: str = "real",
    component: str = "real",
    axis_offset_deg: float = 0.0,
    energy_min: float | None = None,
    energy_max: float | None = None,
    baseline_widths_eV: tuple[float, ...] = _DEFAULT_FEATURE_BASELINE_WIDTHS_EV,
    smooth_width_eV: float = 0.025,
    min_z: float = 3.0,
    min_prominence_z: float = 2.0,
    min_separation_eV: float = 0.04,
    max_features_per_term: int = 8,
) -> dict[str, Any]:
    """Scan collapsed anisotropy spectra for candidate features.

    This is meant as a discovery pass before choosing a narrow physical fit
    window. It scans both linear dichroism and linear birefringence by default,
    using the same twofold rotation collapse as the split-transition fitter.
    """
    terms = result["terms"]
    features: list[dict[str, Any]] = []
    skipped_terms: list[dict[str, str]] = []

    for term_prefix in term_prefixes:
        x_name = f"{term_prefix}_x"
        y_name = f"{term_prefix}_y"
        if x_name not in terms or y_name not in terms:
            skipped_terms.append(
                {
                    "term_prefix": term_prefix,
                    "reason": f"Missing {x_name!r} or {y_name!r}.",
                }
            )
            continue

        collapsed = collapse_twofold_anisotropy_spectrum(
            terms[x_name],
            terms[y_name],
            result["rotations_deg"],
            value_part=vector_part,
            axis_offset_deg=axis_offset_deg,
        )
        scan = detect_spectral_features(
            result["energy_eV"],
            collapsed["spectrum"],
            scatter=collapsed["scatter"],
            energy_min=energy_min,
            energy_max=energy_max,
            component=component,
            baseline_widths_eV=baseline_widths_eV,
            smooth_width_eV=smooth_width_eV,
            min_z=min_z,
            min_prominence_z=min_prominence_z,
            min_separation_eV=min_separation_eV,
            max_features=max_features_per_term,
        )
        for per_term_rank, feature in enumerate(scan["features"], start=1):
            candidate = dict(feature)
            candidate["term_prefix"] = term_prefix
            candidate["per_term_rank"] = per_term_rank
            features.append(candidate)

    features.sort(key=lambda feature: float(feature["score"]), reverse=True)
    for rank, feature in enumerate(features, start=1):
        feature["rank"] = rank

    return {
        "settings": {
            "term_prefixes": tuple(term_prefixes),
            "vector_part": vector_part,
            "component": component,
            "axis_offset_deg": float(axis_offset_deg),
            "energy_window_eV": (energy_min, energy_max),
            "baseline_widths_eV": tuple(float(width) for width in baseline_widths_eV),
            "smooth_width_eV": float(smooth_width_eV),
            "min_z": float(min_z),
            "min_prominence_z": float(min_prominence_z),
            "min_separation_eV": float(min_separation_eV),
            "max_features_per_term": int(max_features_per_term),
        },
        "features": features,
        "skipped_terms": skipped_terms,
    }


def filter_feature_scan_by_energy(
    feature_scan: dict[str, Any],
    *,
    energy_min: float | None = None,
    energy_max: float | None = None,
) -> dict[str, Any]:
    """Return a feature scan with candidates limited to an energy window.

    Feature detection itself is often more stable when the baseline/noise
    estimate sees the broader spectrum. This helper lets callers keep that
    broader context while using only candidates inside the requested physical
    fitting window for Delta Vb estimation.
    """
    if energy_min is None and energy_max is None:
        return feature_scan

    lower = -np.inf if energy_min is None else float(energy_min)
    upper = np.inf if energy_max is None else float(energy_max)
    filtered_features: list[dict[str, Any]] = []
    for feature in feature_scan.get("features", []):
        energy = float(feature.get("energy_eV", np.nan))
        if np.isfinite(energy) and lower <= energy <= upper:
            filtered = dict(feature)
            filtered["source_rank"] = filtered.get("rank")
            filtered_features.append(filtered)

    filtered_features.sort(key=lambda feature: float(feature.get("score", 0.0)), reverse=True)
    for rank, feature in enumerate(filtered_features, start=1):
        feature["rank"] = rank

    settings = dict(feature_scan.get("settings", {}))
    settings["candidate_energy_window_eV"] = (
        None if energy_min is None else float(energy_min),
        None if energy_max is None else float(energy_max),
    )
    return {
        **feature_scan,
        "settings": settings,
        "features": filtered_features,
    }


def fit_split_transition_to_decomposition(
    result: dict[str, Any],
    term_prefix: str = "linear_dichroism",
    energy_min: float | None = None,
    energy_max: float | None = None,
    vector_part: str = "real",
    fit_component: str = "real",
    axis_offset_deg: float = 0.0,
    exponent: float = -0.5,
    initial: dict[str, float] | None = None,
    max_delta_eV: float | None = 0.30,
) -> dict[str, Any]:
    """Collapse a decomposition's anisotropy vector and fit split transitions."""
    terms = result["terms"]
    x_name = f"{term_prefix}_x"
    y_name = f"{term_prefix}_y"
    if x_name not in terms or y_name not in terms:
        raise KeyError(
            f"Could not find {x_name!r} and {y_name!r}. Available terms: {sorted(terms)}"
        )

    collapsed = collapse_twofold_anisotropy_spectrum(
        terms[x_name],
        terms[y_name],
        result["rotations_deg"],
        value_part=vector_part,
        axis_offset_deg=axis_offset_deg,
    )
    fit = fit_split_transition_spectrum(
        result["energy_eV"],
        collapsed["spectrum"],
        energy_min=energy_min,
        energy_max=energy_max,
        exponent=exponent,
        component=fit_component,
        initial=initial,
        max_delta_eV=max_delta_eV,
    )
    fit["collapsed"] = collapsed
    fit["term_prefix"] = term_prefix
    fit["energy_window_eV"] = (energy_min, energy_max)
    fit["vector_part"] = vector_part
    fit["axis_offset_deg"] = float(axis_offset_deg)
    fit["output_dir"] = result.get("output_dir")
    return fit


def _fit_normalized_split_transition_to_decomposition(
    result: dict[str, Any],
    *,
    term_prefix: str,
    energy_min: float,
    energy_max: float,
    vector_part: str,
    fit_component: str,
    axis_offset_deg: float,
    exponent: float,
    max_delta_eV: float | None,
    initial: dict[str, float] | None = None,
) -> dict[str, Any]:
    terms = result["terms"]
    collapsed = collapse_twofold_anisotropy_spectrum(
        terms[f"{term_prefix}_x"],
        terms[f"{term_prefix}_y"],
        result["rotations_deg"],
        value_part=vector_part,
        axis_offset_deg=axis_offset_deg,
    )
    component_values = _spectrum_component(collapsed["spectrum"], fit_component)
    scale = _windowed_signal_scale(
        np.asarray(result["energy_eV"], dtype=np.float64),
        component_values,
        energy_min,
        energy_max,
    )
    normalized_spectrum = collapsed["spectrum"] / scale
    fit = fit_split_transition_spectrum(
        result["energy_eV"],
        normalized_spectrum,
        energy_min=energy_min,
        energy_max=energy_max,
        exponent=exponent,
        component=fit_component,
        initial=initial,
        max_delta_eV=max_delta_eV,
    )
    fit["collapsed"] = collapsed
    fit["term_prefix"] = term_prefix
    fit["energy_window_eV"] = (energy_min, energy_max)
    fit["vector_part"] = vector_part
    fit["axis_offset_deg"] = float(axis_offset_deg)
    fit["normalization_scale"] = scale
    return fit


def _best_local_split_fit(
    result: dict[str, Any],
    *,
    term_prefix: str,
    energy_min: float,
    energy_max: float,
    feature_energies: list[float],
    vector_part: str,
    fit_component: str,
    axis_offset_deg: float,
    exponent: float,
    max_delta_eV: float | None,
) -> dict[str, Any]:
    fits = []
    errors = []
    for initial in _local_split_initial_guesses(
        feature_energies,
        energy_min,
        energy_max,
        max_delta_eV,
    ):
        try:
            fit = _fit_normalized_split_transition_to_decomposition(
                result,
                term_prefix=term_prefix,
                energy_min=energy_min,
                energy_max=energy_max,
                vector_part=vector_part,
                fit_component=fit_component,
                axis_offset_deg=axis_offset_deg,
                exponent=exponent,
                max_delta_eV=max_delta_eV,
                initial=initial,
            )
        except Exception as exc:
            errors.append(str(exc))
            continue
        fits.append(fit)

    if not fits:
        raise RuntimeError("; ".join(errors) if errors else "No local split fits ran.")

    successful = [fit for fit in fits if fit["success"]]
    candidate_fits = successful if successful else fits
    best = min(candidate_fits, key=lambda fit: float(fit["rmse"]))
    best_rmse = float(best["rmse"])
    near_best = [
        fit
        for fit in candidate_fits
        if float(fit["rmse"]) <= best_rmse * 1.10 + 1.0e-12
    ]
    near_best_delta = [
        fit["parameters"]["delta_meV"]
        for fit in near_best
        if np.isfinite(fit["parameters"]["delta_meV"])
    ]
    best["n_initial_guesses"] = len(fits)
    best["candidate_delta_meV"] = [
        fit["parameters"]["delta_meV"]
        for fit in fits
        if np.isfinite(fit["parameters"]["delta_meV"])
    ]
    best["near_best_delta_meV"] = near_best_delta
    if len(near_best_delta) > 1:
        best["delta_std_meV"] = float(np.nanstd(near_best_delta))
    else:
        best["delta_std_meV"] = 0.0
    return best


def fit_kk_consistent_split_to_decomposition(
    result: dict[str, Any],
    *,
    energy_min: float,
    energy_max: float,
    vector_part: str = "real",
    axis_offset_deg: float = 0.0,
    exponent: float = -0.5,
    max_delta_eV: float | None = 0.20,
    initial: dict[str, float] | None = None,
    loss: str = "soft_l1",
) -> dict[str, Any]:
    """Jointly fit the LD and LB on-axis spectra with one KK-consistent model."""
    terms = result["terms"]
    rotations = result["rotations_deg"]
    energy = np.asarray(result["energy_eV"], dtype=np.float64)

    ld = collapse_twofold_anisotropy_spectrum(
        terms["linear_dichroism_x"],
        terms["linear_dichroism_y"],
        rotations,
        value_part=vector_part,
        axis_offset_deg=axis_offset_deg,
    )
    lb = collapse_twofold_anisotropy_spectrum(
        terms["linear_birefringence_x"],
        terms["linear_birefringence_y"],
        rotations,
        value_part=vector_part,
        axis_offset_deg=axis_offset_deg,
    )
    fit = fit_kk_consistent_split_spectra(
        energy,
        np.real(ld["spectrum"]),
        np.real(lb["spectrum"]),
        energy_min=energy_min,
        energy_max=energy_max,
        exponent=exponent,
        max_delta_eV=max_delta_eV,
        initial=initial,
        loss=loss,
    )
    fit["vector_part"] = vector_part
    fit["axis_offset_deg"] = float(axis_offset_deg)
    return fit


def _estimate_energy_windows_overlap(
    first: dict[str, Any],
    second: dict[str, Any],
) -> bool:
    first_window = first.get("energy_window_eV", (np.nan, np.nan))
    second_window = second.get("energy_window_eV", (np.nan, np.nan))
    if len(first_window) < 2 or len(second_window) < 2:
        return False
    lower = max(float(first_window[0]), float(second_window[0]))
    upper = min(float(first_window[1]), float(second_window[1]))
    return bool(np.isfinite(lower) and np.isfinite(upper) and lower <= upper)


def _splitting_result_basis(
    energy_windows_overlap: bool,
    within_agreement_tolerance: bool,
) -> str:
    if energy_windows_overlap and within_agreement_tolerance:
        return "ld_lb_value_and_energy_agreement"
    if energy_windows_overlap:
        return "ld_lb_energy_agreement_split_mismatch"
    return "ld_lb_value_agreement"


def summarize_splitting_consensus(
    estimates: list[dict[str, Any]],
    *,
    agreement_tolerance_meV: float = 7.5,
    transition_tolerance_meV: float = 12.0,
) -> dict[str, Any]:
    """Combine close LD/LB splitting estimates into consensus values."""
    usable = [
        estimate
        for estimate in estimates
        if estimate.get("success")
        and np.isfinite(float(estimate.get("splitting_meV", np.nan)))
        and estimate.get("term_prefix") in {"linear_dichroism", "linear_birefringence"}
    ]
    lds = [estimate for estimate in usable if estimate.get("term_prefix") == "linear_dichroism"]
    lbs = [
        estimate
        for estimate in usable
        if estimate.get("term_prefix") == "linear_birefringence"
    ]

    pair_candidates: list[dict[str, Any]] = []
    for ld_estimate in lds:
        for lb_estimate in lbs:
            values = [
                float(ld_estimate["splitting_meV"]),
                float(lb_estimate["splitting_meV"]),
            ]
            difference = abs(values[0] - values[1])
            energy_windows_overlap = _estimate_energy_windows_overlap(
                ld_estimate, lb_estimate
            )
            within_agreement_tolerance = difference <= float(agreement_tolerance_meV)
            if not (energy_windows_overlap or within_agreement_tolerance):
                continue

            mean = float(np.mean(values))
            std = float(np.std(values))
            all_stable = all(
                estimate.get("fit_stability") == "stable"
                for estimate in (ld_estimate, lb_estimate)
            )
            all_paired = all(
                estimate.get("assignment_quality") == "paired_features"
                for estimate in (ld_estimate, lb_estimate)
            )
            if all_stable and all_paired and energy_windows_overlap and within_agreement_tolerance:
                confidence = "high"
            elif (
                energy_windows_overlap
                and within_agreement_tolerance
                and (all_stable or all_paired)
            ):
                confidence = "medium"
            else:
                confidence = "provisional"

            lower_transitions = [
                float(ld_estimate.get("lower_transition_eV", np.nan)),
                float(lb_estimate.get("lower_transition_eV", np.nan)),
            ]
            upper_transitions = [
                float(ld_estimate.get("upper_transition_eV", np.nan)),
                float(lb_estimate.get("upper_transition_eV", np.nan)),
            ]
            centers = [
                float(ld_estimate.get("center_eV", np.nan)),
                float(lb_estimate.get("center_eV", np.nan)),
            ]
            finite_lowers = [value for value in lower_transitions if np.isfinite(value)]
            finite_uppers = [value for value in upper_transitions if np.isfinite(value)]
            finite_centers = [value for value in centers if np.isfinite(value)]
            lower_transition_spread_meV = (
                1000.0 * float(max(finite_lowers) - min(finite_lowers))
                if len(finite_lowers) > 1
                else 0.0
            )
            upper_transition_spread_meV = (
                1000.0 * float(max(finite_uppers) - min(finite_uppers))
                if len(finite_uppers) > 1
                else 0.0
            )
            center_spread_meV = (
                1000.0 * float(max(finite_centers) - min(finite_centers))
                if len(finite_centers) > 1
                else 0.0
            )
            derivative_splits = [
                _finite_or_nan(ld_estimate.get("direct_derivative_splitting_meV")),
                _finite_or_nan(lb_estimate.get("direct_derivative_splitting_meV")),
            ]
            finite_derivative_splits = [
                value for value in derivative_splits if np.isfinite(value)
            ]
            derivative_spread_meV = (
                float(max(finite_derivative_splits) - min(finite_derivative_splits))
                if len(finite_derivative_splits) > 1
                else np.nan
            )
            derivative_splitting_meV = (
                float(np.mean(finite_derivative_splits))
                if finite_derivative_splits
                else np.nan
            )
            derivative_component_agreement = bool(
                len(finite_derivative_splits) >= 2
                and derivative_spread_meV <= float(agreement_tolerance_meV)
            )
            derivative_confidences = [
                str(ld_estimate.get("direct_derivative_confidence", "")),
                str(lb_estimate.get("direct_derivative_confidence", "")),
            ]
            derivative_confidence_order = {"high": 0, "medium": 1, "provisional": 2}
            finite_derivative_confidences = [
                value for value in derivative_confidences if value in derivative_confidence_order
            ]
            derivative_confidence = (
                max(
                    finite_derivative_confidences,
                    key=lambda value: derivative_confidence_order[value],
                )
                if finite_derivative_confidences
                else ""
            )
            transitions_within_tolerance = bool(
                lower_transition_spread_meV <= float(transition_tolerance_meV)
                and upper_transition_spread_meV <= float(transition_tolerance_meV)
            )
            if not transitions_within_tolerance:
                confidence = "provisional"
            if energy_windows_overlap:
                bandgap_eV = float(np.mean(finite_lowers)) if finite_lowers else np.nan
                bandgap_spread_eV = (
                    float(max(finite_lowers) - min(finite_lowers))
                    if len(finite_lowers) > 1
                    else 0.0
                )
                upper_transition_eV = (
                    float(np.mean(finite_uppers)) if finite_uppers else np.nan
                )
                upper_transition_spread_eV = (
                    float(max(finite_uppers) - min(finite_uppers))
                    if len(finite_uppers) > 1
                    else 0.0
                )
                center_eV = float(np.mean(finite_centers)) if finite_centers else np.nan
            else:
                bandgap_eV = np.nan
                bandgap_spread_eV = np.nan
                upper_transition_eV = np.nan
                upper_transition_spread_eV = np.nan
                center_eV = np.nan

            pair_candidates.append(
                {
                    "bandgap_eV": bandgap_eV,
                    "bandgap_spread_eV": bandgap_spread_eV,
                    "upper_transition_eV": upper_transition_eV,
                    "upper_transition_spread_eV": upper_transition_spread_eV,
                    "center_eV": center_eV,
                    "splitting_meV": mean,
                    "spread_meV": difference,
                    "std_meV": std,
                    "agreement_tolerance_meV": float(agreement_tolerance_meV),
                    "transition_tolerance_meV": float(transition_tolerance_meV),
                    "lower_transition_spread_meV": lower_transition_spread_meV,
                    "upper_transition_spread_meV": upper_transition_spread_meV,
                    "center_spread_meV": center_spread_meV,
                    "transitions_within_tolerance": transitions_within_tolerance,
                    "within_agreement_tolerance": within_agreement_tolerance,
                    "confidence": confidence,
                    "basis": _splitting_result_basis(
                        energy_windows_overlap,
                        within_agreement_tolerance,
                    ),
                    "energy_windows_overlap": energy_windows_overlap,
                    "component_estimate_ranks": [
                        int(ld_estimate.get("rank", 0)),
                        int(lb_estimate.get("rank", 0)),
                    ],
                    "component_terms": [
                        str(ld_estimate["term_prefix"]),
                        str(lb_estimate["term_prefix"]),
                    ],
                    "component_splittings_meV": values,
                    "component_lower_transition_eV": lower_transitions,
                    "component_upper_transition_eV": upper_transitions,
                    "component_center_eV": centers,
                    "component_direct_derivative_splittings_meV": derivative_splits,
                    "component_direct_derivative_confidence": derivative_confidences,
                    "direct_derivative_splitting_meV": derivative_splitting_meV,
                    "direct_derivative_spread_meV": derivative_spread_meV,
                    "direct_derivative_component_agreement": derivative_component_agreement,
                    "direct_derivative_confidence": derivative_confidence,
                    "component_assignment_quality": [
                        str(ld_estimate.get("assignment_quality", "")),
                        str(lb_estimate.get("assignment_quality", "")),
                    ],
                    "component_fit_stability": [
                        str(ld_estimate.get("fit_stability", "")),
                        str(lb_estimate.get("fit_stability", "")),
                    ],
                    "component_energy_windows_eV": [
                        tuple(ld_estimate.get("energy_window_eV", ())),
                        tuple(lb_estimate.get("energy_window_eV", ())),
                    ],
                }
            )

    confidence_order = {"high": 0, "medium": 1, "provisional": 2}
    pair_candidates.sort(
        key=lambda item: (
            not bool(item["energy_windows_overlap"]),
            not bool(item["within_agreement_tolerance"]),
            not bool(item["transitions_within_tolerance"]),
            confidence_order.get(str(item["confidence"]), 9),
            float(item["spread_meV"]),
            min(item["component_estimate_ranks"]),
        )
    )
    for rank, item in enumerate(pair_candidates, start=1):
        item["rank"] = rank
        _add_delta_vb_recommendation(item)

    matched_ranks = {
        rank
        for item in pair_candidates
        for rank in item["component_estimate_ranks"]
        if rank
    }
    unmatched = [
        {
            "rank": int(estimate.get("rank", 0)),
            "term_prefix": str(estimate.get("term_prefix", "")),
            "splitting_meV": float(estimate.get("splitting_meV", np.nan)),
            "assignment_quality": str(estimate.get("assignment_quality", "")),
            "fit_stability": str(estimate.get("fit_stability", "")),
        }
        for estimate in usable
        if int(estimate.get("rank", 0)) not in matched_ranks
    ]

    return {
        "settings": {
            "agreement_tolerance_meV": float(agreement_tolerance_meV),
            "transition_tolerance_meV": float(transition_tolerance_meV),
            "required_terms": ("linear_dichroism", "linear_birefringence"),
        },
        "primary": pair_candidates[0] if pair_candidates else None,
        "matches": pair_candidates,
        "unmatched_estimates": unmatched,
        "note": (
            "The primary result prefers a linear-dichroism and "
            "linear-birefringence pair from overlapping energy windows, then "
            "the closest splitting agreement. If the overlapping pair has an "
            "LD/LB splitting spread above the tolerance, Eg is still assigned "
            "from the shared energy region, and a stable single-channel "
            "transition separation can be used as a provisional Delta Vb. If "
            "no overlapping pair is found, it falls back to the closest "
            "numerical agreement without assigning Eg."
        ),
    }


def _finite_or_nan(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def _component_delta_vb_candidates(primary: dict[str, Any]) -> list[dict[str, Any]]:
    """Return stable single-channel transition-separation candidates."""
    terms = [str(value) for value in primary.get("component_terms", [])]
    values = [
        _finite_or_nan(value)
        for value in primary.get("component_splittings_meV", [])
    ]
    lowers = [
        _finite_or_nan(value)
        for value in primary.get("component_lower_transition_eV", [])
    ]
    uppers = [
        _finite_or_nan(value)
        for value in primary.get("component_upper_transition_eV", [])
    ]
    qualities = [
        str(value) for value in primary.get("component_assignment_quality", [])
    ]
    stabilities = [
        str(value) for value in primary.get("component_fit_stability", [])
    ]
    ranks: list[int] = []
    for value in primary.get("component_estimate_ranks", []):
        try:
            ranks.append(int(value))
        except (TypeError, ValueError):
            ranks.append(999)

    candidates: list[dict[str, Any]] = []
    for index, term in enumerate(terms):
        value = values[index] if index < len(values) else np.nan
        lower = lowers[index] if index < len(lowers) else np.nan
        upper = uppers[index] if index < len(uppers) else np.nan
        stability = stabilities[index] if index < len(stabilities) else ""
        if (
            not np.isfinite(value)
            or not np.isfinite(lower)
            or not np.isfinite(upper)
            or upper <= lower
            or stability != "stable"
        ):
            continue
        transition_delta_meV = 1000.0 * (upper - lower)
        if abs(transition_delta_meV - value) > max(0.5, 0.02 * abs(value)):
            continue
        candidates.append(
            {
                "term": term,
                "value": float(value),
                "lower_transition_eV": float(lower),
                "upper_transition_eV": float(upper),
                "assignment_quality": qualities[index] if index < len(qualities) else "",
                "fit_stability": stability,
                "rank": ranks[index] if index < len(ranks) else 999,
            }
        )

    candidates.sort(
        key=lambda item: (
            item["assignment_quality"] != "paired_features",
            item["term"] != "linear_dichroism",
            int(item["rank"]),
        )
    )
    return candidates


def _add_delta_vb_recommendation(primary: dict[str, Any]) -> None:
    """Add conservative Delta Vb recommendation fields to a consensus match."""
    warnings_out: list[str] = []
    tolerance = float(primary.get("agreement_tolerance_meV", 7.5))
    transition_tolerance = float(primary.get("transition_tolerance_meV", 12.0))
    spread = _finite_or_nan(primary.get("spread_meV"))
    lower_spread = _finite_or_nan(primary.get("lower_transition_spread_meV"))
    upper_spread = _finite_or_nan(primary.get("upper_transition_spread_meV"))
    spread_text = f"{spread:.2f}" if np.isfinite(spread) else "unknown"
    tolerance_text = f"{tolerance:.2f}" if np.isfinite(tolerance) else "unknown"
    transition_tolerance_text = (
        f"{transition_tolerance:.2f}" if np.isfinite(transition_tolerance) else "unknown"
    )

    if not bool(primary.get("energy_windows_overlap")):
        warnings_out.append("LD/LB energy windows do not overlap.")
    if not bool(primary.get("within_agreement_tolerance")):
        warnings_out.append(
            f"LD/LB independent split spread is {spread_text} meV, above "
            f"the {tolerance_text} meV tolerance."
        )
    if not bool(primary.get("transitions_within_tolerance")):
        lower_text = f"{lower_spread:.2f}" if np.isfinite(lower_spread) else "unknown"
        upper_text = f"{upper_spread:.2f}" if np.isfinite(upper_spread) else "unknown"
        warnings_out.append(
            "LD/LB fitted transition energies do not align "
            f"(lower spread {lower_text} meV, upper spread {upper_text} meV; "
            f"tolerance {transition_tolerance_text} meV)."
        )

    qualities = [str(value) for value in primary.get("component_assignment_quality", [])]
    stabilities = [str(value) for value in primary.get("component_fit_stability", [])]
    component_values = [
        value
        for value in (
            _finite_or_nan(item)
            for item in primary.get("component_splittings_meV", [])
        )
        if np.isfinite(value)
    ]
    if len(component_values) < 2:
        warnings_out.append("Fewer than two finite LD/LB component splits were available.")
    if any(value != "paired_features" for value in qualities):
        warnings_out.append(
            "At least one component split came from a single feature; a "
            "two-transition split is underdetermined there."
        )
    if any(value != "stable" for value in stabilities):
        warnings_out.append(
            "At least one component fit has multiple local minima."
        )

    kk_value = _finite_or_nan(primary.get("kk_splitting_meV"))
    if primary.get("kk_fit_success") is False:
        message = str(primary.get("kk_fit_message", "")).strip()
        if message:
            warnings_out.append(f"Joint KK LD/LB validation failed: {message}")
        else:
            warnings_out.append("Joint KK LD/LB validation failed.")
    if np.isfinite(kk_value) and component_values:
        kk_distance = max(abs(kk_value - value) for value in component_values)
        primary["kk_component_max_difference_meV"] = float(kk_distance)
        if kk_distance > max(tolerance, spread if np.isfinite(spread) else 0.0):
            warnings_out.append(
                "Joint KK split disagrees with one or more independent "
                "LD/LB component splits."
            )

    direct_value = _finite_or_nan(primary.get("direct_derivative_splitting_meV"))
    direct_spread = _finite_or_nan(primary.get("direct_derivative_spread_meV"))
    direct_confidence = str(primary.get("direct_derivative_confidence", ""))
    direct_usable = bool(
        np.isfinite(direct_value)
        and bool(primary.get("direct_derivative_component_agreement"))
        and direct_confidence in {"high", "medium"}
    )
    if np.isfinite(direct_value) and not direct_usable:
        primary["direct_derivative_rejected_reason"] = (
            "LD/LB derivative peak separation did not reach medium confidence "
            "or component agreement."
        )

    recommendation_values: list[float] = []
    recommendation_sources: list[str] = []
    recommendation_notes: list[str] = []
    agreement_limit = max(tolerance, transition_tolerance)

    if not warnings_out:
        split_value = _finite_or_nan(primary.get("splitting_meV"))
        if np.isfinite(split_value):
            recommendation_values.append(split_value)
            recommendation_sources.append("independent_ld_lb_mean")
        if np.isfinite(kk_value) and np.isfinite(split_value):
            if abs(kk_value - split_value) <= agreement_limit:
                recommendation_values.append(float(kk_value))
                recommendation_sources.append("joint_kk")
            else:
                warnings_out.append("Joint KK split is outside the recommendation tolerance.")
        if direct_usable and np.isfinite(split_value):
            if abs(direct_value - split_value) <= agreement_limit:
                recommendation_values.append(float(direct_value))
                recommendation_sources.append("direct_derivative")
            else:
                recommendation_notes.append(
                    "Direct derivative split was not fused because it is outside "
                    "the recommendation tolerance."
                )
    else:
        if direct_usable:
            direct_values = [float(direct_value)]
            if np.isfinite(kk_value):
                if abs(float(kk_value) - float(direct_value)) <= agreement_limit:
                    direct_values.append(float(kk_value))
                    recommendation_sources.append("joint_kk")
                else:
                    recommendation_notes.append(
                        "Joint KK split was not fused with the direct derivative split."
                    )
            recommendation_values = direct_values
            recommendation_sources.insert(0, "direct_derivative_ld_lb_consensus")
            recommendation_notes.append(
                "Independent split-transition checks were inconclusive; using "
                "the direct dD/dE LD/LB peak separation instead."
            )
        else:
            component_candidates = _component_delta_vb_candidates(primary)
            if component_candidates:
                best_component = component_candidates[0]
                recommendation_values = [float(best_component["value"])]
                recommendation_sources = [
                    f"{best_component['term']}_transition_separation"
                ]
                primary["single_component_delta_vb_meV"] = float(
                    best_component["value"]
                )
                primary["single_component_delta_source"] = str(best_component["term"])
                primary["single_component_lower_transition_eV"] = float(
                    best_component["lower_transition_eV"]
                )
                primary["single_component_upper_transition_eV"] = float(
                    best_component["upper_transition_eV"]
                )
                recommendation_notes.append(
                    "LD/LB consensus checks were inconclusive; using the stable "
                    f"{best_component['term'].replace('_', ' ')} transition "
                    "separation as a provisional Delta Vb."
                )

    finite_recommendations = [
        value for value in recommendation_values if np.isfinite(value)
    ]
    if finite_recommendations:
        primary["recommended_delta_vb_meV"] = float(np.mean(finite_recommendations))
        primary["recommended_delta_source"] = "+".join(recommendation_sources)
        primary["recommendation_components_meV"] = finite_recommendations
        primary["recommendation_spread_meV"] = (
            float(max(finite_recommendations) - min(finite_recommendations))
            if len(finite_recommendations) > 1
            else 0.0
        )
        primary["requires_manual_delta_vb"] = False
    else:
        primary["recommended_delta_vb_meV"] = np.nan
        primary["recommended_delta_source"] = "manual_review_required"
        primary["recommendation_components_meV"] = []
        primary["recommendation_spread_meV"] = np.nan
        primary["requires_manual_delta_vb"] = True
    primary["direct_derivative_usable"] = direct_usable
    primary["direct_derivative_agreement_limit_meV"] = float(agreement_limit)
    primary["recommendation_notes"] = recommendation_notes
    primary["math_warnings"] = warnings_out


def estimate_valence_band_splittings(
    result: dict[str, Any],
    feature_scan: dict[str, Any] | None = None,
    *,
    term_prefixes: tuple[str, ...] = ("linear_dichroism", "linear_birefringence"),
    vector_part: str = "real",
    fit_component: str = "real",
    axis_offset_deg: float = 0.0,
    exponent: float = -0.5,
    max_delta_eV: float | None = 0.20,
    group_span_eV: float = 0.10,
    window_padding_eV: float = 0.08,
    min_window_width_eV: float = 0.16,
    max_estimates: int = 6,
    consensus_tolerance_meV: float = 7.5,
    transition_tolerance_meV: float = 12.0,
) -> dict[str, Any]:
    """Estimate valence-band splittings from local split-transition fits.

    The returned ``splitting_meV`` is the fitted transition separation. It is a
    valence-band splitting only under the physical assignment that the two
    fitted optical transitions share the same conduction-band final state.

    A Kramers-Kronig-consistent joint LD/LB fit
    (:func:`fit_kk_consistent_split_to_decomposition`) is run on the primary
    energy window and reported under ``kk_consistent_split``.
    """
    energy = np.asarray(result["energy_eV"], dtype=np.float64)
    if energy.size == 0:
        return {
            "settings": {
                "term_prefixes": tuple(term_prefixes),
                "vector_part": vector_part,
                "fit_component": fit_component,
                "axis_offset_deg": float(axis_offset_deg),
                "exponent": float(exponent),
                "max_delta_eV": None if max_delta_eV is None else float(max_delta_eV),
                "group_span_eV": float(group_span_eV),
                "window_padding_eV": float(window_padding_eV),
                "min_window_width_eV": float(min_window_width_eV),
                "max_estimates": int(max_estimates),
                "consensus_tolerance_meV": float(consensus_tolerance_meV),
                "transition_tolerance_meV": float(transition_tolerance_meV),
            },
            "estimates": [],
            "consensus": summarize_splitting_consensus(
                [],
                agreement_tolerance_meV=consensus_tolerance_meV,
                transition_tolerance_meV=transition_tolerance_meV,
            ),
            "results": summarize_splitting_consensus(
                [],
                agreement_tolerance_meV=consensus_tolerance_meV,
                transition_tolerance_meV=transition_tolerance_meV,
            ),
            "message": "No energy points available.",
            "note": (
                "splitting_meV is a valence-band splitting only if the two "
                "transitions share the same conduction-band final state."
            ),
        }

    if feature_scan is None:
        feature_scan = detect_decomposition_features(
            result,
            term_prefixes=term_prefixes,
            vector_part=vector_part,
            component=fit_component,
            axis_offset_deg=axis_offset_deg,
        )

    scan_settings = feature_scan.get("settings", {})
    vector_part = str(scan_settings.get("vector_part", vector_part))
    fit_component = str(scan_settings.get("component", fit_component))
    axis_offset_deg = float(scan_settings.get("axis_offset_deg", axis_offset_deg))

    candidate_features = [
        feature
        for feature in feature_scan.get("features", [])
        if feature.get("term_prefix") in term_prefixes
    ]

    groups: list[list[dict[str, Any]]] = []
    for term_prefix in term_prefixes:
        term_features = sorted(
            [
                feature
                for feature in candidate_features
                if feature.get("term_prefix") == term_prefix
            ],
            key=lambda feature: float(feature["energy_eV"]),
        )
        current: list[dict[str, Any]] = []
        for feature in term_features:
            if not current:
                current = [feature]
                continue
            if float(feature["energy_eV"]) - float(current[0]["energy_eV"]) <= float(group_span_eV):
                current.append(feature)
            else:
                groups.append(current)
                current = [feature]
        if current:
            groups.append(current)

    groups.sort(
        key=lambda group: max(float(feature.get("score", 0.0)) for feature in group),
        reverse=True,
    )

    e_min = float(np.nanmin(energy))
    e_max = float(np.nanmax(energy))
    estimates: list[dict[str, Any]] = []

    for group in groups[: int(max_estimates)]:
        term_prefix = str(group[0]["term_prefix"])
        feature_energies = [float(feature["energy_eV"]) for feature in group]
        center = 0.5 * (min(feature_energies) + max(feature_energies))
        lower = min(feature_energies) - float(window_padding_eV)
        upper = max(feature_energies) + float(window_padding_eV)
        if upper - lower < float(min_window_width_eV):
            half_width = 0.5 * float(min_window_width_eV)
            lower = center - half_width
            upper = center + half_width
        lower = max(e_min, lower)
        upper = min(e_max, upper)

        estimate: dict[str, Any] = {
            "term_prefix": term_prefix,
            "feature_energies_eV": feature_energies,
            "feature_ranks": [int(feature["rank"]) for feature in group if "rank" in feature],
            "feature_kinds": [str(feature.get("kind", "")) for feature in group],
            "energy_window_eV": (float(lower), float(upper)),
            "fit_component": fit_component,
            "vector_part": vector_part,
            "axis_offset_deg": float(axis_offset_deg),
            "assignment_quality": (
                "paired_features" if len(feature_energies) >= 2 else "single_feature"
            ),
        }
        try:
            direct = estimate_direct_derivative_split_to_decomposition(
                result,
                term_prefix=term_prefix,
                energy_min=lower,
                energy_max=upper,
                vector_part=vector_part,
                component=fit_component,
                axis_offset_deg=axis_offset_deg,
                max_delta_eV=max_delta_eV,
            )
        except Exception as exc:
            direct = {"success": False, "message": str(exc)}
        estimate.update(
            {
                "direct_derivative_success": bool(direct.get("success")),
                "direct_derivative_message": str(direct.get("message", "")),
                "direct_derivative_splitting_meV": direct.get("splitting_meV", np.nan),
                "direct_derivative_splitting_eV": direct.get("splitting_eV", np.nan),
                "direct_derivative_lower_peak_eV": direct.get("lower_peak_eV", np.nan),
                "direct_derivative_upper_peak_eV": direct.get("upper_peak_eV", np.nan),
                "direct_derivative_center_eV": direct.get("center_eV", np.nan),
                "direct_derivative_confidence": str(direct.get("confidence", "")),
                "direct_derivative_score": direct.get("score", np.nan),
                "direct_derivative_weakest_peak_z": direct.get("weakest_peak_z", np.nan),
                "direct_derivative_weakest_peak_prominence_z": direct.get(
                    "weakest_peak_prominence_z",
                    np.nan,
                ),
                "direct_derivative_n_peaks": int(direct.get("n_candidate_peaks", 0)),
            }
        )
        try:
            fit = _best_local_split_fit(
                result,
                term_prefix=term_prefix,
                energy_min=lower,
                energy_max=upper,
                feature_energies=feature_energies,
                vector_part=vector_part,
                fit_component=fit_component,
                axis_offset_deg=axis_offset_deg,
                exponent=exponent,
                max_delta_eV=max_delta_eV,
            )
        except Exception as exc:
            estimate.update(
                {
                    "success": False,
                    "message": str(exc),
                    "splitting_meV": np.nan,
                }
            )
            estimates.append(estimate)
            continue

        params = fit["parameters"]
        delta_std_meV = float(fit["delta_std_meV"])
        fit_stability = (
            "stable"
            if delta_std_meV <= max(5.0, 0.25 * abs(params["delta_meV"]))
            else "multi_minimum"
        )
        estimate.update(
            {
                "success": bool(fit["success"]),
                "message": fit["message"],
                "center_eV": params["center_eV"],
                "splitting_eV": params["delta_eV"],
                "splitting_meV": params["delta_meV"],
                "lower_transition_eV": params["lower_transition_eV"],
                "upper_transition_eV": params["upper_transition_eV"],
                "broadening_eV": params["broadening_eV"],
                "broadening_meV": 1000.0 * params["broadening_eV"],
                "rmse": fit["rmse"],
                "mae": fit["mae"],
                "n_points": fit["n_points"],
                "normalization_scale": fit["normalization_scale"],
                "n_initial_guesses": fit["n_initial_guesses"],
                "candidate_delta_meV": fit["candidate_delta_meV"],
                "near_best_delta_meV": fit["near_best_delta_meV"],
                "delta_std_meV": delta_std_meV,
                "fit_stability": fit_stability,
            }
        )
        estimates.append(estimate)

    for rank, estimate in enumerate(estimates, start=1):
        estimate["rank"] = rank

    consensus = summarize_splitting_consensus(
        estimates,
        agreement_tolerance_meV=consensus_tolerance_meV,
        transition_tolerance_meV=transition_tolerance_meV,
    )

    kk_split: dict[str, Any] | None = None
    primary = consensus.get("primary")
    if primary is not None:
        windows = [
            tuple(window)
            for window in primary.get("component_energy_windows_eV", [])
            if len(tuple(window)) == 2
            and all(np.isfinite(float(value)) for value in window)
        ]
        if windows:
            kk_lower = min(float(window[0]) for window in windows)
            kk_upper = max(float(window[1]) for window in windows)
            try:
                kk_split = fit_kk_consistent_split_to_decomposition(
                    result,
                    energy_min=kk_lower,
                    energy_max=kk_upper,
                    vector_part=vector_part,
                    axis_offset_deg=axis_offset_deg,
                    exponent=exponent,
                    max_delta_eV=max_delta_eV,
                )
            except Exception as exc:
                kk_split = {"success": False, "message": str(exc)}
        if kk_split is not None and kk_split.get("success"):
            kk_params = kk_split["parameters"]
            primary["kk_fit_success"] = True
            primary["kk_splitting_meV"] = kk_params["delta_meV"]
            primary["kk_center_eV"] = kk_params["center_eV"]
            primary["kk_lower_transition_eV"] = kk_params["lower_transition_eV"]
            primary["kk_upper_transition_eV"] = kk_params["upper_transition_eV"]
            primary["kk_ld_rmse"] = kk_split["ld_rmse"]
            primary["kk_lb_rmse"] = kk_split["lb_rmse"]
        elif kk_split is not None:
            primary["kk_fit_success"] = False
            primary["kk_fit_message"] = str(kk_split.get("message", ""))
        _add_delta_vb_recommendation(primary)
    consensus["kk_consistent_split"] = kk_split

    return {
        "settings": {
            "term_prefixes": tuple(term_prefixes),
            "vector_part": vector_part,
            "fit_component": fit_component,
            "axis_offset_deg": float(axis_offset_deg),
            "exponent": float(exponent),
            "max_delta_eV": None if max_delta_eV is None else float(max_delta_eV),
            "group_span_eV": float(group_span_eV),
            "window_padding_eV": float(window_padding_eV),
            "min_window_width_eV": float(min_window_width_eV),
            "max_estimates": int(max_estimates),
            "consensus_tolerance_meV": float(consensus_tolerance_meV),
            "transition_tolerance_meV": float(transition_tolerance_meV),
        },
        "estimates": estimates,
        "consensus": consensus,
        "results": consensus,
        "message": "ok",
        "note": (
            "splitting_meV is a valence-band splitting only if the two "
            "transitions share the same conduction-band final state. "
            "recommended_delta_vb_meV uses the stable LD/LB split-transition "
            "mean when available, otherwise a stable single-channel "
            "transition separation or a medium/high-confidence direct dD/dE "
            "LD/LB peak-separation estimate can provide the fallback. "
            "kk_consistent_split is the joint Kramers-Kronig LD/LB fit splitting."
        ),
    }


def _report_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ";".join(str(_report_value(item)) for item in value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if np.isfinite(value):
            return f"{value:.10g}"
        return ""
    return value


def build_short_report_rows(
    result: dict[str, Any],
    *,
    fit: dict[str, Any] | None = None,
    feature_scan: dict[str, Any] | None = None,
    splitting_estimates: dict[str, Any] | None = None,
    dat_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Build compact spreadsheet rows with the main analysis outputs."""
    energy = np.asarray(result.get("energy_eV", []), dtype=np.float64)
    rotations = np.asarray(result.get("rotations_deg", []), dtype=np.float64)
    rows: list[dict[str, Any]] = []

    def add(section: str, metric: str, value: Any, unit: str = "", note: str = "") -> None:
        rows.append(
            {
                "section": section,
                "metric": metric,
                "value": _report_value(value),
                "unit": unit,
                "note": note,
            }
        )

    add("input", "data_file", "" if dat_path is None else str(dat_path))
    add("input", "n_energy", int(energy.size))
    add("input", "n_rotations", int(rotations.size))
    if energy.size:
        add("input", "energy_min", float(np.nanmin(energy)), "eV")
        add("input", "energy_max", float(np.nanmax(energy)), "eV")
    add("input", "thickness", result.get("thickness", ""), "", "blank means integrated")

    primary = None
    if splitting_estimates is not None:
        primary = splitting_estimates.get("results", {}).get("primary")
        if primary is None:
            primary = splitting_estimates.get("consensus", {}).get("primary")
    if primary:
        add("main_result", "Eg", primary.get("bandgap_eV"), "eV", "mean lower transition")
        add(
            "main_result",
            "Eg_spread",
            primary.get("bandgap_spread_eV"),
            "eV",
            "spread between LD and LB lower transitions",
        )
        add("main_result", "splitting", primary.get("splitting_meV"), "meV")
        add(
            "main_result",
            "recommended_delta_vb",
            primary.get("recommended_delta_vb_meV"),
            "meV",
            primary.get("recommended_delta_source", ""),
        )
        add(
            "main_result",
            "kk_splitting",
            primary.get("kk_splitting_meV"),
            "meV",
            "joint Kramers-Kronig LD/LB fit",
        )
        add(
            "main_result",
            "direct_derivative_splitting",
            primary.get("direct_derivative_splitting_meV"),
            "meV",
            "LD/LB dD/dE peak-separation estimate",
        )
        add(
            "main_result",
            "direct_derivative_spread",
            primary.get("direct_derivative_spread_meV"),
            "meV",
            "difference between LD and LB direct derivative splits",
        )
        add(
            "main_result",
            "kk_component_max_difference",
            primary.get("kk_component_max_difference_meV"),
            "meV",
            "largest difference between joint KK and independent LD/LB splits",
        )
        add(
            "main_result",
            "requires_manual_delta_vb",
            primary.get("requires_manual_delta_vb", ""),
        )
        add(
            "main_result",
            "splitting_spread",
            primary.get("spread_meV"),
            "meV",
            "LD/LB splitting difference",
        )
        add(
            "main_result",
            "lower_transition_spread",
            primary.get("lower_transition_spread_meV"),
            "meV",
            "LD/LB lower-transition mismatch",
        )
        add(
            "main_result",
            "upper_transition_spread",
            primary.get("upper_transition_spread_meV"),
            "meV",
            "LD/LB upper-transition mismatch",
        )
        add(
            "main_result",
            "transitions_within_tolerance",
            primary.get("transitions_within_tolerance", ""),
        )
        add("main_result", "upper_transition", primary.get("upper_transition_eV"), "eV")
        add("main_result", "center", primary.get("center_eV"), "eV")
        add("main_result", "confidence", primary.get("confidence", ""))
        add("main_result", "basis", primary.get("basis", ""))
        add(
            "main_result",
            "within_agreement_tolerance",
            primary.get("within_agreement_tolerance", ""),
        )
        add(
            "main_result",
            "agreement_tolerance",
            primary.get("agreement_tolerance_meV", ""),
            "meV",
        )
        add("main_result", "math_warnings", primary.get("math_warnings", []))
        add("main_result", "recommendation_notes", primary.get("recommendation_notes", []))
        add(
            "main_result",
            "component_splittings",
            primary.get("component_splittings_meV", []),
            "meV",
        )
        add(
            "main_result",
            "component_lower_transitions",
            primary.get("component_lower_transition_eV", []),
            "eV",
        )

    if splitting_estimates is not None:
        for estimate in splitting_estimates.get("estimates", [])[:6]:
            prefix = f"estimate_{estimate.get('rank', '')}_{estimate.get('term_prefix', '')}"
            add(prefix, "splitting", estimate.get("splitting_meV"), "meV")
            add(
                prefix,
                "direct_derivative_splitting",
                estimate.get("direct_derivative_splitting_meV"),
                "meV",
            )
            add(prefix, "lower_transition", estimate.get("lower_transition_eV"), "eV")
            add(prefix, "upper_transition", estimate.get("upper_transition_eV"), "eV")
            add(prefix, "assignment_quality", estimate.get("assignment_quality", ""))
            add(prefix, "fit_stability", estimate.get("fit_stability", ""))

    if feature_scan is not None:
        for feature in feature_scan.get("features", [])[:6]:
            prefix = f"feature_{feature.get('rank', '')}_{feature.get('term_prefix', '')}"
            add(prefix, "energy", feature.get("energy_eV"), "eV", feature.get("kind", ""))
            add(prefix, "score", feature.get("score", ""))
            add(prefix, "z_score", feature.get("z_score", ""))
            add(prefix, "scatter_ratio", feature.get("scatter_ratio", ""))

    if fit is not None:
        params = fit.get("parameters", {})
        window = fit.get("energy_window_eV", ("", ""))
        add("selected_split_fit", "window_min", window[0] if len(window) > 0 else "", "eV")
        add("selected_split_fit", "window_max", window[1] if len(window) > 1 else "", "eV")
        add("selected_split_fit", "center", params.get("center_eV"), "eV")
        add("selected_split_fit", "delta", params.get("delta_meV"), "meV")
        add("selected_split_fit", "lower_transition", params.get("lower_transition_eV"), "eV")
        add("selected_split_fit", "upper_transition", params.get("upper_transition_eV"), "eV")
        add("selected_split_fit", "rmse", fit.get("rmse", ""))

    warnings_list = list(result.get("diagnostics", {}).get("warnings", []))
    warnings_list += list(result.get("dat_diagnostics", {}).get("warnings", []))
    add("diagnostics", "warning_count", len(warnings_list))
    if warnings_list:
        add("diagnostics", "first_warning", warnings_list[0])

    return rows


def plot_term_vs_energy(
    result: dict[str, Any], term_name: str, rotation_index: int | None = None
) -> Any:
    """Plot an extracted generator term versus photon energy.

    If rotation_index is None, all rotations are plotted. If result contains an
    output_dir entry, the figure is saved there as a PNG.
    """
    import matplotlib.pyplot as plt

    terms = result["terms"]
    if term_name not in terms:
        raise KeyError(f"Unknown term {term_name!r}. Available terms: {sorted(terms)}")

    energy = np.asarray(result["energy_eV"])
    rotations = np.asarray(result["rotations_deg"])
    values = np.asarray(terms[term_name])
    if values.shape != (len(rotations), len(energy)):
        raise ValueError(
            f"Term {term_name!r} must have shape "
            f"({len(rotations)}, {len(energy)}); got {values.shape}."
        )

    fig, ax = plt.subplots()
    label = _term_label(term_name)
    unit_label = _term_unit_label(result, term_name)

    if rotation_index is None:
        rotation_indices = range(len(rotations))
    else:
        if rotation_index < 0 or rotation_index >= len(rotations):
            raise IndexError(
                f"rotation_index {rotation_index} is out of range for {len(rotations)} rotations."
            )
        rotation_indices = [rotation_index]

    for idx in rotation_indices:
        real_values, imag_values = _coerce_for_plot(values[idx])
        ax.plot(energy, real_values, label=f"{rotations[idx]:g} deg")
        if imag_values is not None:
            ax.plot(energy, imag_values, linestyle="--", label=f"{rotations[idx]:g} deg imag")

    ax.set_xlabel("Photon energy (eV)")
    ax.set_ylabel(f"{label} ({unit_label})")
    ax.set_title(f"{label} vs energy")
    ax.legend()
    fig.tight_layout()

    suffix = "all_rotations" if rotation_index is None else f"rotation_{rotation_index}"
    _save_figure_if_requested(fig, result, f"{_safe_filename(term_name)}_vs_energy_{suffix}.png")
    return fig


def plot_rotation_dependence(
    result: dict[str, Any], term_name: str, energy_index: int | None = None
) -> Any:
    """Plot an extracted generator term versus sample rotation angle.

    If energy_index is None, the middle energy point is used. If result contains
    an output_dir entry, the figure is saved there as a PNG.
    """
    import matplotlib.pyplot as plt

    terms = result["terms"]
    if term_name not in terms:
        raise KeyError(f"Unknown term {term_name!r}. Available terms: {sorted(terms)}")

    energy = np.asarray(result["energy_eV"])
    rotations = np.asarray(result["rotations_deg"])
    values = np.asarray(terms[term_name])
    if values.shape != (len(rotations), len(energy)):
        raise ValueError(
            f"Term {term_name!r} must have shape "
            f"({len(rotations)}, {len(energy)}); got {values.shape}."
        )

    if energy_index is None:
        energy_index = len(energy) // 2
    if energy_index < 0 or energy_index >= len(energy):
        raise IndexError(f"energy_index {energy_index} is out of range for {len(energy)} energies.")

    fig, ax = plt.subplots()
    label = _term_label(term_name)
    unit_label = _term_unit_label(result, term_name)
    real_values, imag_values = _coerce_for_plot(values[:, energy_index])

    ax.plot(rotations, real_values, marker="o", label="real")
    if imag_values is not None:
        ax.plot(rotations, imag_values, marker="o", linestyle="--", label="imag")
        ax.legend()

    ax.set_xlabel("Sample rotation (deg)")
    ax.set_ylabel(f"{label} ({unit_label})")
    ax.set_title(f"{label} at {energy[energy_index]:g} eV")
    fig.tight_layout()

    _save_figure_if_requested(
        fig,
        result,
        f"{_safe_filename(term_name)}_rotation_energy_{energy_index}.png",
    )
    return fig


def plot_decomposition_overview(result: dict[str, Any]) -> Any:
    """Plot a compact overview of key dichroism and birefringence terms.

    The overview shows energy dependence for all rotations for six commonly
    inspected terms. If result contains an output_dir entry, the figure is saved
    there as a PNG.
    """
    import matplotlib.pyplot as plt

    overview_terms = [
        "linear_dichroism_magnitude",
        "dichroism_axis_angle_deg",
        "linear_birefringence_magnitude",
        "birefringence_axis_angle_deg",
        "circular_dichroism",
        "circular_birefringence",
    ]
    terms = result["terms"]
    missing = [term for term in overview_terms if term not in terms]
    if missing:
        raise KeyError(f"Missing terms required for overview: {missing}")

    energy = np.asarray(result["energy_eV"])
    rotations = np.asarray(result["rotations_deg"])
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True)

    for ax, term_name in zip(axes.flat, overview_terms):
        values = np.asarray(terms[term_name])
        for idx, rotation in enumerate(rotations):
            real_values, imag_values = _coerce_for_plot(values[idx])
            ax.plot(energy, real_values, label=f"{rotation:g} deg")
            if imag_values is not None:
                ax.plot(energy, imag_values, linestyle="--", label=f"{rotation:g} deg imag")

        label = _term_label(term_name)
        unit_label = _term_unit_label(result, term_name)
        ax.set_title(label)
        ax.set_ylabel(unit_label)
        ax.grid(True, alpha=0.3)

    for ax in axes[-1, :]:
        ax.set_xlabel("Photon energy (eV)")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4))
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    _save_figure_if_requested(fig, result, "decomposition_overview.png")
    return fig


def plot_split_transition_fit(fit_result: dict[str, Any], filename: str | None = None) -> Any:
    """Plot a split-transition fit and residuals."""
    import matplotlib.pyplot as plt

    energy = np.asarray(fit_result["energy_eV"], dtype=np.float64)
    values = np.asarray(fit_result["values"])
    fitted = np.asarray(fit_result["fitted_values"])
    residuals = np.asarray(fit_result["residuals"])
    params = fit_result["parameters"]

    fig, (ax, residual_ax) = plt.subplots(
        2,
        1,
        figsize=(8, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax.plot(energy, np.real(values), label="data")
    ax.plot(energy, np.real(fitted), label="split-transition fit")
    if np.iscomplexobj(values) and fit_result.get("component") == "complex":
        ax.plot(energy, np.imag(values), linestyle=":", label="data imag")
        ax.plot(energy, np.imag(fitted), linestyle="--", label="fit imag")

    residual_ax.axhline(0.0, color="0.3", linewidth=0.8)
    residual_ax.plot(energy, np.real(residuals), label="residual")
    if np.iscomplexobj(residuals) and fit_result.get("component") == "complex":
        residual_ax.plot(energy, np.imag(residuals), linestyle="--", label="residual imag")

    title_bits = [
        fit_result.get("term_prefix", "spectrum").replace("_", " "),
        f"delta = {params['delta_meV']:.1f} meV",
        f"E = {params['center_eV']:.4f} eV",
        f"Gamma = {1000.0 * params['broadening_eV']:.1f} meV",
    ]
    ax.set_title(", ".join(title_bits))
    ax.set_ylabel(fit_result.get("component", "real"))
    ax.legend()
    ax.grid(True, alpha=0.3)

    residual_ax.set_xlabel("Photon energy (eV)")
    residual_ax.set_ylabel("resid.")
    residual_ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output_dir = fit_result.get("output_dir")
    if output_dir:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        if filename is None:
            term_prefix = fit_result.get("term_prefix", "split_transition")
            window = fit_result.get("energy_window_eV", (None, None))
            if window[0] is None or window[1] is None:
                window_suffix = "full"
            else:
                window_suffix = f"{float(window[0]):.3f}_{float(window[1]):.3f}"
            filename = f"{_safe_filename(term_prefix)}_split_fit_{window_suffix}.png"
        fig.savefig(path / filename, bbox_inches="tight", dpi=150)

    return fig


def plot_feature_scan(
    result: dict[str, Any],
    feature_scan: dict[str, Any],
    filename: str | None = "feature_candidates.png",
) -> Any:
    """Plot collapsed anisotropy spectra with detected feature candidates."""
    import matplotlib.pyplot as plt

    settings = feature_scan.get("settings", {})
    term_prefixes = tuple(settings.get("term_prefixes", ("linear_dichroism",)))
    vector_part = settings.get("vector_part", "real")
    component = settings.get("component", "real")
    axis_offset_deg = float(settings.get("axis_offset_deg", 0.0))
    energy = np.asarray(result["energy_eV"], dtype=np.float64)
    features = list(feature_scan.get("features", []))

    available_terms = []
    for term_prefix in term_prefixes:
        if f"{term_prefix}_x" in result["terms"] and f"{term_prefix}_y" in result["terms"]:
            available_terms.append(term_prefix)
    if not available_terms:
        raise KeyError("No feature-scan term prefixes are available in result['terms'].")

    fig, axes = plt.subplots(
        len(available_terms),
        1,
        figsize=(9, max(3.5, 3.0 * len(available_terms))),
        sharex=True,
    )
    if len(available_terms) == 1:
        axes = [axes]

    for ax, term_prefix in zip(axes, available_terms):
        collapsed = collapse_twofold_anisotropy_spectrum(
            result["terms"][f"{term_prefix}_x"],
            result["terms"][f"{term_prefix}_y"],
            result["rotations_deg"],
            value_part=vector_part,
            axis_offset_deg=axis_offset_deg,
        )
        spectrum = _spectrum_component(collapsed["spectrum"], component)
        ax.plot(energy, spectrum, linewidth=1.0, label=component)
        term_features = [
            feature for feature in features if feature.get("term_prefix") == term_prefix
        ]
        for feature in term_features:
            e_v = float(feature["energy_eV"])
            y_value = float(feature["component_value"])
            marker = "v" if feature.get("kind") == "dip" else "^"
            ax.scatter(e_v, y_value, marker=marker, s=45, zorder=3)
            ax.annotate(
                f"{e_v:.3f}",
                xy=(e_v, y_value),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        label = term_prefix.replace("_", " ")
        ax.set_title(f"{label} collapsed {component} feature candidates")
        ax.set_ylabel(f"collapsed {component}")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Photon energy (eV)")
    fig.tight_layout()

    output_dir = result.get("output_dir")
    if output_dir and filename:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(path / filename, bbox_inches="tight", dpi=150)

    return fig


def _generator_from_terms(
    *,
    a: float = 0.0,
    linear_dichroism_x: float = 0.0,
    linear_dichroism_y: float = 0.0,
    circular_dichroism: float = 0.0,
    linear_birefringence_x: float = 0.0,
    linear_birefringence_y: float = 0.0,
    circular_birefringence: float = 0.0,
) -> np.ndarray:
    """Build a generator using the same convention as decompose_generator_terms."""
    L = np.zeros((4, 4), dtype=np.float64)
    L += a * np.eye(4)
    L[0, 1] = L[1, 0] = linear_dichroism_x
    L[0, 2] = L[2, 0] = linear_dichroism_y
    L[0, 3] = L[3, 0] = circular_dichroism
    L[2, 3] = linear_birefringence_x
    L[3, 2] = -linear_birefringence_x
    L[3, 1] = linear_birefringence_y
    L[1, 3] = -linear_birefringence_y
    L[1, 2] = circular_birefringence
    L[2, 1] = -circular_birefringence
    return L


def _demo_case(name: str, L: np.ndarray, expected_terms: dict[str, float]) -> None:
    M = expm(L)
    result = decompose_mueller_log(M, normalize=True)
    recovered = result["terms"]
    L_aniso_expected, _ = remove_isotropic_part(L)

    generator_error = np.linalg.norm(result["L_aniso"] - L_aniso_expected)
    term_errors = {
        key: abs(float(np.real(recovered[key])) - expected)
        for key, expected in expected_terms.items()
    }
    max_term_error = max(term_errors.values()) if term_errors else 0.0

    if generator_error > 1.0e-10 or max_term_error > 1.0e-10:
        raise AssertionError(
            f"{name} failed: generator_error={generator_error:.3e}, "
            f"max_term_error={max_term_error:.3e}, term_errors={term_errors}"
        )

    print(
        f"{name}: generator_error={generator_error:.3e}, "
        f"max_term_error={max_term_error:.3e}"
    )


def _run_synthetic_demo() -> None:
    print("Running synthetic logarithmic Mueller decomposition demo")

    _demo_case(
        "pure linear birefringence",
        _generator_from_terms(linear_birefringence_x=0.035),
        {"linear_birefringence_x": 0.035},
    )
    _demo_case(
        "pure linear dichroism",
        _generator_from_terms(linear_dichroism_x=0.025),
        {"linear_dichroism_x": 0.025},
    )
    _demo_case(
        "combined weak birefringence and dichroism",
        _generator_from_terms(
            linear_dichroism_x=0.015,
            linear_dichroism_y=-0.008,
            linear_birefringence_x=0.028,
            linear_birefringence_y=0.011,
        ),
        {
            "linear_dichroism_x": 0.015,
            "linear_dichroism_y": -0.008,
            "linear_birefringence_x": 0.028,
            "linear_birefringence_y": 0.011,
        },
    )

    energy = np.linspace(0.8, 5.8, 9)
    rotations = np.arange(0.0, 180.0, 45.0)
    mueller = np.empty((len(rotations), len(energy), 4, 4), dtype=np.float64)
    expected_ldx = np.empty((len(rotations), len(energy)), dtype=np.float64)
    expected_ldy = np.empty_like(expected_ldx)
    expected_lbx = np.empty_like(expected_ldx)
    expected_lby = np.empty_like(expected_ldx)

    for rotation_index, theta_deg in enumerate(rotations):
        theta = np.deg2rad(theta_deg)
        for energy_index, e_v in enumerate(energy):
            ld_magnitude = 0.010 + 0.002 * (e_v - energy.mean())
            lb_magnitude = 0.020 + 0.006 * np.exp(-((e_v - 3.0) / 1.5) ** 2)
            expected_ldx[rotation_index, energy_index] = ld_magnitude * np.cos(2.0 * theta)
            expected_ldy[rotation_index, energy_index] = ld_magnitude * np.sin(2.0 * theta)
            expected_lbx[rotation_index, energy_index] = lb_magnitude * np.cos(2.0 * theta + 0.3)
            expected_lby[rotation_index, energy_index] = lb_magnitude * np.sin(2.0 * theta + 0.3)

            L = _generator_from_terms(
                linear_dichroism_x=expected_ldx[rotation_index, energy_index],
                linear_dichroism_y=expected_ldy[rotation_index, energy_index],
                linear_birefringence_x=expected_lbx[rotation_index, energy_index],
                linear_birefringence_y=expected_lby[rotation_index, energy_index],
            )
            mueller[rotation_index, energy_index] = expm(L)

    dataset = decompose_dataset(mueller, energy, rotations)
    checks = {
        "linear_dichroism_x": (dataset["terms"]["linear_dichroism_x"], expected_ldx),
        "linear_dichroism_y": (dataset["terms"]["linear_dichroism_y"], expected_ldy),
        "linear_birefringence_x": (dataset["terms"]["linear_birefringence_x"], expected_lbx),
        "linear_birefringence_y": (dataset["terms"]["linear_birefringence_y"], expected_lby),
    }
    max_errors = {
        name: float(np.max(np.abs(np.real(recovered) - expected)))
        for name, (recovered, expected) in checks.items()
    }
    worst_error = max(max_errors.values())
    if worst_error > 1.0e-10:
        raise AssertionError(f"rotation-dependent anisotropy failed: {max_errors}")

    print(f"rotation-dependent anisotropy: max_term_error={worst_error:.3e}")

    split_energy = np.linspace(1.85, 2.15, 241)
    split_expected = {
        "center_eV": 2.000,
        "delta_eV": 0.060,
        "broadening_eV": 0.025,
        "amplitude_low": 0.040,
        "amplitude_high": -0.032,
        "phase_rad": 0.35,
        "offset": 0.002,
        "slope": -0.004,
    }
    split_values = split_transition_model(
        split_energy,
        **split_expected,
        exponent=-0.5,
        component="real",
    )
    split_fit = fit_split_transition_spectrum(
        split_energy,
        split_values,
        exponent=-0.5,
        component="real",
        initial={
            "center_eV": 2.005,
            "delta_eV": 0.050,
            "broadening_eV": 0.020,
        },
        max_delta_eV=0.15,
    )
    split_delta_error = abs(
        split_fit["parameters"]["delta_eV"] - split_expected["delta_eV"]
    )
    if split_delta_error > 5.0e-3:
        raise AssertionError(
            "split-transition fit failed: "
            f"delta_error={split_delta_error:.3e}, params={split_fit['parameters']}"
        )
    print(
        "split-transition synthetic fit: "
        f"delta={split_fit['parameters']['delta_meV']:.2f} meV, "
        f"delta_error={1000.0 * split_delta_error:.2f} meV"
    )
    direct_delta_eV = 0.065
    direct_center_eV = 2.000
    direct_low = direct_center_eV - 0.5 * direct_delta_eV
    direct_high = direct_center_eV + 0.5 * direct_delta_eV
    direct_values = (
        0.020 * np.tanh((split_energy - direct_low) / 0.006)
        - 0.018 * np.tanh((split_energy - direct_high) / 0.006)
        + 0.003 * (split_energy - direct_center_eV)
    )
    direct_fit = estimate_direct_derivative_split_spectrum(
        split_energy,
        direct_values,
        smooth_width_eV=0.012,
        max_delta_eV=0.12,
    )
    direct_delta_error = abs(direct_fit["splitting_eV"] - direct_delta_eV)
    if (not direct_fit["success"]) or direct_delta_error > 3.0e-3:
        raise AssertionError(
            "direct derivative split failed: "
            f"delta_error={direct_delta_error:.3e}, fit={direct_fit}"
        )
    print(
        "direct derivative synthetic split: "
        f"delta={direct_fit['splitting_meV']:.2f} meV, "
        f"delta_error={1000.0 * direct_delta_error:.2f} meV"
    )
    if dataset["diagnostics"]["warnings"]:
        print("diagnostic warnings:")
        for warning in dataset["diagnostics"]["warnings"]:
            print(f"  - {warning}")
    else:
        print("diagnostic warnings: none")


if __name__ == "__main__":
    _run_synthetic_demo()
