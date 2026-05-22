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
from scipy.linalg import expm, logm
from scipy.optimize import least_squares
from scipy.signal import find_peaks, peak_prominences, savgol_filter


_MATRIX_SHAPE = (4, 4)
_HC_EV_NM = 1239.8419843320026
_M00_ATOL = 1.0e-12
_CONDITION_WARNING_THRESHOLD = 1.0e12
_IMAG_ABS_WARNING_THRESHOLD = 1.0e-10
_IMAG_REL_WARNING_THRESHOLD = 1.0e-8
_RECONSTRUCTION_WARNING_THRESHOLD = 1.0e-8
_REAL_IF_CLOSE_ABS = 1.0e-12
_REAL_IF_CLOSE_REL = 1.0e-10
_SPLIT_TRANSITION_PARAMETER_NAMES = (
    "center_eV",
    "delta_eV",
    "broadening_eV",
    "amplitude_low",
    "amplitude_high",
    "phase_rad",
    "offset",
    "slope",
)
_DEFAULT_FEATURE_BASELINE_WIDTHS_EV = (0.08, 0.12, 0.20, 0.35, 0.60, 0.90)


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


def _as_numeric_array(values: Any) -> np.ndarray:
    """Return values as float64 or complex128 without discarding complex input."""
    arr = np.asarray(values)
    if np.iscomplexobj(arr):
        return arr.astype(np.complex128, copy=False)
    return arr.astype(np.float64, copy=False)


def _validate_mueller_shape(M: np.ndarray, name: str = "M") -> None:
    if M.ndim < 2 or M.shape[-2:] != _MATRIX_SHAPE:
        raise ValueError(
            f"{name} must have shape (4, 4) or (..., 4, 4); got {M.shape}."
        )


def _is_single_matrix(M: np.ndarray) -> bool:
    return M.ndim == 2


def _flatten_matrices(M: np.ndarray) -> tuple[np.ndarray, tuple[int, ...], bool]:
    single = _is_single_matrix(M)
    batch_shape = () if single else M.shape[:-2]
    return M.reshape((-1, 4, 4)), batch_shape, single


def _restore_batch(values: np.ndarray, batch_shape: tuple[int, ...], single: bool) -> Any:
    arr = np.asarray(values).reshape(batch_shape)
    if single:
        return arr[()]
    return arr


def _mask_count(mask: np.ndarray) -> int:
    return int(np.count_nonzero(np.asarray(mask)))


def _sample_indices(mask: np.ndarray, max_items: int = 8) -> str:
    mask_arr = np.asarray(mask)
    if mask_arr.shape == ():
        return "[()]" if bool(mask_arr) else "[]"
    indices = np.argwhere(mask_arr)
    shown = [tuple(int(v) for v in row) for row in indices[:max_items]]
    suffix = ", ..." if len(indices) > max_items else ""
    return f"{shown}{suffix}"


def _add_mask_warning(
    warnings_out: list[str],
    mask: np.ndarray,
    message: str,
    total_count: int,
) -> None:
    count = _mask_count(mask)
    if count:
        warnings_out.append(
            f"{message}: {count}/{total_count} matrix/matrices; "
            f"indices {_sample_indices(mask)}."
        )


def _matrix_norms(flat_matrices: np.ndarray) -> np.ndarray:
    return np.array([np.linalg.norm(matrix) for matrix in flat_matrices], dtype=np.float64)


def _real_if_close_per_matrix(
    matrices: np.ndarray,
    abs_tol: float = _REAL_IF_CLOSE_ABS,
    rel_tol: float = _REAL_IF_CLOSE_REL,
) -> np.ndarray:
    """Drop imaginary parts only for matrices whose imaginary norm is negligible."""
    arr = np.asarray(matrices)
    if not np.iscomplexobj(arr):
        return arr

    flat, batch_shape, single = _flatten_matrices(arr)
    out = flat.astype(np.complex128, copy=True)
    close_mask = np.zeros(flat.shape[0], dtype=bool)

    for index, matrix in enumerate(flat):
        imag_norm = np.linalg.norm(np.imag(matrix))
        real_norm = np.linalg.norm(np.real(matrix))
        close_mask[index] = imag_norm <= abs_tol + rel_tol * max(1.0, real_norm)
        if close_mask[index]:
            out[index] = np.real(matrix)

    reshaped = out.reshape((1, 4, 4) if single else batch_shape + (4, 4))
    if bool(np.all(close_mask)):
        return np.real(reshaped[0] if single else reshaped)
    return reshaped[0] if single else reshaped


def _angle_from_components(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return 0.5 atan2(y, x) in degrees, or NaN for significant complex input."""
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    angle = 0.5 * np.degrees(np.arctan2(np.real(y_arr), np.real(x_arr)))

    if np.iscomplexobj(x_arr) or np.iscomplexobj(y_arr):
        scale = np.maximum.reduce(
            [
                np.ones_like(np.real(x_arr), dtype=np.float64),
                np.abs(np.real(x_arr)),
                np.abs(np.real(y_arr)),
            ]
        )
        significant_imag = (
            np.abs(np.imag(x_arr)) > _REAL_IF_CLOSE_ABS + _REAL_IF_CLOSE_REL * scale
        ) | (
            np.abs(np.imag(y_arr)) > _REAL_IF_CLOSE_ABS + _REAL_IF_CLOSE_REL * scale
        )
        angle = np.where(significant_imag, np.nan, angle)

    return angle


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


def normalize_mueller(M: np.ndarray) -> np.ndarray:
    """Normalize Mueller matrices by their M[0, 0] element.

    Parameters
    ----------
    M:
        A single Mueller matrix with shape (4, 4), or a batch with shape
        (..., 4, 4).

    Returns
    -------
    numpy.ndarray
        M / M[0, 0] for every matrix in the batch.

    Notes
    -----
    For transmission data, M[0, 0] carries the absolute transmission scale. A
    scalar scale factor in M appears as an identity-matrix contribution in
    log(M), so this normalization removes absolute transmission before
    extracting anisotropic polarization effects. If M[0, 0] is close to zero,
    the caller should inspect diagnostics from decompose_mueller_log.
    """
    arr = _as_numeric_array(M)
    _validate_mueller_shape(arr)
    denominator = arr[..., 0, 0][..., np.newaxis, np.newaxis]
    with np.errstate(divide="ignore", invalid="ignore"):
        return arr / denominator


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


def _matrix_log_batch_with_status(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = _as_numeric_array(M)
    _validate_mueller_shape(arr)
    flat, batch_shape, single = _flatten_matrices(arr)

    logs = np.empty((flat.shape[0], 4, 4), dtype=np.complex128)
    failures = np.zeros(flat.shape[0], dtype=bool)

    for index, matrix in enumerate(flat):
        if not np.all(np.isfinite(matrix)):
            logs[index] = np.nan + 0.0j
            failures[index] = True
            continue
        try:
            logged = logm(matrix)
        except Exception:
            logs[index] = np.nan + 0.0j
            failures[index] = True
            continue

        logs[index] = logged
        if not np.all(np.isfinite(logs[index])):
            failures[index] = True

    if single:
        return logs[0], failures.reshape(())
    return logs.reshape(batch_shape + (4, 4)), failures.reshape(batch_shape)


def matrix_log_batch(M: np.ndarray) -> np.ndarray:
    """Compute scipy.linalg.logm for one Mueller matrix or a batch.

    Parameters
    ----------
    M:
        Shape (4, 4) or (..., 4, 4).

    Returns
    -------
    numpy.ndarray
        Matrix logarithms with the same leading batch shape as M. Complex
        values are retained because a significant imaginary part can indicate a
        branch, noise, or physical-consistency issue that should be diagnosed.
    """
    logged, _ = _matrix_log_batch_with_status(M)
    return logged


def reconstruction_error(M: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Return ||expm(L) - M|| / ||M|| for one matrix or a batch.

    The reconstruction error checks whether the computed logarithm is a useful
    generator of the input matrix. For differential matrices m = log(M) / d,
    pass the integrated generator L = m d to this function.
    """
    M_arr = _as_numeric_array(M)
    L_arr = _as_numeric_array(L)
    _validate_mueller_shape(M_arr, "M")
    _validate_mueller_shape(L_arr, "L")
    if M_arr.shape != L_arr.shape:
        raise ValueError(f"M and L must have the same shape; got {M_arr.shape} and {L_arr.shape}.")

    flat_M, batch_shape, single = _flatten_matrices(M_arr)
    flat_L, _, _ = _flatten_matrices(L_arr)
    errors = np.empty(flat_M.shape[0], dtype=np.float64)

    for index, (matrix, generator) in enumerate(zip(flat_M, flat_L)):
        if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(generator)):
            errors[index] = np.nan
            continue
        try:
            reconstructed = expm(generator)
        except Exception:
            errors[index] = np.nan
            continue

        denominator = np.linalg.norm(matrix)
        numerator = np.linalg.norm(reconstructed - matrix)
        if denominator == 0.0:
            errors[index] = 0.0 if numerator == 0.0 else np.inf
        else:
            errors[index] = float(numerator / denominator)

    return _restore_batch(errors, batch_shape, single)


def remove_isotropic_part(L: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove the scalar isotropic part trace(L) / 4 from a generator.

    Parameters
    ----------
    L:
        Integrated generator L or differential generator m with shape
        (4, 4) or (..., 4, 4).

    Returns
    -------
    tuple
        (L_aniso, isotropic_attenuation), where

            L_aniso = L - trace(L) / 4 * I.

    Notes
    -----
    The trace term represents scalar isotropic attenuation in this effective
    generator convention. Subtracting it leaves the anisotropic generator terms
    used for dichroism and retardance interpretation.
    """
    arr = _as_numeric_array(L)
    _validate_mueller_shape(arr, "L")
    isotropic = np.trace(arr, axis1=-2, axis2=-1) / 4.0
    eye = np.eye(4, dtype=arr.dtype)
    anisotropic = arr - np.asarray(isotropic)[..., np.newaxis, np.newaxis] * eye
    return anisotropic, isotropic


def decompose_generator_terms(L: np.ndarray) -> dict[str, np.ndarray]:
    """Extract approximate differential Mueller generator terms.

    The extraction uses the convention

        L =
        [[ a,  LD_x,  LD_y,  CD  ],
         [ LD_x, a,   CB,   -LB_y],
         [ LD_y, -CB, a,    LB_x ],
         [ CD,   LB_y,-LB_x, a    ]]

    where LD_x and LD_y are linear dichroism components, CD is circular
    dichroism, LB_x and LB_y are linear birefringence / linear retardance
    components, CB is circular birefringence / optical rotation, and a is
    isotropic attenuation.

    This is a practical extraction convention for an effective differential
    generator. Signs, axis definitions, Stokes-vector ordering, and handedness
    must be checked against the exact Mueller convention used by the RC2 export
    pipeline before assigning final physical signs.
    """
    arr = _as_numeric_array(L)
    _validate_mueller_shape(arr, "L")

    isotropic_attenuation = np.trace(arr, axis1=-2, axis2=-1) / 4.0
    linear_dichroism_x = 0.5 * (arr[..., 0, 1] + arr[..., 1, 0])
    linear_dichroism_y = 0.5 * (arr[..., 0, 2] + arr[..., 2, 0])
    circular_dichroism = 0.5 * (arr[..., 0, 3] + arr[..., 3, 0])
    linear_birefringence_x = 0.5 * (arr[..., 2, 3] - arr[..., 3, 2])
    linear_birefringence_y = 0.5 * (arr[..., 3, 1] - arr[..., 1, 3])
    circular_birefringence = 0.5 * (arr[..., 1, 2] - arr[..., 2, 1])

    linear_dichroism_magnitude = np.sqrt(linear_dichroism_x**2 + linear_dichroism_y**2)
    linear_birefringence_magnitude = np.sqrt(
        linear_birefringence_x**2 + linear_birefringence_y**2
    )
    dichroism_axis_angle_deg = _angle_from_components(
        linear_dichroism_x, linear_dichroism_y
    )
    birefringence_axis_angle_deg = _angle_from_components(
        linear_birefringence_x, linear_birefringence_y
    )

    return {
        "isotropic_attenuation": isotropic_attenuation,
        "linear_dichroism_x": linear_dichroism_x,
        "linear_dichroism_y": linear_dichroism_y,
        "circular_dichroism": circular_dichroism,
        "linear_birefringence_x": linear_birefringence_x,
        "linear_birefringence_y": linear_birefringence_y,
        "circular_birefringence": circular_birefringence,
        "linear_dichroism_magnitude": linear_dichroism_magnitude,
        "linear_birefringence_magnitude": linear_birefringence_magnitude,
        "dichroism_axis_angle_deg": dichroism_axis_angle_deg,
        "birefringence_axis_angle_deg": birefringence_axis_angle_deg,
    }


def _condition_numbers(M: np.ndarray) -> np.ndarray:
    flat, batch_shape, single = _flatten_matrices(M)
    values = np.empty(flat.shape[0], dtype=np.float64)
    for index, matrix in enumerate(flat):
        if not np.all(np.isfinite(matrix)):
            values[index] = np.inf
            continue
        try:
            values[index] = float(np.linalg.cond(matrix))
        except Exception:
            values[index] = np.inf
    return np.asarray(_restore_batch(values, batch_shape, single))


def _generator_imaginary_diagnostics(L: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat, batch_shape, single = _flatten_matrices(np.asarray(L))
    imag_norm = _matrix_norms(np.imag(flat))
    total_norm = _matrix_norms(flat)
    relative = imag_norm / np.maximum(1.0, total_norm)
    return (
        np.asarray(_restore_batch(imag_norm, batch_shape, single)),
        np.asarray(_restore_batch(relative, batch_shape, single)),
    )


def _any_nonfinite_by_matrix(M: np.ndarray) -> np.ndarray:
    flat, batch_shape, single = _flatten_matrices(M)
    mask = np.array([not np.all(np.isfinite(matrix)) for matrix in flat], dtype=bool)
    return np.asarray(_restore_batch(mask, batch_shape, single))


def decompose_mueller_log(
    M: np.ndarray,
    thickness: float | None = None,
    normalize: bool = True,
    remove_isotropic_attenuation: bool = True,
    real_if_close: bool = True,
    check_physical: bool = True,
) -> dict[str, Any]:
    """Logarithmically decompose one Mueller matrix or a batch.

    Parameters
    ----------
    M:
        A single measured transmission Mueller matrix with shape (4, 4), or a
        stack with shape (..., 4, 4).
    thickness:
        Optional sample thickness. If omitted, this returns the integrated
        generator L = log(M). If provided, this returns the effective
        differential generator m = log(M) / thickness. The units of m are the
        inverse of the units used for thickness.
    normalize:
        If True, divide every Mueller matrix by M[0, 0] before taking the
        logarithm. This removes absolute transmission. The removed scalar scale
        appears as an isotropic identity contribution in the generator.
    remove_isotropic_attenuation:
        If True, also return an anisotropic generator with trace(L) / 4 removed.
    real_if_close:
        If True, tiny numerical imaginary parts from scipy.linalg.logm are
        converted to real values. Significant imaginary parts are retained and
        reported in diagnostics.
    check_physical:
        If True, populate diagnostic warnings for near-zero M[0, 0],
        non-finite entries, large condition number, significant imaginary log
        components, and poor expm(logm(M)) reconstruction.

    Returns
    -------
    dict
        Dictionary containing generator_full, generator_aniso, L_full, L_aniso,
        isotropic_attenuation, terms, diagnostics, normalized_mueller,
        thickness, and is_differential.

    Notes
    -----
    This is an effective homogeneous-slab decomposition. For layered,
    depolarizing, strongly scattering, or multiple-reflection dominated samples,
    the result is an effective integrated generator of the measured Mueller
    matrix, not necessarily a unique local material tensor.
    """
    M_input = _as_numeric_array(M)
    _validate_mueller_shape(M_input)
    flat_input, batch_shape, single = _flatten_matrices(M_input)
    total_count = flat_input.shape[0]

    if thickness is not None:
        thickness_value = float(thickness)
        if not np.isfinite(thickness_value) or thickness_value <= 0.0:
            raise ValueError("thickness must be a finite positive scalar when provided.")
    else:
        thickness_value = None

    m00 = M_input[..., 0, 0]
    m00_close = np.abs(m00) <= _M00_ATOL
    nonfinite_input = _any_nonfinite_by_matrix(M_input)

    M_for_log = normalize_mueller(M_input) if normalize else M_input.copy()
    nonfinite_normalized = _any_nonfinite_by_matrix(M_for_log)
    condition_number = _condition_numbers(M_for_log)

    log_M, logm_failures = _matrix_log_batch_with_status(M_for_log)
    reconstruction = np.asarray(reconstruction_error(M_for_log, log_M))

    generator_full = log_M / thickness_value if thickness_value is not None else log_M
    imag_norm, imag_relative_norm = _generator_imaginary_diagnostics(generator_full)

    high_imaginary = (
        (imag_norm > _IMAG_ABS_WARNING_THRESHOLD)
        & (imag_relative_norm > _IMAG_REL_WARNING_THRESHOLD)
    )
    large_condition = condition_number > _CONDITION_WARNING_THRESHOLD
    large_reconstruction_error = reconstruction > _RECONSTRUCTION_WARNING_THRESHOLD

    if real_if_close:
        generator_full = _real_if_close_per_matrix(generator_full)

    if remove_isotropic_attenuation:
        generator_aniso, isotropic_attenuation = remove_isotropic_part(generator_full)
    else:
        generator_aniso = np.array(generator_full, copy=True)
        isotropic_attenuation = np.trace(generator_full, axis1=-2, axis2=-1) / 4.0

    diagnostics_warnings: list[str] = []
    if check_physical:
        _add_mask_warning(
            diagnostics_warnings,
            np.asarray(m00_close),
            "M[0, 0] is close to zero",
            total_count,
        )
        _add_mask_warning(
            diagnostics_warnings,
            np.asarray(nonfinite_input),
            "Input Mueller matrix contains non-finite values",
            total_count,
        )
        _add_mask_warning(
            diagnostics_warnings,
            np.asarray(nonfinite_normalized),
            "Normalized Mueller matrix contains non-finite values",
            total_count,
        )
        _add_mask_warning(
            diagnostics_warnings,
            np.asarray(large_condition),
            f"Condition number exceeds {_CONDITION_WARNING_THRESHOLD:.1e}",
            total_count,
        )
        _add_mask_warning(
            diagnostics_warnings,
            np.asarray(high_imaginary),
            "Matrix logarithm has significant imaginary component",
            total_count,
        )
        _add_mask_warning(
            diagnostics_warnings,
            np.asarray(large_reconstruction_error),
            f"expm(logm(M)) reconstruction error exceeds {_RECONSTRUCTION_WARNING_THRESHOLD:.1e}",
            total_count,
        )
        _add_mask_warning(
            diagnostics_warnings,
            np.asarray(logm_failures),
            "Matrix logarithm failed or returned non-finite values",
            total_count,
        )

    diagnostics = {
        "warnings": diagnostics_warnings,
        "input_shape": M_input.shape,
        "batch_shape": batch_shape,
        "normalized": normalize,
        "thickness": thickness_value,
        "m00": m00,
        "m00_close_to_zero": m00_close,
        "nonfinite_input": nonfinite_input,
        "nonfinite_normalized": nonfinite_normalized,
        "condition_number": condition_number,
        "large_condition_number": large_condition,
        "logm_failures": logm_failures,
        "logm_imag_norm": imag_norm,
        "logm_imag_relative_norm": imag_relative_norm,
        "high_imaginary_part": high_imaginary,
        "reconstruction_error": reconstruction,
        "large_reconstruction_error": large_reconstruction_error,
        "thresholds": {
            "m00_atol": _M00_ATOL,
            "condition_number": _CONDITION_WARNING_THRESHOLD,
            "imag_abs": _IMAG_ABS_WARNING_THRESHOLD,
            "imag_relative": _IMAG_REL_WARNING_THRESHOLD,
            "reconstruction_error": _RECONSTRUCTION_WARNING_THRESHOLD,
        },
    }

    return {
        "generator_full": generator_full,
        "generator_aniso": generator_aniso,
        "L_full": generator_full,
        "L_aniso": generator_aniso,
        "isotropic_attenuation": isotropic_attenuation,
        "terms": decompose_generator_terms(generator_full),
        "diagnostics": diagnostics,
        "normalized_mueller": M_for_log,
        "thickness": thickness_value,
        "is_differential": thickness_value is not None,
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


def _spectrum_component(values: np.ndarray, component: str) -> np.ndarray:
    arr = np.asarray(values)
    if component == "real":
        return np.real(arr)
    if component == "imag":
        return np.imag(arr)
    if component == "abs":
        return np.abs(arr)
    if component == "complex":
        return arr.astype(np.complex128, copy=False)
    raise ValueError("component must be one of: 'real', 'imag', 'abs', 'complex'.")


def _stack_residuals(residual: np.ndarray) -> np.ndarray:
    arr = np.asarray(residual)
    if np.iscomplexobj(arr):
        return np.concatenate([np.real(arr), np.imag(arr)])
    return arr.astype(np.float64, copy=False)


def _mad_scale(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1.0
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    if mad > 0.0:
        return 1.4826 * mad
    std = float(np.std(arr))
    if std > 0.0:
        return std
    return max(1.0, abs(median))


def _energy_step_eV(energy: np.ndarray) -> float:
    sorted_energy = np.sort(np.asarray(energy, dtype=np.float64))
    diffs = np.diff(sorted_energy)
    diffs = np.abs(diffs[np.isfinite(diffs) & (np.abs(diffs) > 0.0)])
    if diffs.size == 0:
        return 1.0
    return float(np.nanmedian(diffs))


def _odd_window_points(
    energy: np.ndarray,
    width_eV: float,
    *,
    minimum: int = 5,
) -> int:
    step = _energy_step_eV(energy)
    if not np.isfinite(step) or step <= 0.0:
        step = 1.0
    points = max(int(minimum), int(round(abs(float(width_eV)) / step)))
    if points % 2 == 0:
        points += 1
    return points


def _running_nanmedian(values: np.ndarray, window_points: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    output = np.empty_like(arr)
    half_width = max(0, int(window_points) // 2)
    for index in range(arr.size):
        start = max(0, index - half_width)
        stop = min(arr.size, index + half_width + 1)
        output[index] = np.nanmedian(arr[start:stop])
    return output


def _running_mad_scale(values: np.ndarray, window_points: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    output = np.empty_like(arr)
    half_width = max(0, int(window_points) // 2)
    for index in range(arr.size):
        start = max(0, index - half_width)
        stop = min(arr.size, index + half_width + 1)
        window = arr[start:stop]
        median = float(np.nanmedian(window))
        mad = float(np.nanmedian(np.abs(window - median)))
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale <= 0.0:
            scale = float(np.nanstd(window))
        output[index] = scale if np.isfinite(scale) else np.nan
    return output


def _critical_point_complex_profile(
    energy_eV: np.ndarray,
    transition_energy_eV: float,
    broadening_eV: float,
    exponent: float,
) -> np.ndarray:
    energy = np.asarray(energy_eV, dtype=np.float64)
    gamma = float(broadening_eV)
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("broadening_eV must be finite and positive.")

    z = energy - float(transition_energy_eV) + 1j * gamma
    if abs(float(exponent)) <= 1.0e-14:
        profile = np.log(z)
    else:
        profile = z ** float(exponent)

    scale = float(np.nanmax(np.abs(profile))) if profile.size else 1.0
    if scale > 0.0 and np.isfinite(scale):
        profile = profile / scale
    return profile


def critical_point_profile(
    energy_eV: np.ndarray,
    transition_energy_eV: float,
    broadening_eV: float,
    exponent: float = -0.5,
    phase_rad: float = 0.0,
    component: str = "real",
) -> np.ndarray:
    """Return a normalized complex critical-point oscillator profile.

    ``exponent=-0.5`` is the common 3D M0 Aspnes critical-point shape;
    ``exponent=0`` uses a logarithmic profile. The returned profile is
    normalized over the supplied energy axis so fitted amplitudes are on the
    same scale as the input spectrum.
    """
    profile = np.exp(1j * float(phase_rad)) * _critical_point_complex_profile(
        energy_eV,
        transition_energy_eV,
        broadening_eV,
        exponent,
    )
    return _spectrum_component(profile, component)


def split_transition_model(
    energy_eV: np.ndarray,
    center_eV: float,
    delta_eV: float,
    broadening_eV: float,
    amplitude_low: float,
    amplitude_high: float,
    phase_rad: float = 0.0,
    offset: float = 0.0,
    slope: float = 0.0,
    exponent: float = -0.5,
    component: str = "real",
) -> np.ndarray:
    """Model a spectrum as two polarization-dependent critical points.

    The two transitions are centered at ``center_eV - delta_eV / 2`` and
    ``center_eV + delta_eV / 2``. For a linear-dichroism spectrum, the two
    amplitudes naturally represent the response of the two orthogonal
    polarization channels in the anisotropic difference signal.
    """
    energy = np.asarray(energy_eV, dtype=np.float64)
    center = float(center_eV)
    delta = abs(float(delta_eV))
    low_energy = center - 0.5 * delta
    high_energy = center + 0.5 * delta

    low_profile = _critical_point_complex_profile(
        energy, low_energy, broadening_eV, exponent
    )
    high_profile = _critical_point_complex_profile(
        energy, high_energy, broadening_eV, exponent
    )
    model = np.exp(1j * float(phase_rad)) * (
        float(amplitude_low) * low_profile + float(amplitude_high) * high_profile
    )

    energy_ref = float(np.nanmean(energy)) if energy.size else 0.0
    model = model + float(offset) + float(slope) * (energy - energy_ref)
    return _spectrum_component(model, component)


def _split_transition_initial_guess(
    energy: np.ndarray,
    values: np.ndarray,
    max_delta_eV: float | None,
) -> dict[str, float]:
    y = np.asarray(values)
    y_real = np.real(y) if np.iscomplexobj(y) else y.astype(np.float64, copy=False)
    finite = np.isfinite(energy) & np.isfinite(y_real)
    e = energy[finite]
    yr = y_real[finite]

    if e.size < 8:
        raise ValueError("At least 8 finite data points are required for a split fit.")

    span = float(e[-1] - e[0])
    step = float(np.nanmedian(np.diff(e))) if e.size > 1 else span
    baseline = float(np.nanmedian(yr))
    detrended = yr - baseline
    scale = float(np.nanpercentile(np.abs(detrended), 90))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = max(float(np.nanstd(yr)), 1.0)

    peak_index = int(np.nanargmax(np.abs(detrended)))
    delta_guess = min(0.040, max(0.010, 0.15 * span))
    if max_delta_eV is not None:
        delta_guess = min(delta_guess, 0.5 * float(max_delta_eV))
    broadening_guess = max(0.010, 3.0 * abs(step))

    return {
        "center_eV": float(e[peak_index]),
        "delta_eV": delta_guess,
        "broadening_eV": broadening_guess,
        "amplitude_low": scale,
        "amplitude_high": -scale,
        "phase_rad": 0.0,
        "offset": baseline,
        "slope": 0.0,
    }


def _split_transition_bounds(
    energy: np.ndarray,
    values: np.ndarray,
    max_delta_eV: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    e_min = float(np.nanmin(energy))
    e_max = float(np.nanmax(energy))
    span = max(e_max - e_min, 1.0e-6)
    step = float(np.nanmedian(np.diff(energy))) if energy.size > 1 else span
    y = np.asarray(values)
    y_scale = float(np.nanpercentile(np.abs(_stack_residuals(y)), 95))
    if not np.isfinite(y_scale) or y_scale <= 0.0:
        y_scale = 1.0

    delta_upper = 0.25 * span if max_delta_eV is None else float(max_delta_eV)
    delta_upper = max(delta_upper, max(0.010, 2.0 * abs(step)))
    broadening_lower = max(1.0e-5, 0.25 * abs(step))
    broadening_upper = max(0.5 * span, 10.0 * broadening_lower)
    amplitude_bound = 20.0 * y_scale
    offset_bound = 20.0 * y_scale
    slope_bound = 20.0 * y_scale / span

    lower = np.array(
        [
            e_min,
            0.0,
            broadening_lower,
            -amplitude_bound,
            -amplitude_bound,
            -np.pi,
            -offset_bound,
            -slope_bound,
        ],
        dtype=np.float64,
    )
    upper = np.array(
        [
            e_max,
            delta_upper,
            broadening_upper,
            amplitude_bound,
            amplitude_bound,
            np.pi,
            offset_bound,
            slope_bound,
        ],
        dtype=np.float64,
    )
    return lower, upper


def fit_split_transition_spectrum(
    energy_eV: np.ndarray,
    values: np.ndarray,
    energy_min: float | None = None,
    energy_max: float | None = None,
    exponent: float = -0.5,
    component: str = "real",
    initial: dict[str, float] | None = None,
    max_delta_eV: float | None = 0.30,
    robust_scale: float | None = None,
    loss: str = "soft_l1",
) -> dict[str, Any]:
    """Fit one anisotropic spectrum with the split-transition model.

    The fitted ``delta_eV`` is the transition separation. Use
    ``delta_meV = 1000 * delta_eV`` from the returned ``parameters`` dict for
    the usual meV scale check.
    """
    energy = np.asarray(energy_eV, dtype=np.float64)
    raw_values = np.asarray(values)
    if energy.ndim != 1:
        raise ValueError(f"energy_eV must be one-dimensional; got {energy.shape}.")
    if raw_values.shape != energy.shape:
        raise ValueError(
            f"values must have shape {energy.shape}; got {raw_values.shape}."
        )

    target = _spectrum_component(raw_values, component)
    finite = np.isfinite(energy) & np.isfinite(_stack_residuals(target)[: energy.size])
    if np.iscomplexobj(target):
        finite = np.isfinite(energy) & np.isfinite(np.real(target)) & np.isfinite(np.imag(target))
    if energy_min is not None:
        finite &= energy >= float(energy_min)
    if energy_max is not None:
        finite &= energy <= float(energy_max)

    fit_energy = energy[finite]
    fit_values = target[finite]
    if fit_energy.size < len(_SPLIT_TRANSITION_PARAMETER_NAMES) + 2:
        raise ValueError(
            "Not enough finite points in the requested fit window for the "
            "eight-parameter split-transition fit."
        )

    order = np.argsort(fit_energy)
    fit_energy = fit_energy[order]
    fit_values = fit_values[order]

    guess = _split_transition_initial_guess(fit_energy, fit_values, max_delta_eV)
    if initial is not None:
        unknown = set(initial) - set(_SPLIT_TRANSITION_PARAMETER_NAMES)
        if unknown:
            raise ValueError(f"Unknown initial parameter(s): {sorted(unknown)}")
        guess.update({key: float(value) for key, value in initial.items()})

    lower, upper = _split_transition_bounds(fit_energy, fit_values, max_delta_eV)
    p0 = np.array([guess[name] for name in _SPLIT_TRANSITION_PARAMETER_NAMES], dtype=np.float64)
    p0 = np.clip(p0, lower + 1.0e-12, upper - 1.0e-12)

    if robust_scale is None:
        residual_seed = _stack_residuals(fit_values)
        robust_scale = _mad_scale(residual_seed - np.nanmedian(residual_seed))
    robust_scale = max(float(robust_scale), 1.0e-12)

    def residuals(params: np.ndarray) -> np.ndarray:
        model = split_transition_model(
            fit_energy,
            *params,
            exponent=exponent,
            component=component,
        )
        return _stack_residuals(model - fit_values)

    opt = least_squares(
        residuals,
        p0,
        bounds=(lower, upper),
        loss=loss,
        f_scale=robust_scale,
        max_nfev=5000,
    )

    parameters = {
        name: float(value)
        for name, value in zip(_SPLIT_TRANSITION_PARAMETER_NAMES, opt.x)
    }
    parameters["delta_meV"] = 1000.0 * parameters["delta_eV"]
    parameters["lower_transition_eV"] = (
        parameters["center_eV"] - 0.5 * parameters["delta_eV"]
    )
    parameters["upper_transition_eV"] = (
        parameters["center_eV"] + 0.5 * parameters["delta_eV"]
    )

    fitted_values = split_transition_model(
        fit_energy,
        **{name: parameters[name] for name in _SPLIT_TRANSITION_PARAMETER_NAMES},
        exponent=exponent,
        component=component,
    )
    residual = fitted_values - fit_values
    residual_vector = _stack_residuals(residual)
    rmse = float(np.sqrt(np.mean(residual_vector**2)))
    mae = float(np.mean(np.abs(residual_vector)))

    covariance = None
    if opt.jac.shape[0] > opt.jac.shape[1]:
        try:
            _, singular_values, vt = np.linalg.svd(opt.jac, full_matrices=False)
            threshold = np.finfo(float).eps * max(opt.jac.shape) * singular_values[0]
            keep = singular_values > threshold
            if np.any(keep):
                cov = (vt[keep].T / singular_values[keep] ** 2) @ vt[keep]
                cov *= 2.0 * opt.cost / max(1, opt.jac.shape[0] - opt.jac.shape[1])
                covariance = cov
        except Exception:
            covariance = None

    return {
        "success": bool(opt.success),
        "message": opt.message,
        "parameters": parameters,
        "parameter_names": _SPLIT_TRANSITION_PARAMETER_NAMES,
        "initial_parameters": guess,
        "energy_eV": fit_energy,
        "values": fit_values,
        "fitted_values": fitted_values,
        "residuals": residual,
        "rmse": rmse,
        "mae": mae,
        "n_points": int(fit_energy.size),
        "component": component,
        "exponent": float(exponent),
        "loss": loss,
        "robust_scale": robust_scale,
        "optimizer": opt,
        "covariance": covariance,
    }


def detect_spectral_features(
    energy_eV: np.ndarray,
    values: np.ndarray,
    *,
    scatter: np.ndarray | None = None,
    energy_min: float | None = None,
    energy_max: float | None = None,
    component: str = "real",
    baseline_widths_eV: tuple[float, ...] = _DEFAULT_FEATURE_BASELINE_WIDTHS_EV,
    smooth_width_eV: float = 0.025,
    min_z: float = 3.0,
    min_prominence_z: float = 2.0,
    min_separation_eV: float = 0.04,
    max_features: int = 12,
) -> dict[str, Any]:
    """Detect candidate spectral features in a one-dimensional spectrum.

    The detector subtracts rolling-median baselines at several energy scales,
    smooths the residual, and reports local peaks/dips whose amplitude and
    prominence are large compared with the local median-absolute-deviation
    scale. Optional ``scatter`` values are used only for ranking and reporting,
    so features are not discarded solely because rotation scatter is large.
    """
    if component == "complex":
        raise ValueError("Feature detection needs a scalar component: real, imag, or abs.")

    energy = np.asarray(energy_eV, dtype=np.float64)
    raw_values = np.asarray(values)
    if energy.ndim != 1:
        raise ValueError(f"energy_eV must be one-dimensional; got {energy.shape}.")
    if raw_values.shape != energy.shape:
        raise ValueError(f"values must have shape {energy.shape}; got {raw_values.shape}.")

    y = np.asarray(_spectrum_component(raw_values, component), dtype=np.float64)
    scatter_abs = None
    if scatter is not None:
        scatter_arr = np.asarray(scatter)
        if scatter_arr.shape != energy.shape:
            raise ValueError(f"scatter must have shape {energy.shape}; got {scatter_arr.shape}.")
        scatter_abs = np.abs(scatter_arr).astype(np.float64, copy=False)

    finite = np.isfinite(energy) & np.isfinite(y)
    if energy_min is not None:
        finite &= energy >= float(energy_min)
    if energy_max is not None:
        finite &= energy <= float(energy_max)

    scan_energy = energy[finite]
    scan_values = y[finite]
    scan_scatter = None if scatter_abs is None else scatter_abs[finite]
    if scan_energy.size < 7:
        return {
            "settings": {
                "component": component,
                "energy_window_eV": (energy_min, energy_max),
                "baseline_widths_eV": tuple(float(width) for width in baseline_widths_eV),
                "smooth_width_eV": float(smooth_width_eV),
                "min_z": float(min_z),
                "min_prominence_z": float(min_prominence_z),
                "min_separation_eV": float(min_separation_eV),
                "max_features": int(max_features),
            },
            "n_points": int(scan_energy.size),
            "features": [],
            "message": "Not enough finite points for feature detection.",
        }

    order = np.argsort(scan_energy)
    scan_energy = scan_energy[order]
    scan_values = scan_values[order]
    if scan_scatter is not None:
        scan_scatter = scan_scatter[order]

    distance_points = max(
        1,
        _odd_window_points(scan_energy, min_separation_eV, minimum=1),
    )
    all_features: list[dict[str, float | str | None]] = []

    for baseline_width in baseline_widths_eV:
        baseline_points = _odd_window_points(scan_energy, baseline_width, minimum=5)
        if baseline_points >= scan_values.size:
            continue

        baseline = _running_nanmedian(scan_values, baseline_points)
        residual = scan_values - baseline
        smooth_points = _odd_window_points(scan_energy, smooth_width_eV, minimum=5)
        if smooth_points < scan_values.size:
            smoothed = savgol_filter(residual, smooth_points, 2, mode="interp")
        else:
            smoothed = residual

        local_noise = _running_mad_scale(residual, baseline_points)
        finite_residual = residual[np.isfinite(residual)]
        if finite_residual.size:
            residual_median = float(np.nanmedian(finite_residual))
            global_noise = 1.4826 * float(
                np.nanmedian(np.abs(finite_residual - residual_median))
            )
            if not np.isfinite(global_noise) or global_noise <= 0.0:
                global_noise = float(np.nanstd(finite_residual))
        else:
            global_noise = np.nan
        if not np.isfinite(global_noise) or global_noise <= 0.0:
            global_noise = np.finfo(float).eps

        valid_noise = local_noise[np.isfinite(local_noise) & (local_noise > 0.0)]
        if valid_noise.size:
            noise_floor = 0.10 * float(np.nanmedian(valid_noise))
        else:
            noise_floor = 0.10 * global_noise
        noise_floor = max(noise_floor, 0.10 * global_noise, np.finfo(float).eps)
        local_noise = np.where(
            np.isfinite(local_noise) & (local_noise > 0.0),
            local_noise,
            noise_floor,
        )
        local_noise = np.maximum(local_noise, noise_floor)

        abs_smoothed = np.abs(smoothed)
        peaks, _ = find_peaks(abs_smoothed, distance=distance_points)
        if peaks.size == 0:
            continue
        prominences = peak_prominences(abs_smoothed, peaks)[0]

        for peak_index, prominence in zip(peaks, prominences):
            amplitude = float(abs_smoothed[peak_index])
            noise = float(local_noise[peak_index])
            z_score = amplitude / noise
            prominence_z = float(prominence) / noise
            if z_score < float(min_z) or prominence_z < float(min_prominence_z):
                continue

            scatter_value = None
            scatter_ratio = None
            if scan_scatter is not None:
                candidate_scatter = float(scan_scatter[peak_index])
                if np.isfinite(candidate_scatter):
                    scatter_value = candidate_scatter
                    if candidate_scatter > 0.0:
                        scatter_ratio = amplitude / candidate_scatter

            score = z_score * np.sqrt(max(prominence_z, 1.0))
            if scatter_ratio is not None:
                score *= np.sqrt(max(scatter_ratio, 0.05))

            all_features.append(
                {
                    "energy_eV": float(scan_energy[peak_index]),
                    "kind": "peak" if float(smoothed[peak_index]) >= 0.0 else "dip",
                    "component": component,
                    "component_value": float(scan_values[peak_index]),
                    "baseline_value": float(baseline[peak_index]),
                    "detrended_value": float(smoothed[peak_index]),
                    "amplitude_abs": amplitude,
                    "prominence_abs": float(prominence),
                    "local_noise": noise,
                    "z_score": float(z_score),
                    "prominence_z": float(prominence_z),
                    "rotation_scatter_abs": scatter_value,
                    "scatter_ratio": None if scatter_ratio is None else float(scatter_ratio),
                    "baseline_width_eV": float(baseline_width),
                    "score": float(score),
                }
            )

    all_features.sort(key=lambda feature: float(feature["score"]), reverse=True)
    merged_features: list[dict[str, float | str | None | int]] = []
    for feature in all_features:
        if all(
            abs(float(feature["energy_eV"]) - float(existing["energy_eV"]))
            >= float(min_separation_eV)
            for existing in merged_features
        ):
            feature = dict(feature)
            feature["rank"] = len(merged_features) + 1
            merged_features.append(feature)
        if len(merged_features) >= int(max_features):
            break

    return {
        "settings": {
            "component": component,
            "energy_window_eV": (energy_min, energy_max),
            "baseline_widths_eV": tuple(float(width) for width in baseline_widths_eV),
            "smooth_width_eV": float(smooth_width_eV),
            "min_z": float(min_z),
            "min_prominence_z": float(min_prominence_z),
            "min_separation_eV": float(min_separation_eV),
            "max_features": int(max_features),
        },
        "n_points": int(scan_energy.size),
        "features": merged_features,
        "message": "ok",
    }


def collapse_twofold_anisotropy_spectrum(
    component_x: np.ndarray,
    component_y: np.ndarray,
    rotations_deg: np.ndarray,
    value_part: str = "real",
    axis_offset_deg: float = 0.0,
) -> dict[str, np.ndarray]:
    """Rotate a twofold anisotropy vector into the sample frame and average it.

    ``component_x`` and ``component_y`` should have shape
    ``(n_rotations, n_energy)``. The returned ``spectrum`` is complex: its real
    part is the anisotropy along the chosen sample axis, while its imaginary
    part is the residual quadrature/cross-axis component after rotation
    collapse.
    """
    x = _spectrum_component(np.asarray(component_x), value_part)
    y = _spectrum_component(np.asarray(component_y), value_part)
    rotations = np.asarray(rotations_deg, dtype=np.float64)

    if x.shape != y.shape:
        raise ValueError(f"component_x and component_y shapes differ: {x.shape}, {y.shape}.")
    if x.ndim != 2:
        raise ValueError(f"components must have shape (n_rotations, n_energy); got {x.shape}.")
    if x.shape[0] != rotations.size:
        raise ValueError(
            f"rotations_deg length {rotations.size} does not match component axis {x.shape[0]}."
        )

    theta = np.deg2rad(rotations + float(axis_offset_deg))
    rotated = (x + 1j * y) * np.exp(-2j * theta)[:, np.newaxis]
    return {
        "rotated_spectra": rotated,
        "spectrum": np.nanmean(rotated, axis=0),
        "scatter": np.nanstd(rotated, axis=0),
        "value_part": value_part,
        "axis_offset_deg": float(axis_offset_deg),
    }


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


def _windowed_signal_scale(
    energy: np.ndarray,
    values: np.ndarray,
    energy_min: float,
    energy_max: float,
) -> float:
    component_values = np.asarray(values)
    finite = np.isfinite(energy)
    finite &= energy >= float(energy_min)
    finite &= energy <= float(energy_max)
    if np.iscomplexobj(component_values):
        finite &= np.isfinite(np.real(component_values)) & np.isfinite(
            np.imag(component_values)
        )
    else:
        finite &= np.isfinite(component_values)

    window_values = _stack_residuals(component_values[finite])
    window_values = window_values[np.isfinite(window_values)]
    if window_values.size == 0:
        return 1.0
    centered = window_values - float(np.nanmedian(window_values))
    scale = float(np.nanpercentile(np.abs(centered), 95))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.nanmax(np.abs(window_values)))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return scale


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


def _local_split_initial_guesses(
    feature_energies: list[float],
    energy_min: float,
    energy_max: float,
    max_delta_eV: float | None,
) -> list[dict[str, float] | None]:
    center = 0.5 * (min(feature_energies) + max(feature_energies))
    span = max(float(energy_max) - float(energy_min), 1.0e-6)
    feature_span = max(feature_energies) - min(feature_energies)
    delta_limit = float(max_delta_eV) if max_delta_eV is not None else 0.5 * span
    delta_guesses = [
        feature_span if feature_span > 0.0 else 0.040,
        0.75 * feature_span if feature_span > 0.0 else 0.020,
        1.25 * feature_span if feature_span > 0.0 else 0.060,
        min(0.040, delta_limit),
    ]
    broadening_guesses = (0.010, 0.020, 0.040)

    guesses: list[dict[str, float] | None] = [None]
    seen: set[tuple[float, float, float]] = set()
    for delta_guess in delta_guesses:
        delta = max(0.0, min(float(delta_guess), delta_limit))
        if delta <= 0.0:
            continue
        for broadening in broadening_guesses:
            key = (round(center, 8), round(delta, 8), round(broadening, 8))
            if key in seen:
                continue
            seen.add(key)
            guesses.append(
                {
                    "center_eV": center,
                    "delta_eV": delta,
                    "broadening_eV": broadening,
                }
            )
    return guesses


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
        fit["parameters"]["delta_meV"] for fit in fits if np.isfinite(fit["parameters"]["delta_meV"])
    ]
    best["near_best_delta_meV"] = near_best_delta
    if len(near_best_delta) > 1:
        best["delta_std_meV"] = float(np.nanstd(near_best_delta))
    else:
        best["delta_std_meV"] = 0.0
    return best


def kk_split_model(
    energy_eV: np.ndarray,
    center_eV: float,
    delta_eV: float,
    broadening_eV: float,
    amplitude_low: float,
    amplitude_high: float,
    phase_rad: float = 0.0,
    exponent: float = -0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (linear_dichroism, linear_birefringence) from one shared model.

    A single complex anisotropic susceptibility is built from two critical
    points at ``center_eV -/+ delta_eV / 2``. Linear dichroism is its
    absorptive (imaginary) part and linear birefringence its dispersive (real)
    part. Because both channels are generated from the *same* transition
    energies, broadening, and amplitudes, the model is Kramers-Kronig
    consistent by construction: LD and LB cannot prefer different transition
    energies. ``phase_rad`` is a single shared phase that fixes which quadrature
    is absorptive without breaking the 90-degree LD/LB relationship.
    """
    energy = np.asarray(energy_eV, dtype=np.float64)
    center = float(center_eV)
    delta = abs(float(delta_eV))
    low_profile = _critical_point_complex_profile(
        energy, center - 0.5 * delta, broadening_eV, exponent
    )
    high_profile = _critical_point_complex_profile(
        energy, center + 0.5 * delta, broadening_eV, exponent
    )
    susceptibility = np.exp(1j * float(phase_rad)) * (
        float(amplitude_low) * low_profile + float(amplitude_high) * high_profile
    )
    return np.imag(susceptibility), np.real(susceptibility)


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
    """Jointly fit the LD and LB on-axis spectra with one KK-consistent model.

    Unlike fitting LD and LB independently and then averaging their separate
    transition energies, this shares ``center_eV``, ``delta_eV``, and
    ``broadening_eV`` across both channels, so the returned ``delta_eV`` is the
    splitting that simultaneously explains dichroism and birefringence. Each
    channel keeps its own amplitude, linear baseline, and robust scale.
    """
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
    ld_values = np.real(ld["spectrum"])
    lb_values = np.real(lb["spectrum"])

    finite = (
        np.isfinite(energy)
        & np.isfinite(ld_values)
        & np.isfinite(lb_values)
        & (energy >= float(energy_min))
        & (energy <= float(energy_max))
    )
    fit_energy = energy[finite]
    ld_target = ld_values[finite]
    lb_target = lb_values[finite]
    if fit_energy.size < 10:
        raise ValueError("Not enough finite points for a joint KK split fit.")

    order = np.argsort(fit_energy)
    fit_energy = fit_energy[order]
    ld_target = ld_target[order]
    lb_target = lb_target[order]

    ld_scale = max(_mad_scale(ld_target - np.nanmedian(ld_target)), 1.0e-12)
    lb_scale = max(_mad_scale(lb_target - np.nanmedian(lb_target)), 1.0e-12)
    energy_ref = float(np.nanmean(fit_energy))

    e_min = float(np.nanmin(fit_energy))
    e_max = float(np.nanmax(fit_energy))
    span = max(e_max - e_min, 1.0e-6)
    step = float(np.nanmedian(np.diff(fit_energy)))
    delta_upper = 0.25 * span if max_delta_eV is None else float(max_delta_eV)
    delta_upper = max(delta_upper, max(0.010, 2.0 * abs(step)))
    broadening_lower = max(1.0e-5, 0.25 * abs(step))
    broadening_upper = max(0.5 * span, 10.0 * broadening_lower)
    amplitude_bound = 20.0 * max(ld_scale, lb_scale)

    # parameter order: center, delta, broadening, amp_low, amp_high, phase,
    #                   off_ld, slope_ld, off_lb, slope_lb
    lower = np.array(
        [e_min, 0.0, broadening_lower, -amplitude_bound, -amplitude_bound, -np.pi,
         -20.0 * ld_scale, -20.0 * ld_scale / span,
         -20.0 * lb_scale, -20.0 * lb_scale / span],
        dtype=np.float64,
    )
    upper = np.array(
        [e_max, delta_upper, broadening_upper, amplitude_bound, amplitude_bound, np.pi,
         20.0 * ld_scale, 20.0 * ld_scale / span,
         20.0 * lb_scale, 20.0 * lb_scale / span],
        dtype=np.float64,
    )

    peak_index = int(np.nanargmax(np.abs(ld_target - np.nanmedian(ld_target))))
    guess = {
        "center_eV": float(fit_energy[peak_index]),
        "delta_eV": min(0.040, 0.5 * delta_upper),
        "broadening_eV": max(0.010, 3.0 * abs(step)),
        "amplitude_low": ld_scale,
        "amplitude_high": -ld_scale,
        "phase_rad": 0.0,
        "offset_ld": float(np.nanmedian(ld_target)),
        "slope_ld": 0.0,
        "offset_lb": float(np.nanmedian(lb_target)),
        "slope_lb": 0.0,
    }
    if initial:
        guess.update({k: float(v) for k, v in initial.items() if k in guess})
    p0 = np.clip(
        np.array(list(guess.values()), dtype=np.float64),
        lower + 1.0e-12,
        upper - 1.0e-12,
    )

    def residuals(params: np.ndarray) -> np.ndarray:
        (center, delta, broadening, amp_low, amp_high, phase,
         off_ld, slope_ld, off_lb, slope_lb) = params
        ld_model, lb_model = kk_split_model(
            fit_energy, center, delta, broadening, amp_low, amp_high, phase,
            exponent=exponent,
        )
        ld_model = ld_model + off_ld + slope_ld * (fit_energy - energy_ref)
        lb_model = lb_model + off_lb + slope_lb * (fit_energy - energy_ref)
        return np.concatenate(
            [(ld_model - ld_target) / ld_scale, (lb_model - lb_target) / lb_scale]
        )

    opt = least_squares(
        residuals, p0, bounds=(lower, upper), loss=loss, f_scale=1.0, max_nfev=5000
    )
    names = (
        "center_eV", "delta_eV", "broadening_eV", "amplitude_low", "amplitude_high",
        "phase_rad", "offset_ld", "slope_ld", "offset_lb", "slope_lb",
    )
    parameters = {name: float(value) for name, value in zip(names, opt.x)}
    parameters["delta_eV"] = abs(parameters["delta_eV"])
    parameters["delta_meV"] = 1000.0 * parameters["delta_eV"]
    parameters["lower_transition_eV"] = parameters["center_eV"] - 0.5 * parameters["delta_eV"]
    parameters["upper_transition_eV"] = parameters["center_eV"] + 0.5 * parameters["delta_eV"]

    final = residuals(opt.x)
    n_per_channel = fit_energy.size
    ld_residual = final[:n_per_channel]
    lb_residual = final[n_per_channel:]
    rmse = float(np.sqrt(np.mean(final**2)))
    ld_rmse = float(np.sqrt(np.mean(ld_residual**2)))
    lb_rmse = float(np.sqrt(np.mean(lb_residual**2)))

    return {
        "success": bool(opt.success),
        "message": opt.message,
        "parameters": parameters,
        "energy_eV": fit_energy,
        "ld_values": ld_target,
        "lb_values": lb_target,
        "energy_window_eV": (float(energy_min), float(energy_max)),
        "rmse": rmse,
        "ld_rmse": ld_rmse,
        "lb_rmse": lb_rmse,
        "n_points": int(fit_energy.size),
        "exponent": float(exponent),
        "vector_part": vector_part,
        "axis_offset_deg": float(axis_offset_deg),
    }


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
            "from the shared energy region, but the splitting is provisional. "
            "If no overlapping pair is found, it falls back to the closest "
            "numerical agreement without assigning Eg."
        ),
    }


def _finite_or_nan(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


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

    if warnings_out:
        primary["recommended_delta_vb_meV"] = np.nan
        primary["recommended_delta_source"] = "manual_review_required"
    else:
        primary["recommended_delta_vb_meV"] = float(primary["splitting_meV"])
        if np.isfinite(kk_value):
            primary["recommended_delta_source"] = "stable_ld_lb_mean_kk_validated"
        else:
            primary["recommended_delta_source"] = "stable_ld_lb_independent_mean"
    primary["math_warnings"] = warnings_out
    primary["requires_manual_delta_vb"] = bool(warnings_out)


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
            "recommended_delta_vb_meV is filled only when the LD/LB pair passes "
            "the stability, assignment, agreement, and joint KK consistency checks. "
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
    if dataset["diagnostics"]["warnings"]:
        print("diagnostic warnings:")
        for warning in dataset["diagnostics"]["warnings"]:
            print(f"  - {warning}")
    else:
        print("diagnostic warnings: none")


if __name__ == "__main__":
    _run_synthetic_demo()
