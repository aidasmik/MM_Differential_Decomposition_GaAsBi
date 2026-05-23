"""Numerical routines for logarithmic Mueller decomposition.

This module is intentionally UI- and file-format-free: it contains array
validation, matrix logarithms, spectral smoothing, feature detection, and
split-transition fitting. The application layer in differential_decomposition.py
handles Woollam file parsing, reporting, and plotting.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import expm, logm
from scipy.optimize import least_squares
from scipy.signal import find_peaks, peak_prominences, savgol_filter


__all__ = (
    "normalize_mueller",
    "matrix_log_batch",
    "reconstruction_error",
    "remove_isotropic_part",
    "decompose_generator_terms",
    "decompose_mueller_log",
    "critical_point_profile",
    "split_transition_model",
    "fit_split_transition_spectrum",
    "estimate_direct_derivative_split_spectrum",
    "detect_spectral_features",
    "collapse_twofold_anisotropy_spectrum",
    "kk_split_model",
    "fit_kk_consistent_split_spectra",
)


_MATRIX_SHAPE = (4, 4)
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


# Array and Mueller-matrix helpers

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



# Logarithmic Mueller decomposition

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



# Spectrum processing

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


def _local_polynomial_smooth_and_derivative(
    energy: np.ndarray,
    values: np.ndarray,
    window_points: int,
    *,
    polyorder: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth y(E) and estimate dy/dE with local polynomial regression."""
    e = np.asarray(energy, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if e.shape != y.shape:
        raise ValueError(f"energy and values shapes differ: {e.shape} vs {y.shape}.")
    if e.ndim != 1:
        raise ValueError("energy and values must be one-dimensional.")

    n_points = e.size
    if n_points == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    window = max(int(window_points), int(polyorder) + 2)
    if window % 2 == 0:
        window += 1
    window = min(window, n_points if n_points % 2 else max(1, n_points - 1))
    half_width = max(1, window // 2)
    smooth = np.full(n_points, np.nan, dtype=np.float64)
    derivative = np.full(n_points, np.nan, dtype=np.float64)

    for index in range(n_points):
        start = max(0, index - half_width)
        stop = min(n_points, index + half_width + 1)
        if stop - start < int(polyorder) + 2:
            missing = int(polyorder) + 2 - (stop - start)
            start = max(0, start - missing)
            stop = min(n_points, stop + missing)

        x = e[start:stop] - e[index]
        local_y = y[start:stop]
        finite = np.isfinite(x) & np.isfinite(local_y)
        if np.count_nonzero(finite) < 2:
            continue

        x = x[finite]
        local_y = local_y[finite]
        degree = min(int(polyorder), x.size - 1)
        try:
            span = max(float(np.nanmax(np.abs(x))), np.finfo(float).eps)
            weights = np.exp(-0.5 * (x / max(0.5 * span, np.finfo(float).eps)) ** 2)
            coeff = np.polyfit(x, local_y, degree, w=weights)
            smooth[index] = float(np.polyval(coeff, 0.0))
            if degree >= 1:
                derivative[index] = float(np.polyval(np.polyder(coeff), 0.0))
        except Exception:
            smooth[index] = float(np.nanmedian(local_y))
            derivative[index] = np.nan

    return smooth, derivative


def _quadratic_refined_peak_energy(
    energy: np.ndarray,
    magnitude: np.ndarray,
    peak_index: int,
) -> float:
    """Return a parabolic peak-position refinement around a sampled maximum."""
    e = np.asarray(energy, dtype=np.float64)
    y = np.asarray(magnitude, dtype=np.float64)
    index = int(peak_index)
    if index <= 0 or index >= e.size - 1:
        return float(e[index])

    local_e = e[index - 1 : index + 2]
    local_y = y[index - 1 : index + 2]
    if not (np.all(np.isfinite(local_e)) and np.all(np.isfinite(local_y))):
        return float(e[index])

    x = local_e - e[index]
    try:
        a, b, _ = np.polyfit(x, local_y, 2)
    except Exception:
        return float(e[index])
    if not np.isfinite(a) or not np.isfinite(b) or a >= 0.0:
        return float(e[index])

    vertex = -b / (2.0 * a)
    if float(np.nanmin(x)) <= vertex <= float(np.nanmax(x)):
        return float(e[index] + vertex)
    return float(e[index])



# Critical-point and split-transition fitting

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


def estimate_direct_derivative_split_spectrum(
    energy_eV: np.ndarray,
    values: np.ndarray,
    *,
    energy_min: float | None = None,
    energy_max: float | None = None,
    component: str = "real",
    smooth_width_eV: float = 0.025,
    min_delta_eV: float | None = None,
    max_delta_eV: float | None = 0.20,
    min_peak_z: float = 2.0,
    min_prominence_z: float = 1.0,
    max_peaks: int = 8,
) -> dict[str, Any]:
    """Estimate a split directly from peaks in d(spectrum)/dE.

    This is the model-light version of the visual workflow: smooth the scalar
    anisotropy spectrum as a function of photon energy, compute dD/dE with a
    local polynomial so non-uniform energy grids are handled correctly, then
    refine the two derivative extrema by a local quadratic peak fit.
    """
    if component == "complex":
        raise ValueError("Direct derivative split needs a scalar component.")

    energy = np.asarray(energy_eV, dtype=np.float64)
    raw_values = np.asarray(values)
    if energy.ndim != 1:
        raise ValueError(f"energy_eV must be one-dimensional; got {energy.shape}.")
    if raw_values.shape != energy.shape:
        raise ValueError(f"values must have shape {energy.shape}; got {raw_values.shape}.")

    y = np.asarray(_spectrum_component(raw_values, component), dtype=np.float64)
    finite = np.isfinite(energy) & np.isfinite(y)
    if energy_min is not None:
        finite &= energy >= float(energy_min)
    if energy_max is not None:
        finite &= energy <= float(energy_max)

    fit_energy = energy[finite]
    fit_values = y[finite]
    if fit_energy.size < 9:
        return {
            "success": False,
            "message": "Not enough finite points for direct derivative split.",
            "splitting_eV": np.nan,
            "splitting_meV": np.nan,
            "energy_window_eV": (energy_min, energy_max),
            "component": component,
        }

    order = np.argsort(fit_energy)
    fit_energy = fit_energy[order]
    fit_values = fit_values[order]
    span = float(fit_energy[-1] - fit_energy[0])
    step = _energy_step_eV(fit_energy)
    smooth_points = _odd_window_points(fit_energy, smooth_width_eV, minimum=7)
    smooth, derivative = _local_polynomial_smooth_and_derivative(
        fit_energy,
        fit_values,
        smooth_points,
        polyorder=3,
    )

    abs_derivative = np.abs(derivative)
    valid = np.isfinite(abs_derivative)
    if np.count_nonzero(valid) < 5:
        return {
            "success": False,
            "message": "Derivative could not be estimated robustly.",
            "splitting_eV": np.nan,
            "splitting_meV": np.nan,
            "energy_window_eV": (float(fit_energy[0]), float(fit_energy[-1])),
            "component": component,
            "smooth_width_eV": float(smooth_width_eV),
        }

    derivative_scale = _mad_scale(derivative[valid] - np.nanmedian(derivative[valid]))
    derivative_scale = max(float(derivative_scale), np.finfo(float).eps)
    local_min_delta = (
        max(2.0 * step, 0.006) if min_delta_eV is None else float(min_delta_eV)
    )
    local_max_delta = 0.5 * span if max_delta_eV is None else float(max_delta_eV)
    local_max_delta = max(local_max_delta, local_min_delta)
    distance_points = max(
        1,
        _odd_window_points(fit_energy, local_min_delta, minimum=1),
    )

    peak_signal = np.where(valid, abs_derivative, 0.0)
    peak_indices, _ = find_peaks(peak_signal, distance=distance_points)
    if peak_indices.size == 0:
        return {
            "success": False,
            "message": "No derivative peaks found.",
            "splitting_eV": np.nan,
            "splitting_meV": np.nan,
            "energy_window_eV": (float(fit_energy[0]), float(fit_energy[-1])),
            "component": component,
            "smooth_width_eV": float(smooth_width_eV),
        }

    prominences = peak_prominences(peak_signal, peak_indices)[0]
    peaks: list[dict[str, Any]] = []
    for peak_index, prominence in zip(peak_indices, prominences):
        amplitude = float(abs_derivative[peak_index])
        if not np.isfinite(amplitude):
            continue
        z_score = amplitude / derivative_scale
        prominence_z = float(prominence) / derivative_scale
        if z_score < float(min_peak_z) or prominence_z < float(min_prominence_z):
            continue
        refined_energy = _quadratic_refined_peak_energy(
            fit_energy,
            peak_signal,
            int(peak_index),
        )
        peaks.append(
            {
                "energy_eV": refined_energy,
                "sample_energy_eV": float(fit_energy[peak_index]),
                "derivative_value": float(derivative[peak_index]),
                "amplitude_abs": amplitude,
                "prominence_abs": float(prominence),
                "z_score": float(z_score),
                "prominence_z": float(prominence_z),
                "score": float(z_score * np.sqrt(max(prominence_z, 1.0))),
                "index": int(peak_index),
                "kind": "positive" if derivative[peak_index] >= 0.0 else "negative",
            }
        )

    peaks.sort(key=lambda item: float(item["score"]), reverse=True)
    peaks = peaks[: int(max_peaks)]
    if len(peaks) < 2:
        return {
            "success": False,
            "message": "Fewer than two derivative peaks passed the thresholds.",
            "splitting_eV": np.nan,
            "splitting_meV": np.nan,
            "energy_window_eV": (float(fit_energy[0]), float(fit_energy[-1])),
            "component": component,
            "smooth_width_eV": float(smooth_width_eV),
            "derivative_peaks": peaks,
        }

    pair_candidates: list[dict[str, Any]] = []
    for first_index, first in enumerate(peaks):
        for second in peaks[first_index + 1 :]:
            e1 = float(first["energy_eV"])
            e2 = float(second["energy_eV"])
            delta = abs(e2 - e1)
            if delta < local_min_delta or delta > local_max_delta:
                continue
            weaker_score = min(float(first["score"]), float(second["score"]))
            stronger_score = max(float(first["score"]), float(second["score"]))
            balance = weaker_score / max(stronger_score, np.finfo(float).eps)
            score = weaker_score * np.sqrt(max(balance, 0.05))
            if np.sign(first["derivative_value"]) != np.sign(second["derivative_value"]):
                score *= 1.05
            lower_peak, upper_peak = sorted((first, second), key=lambda item: item["energy_eV"])
            pair_candidates.append(
                {
                    "lower_peak_eV": float(lower_peak["energy_eV"]),
                    "upper_peak_eV": float(upper_peak["energy_eV"]),
                    "splitting_eV": float(delta),
                    "splitting_meV": float(1000.0 * delta),
                    "score": float(score),
                    "peak_scores": [float(first["score"]), float(second["score"])],
                    "peak_z_scores": [
                        float(first["z_score"]),
                        float(second["z_score"]),
                    ],
                    "peak_prominence_z": [
                        float(first["prominence_z"]),
                        float(second["prominence_z"]),
                    ],
                    "peak_kinds": [str(first["kind"]), str(second["kind"])],
                }
            )

    if not pair_candidates:
        return {
            "success": False,
            "message": "No derivative peak pair satisfied the split bounds.",
            "splitting_eV": np.nan,
            "splitting_meV": np.nan,
            "energy_window_eV": (float(fit_energy[0]), float(fit_energy[-1])),
            "component": component,
            "smooth_width_eV": float(smooth_width_eV),
            "derivative_peaks": peaks,
        }

    pair_candidates.sort(key=lambda item: float(item["score"]), reverse=True)
    best = pair_candidates[0]
    weakest_z = min(float(value) for value in best["peak_z_scores"])
    weakest_prominence_z = min(float(value) for value in best["peak_prominence_z"])
    if weakest_z >= 6.0 and weakest_prominence_z >= 4.0:
        confidence = "high"
    elif weakest_z >= 4.0 and weakest_prominence_z >= 2.5:
        confidence = "medium"
    else:
        confidence = "provisional"

    return {
        "success": True,
        "message": "ok",
        "splitting_eV": float(best["splitting_eV"]),
        "splitting_meV": float(best["splitting_meV"]),
        "lower_peak_eV": float(best["lower_peak_eV"]),
        "upper_peak_eV": float(best["upper_peak_eV"]),
        "center_eV": 0.5 * (float(best["lower_peak_eV"]) + float(best["upper_peak_eV"])),
        "confidence": confidence,
        "score": float(best["score"]),
        "weakest_peak_z": weakest_z,
        "weakest_peak_prominence_z": weakest_prominence_z,
        "energy_window_eV": (float(fit_energy[0]), float(fit_energy[-1])),
        "component": component,
        "smooth_width_eV": float(smooth_width_eV),
        "min_delta_eV": float(local_min_delta),
        "max_delta_eV": float(local_max_delta),
        "n_points": int(fit_energy.size),
        "n_candidate_peaks": int(len(peaks)),
        "derivative_peaks": peaks,
        "pair_candidates": pair_candidates[: int(max_peaks)],
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


def fit_kk_consistent_split_spectra(
    energy_eV: np.ndarray,
    linear_dichroism: np.ndarray,
    linear_birefringence: np.ndarray,
    *,
    energy_min: float,
    energy_max: float,
    exponent: float = -0.5,
    max_delta_eV: float | None = 0.20,
    initial: dict[str, float] | None = None,
    loss: str = "soft_l1",
) -> dict[str, Any]:
    """Jointly fit LD and LB spectra with one KK-consistent split model."""
    energy = np.asarray(energy_eV, dtype=np.float64)
    ld_values = np.asarray(linear_dichroism, dtype=np.float64)
    lb_values = np.asarray(linear_birefringence, dtype=np.float64)
    if energy.ndim != 1:
        raise ValueError(f"energy_eV must be one-dimensional; got {energy.shape}.")
    if ld_values.shape != energy.shape:
        raise ValueError(f"linear_dichroism must have shape {energy.shape}; got {ld_values.shape}.")
    if lb_values.shape != energy.shape:
        raise ValueError(
            f"linear_birefringence must have shape {energy.shape}; "
            f"got {lb_values.shape}."
        )

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

    lower = np.array(
        [
            e_min,
            0.0,
            broadening_lower,
            -amplitude_bound,
            -amplitude_bound,
            -np.pi,
            -20.0 * ld_scale,
            -20.0 * ld_scale / span,
            -20.0 * lb_scale,
            -20.0 * lb_scale / span,
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
            20.0 * ld_scale,
            20.0 * ld_scale / span,
            20.0 * lb_scale,
            20.0 * lb_scale / span,
        ],
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
        guess.update({key: float(value) for key, value in initial.items() if key in guess})
    p0 = np.clip(
        np.array(list(guess.values()), dtype=np.float64),
        lower + 1.0e-12,
        upper - 1.0e-12,
    )

    def residuals(params: np.ndarray) -> np.ndarray:
        (
            center,
            delta,
            broadening,
            amp_low,
            amp_high,
            phase,
            off_ld,
            slope_ld,
            off_lb,
            slope_lb,
        ) = params
        ld_model, lb_model = kk_split_model(
            fit_energy,
            center,
            delta,
            broadening,
            amp_low,
            amp_high,
            phase,
            exponent=exponent,
        )
        ld_model = ld_model + off_ld + slope_ld * (fit_energy - energy_ref)
        lb_model = lb_model + off_lb + slope_lb * (fit_energy - energy_ref)
        return np.concatenate(
            [(ld_model - ld_target) / ld_scale, (lb_model - lb_target) / lb_scale]
        )

    opt = least_squares(
        residuals,
        p0,
        bounds=(lower, upper),
        loss=loss,
        f_scale=1.0,
        max_nfev=5000,
    )
    names = (
        "center_eV",
        "delta_eV",
        "broadening_eV",
        "amplitude_low",
        "amplitude_high",
        "phase_rad",
        "offset_ld",
        "slope_ld",
        "offset_lb",
        "slope_lb",
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
    }
