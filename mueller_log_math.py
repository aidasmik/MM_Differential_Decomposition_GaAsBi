"""Pure Mueller-matrix logarithmic decomposition math."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import expm, logm


__all__ = (
    "normalize_mueller",
    "matrix_log_batch",
    "reconstruction_error",
    "remove_isotropic_part",
    "decompose_generator_terms",
    "decompose_mueller_log",
)


_MATRIX_SHAPE = (4, 4)
_M00_ATOL = 1.0e-12
_CONDITION_WARNING_THRESHOLD = 1.0e12
_IMAG_ABS_WARNING_THRESHOLD = 1.0e-10
_IMAG_REL_WARNING_THRESHOLD = 1.0e-8
_RECONSTRUCTION_WARNING_THRESHOLD = 1.0e-8
_REAL_IF_CLOSE_ABS = 1.0e-12
_REAL_IF_CLOSE_REL = 1.0e-10

_STOKES_I, _STOKES_Q, _STOKES_U, _STOKES_V = range(4)
_LD_X_PAIR = (_STOKES_I, _STOKES_Q)
_LD_Y_PAIR = (_STOKES_I, _STOKES_U)
_CD_PAIR = (_STOKES_I, _STOKES_V)
_LB_X_PAIR = (_STOKES_U, _STOKES_V)
_LB_Y_PAIR = (_STOKES_V, _STOKES_Q)
_CB_PAIR = (_STOKES_Q, _STOKES_U)


def _as_numeric_array(values: Any) -> np.ndarray:
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


def _symmetric_generator_component(
    L: np.ndarray, pair: tuple[int, int]
) -> np.ndarray:
    row, col = pair
    return 0.5 * (L[..., row, col] + L[..., col, row])


def _antisymmetric_generator_component(
    L: np.ndarray, positive_pair: tuple[int, int]
) -> np.ndarray:
    row, col = positive_pair
    return 0.5 * (L[..., row, col] - L[..., col, row])


def normalize_mueller(M: np.ndarray) -> np.ndarray:
    """Normalize each Mueller matrix by M[0, 0]."""
    arr = _as_numeric_array(M)
    _validate_mueller_shape(arr)
    denominator = arr[..., 0, 0][..., np.newaxis, np.newaxis]
    with np.errstate(divide="ignore", invalid="ignore"):
        return arr / denominator


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
    logged, _ = _matrix_log_batch_with_status(M)
    return logged


def reconstruction_error(M: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Return ||expm(L) - M|| / ||M|| for one matrix or a batch."""
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
    """Return L with trace(L) / 4 removed, plus the removed scalar term."""
    arr = _as_numeric_array(L)
    _validate_mueller_shape(arr, "L")
    isotropic = np.trace(arr, axis1=-2, axis2=-1) / 4.0
    eye = np.eye(4, dtype=arr.dtype)
    anisotropic = arr - np.asarray(isotropic)[..., np.newaxis, np.newaxis] * eye
    return anisotropic, isotropic


def decompose_generator_terms(L: np.ndarray) -> dict[str, np.ndarray]:
    """Extract LD, CD, LB, and CB terms from an effective Mueller generator.

    Rows and columns use Stokes order (I, Q, U, V):

        [[ a,  LD_x,  LD_y,  CD  ],
         [ LD_x, a,   CB,   -LB_y],
         [ LD_y, -CB, a,    LB_x ],
         [ CD,   LB_y,-LB_x, a    ]]
    """
    arr = _as_numeric_array(L)
    _validate_mueller_shape(arr, "L")

    isotropic_attenuation = np.trace(arr, axis1=-2, axis2=-1) / 4.0
    linear_dichroism_x = _symmetric_generator_component(arr, _LD_X_PAIR)
    linear_dichroism_y = _symmetric_generator_component(arr, _LD_Y_PAIR)
    circular_dichroism = _symmetric_generator_component(arr, _CD_PAIR)
    linear_birefringence_x = _antisymmetric_generator_component(arr, _LB_X_PAIR)
    linear_birefringence_y = _antisymmetric_generator_component(arr, _LB_Y_PAIR)
    circular_birefringence = _antisymmetric_generator_component(arr, _CB_PAIR)

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
    """Compute the effective logarithmic Mueller generator and diagnostics."""
    M_input = _as_numeric_array(M)
    _validate_mueller_shape(M_input)
    flat_input, batch_shape, _ = _flatten_matrices(M_input)
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
