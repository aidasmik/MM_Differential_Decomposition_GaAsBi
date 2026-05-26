"""Git-friendly database for accepted GaAsBi analysis results."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any

import numpy as np


DEFAULT_DATABASE_DIR = Path("shared_database")
JSONL_NAME = "selected_results.jsonl"
CSV_NAME = "selected_results.csv"
SUMMARY_NAME = "comparison_summary.csv"
COMPARISON_DIR_NAME = "comparisons"
GITHUB_DATABASE_URL = "https://github.com/aidasmik/MM_Differential_Decomposition_GaAsBi.git"
README_START = "<!-- comparison-plots:start -->"
README_END = "<!-- comparison-plots:end -->"
GIT_TRACKED_PATHS = (
    JSONL_NAME,
    CSV_NAME,
    COMPARISON_DIR_NAME,
    "README.md",
)

CSV_FIELDS = [
    "record_id",
    "sample_id",
    "source_file",
    "measurement_datetime",
    "composition",
    "bi_percent",
    "temperature_C",
    "strain_r_value",
    "thickness",
    "analysis_thickness",
    "eg_eV",
    "valence_band_splitting_meV",
    "delta_vb_per_bi_meV_per_percent",
    "recommended_delta_vb_meV",
    "recommended_delta_source",
    "requires_manual_delta_vb",
    "selected_delta_source",
    "selected_delta_label",
    "selected_delta_rank",
    "selected_delta_term",
    "selected_delta_lower_transition_eV",
    "selected_delta_upper_transition_eV",
    "selected_delta_center_eV",
    "math_warnings",
    "kk_splitting_meV",
    "direct_derivative_splitting_meV",
    "direct_derivative_spread_meV",
    "direct_derivative_confidence",
    "recommendation_spread_meV",
    "recommendation_components_meV",
    "recommendation_notes",
    "eg_spread_eV",
    "splitting_spread_meV",
    "lower_transition_spread_meV",
    "upper_transition_spread_meV",
    "upper_transition_eV",
    "center_eV",
    "selected_result_rank",
    "status",
    "confidence",
    "basis",
    "n_rotations",
    "energy_min_eV",
    "energy_max_eV",
    "component_splittings_meV",
    "component_lower_transition_eV",
    "component_upper_transition_eV",
    "analysis_summary_path",
    "output_dir",
    "analyst",
    "notes",
    "saved_at",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _finite_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    slug = slug.strip("._-")
    return slug or "record"


def default_sample_id(dat_path: str | Path | None) -> str:
    if dat_path is None:
        return ""
    return Path(dat_path).stem


def infer_temperature_c(text: str | None) -> float | None:
    if not text:
        return None
    match = re.search(r"(?<![A-Za-z0-9])(-?\d+(?:\.\d+)?)\s*C(?![A-Za-z])", text)
    if not match:
        return None
    return _finite_float(match.group(1))


def infer_measurement_datetime(text: str | None) -> str:
    if not text:
        return ""
    match = re.search(r"\[(\d{4}-\d{2}-\d{2}),(\d{2})(\d{2})(\d{2})\]", text)
    if not match:
        return ""
    date, hour, minute, second = match.groups()
    return f"{date}T{hour}:{minute}:{second}"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and not np.isfinite(value):
        return ""
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, default=_json_default, separators=(",", ":"))
    return str(value)


def _ratio_or_none(numerator: Any, denominator: Any) -> float | None:
    top = _finite_float(numerator)
    bottom = _finite_float(denominator)
    if top is None or bottom is None or bottom == 0.0:
        return None
    return top / bottom


def _jsonl_path(database_dir: Path) -> Path:
    return database_dir / JSONL_NAME


def _csv_path(database_dir: Path) -> Path:
    return database_dir / CSV_NAME


def load_records(database_dir: str | Path = DEFAULT_DATABASE_DIR) -> list[dict[str, Any]]:
    db_dir = Path(database_dir)
    path = _jsonl_path(db_dir)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL record at {path}:{line_number}: {exc}") from exc
            if isinstance(record, dict):
                records.append(record)
    return records


def write_records_csv(
    records: list[dict[str, Any]],
    database_dir: str | Path = DEFAULT_DATABASE_DIR,
) -> Path:
    db_dir = Path(database_dir)
    db_dir.mkdir(parents=True, exist_ok=True)
    path = _csv_path(db_dir)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            row_record = _normalize_record_for_save(record)
            writer.writerow({field: _csv_value(row_record.get(field)) for field in CSV_FIELDS})
    return path


def _write_jsonl(records: list[dict[str, Any]], database_dir: Path) -> Path:
    path = _jsonl_path(database_dir)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=True,
                    sort_keys=True,
                    default=_json_default,
                    separators=(",", ":"),
                )
            )
            handle.write("\n")
    return path


def _as_float_list(values: Any) -> list[float]:
    if not isinstance(values, (list, tuple)):
        return []
    out: list[float] = []
    for value in values:
        number = _finite_float(value)
        if number is not None:
            out.append(number)
    return out


def _normalize_record_for_save(record: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    normalized["record_id"] = _slug(
        str(normalized.get("record_id") or normalized.get("sample_id"))
    )
    normalized["bi_percent"] = _finite_float(normalized.get("bi_percent"))
    normalized["temperature_C"] = _finite_float(normalized.get("temperature_C"))
    normalized["strain_r_value"] = _finite_float(normalized.get("strain_r_value"))
    normalized["thickness"] = _finite_float(normalized.get("thickness"))
    normalized["analysis_thickness"] = _finite_float(normalized.get("analysis_thickness"))
    normalized["eg_eV"] = _finite_float(normalized.get("eg_eV"))
    normalized["valence_band_splitting_meV"] = _finite_float(
        normalized.get("valence_band_splitting_meV")
    )
    normalized["delta_vb_per_bi_meV_per_percent"] = _ratio_or_none(
        normalized.get("valence_band_splitting_meV"),
        normalized.get("bi_percent"),
    )
    return normalized


def build_record(
    *,
    summary: dict[str, Any],
    selected_match: dict[str, Any] | None,
    sample_id: str,
    eg_eV: float,
    valence_band_splitting_meV: float,
    composition: str = "",
    bi_percent: float | None = None,
    temperature_C: float | None = None,
    strain_r_value: float | None = None,
    thickness: float | None = None,
    status: str = "accepted",
    analyst: str = "",
    notes: str = "",
    analysis_summary_path: str | Path | None = None,
) -> dict[str, Any]:
    dat_path = Path(str(summary.get("dat_path", "")))
    source_name = dat_path.name
    match = selected_match or {}
    record_id = _slug(sample_id)

    return {
        "record_id": record_id,
        "sample_id": sample_id,
        "source_file": source_name,
        "source_path": str(dat_path),
        "measurement_datetime": infer_measurement_datetime(source_name),
        "composition": composition,
        "bi_percent": bi_percent,
        "temperature_C": temperature_C,
        "strain_r_value": strain_r_value,
        "thickness": thickness,
        "analysis_thickness": _finite_float(summary.get("thickness")),
        "eg_eV": float(eg_eV),
        "valence_band_splitting_meV": float(valence_band_splitting_meV),
        "delta_vb_per_bi_meV_per_percent": _ratio_or_none(
            valence_band_splitting_meV,
            bi_percent,
        ),
        "recommended_delta_vb_meV": _finite_float(match.get("recommended_delta_vb_meV")),
        "recommended_delta_source": str(match.get("recommended_delta_source", "")),
        "requires_manual_delta_vb": bool(match.get("requires_manual_delta_vb", False)),
        "selected_delta_source": str(
            match.get("selected_delta_source", match.get("recommended_delta_source", ""))
        ),
        "selected_delta_label": str(match.get("selected_delta_label", "")),
        "selected_delta_rank": match.get("selected_delta_rank", ""),
        "selected_delta_term": str(match.get("selected_delta_term", "")),
        "selected_delta_lower_transition_eV": _finite_float(
            match.get("selected_delta_lower_transition_eV")
        ),
        "selected_delta_upper_transition_eV": _finite_float(
            match.get("selected_delta_upper_transition_eV")
        ),
        "selected_delta_center_eV": _finite_float(match.get("selected_delta_center_eV")),
        "math_warnings": [str(value) for value in match.get("math_warnings", [])],
        "kk_splitting_meV": _finite_float(match.get("kk_splitting_meV")),
        "direct_derivative_splitting_meV": _finite_float(
            match.get("direct_derivative_splitting_meV")
        ),
        "direct_derivative_spread_meV": _finite_float(
            match.get("direct_derivative_spread_meV")
        ),
        "direct_derivative_confidence": str(match.get("direct_derivative_confidence", "")),
        "recommendation_spread_meV": _finite_float(match.get("recommendation_spread_meV")),
        "recommendation_components_meV": _as_float_list(
            match.get("recommendation_components_meV")
        ),
        "recommendation_notes": [
            str(value) for value in match.get("recommendation_notes", [])
        ],
        "eg_spread_eV": _finite_float(match.get("bandgap_spread_eV")),
        "splitting_spread_meV": _finite_float(match.get("spread_meV")),
        "lower_transition_spread_meV": _finite_float(
            match.get("lower_transition_spread_meV")
        ),
        "upper_transition_spread_meV": _finite_float(
            match.get("upper_transition_spread_meV")
        ),
        "upper_transition_eV": _finite_float(match.get("upper_transition_eV")),
        "center_eV": _finite_float(match.get("center_eV")),
        "selected_result_rank": match.get("rank", ""),
        "status": status,
        "confidence": str(match.get("confidence", "")),
        "basis": str(match.get("basis", "")),
        "n_rotations": int(summary.get("n_rotations", 0)),
        "energy_min_eV": _finite_float(summary.get("energy_min_eV")),
        "energy_max_eV": _finite_float(summary.get("energy_max_eV")),
        "component_splittings_meV": _as_float_list(
            match.get("component_splittings_meV")
        ),
        "component_lower_transition_eV": _as_float_list(
            match.get("component_lower_transition_eV")
        ),
        "component_upper_transition_eV": _as_float_list(
            match.get("component_upper_transition_eV")
        ),
        "analysis_summary_path": "" if analysis_summary_path is None else str(analysis_summary_path),
        "output_dir": str(summary.get("output_dir", "")),
        "analyst": analyst,
        "notes": notes,
        "saved_at": _now_iso(),
    }


def upsert_record(
    record: dict[str, Any],
    database_dir: str | Path = DEFAULT_DATABASE_DIR,
) -> dict[str, Path]:
    db_dir = Path(database_dir)
    db_dir.mkdir(parents=True, exist_ok=True)
    record = _normalize_record_for_save(record)
    existing = load_records(db_dir)
    records = [item for item in existing if item.get("record_id") != record["record_id"]]
    records.append(record)
    records.sort(key=lambda item: str(item.get("sample_id", "")))

    jsonl = _write_jsonl(records, db_dir)
    csv_path = write_records_csv(records, db_dir)
    comparison_paths = build_comparison_outputs(db_dir, records=records)
    return {"jsonl": jsonl, "csv": csv_path, **comparison_paths}


def replace_record(
    original_record_id: str,
    record: dict[str, Any],
    database_dir: str | Path = DEFAULT_DATABASE_DIR,
) -> dict[str, Path]:
    db_dir = Path(database_dir)
    db_dir.mkdir(parents=True, exist_ok=True)
    record = _normalize_record_for_save(record)
    existing = load_records(db_dir)
    records = [
        item
        for item in existing
        if str(item.get("record_id", "")) != str(original_record_id)
        and str(item.get("record_id", "")) != str(record["record_id"])
    ]
    records.append(record)
    records.sort(key=lambda item: str(item.get("sample_id", "")))

    jsonl = _write_jsonl(records, db_dir)
    csv_path = write_records_csv(records, db_dir)
    comparison_paths = build_comparison_outputs(db_dir, records=records)
    return {"jsonl": jsonl, "csv": csv_path, **comparison_paths}


def delete_record(
    record_id: str,
    database_dir: str | Path = DEFAULT_DATABASE_DIR,
) -> dict[str, Path | int]:
    db_dir = Path(database_dir)
    records = load_records(db_dir)
    remaining = [
        record
        for record in records
        if str(record.get("record_id", "")) != str(record_id)
    ]
    deleted = len(records) - len(remaining)
    if deleted == 0:
        raise ValueError(f"No database record found with record_id={record_id!r}.")
    jsonl = _write_jsonl(remaining, db_dir)
    csv_path = write_records_csv(remaining, db_dir)
    comparison_paths = build_comparison_outputs(db_dir, records=remaining)
    return {"jsonl": jsonl, "csv": csv_path, "deleted": deleted, **comparison_paths}


def _run_git(
    database_dir: Path,
    args: list[str],
    *,
    check: bool = True,
    timeout: int = 120,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    env.setdefault("GIT_SSH_COMMAND", "ssh -o BatchMode=yes")
    result = subprocess.run(
        ["git", "-C", str(database_dir), *args],
        text=True,
        capture_output=True,
        env=env,
        timeout=timeout,
    )
    if check and result.returncode != 0:
        command = "git " + " ".join(args)
        details = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"{command} failed: {details}")
    return result


def is_git_database(database_dir: str | Path) -> bool:
    db_dir = Path(database_dir)
    if not db_dir.exists():
        return False
    result = _run_git(db_dir, ["rev-parse", "--is-inside-work-tree"], check=False)
    return result.returncode == 0 and result.stdout.strip() == "true"


def git_remote_url(database_dir: str | Path, remote: str = "origin") -> str:
    db_dir = Path(database_dir)
    result = _run_git(db_dir, ["remote", "get-url", remote], check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def _same_github_repo_url(url: str) -> bool:
    normalized = url.strip().removesuffix(".git")
    expected = GITHUB_DATABASE_URL.removesuffix(".git")
    ssh_expected = "git@github.com:aidasmik/MM_Differential_Decomposition_GaAsBi"
    return normalized in {expected, ssh_expected}


def find_database_clone(start_dir: str | Path | None = None) -> Path | None:
    candidates: list[Path] = []
    if start_dir is not None:
        start = Path(start_dir)
        candidates.extend(
            [
                start,
                start / "MM_Differential_Decomposition_GaAsBi",
                start.parent / "MM_Differential_Decomposition_GaAsBi",
            ]
        )
    home = Path.home()
    candidates.extend(
        [
            home / "MM_Differential_Decomposition_GaAsBi",
            home / "Desktop" / "MM_Differential_Decomposition_GaAsBi",
            home / "Desktop" / "FTMC" / "MM_Differential_Decomposition_GaAsBi",
        ]
    )
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if is_git_database(candidate) and _same_github_repo_url(git_remote_url(candidate)):
            return candidate
    return None


def clone_database_repo(
    parent_dir: str | Path,
    *,
    repo_url: str = GITHUB_DATABASE_URL,
    folder_name: str | None = None,
) -> Path:
    parent = Path(parent_dir)
    parent.mkdir(parents=True, exist_ok=True)
    target_name = folder_name or Path(repo_url.removesuffix(".git")).name
    target = parent / target_name
    if target.exists():
        if is_git_database(target):
            return target
        raise ValueError(f"Target exists but is not a Git repository: {target}")
    result = subprocess.run(
        ["git", "clone", repo_url, str(target)],
        text=True,
        capture_output=True,
        timeout=180,
    )
    if result.returncode != 0:
        details = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"git clone failed: {details}")
    return target


def commit_and_push_database(
    database_dir: str | Path = DEFAULT_DATABASE_DIR,
    *,
    commit_message: str = "Update GaAsBi results database",
    remote: str = "origin",
    pull_rebase: bool = True,
) -> dict[str, Any]:
    db_dir = Path(database_dir)
    if not db_dir.exists():
        raise ValueError(f"Database folder does not exist: {db_dir}")
    if not is_git_database(db_dir):
        raise ValueError(
            "Database folder is not a Git repository. Clone the GitHub database "
            f"repo first: {GITHUB_DATABASE_URL}"
        )

    branch_result = _run_git(db_dir, ["symbolic-ref", "--short", "HEAD"])
    branch = branch_result.stdout.strip()
    if not branch or branch == "HEAD":
        raise ValueError("Database repository is in detached HEAD state.")

    remote_url = git_remote_url(db_dir, remote)
    if not remote_url:
        raise ValueError(f"Git remote {remote!r} is not configured.")

    existing_paths = [path for path in GIT_TRACKED_PATHS if (db_dir / path).exists()]
    if not existing_paths:
        raise ValueError("No database files exist yet. Save a result before pushing.")

    _run_git(db_dir, ["add", *existing_paths])
    staged = _run_git(db_dir, ["diff", "--cached", "--quiet"], check=False)
    committed = staged.returncode != 0
    if committed:
        _run_git(db_dir, ["commit", "-m", commit_message])

    if pull_rebase:
        remote_branch = _run_git(
            db_dir,
            ["ls-remote", "--heads", remote, branch],
            check=False,
            timeout=60,
        )
        if remote_branch.returncode == 0 and remote_branch.stdout.strip():
            _run_git(db_dir, ["pull", "--rebase", "--autostash", remote, branch], timeout=180)
    _run_git(db_dir, ["push", remote, branch], timeout=180)

    head = _run_git(db_dir, ["rev-parse", "--short", "HEAD"]).stdout.strip()
    return {
        "database_dir": db_dir,
        "remote": remote,
        "remote_url": remote_url,
        "branch": branch,
        "committed": committed,
        "head": head,
    }


def _valid_xy(
    records: list[dict[str, Any]],
    x_field: str,
    y_field: str,
) -> tuple[list[float], list[float], list[str]]:
    x_values: list[float] = []
    y_values: list[float] = []
    labels: list[str] = []
    for record in records:
        x = _finite_float(record.get(x_field))
        y = _finite_float(record.get(y_field))
        if x is None or y is None:
            continue
        x_values.append(x)
        y_values.append(y)
        labels.append(str(record.get("sample_id", "")))
    return x_values, y_values, labels


def linear_fit_statistics(
    records: list[dict[str, Any]],
    x_field: str,
    y_field: str,
) -> dict[str, float | int | None]:
    x_values, y_values, _ = _valid_xy(records, x_field, y_field)
    stats: dict[str, float | int | None] = {
        "n": len(x_values),
        "slope": None,
        "intercept": None,
        "r_value": None,
        "r_squared": None,
    }
    if len(set(float(value) for value in x_values)) < 2:
        return stats

    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    slope, intercept = np.polyfit(x, y, 1)
    stats["slope"] = float(slope)
    stats["intercept"] = float(intercept)
    if x.size >= 2 and float(np.std(x)) > 0.0 and float(np.std(y)) > 0.0:
        r_value = float(np.corrcoef(x, y)[0, 1])
        stats["r_value"] = r_value
        stats["r_squared"] = r_value * r_value
    return stats


def delta_vb_bi_statistics(
    records: list[dict[str, Any]],
) -> dict[str, float | int | None]:
    return linear_fit_statistics(records, "bi_percent", "valence_band_splitting_meV")


def format_delta_vb_bi_summary(records: list[dict[str, Any]]) -> str:
    stats = delta_vb_bi_statistics(records)
    n_points = int(stats.get("n") or 0)
    slope = stats.get("slope")
    if slope is None:
        return f"Delta Vb/Bi trend: need at least two Bi-tagged records ({n_points} available)."
    r_squared = stats.get("r_squared")
    if r_squared is None:
        fit_text = "R2 n/a"
    else:
        fit_text = f"R2={float(r_squared):.3f}"
    return (
        f"Delta Vb/Bi trend: {float(slope):.2f} meV/%Bi "
        f"from {n_points} records ({fit_text})."
    )


def _all_y(
    records: list[dict[str, Any]],
    y_field: str,
) -> tuple[list[int], list[float], list[str]]:
    x_values: list[int] = []
    y_values: list[float] = []
    labels: list[str] = []
    for index, record in enumerate(records, start=1):
        y = _finite_float(record.get(y_field))
        if y is None:
            continue
        x_values.append(index)
        y_values.append(y)
        labels.append(str(record.get("sample_id", "")))
    return x_values, y_values, labels


def _plot_scatter(
    path: Path,
    x_values: list[float] | list[int],
    y_values: list[float],
    labels: list[str],
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    fit_line: bool = False,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.scatter(x_values, y_values, s=42)
    if fit_line and len(set(float(value) for value in x_values)) >= 2:
        coefficients = np.polyfit(np.asarray(x_values, dtype=float), y_values, 1)
        x_line = np.linspace(min(x_values), max(x_values), 100)
        y_line = coefficients[0] * x_line + coefficients[1]
        ax.plot(x_line, y_line, linestyle="--", linewidth=1.2)
    if len(labels) <= 20:
        for x_value, y_value, label in zip(x_values, y_values, labels):
            ax.annotate(label, (x_value, y_value), xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def update_database_readme(
    database_dir: str | Path = DEFAULT_DATABASE_DIR,
    *,
    records: list[dict[str, Any]] | None = None,
) -> Path:
    db_dir = Path(database_dir)
    records = load_records(db_dir) if records is None else records
    readme_path = db_dir / "README.md"
    existing = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
    if README_START in existing and README_END in existing:
        prefix = existing.split(README_START, 1)[0].rstrip()
        suffix = existing.split(README_END, 1)[1].lstrip()
    else:
        prefix = existing.rstrip() or "# MM Differential Decomposition for GaAsBi"
        suffix = ""

    section = [
        README_START,
        "## Current Comparison Plots",
        "",
        f"Records in database: `{len(records)}`",
        "",
        "These plots are regenerated by the app from `selected_results.jsonl`.",
        "",
        "### Eg vs Delta Vb",
        "",
        "![Eg vs Delta Vb](comparisons/eg_vs_delta_vb.png)",
        "",
        "### Eg vs Bi%",
        "",
        "![Eg vs Bi%](comparisons/eg_vs_bi_percent.png)",
        "",
        "### Delta Vb vs Bi%",
        "",
        "![Delta Vb vs Bi%](comparisons/delta_vb_vs_bi_percent.png)",
        "",
        "Summary table: [`comparisons/comparison_summary.csv`](comparisons/comparison_summary.csv)",
        README_END,
    ]
    content = prefix + "\n\n" + "\n".join(section).rstrip() + "\n"
    if suffix:
        content += "\n" + suffix
    readme_path.write_text(content, encoding="utf-8", newline="\n")
    return readme_path


def build_comparison_outputs(
    database_dir: str | Path = DEFAULT_DATABASE_DIR,
    *,
    records: list[dict[str, Any]] | None = None,
) -> dict[str, Path]:
    db_dir = Path(database_dir)
    db_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(db_dir) if records is None else records
    write_records_csv(records, db_dir)

    comparison_dir = db_dir / COMPARISON_DIR_NAME
    comparison_dir.mkdir(parents=True, exist_ok=True)
    matplotlib_config_dir = Path(tempfile.gettempdir()) / "logdecomp_matplotlib"
    matplotlib_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config_dir))
    paths: dict[str, Path] = {"comparison_dir": comparison_dir}

    summary_path = comparison_dir / SUMMARY_NAME
    eg_values = [value for value in (_finite_float(r.get("eg_eV")) for r in records) if value is not None]
    splitting_values = [
        value
        for value in (_finite_float(r.get("valence_band_splitting_meV")) for r in records)
        if value is not None
    ]
    bi_eg_x, bi_eg_y, _ = _valid_xy(records, "bi_percent", "eg_eV")
    bi_split_x, bi_split_y, _ = _valid_xy(
        records,
        "bi_percent",
        "valence_band_splitting_meV",
    )
    eg_bi_stats = linear_fit_statistics(records, "bi_percent", "eg_eV")
    split_bi_stats = delta_vb_bi_statistics(records)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value", "unit", "note"])
        writer.writerow(["records", len(records), "", ""])
        writer.writerow(["Eg_mean", np.mean(eg_values) if eg_values else "", "eV", ""])
        writer.writerow(["Eg_std", np.std(eg_values) if len(eg_values) > 1 else "", "eV", ""])
        writer.writerow(
            [
                "splitting_mean",
                np.mean(splitting_values) if splitting_values else "",
                "meV",
                "",
            ]
        )
        writer.writerow(
            [
                "splitting_std",
                np.std(splitting_values) if len(splitting_values) > 1 else "",
                "meV",
                "",
            ]
        )
        if eg_bi_stats["slope"] is not None:
            writer.writerow(
                [
                    "Eg_vs_Bi_slope",
                    1000.0 * float(eg_bi_stats["slope"]),
                    "meV/%Bi",
                    "linear fit",
                ]
            )
            writer.writerow(["Eg_vs_Bi_intercept", eg_bi_stats["intercept"], "eV", "linear fit"])
            writer.writerow(["Eg_vs_Bi_R", eg_bi_stats["r_value"], "", "linear fit"])
            writer.writerow(["Eg_vs_Bi_R2", eg_bi_stats["r_squared"], "", "linear fit"])
        if split_bi_stats["slope"] is not None:
            writer.writerow(
                [
                    "DeltaVb_vs_Bi_slope",
                    split_bi_stats["slope"],
                    "meV/%Bi",
                    "linear fit",
                ]
            )
            writer.writerow(
                [
                    "DeltaVb_vs_Bi_intercept",
                    split_bi_stats["intercept"],
                    "meV",
                    "linear fit",
                ]
            )
            writer.writerow(["DeltaVb_vs_Bi_R", split_bi_stats["r_value"], "", "linear fit"])
            writer.writerow(["DeltaVb_vs_Bi_R2", split_bi_stats["r_squared"], "", "linear fit"])
    paths["summary"] = summary_path

    x_values, y_values, labels = _valid_xy(records, "bi_percent", "eg_eV")
    eg_path = comparison_dir / "eg_vs_bi_percent.png"
    if y_values:
        _plot_scatter(
            eg_path,
            x_values,
            y_values,
            labels,
            xlabel="Bi (%)",
            ylabel="Eg (eV)",
            title="Eg vs Bi%",
            fit_line=len(x_values) >= 2,
        )
        paths["eg_vs_bi_plot"] = eg_path
    else:
        x_idx, y_all, labels = _all_y(records, "eg_eV")
        if y_all:
            _plot_scatter(
                eg_path,
                x_idx,
                y_all,
                labels,
                xlabel="Record",
                ylabel="Eg (eV)",
                title="Eg comparison",
            )
            paths["eg_vs_bi_plot"] = eg_path

    x_values, y_values, labels = _valid_xy(
        records,
        "bi_percent",
        "valence_band_splitting_meV",
    )
    splitting_path = comparison_dir / "delta_vb_vs_bi_percent.png"
    if y_values:
        _plot_scatter(
            splitting_path,
            x_values,
            y_values,
            labels,
            xlabel="Bi (%)",
            ylabel="Delta Vb (meV)",
            title="Delta Vb vs Bi%",
            fit_line=len(x_values) >= 2,
        )
        paths["delta_vb_vs_bi_plot"] = splitting_path
    else:
        x_idx, y_all, labels = _all_y(records, "valence_band_splitting_meV")
        if y_all:
            _plot_scatter(
                splitting_path,
                x_idx,
                y_all,
                labels,
                xlabel="Record",
                ylabel="Delta Vb (meV)",
                title="Delta Vb comparison",
            )
            paths["delta_vb_vs_bi_plot"] = splitting_path

    x_values, y_values, labels = _valid_xy(
        records,
        "eg_eV",
        "valence_band_splitting_meV",
    )
    eg_split_path = comparison_dir / "eg_vs_delta_vb.png"
    if y_values:
        _plot_scatter(
            eg_split_path,
            x_values,
            y_values,
            labels,
            xlabel="Eg (eV)",
            ylabel="Delta Vb (meV)",
            title="Eg vs Delta Vb",
            fit_line=len(x_values) >= 2,
        )
        paths["eg_vs_delta_vb_plot"] = eg_split_path

    paths["readme"] = update_database_readme(db_dir, records=records)
    return paths


def sorted_records_for_display(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(record: dict[str, Any]) -> tuple[int, float, str]:
        bi = _finite_float(record.get("bi_percent"))
        return (0 if bi is not None else 1, bi if bi is not None else 0.0, str(record.get("sample_id", "")))

    return sorted(records, key=key)
