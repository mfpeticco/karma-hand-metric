"""Persistence layer for saving and loading metric results."""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .results import MetricResult, RESULT_SCHEMA_VERSION

logger = logging.getLogger(__name__)

SAVES_DIR = "workspace"

# Result files written before the package was renamed (including the published
# camera-ready batch in results/) pickled their classes as "fine_metric.*".
# Rewriting those artifacts would destroy their provenance, so instead we remap
# the module path at load time.
_LEGACY_PACKAGE = "fine_metric"

# MetricResult and StateNode used to live in the `metric` module; they moved to
# `results` when `metric` became the compute module. Redirect the old paths so
# existing pickles (which name the classes under `metric`) still resolve.
_MOVED_TO_RESULTS = {"MetricResult", "StateNode"}


class _CompatUnpickler(pickle.Unpickler):
    """Unpickler that resolves pre-rename ``fine_metric.*`` and pre-split
    ``*.metric`` class paths."""

    def find_class(self, module: str, name: str):
        if module == _LEGACY_PACKAGE or module.startswith(_LEGACY_PACKAGE + "."):
            module = __package__ + module[len(_LEGACY_PACKAGE):]
        if module == __package__ + ".metric" and name in _MOVED_TO_RESULTS:
            module = __package__ + ".results"
        return super().find_class(module, name)


def _load_pickle(f) -> dict:
    return _CompatUnpickler(f).load()


@dataclass
class SavedResult:
    """Metadata about a saved result."""
    name: str
    filepath: Path
    timestamp: datetime
    urdf_name: str
    n_voxels: int
    n_states: int


def _ensure_saves_dir(base_dir: str = ".") -> Path:
    """Ensure the saves directory exists."""
    saves_path = Path(base_dir) / SAVES_DIR
    saves_path.mkdir(parents=True, exist_ok=True)
    return saves_path


def generate_save_name(urdf_path: str) -> str:
    """Generate a save name from URDF path and current timestamp."""
    urdf_name = Path(urdf_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{urdf_name}_{timestamp}"


def save_result(
    result: MetricResult,
    urdf_path: str,
    base_dir: str = ".",
    name: str | None = None,
) -> Path:
    """Save a metric result to disk.

    Args:
        result: The MetricResult to save
        urdf_path: Path to the URDF (used for naming)
        base_dir: Base directory for saves
        name: Optional custom name (otherwise auto-generated)

    Returns:
        Path to the saved file
    """
    saves_dir = _ensure_saves_dir(base_dir)

    if name is None:
        name = generate_save_name(urdf_path)

    # Ensure unique filename
    filepath = saves_dir / f"{name}.pkl"
    counter = 1
    while filepath.exists():
        filepath = saves_dir / f"{name}_{counter}.pkl"
        counter += 1

    # Save with metadata
    data = {
        "result": result,
        "urdf_path": urdf_path,
        "timestamp": datetime.now(),
        "version": RESULT_SCHEMA_VERSION,
    }

    with open(filepath, "wb") as f:
        pickle.dump(data, f)

    logger.info("Saved result to %s", filepath)
    return filepath


def list_saved_results(base_dir: str = ".") -> list[SavedResult]:
    """List all saved results in the saves directory.

    Searches both top-level .pkl files and batch subdirectories.

    Returns:
        List of SavedResult metadata, sorted by timestamp (newest first)
    """
    saves_dir = Path(base_dir) / SAVES_DIR
    if not saves_dir.exists():
        return []

    results = []

    # Search patterns: top-level pkls and subdirectories
    patterns = [
        "*.pkl",                           # Top-level: workspace/*.pkl
        "*/robot_*/current.pkl",           # Batch runs: workspace/<batch>/robot_*/current.pkl
        "*/*.pkl",                         # Batch summary: workspace/<batch>/*.pkl
    ]

    seen_paths: set[Path] = set()
    for pattern in patterns:
        for filepath in saves_dir.glob(pattern):
            if filepath in seen_paths:
                continue
            seen_paths.add(filepath)

            try:
                with open(filepath, "rb") as f:
                    data = _load_pickle(f)

                result: MetricResult = data["result"]
                urdf_path = data.get("urdf_path", "unknown")
                timestamp = data.get("timestamp", datetime.fromtimestamp(filepath.stat().st_mtime))

                # For batch results, include the batch folder in the display name
                rel_path = filepath.relative_to(saves_dir)
                if len(rel_path.parts) > 1:
                    # e.g., "batch_20260130_215904/robot_ARMS/current"
                    name = "/".join(rel_path.parts[:-1]) + "/" + filepath.stem
                else:
                    name = filepath.stem

                results.append(SavedResult(
                    name=name,
                    filepath=filepath,
                    timestamp=timestamp,
                    urdf_name=Path(urdf_path).stem,
                    n_voxels=result.n_voxels_reached,
                    n_states=result.n_states_reached,
                ))
            except Exception as e:
                logger.warning("Failed to load metadata from %s: %s", filepath, e)

    # Sort by timestamp, newest first
    results.sort(key=lambda x: x.timestamp, reverse=True)
    return results


def load_result(filepath: Path | str) -> tuple[MetricResult, dict]:
    """Load a saved metric result.

    Args:
        filepath: Path to the saved .pkl file

    Returns:
        Tuple of (MetricResult, metadata dict)
    """
    filepath = Path(filepath)

    with open(filepath, "rb") as f:
        data = _load_pickle(f)

    result = data["result"]
    metadata = {
        "urdf_path": data.get("urdf_path"),
        "timestamp": data.get("timestamp"),
        "version": data.get("version"),
    }

    logger.info("Loaded result from %s", filepath)
    return result, metadata
