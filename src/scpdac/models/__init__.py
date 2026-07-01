from __future__ import annotations

from importlib.resources import as_file, files
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import scarches as sca

__all__ = [
    "list_models",
    "available_models",
    "models_dir",
    "model_path",
    "load_scanvi_model",
    "classifier_dir",
    "load_classifier_checkpoints",
]

_CLASSIFIER_FILES = {"root": "root.pt", "malignant": "malignant.pt", "non_malignant": "nonmalignant.pt"}


def _scanvi_root() -> Path | None:
    """Return the absolute path to the packaged ScanVI models directory, if present."""
    res = files("scpdac").joinpath("models", "scanvi")
    if not res.exists():
        return None
    with as_file(res) as p:
        return Path(p)


def _alias(name: str) -> str:
    """Normalize a model name to an alias (e.g., 'human_scanvi' -> 'human')."""
    s = name.lower()
    for suf in ("_scanvi_model", "_scanvi", "_model"):
        if s.endswith(suf):
            s = s[: -len(suf)]
            break
    return s


def _is_scanvi_dir(p: Path) -> bool:
    """Return True if the path is a ScanVI save directory."""
    return p.is_dir() and (p / "model.pt").exists()


def _index() -> dict[str, Path]:
    """Build the alias -> path index for packaged ScanVI models."""
    root = _scanvi_root()
    out: dict[str, Path] = {}
    if not root:
        return out
    for p in root.iterdir():
        if _is_scanvi_dir(p):
            out[_alias(p.name)] = p
    return out


_INDEX = _index()


def list_models() -> list[str]:
    """List the directory names of packaged ScanVI models."""
    return sorted(v.name for v in _INDEX.values())


def available_models() -> dict[str, str]:
    """Return a mapping {alias -> directory name} for packaged ScanVI models."""
    return dict(sorted((k, v.name) for k, v in _INDEX.items()))


def models_dir() -> Path | None:
    """Return the absolute path to the packaged ScanVI models directory, if present."""
    return _scanvi_root()


def model_path(*parts: str) -> Path | None:
    """Return a concrete path under the packaged ScanVI models directory, if it exists."""
    root = _scanvi_root()
    if not root:
        return None
    p = root.joinpath(*parts)
    return p if p.exists() else None


def load_scanvi_model(alias: str, *, adata, freeze_dropout: bool = True) -> sca.models.SCANVI:
    """Load a packaged ScanVI model by alias using scvi-tools' SCANVI.load."""
    import scarches as sca

    if not _INDEX:
        raise RuntimeError("No ScanVI models found in package resources.")
    key = _alias(alias)
    if key not in _INDEX:
        raise KeyError(f"Unknown model alias '{alias}'. Available: {sorted(_INDEX.keys())}")

    return sca.models.SCANVI.load_query_data(
        adata=adata, reference_model=str(_INDEX[key]), freeze_dropout=freeze_dropout
    )


def classifier_dir(species: str) -> Path | None:
    """Return the packaged classifier directory for a species, if present."""
    res = files("scpdac").joinpath("models", "classifier", species)
    if not res.exists():
        return None
    with as_file(res) as p:
        return Path(p)


def load_classifier_checkpoints(species: str, map_location: str = "cpu") -> dict[str, dict]:
    """Load the three hierarchical-classifier checkpoints for a species.

    Parameters
    ----------
    species
        ``"human"`` or ``"mouse"``.
    map_location
        Torch ``map_location`` for :func:`torch.load`.

    Returns
    -------
    Mapping ``{"root", "malignant", "non_malignant"}`` to checkpoint dicts
    (each holding ``state_dict``, ``genes``, ``classes``, ...).

    Raises
    ------
    FileNotFoundError
        If the species directory or any of the three checkpoints is missing.
    """
    import torch

    root = classifier_dir(species)
    if root is None:
        raise FileNotFoundError(
            f"No classifier checkpoints for species {species!r}. "
            "Train them with scripts/train_classifier.py and place them under "
            f"src/scpdac/models/classifier/{species}/."
        )
    ckpts: dict[str, dict] = {}
    for key, fname in _CLASSIFIER_FILES.items():
        path = root / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing classifier checkpoint: {path}")
        ckpts[key] = torch.load(path, map_location=map_location, weights_only=False)
    return ckpts
