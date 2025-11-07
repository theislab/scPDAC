from __future__ import annotations

from importlib.resources import as_file, files
from pathlib import Path

__all__ = ["list_models", "available_models", "models_dir", "model_path", "load_scanvi_model"]


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


def load_scanvi_model(alias: str, *, adata):
    """Load a packaged ScanVI model by alias using scvi-tools' SCANVI.load."""
    if not _INDEX:
        raise RuntimeError("No ScanVI models found in package resources.")
    key = _alias(alias)
    if key not in _INDEX:
        raise KeyError(f"Unknown model alias '{alias}'. Available: {sorted(_INDEX.keys())}")
    from scvi.model import SCANVI

    return SCANVI.load(_INDEX[key], adata=adata)
