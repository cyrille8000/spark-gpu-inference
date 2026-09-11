"""Modèles résidents : chargés une fois par worker, libérés ensemble sur OOM."""
from __future__ import annotations

import gc
import logging
import threading
from typing import Callable, TypeVar

log = logging.getLogger("spark.registry")

T = TypeVar("T")
_lock = threading.Lock()
_models: dict[str, object] = {}


def get(name: str, factory: Callable[[], T]) -> T:
    with _lock:
        if name not in _models:
            log.info("chargement du modèle « %s »", name)
            _models[name] = factory()
        return _models[name]  # type: ignore[return-value]


def loaded() -> list[str]:
    return sorted(_models)


def others_loaded(keep: str | None) -> list[str]:
    """Les modèles résidents AUTRES que `keep` — ceux qu'une libération rendrait."""
    return [m for m in loaded() if m != keep]


def release(name: str | None = None) -> None:
    """Libère un modèle (ou tous) et vide le cache CUDA."""
    with _lock:
        names = [name] if name else list(_models)
        for n in names:
            if n in _models:
                log.info("libération du modèle « %s »", n)
                del _models[n]
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:  # noqa: BLE001
        pass


def device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def vram_total_gb() -> float | None:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_properties(0).total_memory / 1e9
    except Exception:  # noqa: BLE001
        return None


def gpu_name() -> str | None:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        return None
