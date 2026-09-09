"""Fonctions audio pures (numpy/scipy seulement — testables sans GPU ni torch)."""
from __future__ import annotations

import logging

import numpy as np
from scipy.signal import butter, sosfilt

log = logging.getLogger("spark.audio")


def is_cuda_oom(exc: BaseException) -> bool:
    name = type(exc).__name__
    msg = str(exc).lower()
    return name == "OutOfMemoryError" or "out of memory" in msg or "cuda_error_out_of_memory" in msg


# --- Prétraitement de la source pour la conversion vocale ---

def preprocess_source(y: np.ndarray, sr: int, highpass_hz: float = 70.0,
                      target_lufs: float = -23.0) -> np.ndarray:
    """Passe-haut 70 Hz + normalisation de sonie (-23 LUFS) + clip. Mono float32."""
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return y
    sos = butter(2, highpass_hz, btype="highpass", fs=sr, output="sos")
    y = sosfilt(sos, y).astype(np.float32)
    try:
        import pyloudnorm as pyln  # dépendance de chatterbox
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(y)
        if np.isfinite(loudness):
            y = pyln.normalize.loudness(y, loudness, target_lufs)
    except Exception as e:  # noqa: BLE001 — la sonie est un confort, pas une exigence
        log.warning("loudness normalisation ignorée: %s", e)
    return np.clip(y, -0.99, 0.99).astype(np.float32)


def fit_length(x: np.ndarray, n: int) -> np.ndarray:
    """Ramène `x` à exactement `n` échantillons (padding zéro ou coupe)."""
    n = max(0, int(n))
    if len(x) >= n:
        return x[:n]
    return np.pad(x, (0, n - len(x)))
