"""Fonctions audio pures (numpy/scipy seulement — testables sans GPU ni torch)."""
from __future__ import annotations

import logging

import numpy as np
from scipy.signal import butter, sosfilt

log = logging.getLogger("spark.audio")

# --- Taille de chunk MDX en fonction de la VRAM (formule reprise de demucs-separate) ---
MODEL_OVERHEAD_GB = 4
SAMPLES_PER_GB = 60_000
SAFETY_MARGIN = 0.9
MIN_CHUNK_SIZE = 50_000
MAX_CHUNK_SIZE = 5_000_000
CHUNK_REDUCTION = 50_000


def chunk_size_for_vram(vram_gb: float | None) -> int:
    """Chunk ONNX initial pour une VRAM donnée (None → minimum sûr)."""
    if not vram_gb or vram_gb <= 0:
        return MIN_CHUNK_SIZE
    available = max(1.0, float(vram_gb) - MODEL_OVERHEAD_GB)
    chunk = int(available * SAMPLES_PER_GB * SAFETY_MARGIN)
    return max(MIN_CHUNK_SIZE, min(MAX_CHUNK_SIZE, chunk))


def next_chunk_size(current: int) -> int | None:
    """Chunk réduit après un OOM ; None si on est déjà au minimum."""
    if current <= MIN_CHUNK_SIZE:
        return None
    return max(MIN_CHUNK_SIZE, current - CHUNK_REDUCTION)


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


def crossfade_append(base: np.ndarray, tail: np.ndarray, ov_n: int) -> np.ndarray:
    """Concatène `tail` à `base` avec un fondu enchaîné linéaire de `ov_n` échantillons."""
    ov_n = int(min(ov_n, len(base), len(tail)))
    if ov_n <= 0:
        return np.concatenate([base, tail])
    f = np.linspace(0, 1, ov_n, dtype=np.float32)
    mixed = base[-ov_n:] * (1 - f) + tail[:ov_n] * f
    return np.concatenate([base[:-ov_n], mixed, tail[ov_n:]])


def fit_length(x: np.ndarray, n: int) -> np.ndarray:
    """Ramène `x` à exactement `n` échantillons (padding zéro ou coupe)."""
    n = max(0, int(n))
    if len(x) >= n:
        return x[:n]
    return np.pad(x, (0, n - len(x)))


def to_stereo(audio: np.ndarray) -> np.ndarray:
    """(samples,) ou (samples, ch) → (samples, 2) float32."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        return np.stack([audio, audio], axis=1)
    if audio.shape[1] == 1:
        return np.repeat(audio, 2, axis=1)
    return audio[:, :2]
