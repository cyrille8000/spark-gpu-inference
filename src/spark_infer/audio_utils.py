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


# --- Fenêtrage de la source pour la conversion vocale ---

def plan_windows(y: np.ndarray, sr: int, window_s: float = 60.0, search_s: float | None = None,
                 frame_s: float = 0.05, hop_s: float = 0.01) -> list[tuple[int, int]]:
    """Bornes `[début, fin)` en échantillons des fenêtres de conversion.

    Chatterbox convertit un fichier EN UNE PASSE et l'attention de son décodeur
    grandit avec le carré de la durée : sur un L4 de 22 Go, 176 s passaient et
    179 s débordaient (mesuré le 2026-09-11). Une fenêtre vise `window_s` ; la
    coupe se pose au creux d'énergie (RMS sur `frame_s`) le plus bas des
    `search_s` dernières secondes avant la cible — entre deux mots ou deux
    segments, jamais au milieu d'un son. Le reliquat final est absorbé dans la
    dernière fenêtre s'il tient dans `search_s` : jamais de fenêtre minuscule.
    Les fenêtres se touchent et couvrent tout : concaténées, elles redonnent la
    durée de la source. Pure (numpy), testée sans GPU.
    """
    n = int(len(y))
    if n == 0:
        return []
    window_s = float(window_s)
    search_s = float(search_s) if search_s is not None else min(10.0, window_s / 4)
    win = max(1, int(round(window_s * sr)))
    search = max(1, int(round(search_s * sr)))
    frame = max(1, int(round(frame_s * sr)))
    hop = max(1, int(round(hop_s * sr)))
    y = np.asarray(y, dtype=np.float32)
    bornes: list[tuple[int, int]] = []
    start = 0
    while n - start > win + search:
        cible = start + win
        lo = max(start + 1, cible - search)
        hi = max(lo, cible - frame)
        meilleur, meilleur_rms = cible, np.inf
        for p in range(lo, hi + 1, hop):
            rms = float(np.sqrt(np.mean(np.square(y[p:p + frame]))))
            if rms < meilleur_rms:
                meilleur, meilleur_rms = p + frame // 2, rms
        bornes.append((start, meilleur))
        start = meilleur
    bornes.append((start, n))
    return bornes
