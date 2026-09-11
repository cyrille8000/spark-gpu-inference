import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spark_infer.audio_utils import fit_length, is_cuda_oom, plan_windows, preprocess_source  # noqa: E402


def test_is_cuda_oom():
    assert is_cuda_oom(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"))
    assert is_cuda_oom(RuntimeError("CUDA_ERROR_OUT_OF_MEMORY"))
    assert not is_cuda_oom(RuntimeError("shape mismatch"))


def test_fit_length():
    x = np.arange(10, dtype=np.float32)
    assert len(fit_length(x, 4)) == 4
    padded = fit_length(x, 15)
    assert len(padded) == 15 and padded[-1] == 0.0


def test_preprocess_source_keeps_length_and_range():
    sr = 16000
    t = np.arange(sr) / sr
    y = 0.5 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 30 * t)  # 30 Hz à couper
    out = preprocess_source(y.astype(np.float32), sr)
    assert out.shape == y.shape and out.dtype == np.float32
    assert np.max(np.abs(out)) <= 0.99
    spec = np.abs(np.fft.rfft(out))
    freqs = np.fft.rfftfreq(len(out), 1 / sr)
    # ordre 2 (12 dB/octave) : 30 Hz est ~1,2 octave sous 70 Hz → rapport 0,6 avant filtre, < 0,2 après
    assert spec[np.argmin(np.abs(freqs - 30))] < 0.2 * spec[np.argmin(np.abs(freqs - 220))]


def _parole_avec_silences(sr: int, duree_s: float, periode_s: float = 7.0, silence_s: float = 0.4) -> np.ndarray:
    """Un bruit continu troué d'un silence de `silence_s` toutes les `periode_s`."""
    rng = np.random.default_rng(7)
    y = rng.uniform(-0.5, 0.5, int(duree_s * sr)).astype(np.float32)
    p, s = int(periode_s * sr), int(silence_s * sr)
    for a in range(p, len(y), p):
        y[a:a + s] = 0.0
    return y


def test_plan_windows_short_source_is_one_window():
    sr = 24000
    y = np.ones(50 * sr, dtype=np.float32)
    assert plan_windows(y, sr, 60.0) == [(0, len(y))]
    assert plan_windows(np.zeros(0, dtype=np.float32), sr) == []
    # 65 s : tient dans fenêtre + marge de recherche → une seule fenêtre, jamais un reliquat minuscule
    assert plan_windows(np.ones(65 * sr, dtype=np.float32), sr, 60.0) == [(0, 65 * sr)]


def test_plan_windows_cuts_in_silences_and_covers_everything():
    sr = 24000
    y = _parole_avec_silences(sr, 179.2)  # le chunk qui a débordé l'L4 le 2026-09-11
    bornes = plan_windows(y, sr, 60.0)
    assert bornes[0][0] == 0 and bornes[-1][1] == len(y)
    assert all(b[0] == a[1] for a, b in zip(bornes, bornes[1:]))  # jointives
    assert 3 <= len(bornes) <= 4
    for a, b in bornes[:-1]:
        assert 50 * sr <= b - a <= 70 * sr  # fenêtre visée 60 s, coupe dans les 10 s avant
    assert 10 * sr < bornes[-1][1] - bornes[-1][0] <= 70 * sr  # reliquat : jamais minuscule, jamais au-dessus
    for _, coupe in bornes[:-1]:
        assert float(np.abs(y[coupe - 120:coupe + 120]).max()) == 0.0  # au creux : dans un silence


def test_plan_windows_without_silence_still_bounded():
    sr = 16000
    y = np.ones(200 * sr, dtype=np.float32)  # aucun creux : la coupe tombe quand même dans la zone de recherche
    bornes = plan_windows(y, sr, 60.0)
    assert bornes[0][0] == 0 and bornes[-1][1] == len(y)
    assert all(b[0] == a[1] for a, b in zip(bornes, bornes[1:]))
    for a, b in bornes[:-1]:
        assert 50 * sr <= b - a <= 70 * sr
    assert 10 * sr < bornes[-1][1] - bornes[-1][0] <= 70 * sr
