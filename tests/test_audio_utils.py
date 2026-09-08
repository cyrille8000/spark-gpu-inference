import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spark_infer.audio_utils import crossfade_append, fit_length, is_cuda_oom, preprocess_source  # noqa: E402


def test_is_cuda_oom():
    assert is_cuda_oom(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"))
    assert is_cuda_oom(RuntimeError("CUDA_ERROR_OUT_OF_MEMORY"))
    assert not is_cuda_oom(RuntimeError("shape mismatch"))


def test_crossfade_append_lengths_and_continuity():
    base = np.ones(100, dtype=np.float32)
    tail = np.zeros(50, dtype=np.float32)
    out = crossfade_append(base, tail, 20)
    assert len(out) == 130
    assert out[79] == 1.0 and out[80] == 1.0 and out[99] == 0.0 and out[100] == 0.0
    assert np.all(np.diff(out[80:100]) <= 0)   # descente monotone dans le fondu
    assert len(crossfade_append(base, tail, 0)) == 150
    assert len(crossfade_append(base, tail, 500)) == 100  # recouvrement borné par la queue


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
