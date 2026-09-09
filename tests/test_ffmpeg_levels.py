"""Les conversions de canaux de l'image sont à GAIN CONSTANT (mesuré, pas supposé).

`-ac` de ffmpeg atténue mono → stéréo de 0,707 et amplifie stéréo → mono de 1,414 ; un WAV mono
passé au modèle puis rendu mono ressortait 3 dB trop bas (constaté le 2026-09-09 sur l'endpoint).
Ces tests exigent ffmpeg/ffprobe dans le PATH ; ils sont sautés sinon.
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spark_infer.io_utils import decode_to_wav, encode_output, ffprobe_channels  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                reason="ffmpeg absent")

SR = 44_100


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)))


@pytest.fixture
def mono_wav(tmp_path: Path) -> Path:
    t = np.arange(SR) / SR
    y = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    p = tmp_path / "mono.wav"
    sf.write(p, y, SR, subtype="FLOAT")
    return p


def test_mono_to_stereo_keeps_level(mono_wav: Path, tmp_path: Path):
    out = decode_to_wav(mono_wav, tmp_path / "st.wav", sr=SR, channels=2)
    m, _ = sf.read(mono_wav)
    st, _ = sf.read(out)
    assert ffprobe_channels(out) == 2
    assert abs(_rms(st[:, 0]) / _rms(m) - 1) < 0.01
    assert abs(_rms(st[:, 1]) / _rms(m) - 1) < 0.01


def test_stereo_to_mono_and_resample_keeps_level(mono_wav: Path, tmp_path: Path):
    st = decode_to_wav(mono_wav, tmp_path / "st.wav", channels=2)
    out = encode_output(st, tmp_path / "out.wav", "wav", mono=True, sr=24_000)
    m, _ = sf.read(mono_wav)
    y, sr = sf.read(out)
    assert sr == 24_000 and y.ndim == 1
    assert abs(_rms(y) / _rms(m) - 1) < 0.02   # 16 bits + rééchantillonnage : ±2 %


def test_round_trip_platform_path(mono_wav: Path, tmp_path: Path):
    """Le chemin plateforme : WAV mono → stéréo pour le modèle → mono 24 kHz en sortie."""
    st = decode_to_wav(mono_wav, tmp_path / "st.wav", sr=SR, channels=2)
    out = encode_output(st, tmp_path / "out.wav", "wav", mono=True, sr=24_000)
    m, _ = sf.read(mono_wav)
    y, _ = sf.read(out)
    assert abs(_rms(y) / _rms(m) - 1) < 0.02
