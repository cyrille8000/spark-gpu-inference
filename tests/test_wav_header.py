import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spark_infer.io_utils import encode_output  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg absent")


def test_wav_output_is_canonical_44_bytes(tmp_path: Path):
    """-bitexact : `data` à l'octet 36, pas de bloc LIST/INFO « Lavf… » (2026-09-11)."""
    sr = 24000
    src = tmp_path / "f32.wav"
    sf.write(src, (0.1 * np.sin(np.arange(sr) * 2 * np.pi * 440 / sr)).astype(np.float32), sr, subtype="FLOAT")
    out = encode_output(src, tmp_path / "out.wav", "wav", True, sr=sr)
    b = out.read_bytes()
    assert b[:4] == b"RIFF" and b[8:12] == b"WAVE" and b[12:16] == b"fmt "
    assert b[36:40] == b"data", b[36:80]
    assert int.from_bytes(b[40:44], "little") == len(b) - 44
