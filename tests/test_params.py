import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spark_infer.io_utils import InputError  # noqa: E402
from spark_infer.params import parse_demucs, parse_task, parse_vc  # noqa: E402

URL = "https://example.com/a.wav"


def test_task_required():
    assert parse_task({"task": "demucs"}) == "demucs"
    assert parse_task({"task": "vc"}) == "vc"
    with pytest.raises(InputError):
        parse_task({"task": "tts"})
    with pytest.raises(InputError):
        parse_task({})


def test_demucs_defaults():
    r = parse_demucs({"audio_url": URL})
    assert r.output_format == "mp3" and r.mono is False and r.output_url is None
    assert r.chunk_size is None and r.overlap == 0.0001 and r.single_onnx is False


def test_demucs_validation():
    with pytest.raises(InputError):
        parse_demucs({})
    with pytest.raises(InputError):
        parse_demucs({"audio_url": "ftp://x/y"})
    with pytest.raises(InputError):
        parse_demucs({"audio_url": URL, "output_format": "flac"})
    with pytest.raises(InputError):
        parse_demucs({"audio_url": URL, "chunk_size": 10})
    r = parse_demucs({"audio_url": URL, "output_url": URL, "mono": "true", "vram_gb": "24", "chunk_size": 300000})
    assert r.mono is True and r.vram_gb == 24.0 and r.chunk_size == 300000


def test_vc_defaults_and_ref_url_alias():
    r = parse_vc({"source_url": URL, "ref_url": URL})
    assert r.ref_urls == [URL] and r.prompt_url is None and r.output_format == "wav"
    p = r.params
    assert (p.n, p.steps, p.temp, p.cfg, p.ref_len, p.overlap, p.preproc, p.seed) == (1, 20, 0.8, None, 10.0, 1.0, True, 1000)


def test_vc_validation():
    with pytest.raises(InputError):
        parse_vc({"source_url": URL})
    with pytest.raises(InputError):
        parse_vc({"source_url": URL, "ref_urls": []})
    with pytest.raises(InputError):
        parse_vc({"source_url": URL, "ref_urls": [URL] * 9})
    with pytest.raises(InputError):
        parse_vc({"source_url": URL, "ref_urls": [URL], "n": 0})
    with pytest.raises(InputError):
        parse_vc({"source_url": URL, "ref_urls": [URL], "steps": "vingt"})
    r = parse_vc({"source_url": URL, "ref_urls": [URL, URL], "prompt_url": URL, "n": 4,
                  "steps": 30, "temp": 0.6, "cfg": 0.7, "ref_len": 8, "preproc": False,
                  "output_format": "mp3", "output_sr": 44100})
    assert len(r.ref_urls) == 2 and r.prompt_url == URL and r.output_sr == 44100
    assert r.params.n == 4 and r.params.cfg == 0.7 and r.params.preproc is False
