"""Orchestration d'un job : téléchargement → modèle → encodage → livraison. Une tâche à la fois."""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from . import registry
from .audio_utils import chunk_size_for_vram, is_cuda_oom, next_chunk_size, to_stereo
from .io_utils import InputError, decode_to_wav, deliver, download, encode_output, ffprobe_duration
from .params import DemucsRequest, VcRequest, parse_demucs, parse_task, parse_vc

log = logging.getLogger("spark.tasks")

MODELS_DIR = Path(os.environ.get("SPARK_MODELS_DIR", "/models"))
DEMUCS_DIR = MODELS_DIR / "demucs"
CHATTERBOX_DIR = MODELS_DIR / "chatterbox"
ECAPA_DIR = MODELS_DIR / "ecapa"
MAX_OOM_ATTEMPTS = 6

Progress = Callable[[dict], None]


def run_task(inp: dict, job_id: str, progress: Progress) -> dict:
    task = parse_task(inp)
    workdir = Path(tempfile.mkdtemp(prefix=f"spark-{task}-", dir=os.environ.get("SPARK_TMPDIR") or None))
    t0 = time.monotonic()
    try:
        if task == "demucs":
            result = _run_demucs(parse_demucs(inp), workdir, job_id, progress)
        else:
            result = _run_vc(parse_vc(inp), workdir, job_id, progress)
        result.update({"status": "completed", "task": task, "job_id": job_id,
                       "elapsed_s": round(time.monotonic() - t0, 2),
                       "device": registry.device(), "models_loaded": registry.loaded()})
        return result
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# ============================================================ DEMUCS

def _load_separator(chunk: int, overlap: float, single_onnx: bool):
    from .demucs_engine import InstrumentalSeparator

    key = f"demucs:{'single' if single_onnx else 'ensemble'}"
    sep = registry.get(key, lambda: InstrumentalSeparator(
        DEMUCS_DIR, registry.device(), chunk_size=chunk, overlap=overlap, single_onnx=single_onnx))
    sep.chunk_size = chunk          # simples attributs lus au moment du demix
    sep.overlap = min(0.99, max(0.0, overlap))
    return sep


def _run_demucs(req: DemucsRequest, workdir: Path, job_id: str, progress: Progress) -> dict:
    from .demucs_engine import SAMPLE_RATE
    import librosa

    def report(pct: int, msg: str) -> None:
        progress({"task": "demucs", "percent": pct, "message": msg})

    raw = workdir / "input.bin"
    download(req.audio_url, raw)
    src_wav = decode_to_wav(raw, workdir / "input.wav")
    duration = ffprobe_duration(src_wav)
    report(5, f"audio décodé ({duration:.1f} s)")

    # même chargement que l'image actuelle : librosa → 44,1 kHz stéréo (rééchantillonnage soxr)
    audio, _sr = librosa.load(str(src_wav), sr=SAMPLE_RATE, mono=False)
    mix = to_stereo(audio.T if audio.ndim == 2 else audio)
    del audio

    chunk = req.chunk_size or chunk_size_for_vram(req.vram_gb or registry.vram_total_gb())
    attempts = 0
    instrumental: np.ndarray | None = None
    while instrumental is None:
        attempts += 1
        try:
            sep = _load_separator(chunk, req.overlap, req.single_onnx)
            log.info("[%s] séparation : chunk=%d tentative=%d", job_id, chunk, attempts)
            instrumental = sep.separate(mix, progress=lambda p: report(5 + int(p * 0.85), "séparation"))
        except Exception as e:  # noqa: BLE001
            if not is_cuda_oom(e) or attempts >= MAX_OOM_ATTEMPTS:
                raise
            registry.release()
            smaller = next_chunk_size(chunk)
            if smaller is None:
                raise RuntimeError("OOM au chunk minimum : audio trop long pour ce GPU") from e
            log.warning("[%s] OOM avec chunk=%d → %d", job_id, chunk, smaller)
            chunk = smaller

    out_wav = workdir / "instrumental_f32.wav"
    sf.write(out_wav, instrumental, SAMPLE_RATE, subtype="FLOAT")
    del instrumental, mix
    report(92, "encodage")

    out_path = workdir / f"instrumental.{req.output_format}"
    encode_output(out_wav, out_path, req.output_format, req.mono)
    delivered = deliver(out_path, req.output_url, req.output_format)
    report(100, "terminé")
    return {
        **delivered,
        "duration_s": round(duration, 3),
        "sample_rate": SAMPLE_RATE,
        "channels": 1 if req.mono else 2,
        "chunk_size": chunk,
        "attempts": attempts,
    }


# ============================================================ VOICE CONVERSION

def _load_converter():
    from .vc_engine import VoiceConverter

    return registry.get("chatterbox_vc", lambda: VoiceConverter(
        CHATTERBOX_DIR, registry.device(), ecapa_dir=ECAPA_DIR if ECAPA_DIR.is_dir() else None))


def _fetch_wav(url: str, workdir: Path, name: str) -> Path:
    raw = workdir / f"{name}.bin"
    download(url, raw)
    return decode_to_wav(raw, workdir / f"{name}.wav", channels=1)


def _run_vc(req: VcRequest, workdir: Path, job_id: str, progress: Progress) -> dict:
    import librosa

    def report(pct: int, msg: str) -> None:
        progress({"task": "vc", "percent": pct, "message": msg})

    source = _fetch_wav(req.source_url, workdir, "source")
    refs = [_fetch_wav(u, workdir, f"ref_{i}") for i, u in enumerate(req.ref_urls)]
    prompt = _fetch_wav(req.prompt_url, workdir, "prompt") if req.prompt_url else None
    if ffprobe_duration(source) <= 0:
        raise InputError("source vide ou illisible")
    report(5, "entrées décodées")

    attempts = 0
    while True:
        attempts += 1
        try:
            vc = _load_converter()
            wav, sr, meta = vc.run(source, refs, prompt, req.params, workdir,
                                   progress=lambda p, m: report(5 + int(p * 0.85), m))
            break
        except Exception as e:  # noqa: BLE001
            if not is_cuda_oom(e) or attempts >= 2:
                raise
            log.warning("[%s] OOM en conversion vocale → libération des modèles et 2e essai", job_id)
            registry.release()

    if req.output_sr and req.output_sr != sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=req.output_sr, res_type="soxr_hq")
        sr = req.output_sr
    out_wav = workdir / "converted_f32.wav"
    sf.write(out_wav, wav.astype(np.float32), sr, subtype="FLOAT")
    report(92, "encodage")

    out_path = workdir / f"converted.{req.output_format}"
    encode_output(out_wav, out_path, req.output_format, mono=True)
    delivered = deliver(out_path, req.output_url, req.output_format)
    report(100, "terminé")
    return {**delivered, **meta, "sample_rate": sr, "channels": 1,
            "duration_s": round(len(wav) / sr, 3), "attempts": attempts}
