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
from .audio_utils import is_cuda_oom
from .io_utils import InputError, decode_to_wav, deliver, download, encode_output, ffprobe_duration
from .params import InstrumentalRequest, VcRequest, parse_instrumental, parse_task, parse_vc

log = logging.getLogger("spark.tasks")

MODELS_DIR = Path(os.environ.get("SPARK_MODELS_DIR", "/models"))
BSROFORMER_DIR = Path(os.environ.get("BS_ROFORMER_MODELS_PATH", str(MODELS_DIR / "bsroformer")))
CHATTERBOX_DIR = MODELS_DIR / "chatterbox"
ECAPA_DIR = MODELS_DIR / "ecapa"

Progress = Callable[[dict], None]


def run_task(inp: dict, job_id: str, progress: Progress) -> dict:
    task = parse_task(inp)
    workdir = Path(tempfile.mkdtemp(prefix=f"spark-{task}-", dir=os.environ.get("SPARK_TMPDIR") or None))
    t0 = time.monotonic()
    try:
        if task == "instrumental":
            result = _run_instrumental(parse_instrumental(inp), workdir, job_id, progress)
        else:
            result = _run_vc(parse_vc(inp), workdir, job_id, progress)
        result.update({"status": "completed", "task": task, "job_id": job_id,
                       "elapsed_s": round(time.monotonic() - t0, 2),
                       "device": registry.device(), "models_loaded": registry.loaded()})
        return result
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _with_oom_retry(job_id: str, what: str, fn: Callable[[], object]) -> tuple[object, int]:
    """Exécute `fn` ; sur OOM CUDA, libère tous les modèles résidents et rejoue une fois."""
    attempts = 0
    while True:
        attempts += 1
        try:
            return fn(), attempts
        except Exception as e:  # noqa: BLE001
            if not is_cuda_oom(e) or attempts >= 2:
                raise
            log.warning("[%s] OOM en %s → libération des modèles et 2e essai", job_id, what)
            registry.release()


# ============================================================ INSTRUMENTAL (BS-Roformer Leap Xe)

def _load_separator():
    from .separation_engine import InstrumentalSeparator

    return registry.get("bs_roformer_leap_xe", lambda: InstrumentalSeparator(BSROFORMER_DIR, registry.device()))


def _run_instrumental(req: InstrumentalRequest, workdir: Path, job_id: str, progress: Progress) -> dict:
    from .separation_engine import MODEL_SLUG, SAMPLE_RATE

    def report(pct: int, msg: str) -> None:
        progress({"task": "instrumental", "percent": pct, "message": msg})

    raw = workdir / "input.bin"
    download(req.audio_url, raw)
    # le modèle travaille en 44,1 kHz stéréo : ffmpeg décode et rééchantillonne (soxr) en une passe
    mix_wav = decode_to_wav(raw, workdir / "mix.wav", sr=SAMPLE_RATE, channels=2)
    duration = ffprobe_duration(mix_wav)
    if duration <= 0:
        raise InputError("audio vide ou illisible")
    report(5, f"audio décodé ({duration:.1f} s)")

    def separate():
        sep = _load_separator()
        report(10, "séparation BS-Roformer")
        return sep.separate(mix_wav, workdir)

    inst_wav, attempts = _with_oom_retry(job_id, "séparation", separate)
    report(90, "encodage")

    out_path = workdir / f"instrumental.{req.output_format}"
    encode_output(Path(inst_wav), out_path, req.output_format, req.mono)
    delivered = deliver(out_path, req.output_url, req.output_format)
    report(100, "terminé")
    return {
        **delivered,
        "model": MODEL_SLUG,
        "duration_s": round(duration, 3),
        "sample_rate": SAMPLE_RATE,
        "channels": 1 if req.mono else 2,
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

    def convert():
        vc = _load_converter()
        return vc.run(source, refs, prompt, req.params, workdir,
                      progress=lambda p, m: report(5 + int(p * 0.85), m))

    (wav, sr, meta), attempts = _with_oom_retry(job_id, "conversion vocale", convert)

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
