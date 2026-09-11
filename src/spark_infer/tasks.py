"""Orchestration d'un job : téléchargement → modèle → encodage → livraison. Une tâche à la fois.

Chaque étape est chronométrée (`timings`, en secondes) : c'est la seule mesure de coût que le
propriétaire retient (2026-09-09), avec `executionTime` du statut RunPod.
"""
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
from .params import OUTPUT_FORMAT, OUTPUT_MONO, OUTPUT_SR, InstrumentalRequest, VcRequest, parse_instrumental, parse_task, parse_vc

log = logging.getLogger("spark.tasks")

MODELS_DIR = Path(os.environ.get("SPARK_MODELS_DIR", "/models"))
BSROFORMER_DIR = Path(os.environ.get("BS_ROFORMER_MODELS_PATH", str(MODELS_DIR / "bsroformer")))
CHATTERBOX_DIR = MODELS_DIR / "chatterbox"

Progress = Callable[[dict], None]


class Timer:
    """Chronomètre par étape : `with t.step("download"): ...` → t.timings["download_s"]."""

    def __init__(self) -> None:
        self.t0 = time.monotonic()
        self.timings: dict[str, float] = {}

    class _Step:
        def __init__(self, timer: "Timer", name: str) -> None:
            self.timer, self.name = timer, name

        def __enter__(self):
            self.start = time.monotonic()
            return self

        def __exit__(self, *exc):
            self.timer.timings[f"{self.name}_s"] = round(
                self.timer.timings.get(f"{self.name}_s", 0.0) + time.monotonic() - self.start, 3)
            return False

    def step(self, name: str) -> "Timer._Step":
        return Timer._Step(self, name)

    def total(self) -> float:
        return round(time.monotonic() - self.t0, 3)


def run_task(inp: dict, job_id: str, progress: Progress) -> dict:
    task = parse_task(inp)
    workdir = Path(tempfile.mkdtemp(prefix=f"spark-{task}-", dir=os.environ.get("SPARK_TMPDIR") or None))
    timer = Timer()
    try:
        if task == "instrumental":
            result = _run_instrumental(parse_instrumental(inp), workdir, job_id, progress, timer)
        else:
            result = _run_vc(parse_vc(inp), workdir, job_id, progress, timer)
        result.update({
            "status": "completed", "task": task, "job_id": job_id,
            "elapsed_s": timer.total(), "timings": timer.timings,
            "device": registry.device(), "gpu_name": registry.gpu_name(),
            "models_loaded": registry.loaded(),
        })
        return result
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _with_oom_retry(job_id: str, what: str, fn: Callable[[], object], keep: str | None = None) -> tuple[object, int]:
    """Exécute `fn` ; sur OOM CUDA, libère les modèles résidents et rejoue UNE fois —
    seulement si un AUTRE modèle que `keep` (celui du job) occupait la carte. Le même
    job, sur la même carte, avec le même modèle seul, redéborde à l'identique :
    rejouer ne fait que payer deux fois (mesuré le 2026-09-11 : 15 min d'L4 pour rien,
    l'image ayant relancé la conversion après le premier OOM jusqu'au timeout Modal)."""
    attempts = 0
    while True:
        attempts += 1
        try:
            return fn(), attempts
        except Exception as e:  # noqa: BLE001
            if not is_cuda_oom(e) or attempts >= 2:
                raise
            autres = registry.others_loaded(keep)
            if not autres:
                log.warning("[%s] OOM en %s, aucun autre modèle résident → pas de 2e essai", job_id, what)
                raise
            log.warning("[%s] OOM en %s → libération de %s et 2e essai", job_id, what, autres)
            registry.release()


def _load(name: str, factory: Callable[[], object], timer: Timer) -> tuple[object, bool]:
    """Modèle résident ; renvoie (modèle, chargé_maintenant). Le chargement est chronométré à part."""
    cold = name not in registry.loaded()
    with timer.step("model_load"):
        model = registry.get(name, factory)
    return model, cold


# ============================================================ INSTRUMENTAL (BS-Roformer Leap Xe)

def _run_instrumental(req: InstrumentalRequest, workdir: Path, job_id: str, progress: Progress,
                      timer: Timer) -> dict:
    from .separation_engine import MODEL_SLUG, SAMPLE_RATE, InstrumentalSeparator

    def report(pct: int, msg: str) -> None:
        progress({"task": "instrumental", "percent": pct, "message": msg})

    with timer.step("download"):
        raw = workdir / "input.bin"
        download(req.audio_url, raw)
    with timer.step("decode"):
        # le modèle travaille en 44,1 kHz stéréo : ffmpeg décode et rééchantillonne (soxr) en une passe
        mix_wav = decode_to_wav(raw, workdir / "mix.wav", sr=SAMPLE_RATE, channels=2)
        duration = ffprobe_duration(mix_wav)
    if duration <= 0:
        raise InputError("audio vide ou illisible")
    report(5, f"audio décodé ({duration:.1f} s)")

    cold_start = False

    def separate():
        nonlocal cold_start
        sep, cold = _load("bs_roformer_leap_xe",
                          lambda: InstrumentalSeparator(BSROFORMER_DIR, registry.device()), timer)
        cold_start = cold_start or cold
        report(10, "séparation BS-Roformer")
        with timer.step("inference"):
            return sep.separate(mix_wav, workdir)

    inst_wav, attempts = _with_oom_retry(job_id, "séparation", separate, keep="bs_roformer_leap_xe")
    report(90, "encodage")

    with timer.step("encode"):
        out_path = workdir / f"instrumental.{OUTPUT_FORMAT}"
        encode_output(Path(inst_wav), out_path, OUTPUT_FORMAT, OUTPUT_MONO, sr=OUTPUT_SR)
    with timer.step("upload"):
        delivered = deliver(out_path, req.output_url, OUTPUT_FORMAT)
    report(100, "terminé")
    return {
        **delivered,
        "model": MODEL_SLUG,
        "duration_s": round(duration, 3),
        "sample_rate": OUTPUT_SR,
        "channels": 1,
        "attempts": attempts,
        "cold_start": cold_start,
    }


# ============================================================ VOICE CONVERSION

def _fetch_wav(url: str, workdir: Path, name: str) -> Path:
    raw = workdir / f"{name}.bin"
    download(url, raw)
    return decode_to_wav(raw, workdir / f"{name}.wav", channels=1)


def _run_vc(req: VcRequest, workdir: Path, job_id: str, progress: Progress, timer: Timer) -> dict:
    from .vc_engine import VoiceConverter

    def report(pct: int, msg: str) -> None:
        progress({"task": "vc", "percent": pct, "message": msg})

    with timer.step("download"):
        source = _fetch_wav(req.source_url, workdir, "source")
        refs = [_fetch_wav(u, workdir, f"ref_{i}") for i, u in enumerate(req.ref_urls)]
        prompt = _fetch_wav(req.prompt_url, workdir, "prompt") if req.prompt_url else None
    if ffprobe_duration(source) <= 0:
        raise InputError("source vide ou illisible")
    report(5, "entrées décodées")

    cold_start = False

    def convert():
        nonlocal cold_start
        vc, cold = _load("chatterbox_vc", lambda: VoiceConverter(CHATTERBOX_DIR, registry.device()), timer)
        cold_start = cold_start or cold
        with timer.step("inference"):
            return vc.run(source, refs, prompt, req.params, workdir,
                          progress=lambda p, m: report(5 + int(p * 0.85), m), cuts_s=req.cuts_s)

    (wav, sr, meta), attempts = _with_oom_retry(job_id, "conversion vocale", convert, keep="chatterbox_vc")

    with timer.step("encode"):
        out_wav = workdir / "converted_f32.wav"
        sf.write(out_wav, wav.astype(np.float32), sr, subtype="FLOAT")
        report(92, "encodage")
        out_path = workdir / f"converted.{OUTPUT_FORMAT}"
        encode_output(out_wav, out_path, OUTPUT_FORMAT, OUTPUT_MONO, sr=OUTPUT_SR)
    with timer.step("upload"):
        delivered = deliver(out_path, req.output_url, OUTPUT_FORMAT)
    report(100, "terminé")
    return {**delivered, **meta, "sample_rate": OUTPUT_SR, "channels": 1,
            "duration_s": round(len(wav) / sr, 3), "attempts": attempts, "cold_start": cold_start}
