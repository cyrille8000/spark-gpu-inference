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
from contextlib import contextmanager
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


# Coût mémoire d'une tâche : une part fixe — contexte CUDA, espaces de travail cuDNN,
# premier exemplaire du modèle — puis un coût par job supplémentaire.
#
# MESURÉ le 2026-09-12 avec les fichiers de la PRODUCTION (WAV, pas MP3 : le chunk de
# conversion vocale est un WAV 24 kHz mono de 120 s, 5,8 Mo ; l'entrée de séparation
# une tranche du WAV source, mono 16 bits à la fréquence d'origine, 150 s, 13 Mo) et
# UNE SEULE TÂCHE PAR POD. Ces deux points comptent :
#
#   — deux tâches sur le même pod NE SE MESURENT PAS. L'allocateur de PyTorch ne rend
#     jamais ce qu'il a réservé, donc la seconde tâche relève la somme des deux. C'est
#     ce qui m'avait fait annoncer 13 à 17 Go pour UNE séparation, et croire que le
#     coût dépendait de la carte : c'était le pool de conversion vocale resté en place.
#     Le chiffre vrai est 5 Go, sur toutes les cartes.
#   — un MP3 de 1,2 Mo masque le téléchargement, donc le temps où le GPU dort. Avec le
#     vrai WAV, le parallélisme rend ×1,78 à trois séparations au lieu de ×1,11.
#
# Relevés bruts (Go réservés, vagues de 1/2/3/4 jobs) :
#   conversion vocale, RTX PRO 4000 Blackwell 25 Go : 6,9 / 7,4 / 8,3 / 9,7
#   séparation, A100 PCIE 40 Go                     : 5,0 / 9,7 / 14,1
# La conversion vocale à un seul job est surévaluée (elle hérite du pic du job
# d'échauffement) : le modèle est calé sur le haut des vagues, qui est fiable.
COUT_MEMOIRE_GB = {
    "chatterbox_vc": (5.0, 1.3),
    "bs_roformer_leap_xe": (0.6, 4.8),
}
# Une tâche inconnue est traitée en gourmande : mieux vaut un job de moins qu'un OOM.
COUT_INCONNU_GB = (12.0, 6.0)
# Part de la carte qu'on ne promet JAMAIS : fragmentation de l'allocateur, pilote,
# et un chunk plus long que celui du banc. Un OOM coûte tout le job, un job de moins
# ne coûte que du débit.
MARGE = 0.85
# Plafond par défaut. Monter au-delà n'accélère personne : CHAQUE job s'allonge à
# proportion. Mesuré avec les fichiers de production — une conversion vocale passe de
# 27 s seule à 95 s quand elles sont quatre, pour ×1,13 de débit ; une séparation de
# 24 s à 41 s à trois, pour ×1,78. Le risque de monter est le timeout de l'hébergeur
# (900 s chez Modal), qui tombe sur un job qui aurait réussi seul. Ce qu'on gagne en
# montant, ce n'est donc pas de la vitesse : c'est le nombre de jobs qu'UNE machine
# absorbe, donc des places — et c'est bien ça qui manque (80 places simultanées).
PLAFOND_DEFAUT = 4


def jobs_pour_vram(modele: str, vram_gb: float | None, force: str | None = None,
                   plafond: int = PLAFOND_DEFAUT) -> int:
    """Combien de jobs de CETTE tâche cette carte peut tenir en même temps.

    `force` l'emporte quand il est posé (mesure, incident, carte exotique) ; sinon la
    valeur se déduit de la mémoire réellement trouvée. Le calcul est par TÂCHE, parce
    que les deux ne coûtent pas la même chose : sur un L4 de 22 Go, quatre conversions
    vocales tiennent (9,7 Go mesurés à quatre) contre trois séparations (14,1 Go).
    Pur, testable sans GPU.
    """
    if force:
        try:
            return max(1, min(32, int(force)))
        except ValueError:
            pass
    if not vram_gb or vram_gb <= 0:
        return 1
    base, par_job = COUT_MEMOIRE_GB.get(modele, COUT_INCONNU_GB)
    reste = vram_gb * MARGE - base
    if reste < par_job:
        return 1
    return max(1, min(plafond, int(reste // par_job)))


def _auto_actif() -> bool:
    """La déduction d'après la carte est OPT-IN, et c'est volontaire.

    Ce fichier est partagé par les trois hébergeurs. Modal et RunPod tournent
    aujourd'hui à UN job par conteneur parce qu'aucun des deux ne pose
    `SPARK_JOBS_PER_GPU` — une déduction active par défaut les ferait passer à
    quatre sans que personne l'ait demandé, et ils marchent bien comme ils sont.
    Seule l'image Vast.ai pose `SPARK_JOBS_AUTO` (Dockerfile.vast) : c'est elle
    qui atterrit sur une carte inconnue à chaque location et qui a besoin de
    s'adapter.
    """
    return os.environ.get("SPARK_JOBS_AUTO", "").strip().lower() in ("1", "true", "vrai", "oui", "yes", "on")


def jobs_per_gpu(modele: str = "chatterbox_vc") -> int:
    """Jobs simultanés pour une tâche, sur la carte de CE conteneur.

    `SPARK_JOBS_PER_GPU` force la valeur partout. Sans lui : la carte décide si
    `SPARK_JOBS_AUTO` est posé, sinon UN seul job — le comportement historique.
    """
    force = os.environ.get("SPARK_JOBS_PER_GPU")
    if force:
        return jobs_pour_vram(modele, None, force)
    if not _auto_actif():
        return 1
    try:
        plafond = int(os.environ.get("SPARK_JOBS_MAX", PLAFOND_DEFAUT))
    except ValueError:
        plafond = PLAFOND_DEFAUT
    # `jobs_pour_vram` raisonne sur UNE carte — un job ne s'etale jamais sur deux GPU.
    # Ce que le conteneur peut absorber, c'est cette place multipliee par le nombre de
    # cartes de la machine (mesure du 2026-09-12 : une machine Vast peut en porter
    # deux, quatre, douze).
    par_carte = jobs_pour_vram(modele, registry.vram_total_gb(), None, max(1, plafond))
    return par_carte * max(1, len(registry.devices()))


def run_task(inp: dict, job_id: str, progress: Progress) -> dict:
    task = parse_task(inp)
    workdir = Path(tempfile.mkdtemp(prefix=f"spark-{task}-", dir=os.environ.get("SPARK_TMPDIR") or None))
    timer = Timer()
    # Le pic mémoire est celui de la CARTE : on ne le remet à zéro que si personne
    # d'autre ne travaille, sinon on fausserait la mesure du voisin.
    seul_au_depart = registry.active() == 0
    if seul_au_depart:
        registry.reset_peak_memory()
    voisins = registry.active()
    try:
        if task == "instrumental":
            result = _run_instrumental(parse_instrumental(inp), workdir, job_id, progress, timer)
        else:
            result = _run_vc(parse_vc(inp), workdir, job_id, progress, timer)
        # Mémoire : `jobs` = le plus grand nombre de jobs qui se sont croisés pendant
        # celui-ci (1 = mesure propre, ce job seul) ; `clean` = le compteur partait de
        # zéro. Sans ça, un pic mesuré à plusieurs serait pris pour le coût d'un job.
        mem = registry.peak_memory_gb()
        if mem is not None:
            mem.update({"jobs": max(1, voisins, registry.active()), "clean": seul_au_depart})
        result.update({
            "status": "completed", "task": task, "job_id": job_id,
            "elapsed_s": timer.total(), "timings": timer.timings,
            "device": registry.device(), "gpu_name": registry.gpu_name(),
            "models_loaded": registry.loaded(), "pools": registry.pool_state(),
            "jobs_per_gpu": jobs_per_gpu("chatterbox_vc" if task == "vc" else "bs_roformer_leap_xe"),
            # Ce que le job a VRAIMENT demandé à la carte, et ce qu'elle offre.
            "gpu_mem": mem, "gpu_mem_total_gb": registry.vram_total_gb(),
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


@contextmanager
def _lease(name: str, factory: Callable[[], object], timer: Timer):
    """Emprunte une instance du modèle POUR CE JOB (rendue à la sortie du bloc) ;
    donne (modèle, chargé_maintenant). L'attente d'une instance libre et le
    chargement sont chronométrés ensemble sous `model_load`."""
    with timer.step("model_load"):
        bail = registry.lease(name, factory, jobs_per_gpu(name))
        model, cold = bail.__enter__()
    try:
        yield model, cold
    finally:
        bail.__exit__(None, None, None)


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
        with _lease("bs_roformer_leap_xe",
                    lambda carte: InstrumentalSeparator(BSROFORMER_DIR, carte), timer) as (sep, cold):
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
        with _lease("chatterbox_vc", lambda carte: VoiceConverter(CHATTERBOX_DIR, carte), timer) as (vc, cold):
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
