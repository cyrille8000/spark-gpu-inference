"""Orchestration d'un job : téléchargement → modèle → encodage → livraison. Une tâche à la fois.

Chaque étape est chronométrée (`timings`, en secondes) : c'est la seule mesure de coût que le
propriétaire retient (2026-09-09), avec `executionTime` du statut RunPod.
"""
from __future__ import annotations

import json
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
from .io_utils import (InputError, decode_to_wav, deliver, download, encode_output, ffprobe_duration,
                       sha256_of, upload_put)
from .params import (OUTPUT_FORMAT, OUTPUT_MONO, OUTPUT_SR, InstrumentalRequest, SpeakingFacesRequest,
                     VcRequest, parse_instrumental, parse_speaking_faces, parse_task, parse_vc)

log = logging.getLogger("spark.tasks")

MODELS_DIR = Path(os.environ.get("SPARK_MODELS_DIR", "/models"))
BSROFORMER_DIR = Path(os.environ.get("BS_ROFORMER_MODELS_PATH", str(MODELS_DIR / "bsroformer")))
CHATTERBOX_DIR = MODELS_DIR / "chatterbox"
LRASD_DIR = MODELS_DIR / "lrasd"

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
# RECALE le 2026-09-12 au soir sur le L4 de Modal, seule carte ou le plafond a ete
# cherche pour les DEUX taches : separation 4,65 / 18,71 / 18,71 Go a 1, 4 et 5 jobs
# (la 6e deborde) ; conversion vocale 7,25 / 10,29 / 12,83 / 17,29 Go a 1, 4, 6 et 8.
COUT_MEMOIRE_GB = {
    "chatterbox_vc": (5.0, 1.6),
    "bs_roformer_leap_xe": (0.6, 3.9),
    # Visages qui parlent : PAS ENCORE MESURÉ (2026-09-15, code écrit sans build ni déploiement).
    # Estimation prudente : S3FD est un VGG16 sur une image réduite (480×270 en 1080p), LR-ASD un
    # réseau léger (3,4 Mo de poids) — l'un et l'autre bien sous une séparation. À caler par le
    # banc (`bench_concurrence.py`) avec une portion de production AVANT d'ouvrir plus d'une place.
    # Attention : cette tâche est aussi bornée par le CPU (décodage vidéo, recadrage, MFCC).
    "lr_asd": (1.5, 1.0),
}
# Une tâche inconnue est traitée en gourmande : mieux vaut un job de moins qu'un OOM.
COUT_INCONNU_GB = (12.0, 6.0)
# Part de la carte qu'on ne promet JAMAIS : fragmentation de l'allocateur, pilote,
# et un chunk plus long que celui du banc. Un OOM coûte tout le job, un job de moins
# ne coûte que du débit.
MARGE = 0.85
# Plafond par défaut : un garde-fou, PAS un réglage. C'est la MÉMOIRE qui décide.
#
# Il valait 4 jusqu'au 2026-09-12, pour protéger le temps d'un job. Le propriétaire a
# tranché l'inverse : ce qu'il veut, c'est le maximum de jobs que la carte permet — un
# gros bloc n'achète pas de la vitesse, il achète des places, et les places sont la
# ressource rare. Un plafond de 4 bridait d'ailleurs une carte de 80 Go exactement
# comme une de 24, ce qui n'a aucun sens.
#
# Ce qui reste vrai et qu'il faut garder en tête : chaque job s'allonge à proportion du
# nombre de jobs. Une séparation seule prend 56 s sur un L4, 210 s quand elles sont
# cinq. Si un hébergeur coupe un job trop long (900 s chez Modal), c'est à LUI de
# baisser le plafond par `SPARK_JOBS_MAX`, pas à la table de le faire pour tout le monde.
PLAFOND_DEFAUT = 32
# Cartes de 16 a 22 Go : UNE place, quelle que soit la tache (decision du proprietaire,
# 2026-09-15 : elles sont acceptees sur Vast, on ne les empile pas). Seuil en Go tels que
# torch les rapporte (total_memory / 1e9) : une carte « 24 Go » en annonce 23,6 a 25,4
# (L4 23,66, 3090 25,3), une « 16 Go » 17,2, une « 20 Go » 21,5.
VRAM_UNE_PLACE_GB = 23.0
# En PRISE, le worker decide seul ses places d'apres CHAQUE carte (decision du proprietaire,
# 2026-09-15 : pas de consigne par defaut du serveur, pas de version par hebergeur) :
# moins de 24 Go → 1 job par carte, 24 Go et plus → 2. Le serveur ne peut que plafonner.
PLACES_PRISE_24GO = 2


def places_prise(vram_gb: float | None) -> int:
    """Places qu'UNE carte ouvre en prise, d'apres sa memoire. Pur."""
    if not vram_gb or vram_gb < VRAM_UNE_PLACE_GB:
        return 1
    return PLACES_PRISE_24GO


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
    if vram_gb < VRAM_UNE_PLACE_GB:
        return 1
    base, par_job = COUT_MEMOIRE_GB.get(modele, COUT_INCONNU_GB)
    reste = vram_gb * MARGE - base
    if reste < par_job:
        return 1
    return max(1, min(plafond, int(reste // par_job)))


# Places ouvertes pendant un LOT (`service.process_lot`). Un lot dit explicitement
# combien de sous-jobs il veut voir tourner ensemble : la taille des pools de modeles
# doit suivre, sinon les sous-jobs s'attendraient sur une seule instance et le lot ne
# serait qu'une file deguisee. Un conteneur ne traite qu'UN lot a la fois, d'ou une
# simple variable de module plutot qu'un reglage par fil.
_PLACES_LOT: int | None = None


@contextmanager
def places_du_lot(n: int):
    """Ouvre `n` places le temps d'un lot, puis remet l'etat d'avant."""
    global _PLACES_LOT
    ancien = _PLACES_LOT
    _PLACES_LOT = max(1, int(n))
    try:
        yield _PLACES_LOT
    finally:
        _PLACES_LOT = ancien


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
    if _PLACES_LOT is not None:
        return _PLACES_LOT
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


MODELE_DE_TACHE = {"vc": "chatterbox_vc", "instrumental": "bs_roformer_leap_xe", "speaking_faces": "lr_asd"}


def places_pour_taches(taches: list[str]) -> int:
    """Combien de sous-jobs peuvent tourner ENSEMBLE sur cette machine.

    Calcule sur la tache la PLUS GOURMANDE du lot : une separation coute ~3,9 Go par
    job, une conversion vocale ~1,6 (mesures du 2026-09-12). Un lot melange se
    dimensionne donc sur la separation, sinon il deborde. Multiplie par le nombre de
    cartes : un job ne s'etale jamais sur deux GPU, mais deux jobs vont sur deux GPU.
    """
    force = os.environ.get("SPARK_JOBS_PER_GPU")
    vram = registry.vram_total_gb()
    noms = {MODELE_DE_TACHE.get(str(t), "inconnue") for t in taches} or {"inconnue"}
    par_carte = min(jobs_pour_vram(n, vram, force, 32) for n in noms)
    return max(1, par_carte * max(1, len(registry.devices())))


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
        elif task == "vc":
            result = _run_vc(parse_vc(inp), workdir, job_id, progress, timer)
        else:
            result = _run_speaking_faces(parse_speaking_faces(inp), workdir, job_id, progress, timer)
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
            "jobs_per_gpu": jobs_per_gpu(MODELE_DE_TACHE[task]),
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


# ============================================================ VISAGES QUI PARLENT (LR-ASD)

def _run_speaking_faces(req: SpeakingFacesRequest, workdir: Path, job_id: str, progress: Progress,
                        timer: Timer) -> dict:
    """Qui parle à l'image, et où. Sortie : `speech` (quand quelqu'un parle, secondes au millième,
    visages fusionnés) et `faces` (où : une suite de boîtes par visage et par passage, en fractions
    de l'image). Les temps sont TOUJOURS ceux de la vidéo d'origine, jamais de l'extrait — c'est ce
    qui permet d'analyser une vidéo longue par portions, en parallèle, et de recoller par simple
    fusion d'intervalles. Contrat complet : docs/TACHE_VISAGES_QUI_PARLENT.md."""
    import soundfile as sf

    from .faces_clip import SR_ANALYSE, clip_bounds, cut, facedet_scale, probe_media, to_analysis_wav
    from .faces_engine import MODEL_NAME, WEIGHTS_ASD, SpeakingFaceDetector
    from .faces_segments import speaking_faces, speech_seconds

    def report(pct: int, msg: str) -> None:
        progress({"task": "speaking_faces", "percent": pct, "message": msg})

    clip_start, clip_end = clip_bounds(req.window, req.margin)
    clip_mp4, raw_wav = workdir / "clip.mp4", workdir / "audio_raw.wav"
    with timer.step("download"):
        # Téléchargement ET découpe ne font qu'un : ffmpeg lit la source par requêtes Range et
        # n'écrit que la portion voulue (25 i/s) et son son — jamais de fichier complet.
        cut(req.video_url, req.audio_url, clip_start, clip_end, clip_mp4, raw_wav)
    report(5, "extrait découpé")

    base: dict = {
        "model": f"{MODEL_NAME}/{WEIGHTS_ASD}",
        "window": {"start": req.window[0], "end": req.window[1]} if req.window else None,
        "clip": {"start": round(clip_start, 3), "end": round(clip_end, 3) if clip_end is not None else None},
    }
    media = probe_media(clip_mp4)
    if media is None:
        # Fenêtre située APRÈS la fin du média : ffmpeg sort en succès avec un conteneur vide. Là
        # où il n'y a pas d'image, il n'y a pas de parole — une portion de trop en fin de découpage
        # ne fait pas échouer toute l'analyse, mais ça se dit dans les logs.
        log.warning("[%s] aucun flux vidéo sur [%.1f, %s] — fenêtre après la fin du média ?",
                    job_id, clip_start, "fin" if clip_end is None else f"{clip_end:.1f}")
        return _deliver_faces({**base, "speech": [], "faces": [], "frames": 0, "scenes": 0, "tracks": 0,
                               "attempts": 1, "cold_start": False}, req.output_url, workdir, timer)

    with timer.step("decode"):
        audio, sr = sf.read(to_analysis_wav(raw_wav, workdir / "audio16k.wav"), dtype="int16")
    if sr != SR_ANALYSE:
        raise RuntimeError(f"audio d'analyse à {sr} Hz au lieu de {SR_ANALYSE}")
    if audio.ndim > 1:
        audio = audio[:, 0]
    scale = facedet_scale(media["width"])
    report(10, f"{media['width']}×{media['height']} à {media['fps']:.3g} i/s, échelle de détection {scale}")

    cold_start = False

    def analyse():
        nonlocal cold_start
        with _lease(MODEL_NAME, lambda carte: SpeakingFaceDetector(LRASD_DIR, carte), timer) as (det, cold):
            cold_start = cold_start or cold
            with timer.step("inference"):
                return det.run(clip_mp4, audio, scale, progress=report)

    (tracks, scores, info), attempts = _with_oom_retry(job_id, "visages qui parlent", analyse, keep=MODEL_NAME)
    timer.timings.update(info["timings"])

    fps, w, h = media["fps"], media["width"], media["height"]
    speech = speech_seconds(tracks, scores, fps, offset=clip_start, window=req.window)
    faces = speaking_faces(tracks, scores, fps, w, h, offset=clip_start, window=req.window)
    report(96, f"{len(speech)} intervalle(s) de parole, {len(faces)} passage(s) de visage")
    base["clip"].update({"fps": round(fps, 3), "width": w, "height": h, "frames": info["frames"]})
    return _deliver_faces({**base, "speech": speech, "faces": faces, "frames": info["frames"],
                           "scenes": info["scenes"], "tracks": info["tracks"],
                           "attempts": attempts, "cold_start": cold_start}, req.output_url, workdir, timer)


def _deliver_faces(result: dict, output_url: str | None, workdir: Path, timer: Timer) -> dict:
    """Le résultat est PETIT (les boîtes sont échantillonnées, 4 000 points au plus) : il voyage
    dans la réponse et dans le rappel `finished`. `output_url` le dépose EN PLUS en JSON sur R2,
    pour qui préfère le relire de là."""
    with timer.step("upload"):
        out = workdir / "speaking_faces.json"
        out.write_text(json.dumps({k: result[k] for k in ("speech", "faces", "window", "clip")},
                                  ensure_ascii=False), encoding="utf-8")
        result.update({"format": "json", "bytes": out.stat().st_size, "sha256": sha256_of(out), "uploaded": False})
        if output_url:
            upload_put(output_url, out, "application/json")
            result["uploaded"] = True
    return result
