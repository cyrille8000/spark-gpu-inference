"""Le traitement d'UN job, indépendant de la plateforme qui l'héberge.

RunPod (`handler.py`) et Modal (`modal_app.py`) appellent la même fonction : parsing,
exécution, erreurs typées, rappels `started` / `heartbeat` / `finished` vers
`callback_url`. Une seule définition du comportement, deux façons d'être invoqué.
"""
from __future__ import annotations

import logging
import os
import time
import traceback
from typing import Callable

from . import registry
from .container_clock import CLOCK
from .io_utils import InputError
from .params import parse_callback, parse_heartbeat_s, parse_meta
from .tasks import run_task
from .webhooks import Heartbeat, JobWebhooks, now_iso

log = logging.getLogger("spark.service")

Progress = Callable[[dict], None]


def process_job(inp: dict, job_id: str, progress: Progress | None = None) -> dict:
    """Exécute le job et renvoie le JSON de sortie (succès ou erreur typée). Ne lève jamais."""
    task = inp.get("task")
    log.info("[%s] job reçu task=%s", job_id, task)
    provider = os.environ.get("SPARK_PROVIDER", "")
    meta = parse_meta(inp)
    warnings: list[str] = []

    # Rappels : lus avec tolérance — un rappel malformé n'empêche pas le job.
    hooks: JobWebhooks | None = None
    try:
        cb = parse_callback(inp)
        if cb is not None:
            hooks = JobWebhooks(cb.url, cb.token, job_id, task, meta, provider)
    except InputError as e:
        log.warning("[%s] callback_url ignorée : %s", job_id, e)
        warnings.append(f"callback_url ignorée : {e}")

    # 1. started — avant tout travail : GPU, conteneur neuf ou réutilisé, modèles déjà résidents.
    started_at = now_iso()
    t0 = time.monotonic()
    # Le job prend sa part du temps conteneur à partir de MAINTENANT (le temps mort
    # qui précède lui revient) ; il la ferme dans le `finally`, quoi qu'il arrive.
    CLOCK.enter(job_id)
    if hooks is not None:
        hooks.started(gpu_name=registry.gpu_name(), device=registry.device(),
                      container_first_job=CLOCK.is_first(), container_uptime_s=CLOCK.uptime(),
                      models_loaded=registry.loaded())

    # 2. heartbeat — un fil qui bat pendant le job, avec la dernière progression.
    hb = Heartbeat(hooks, parse_heartbeat_s(inp))
    report = progress or (lambda _p: None)

    def progress_and_pulse(p: dict) -> None:
        hb.progress = p
        report(p)

    hb.start()
    try:
        result = run_task(inp, job_id, progress_and_pulse)
        log.info("[%s] terminé en %.1f s", job_id, result.get("elapsed_s", 0.0))
    except InputError as e:
        log.warning("[%s] entrée refusée : %s", job_id, e)
        result = {"status": "error", "error": str(e), "code": "bad_input", "job_id": job_id}
    except Exception as e:  # noqa: BLE001
        log.error("[%s] échec : %s\n%s", job_id, e, traceback.format_exc())
        result = {"status": "error", "error": f"{type(e).__name__}: {e}", "code": "internal", "job_id": job_id}
    finally:
        hb.stop()

    # Ce que l'hébergeur FACTURE : la fenêtre conteneur depuis le rapport
    # précédent (boot + attente + ce job), succès comme échec. Voir container_clock.
    container_s, first = CLOCK.leave(job_id)
    result.update({
        "task": result.get("task") or task,
        # Le GPU, succès COMME échec : sans lui, la plateforme facturait un échec
        # au tarif « GPU inconnu » sous l'étiquette du compte (2026-09-11).
        "gpu_name": result.get("gpu_name") or registry.gpu_name(),
        "device": result.get("device") or registry.device(),
        "container_s": round(container_s, 3), "container_first_job": first,
        "elapsed_s": result.get("elapsed_s", round(time.monotonic() - t0, 3)),
        "started_at": started_at, "finished_at": now_iso(),
        "meta": meta, "provider": provider, "heartbeats": hb.sent,
    })
    if warnings:
        result.setdefault("warnings", []).extend(warnings)

    # 3. finished — le résultat entier (sans base64), même JSON que la sortie.
    if hooks is not None:
        result["callback_delivered"] = hooks.finished(result)
    return result
