"""Le traitement d'UN job, indépendant de la plateforme qui l'héberge.

RunPod (`handler.py`) et Modal (`modal_app.py`) appellent la même fonction : parsing,
exécution, erreurs typées, rappel `callback_url`. Une seule définition du comportement,
deux façons d'être invoqué.
"""
from __future__ import annotations

import logging
import traceback
from typing import Callable

from .io_utils import InputError, send_callback
from .params import parse_callback
from .tasks import run_task

log = logging.getLogger("spark.service")

Progress = Callable[[dict], None]


def process_job(inp: dict, job_id: str, progress: Progress | None = None) -> dict:
    """Exécute le job et renvoie le JSON de sortie (succès ou erreur typée). Ne lève jamais."""
    log.info("[%s] job reçu task=%s", job_id, inp.get("task"))
    report = progress or (lambda _p: None)
    try:
        result = run_task(inp, job_id, report)
        log.info("[%s] terminé en %.1f s", job_id, result.get("elapsed_s", 0.0))
    except InputError as e:
        log.warning("[%s] entrée refusée : %s", job_id, e)
        result = {"status": "error", "error": str(e), "code": "bad_input", "job_id": job_id}
    except Exception as e:  # noqa: BLE001
        log.error("[%s] échec : %s\n%s", job_id, e, traceback.format_exc())
        result = {"status": "error", "error": f"{type(e).__name__}: {e}", "code": "internal", "job_id": job_id}

    _notify(inp, job_id, result)
    return result


def _notify(inp: dict, job_id: str, result: dict) -> None:
    """Rappel client de fin de job (succès comme échec). Sans base64 : trop gros pour un webhook."""
    try:
        cb = parse_callback(inp)
    except InputError as e:
        log.warning("[%s] callback_url ignorée : %s", job_id, e)
        result.setdefault("warnings", []).append(f"callback_url ignorée : {e}")
        return
    if cb is None:
        return
    payload = {k: v for k, v in result.items() if k != "audio_base64"}
    payload["task"] = payload.get("task") or inp.get("task")
    result["callback_delivered"] = send_callback(cb.url, cb.token, payload)
