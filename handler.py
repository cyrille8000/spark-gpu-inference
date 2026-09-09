#!/usr/bin/env python3
"""Handler RunPod Serverless — Spark GPU Inference.

Une image, deux tâches choisies par `input.task` :
  - "instrumental" : instrumental seul, BS-Roformer Leap Xe (unwa) via bs-roformer-infer
  - "vc"           : conversion de timbre Chatterbox VC (S3Gen), un seul tirage

Les modèles restent résidents entre deux jobs du même worker (chargés à la première demande).
Fin de job : le résultat (ou l'erreur) est renvoyé à RunPod ET posté sur `input.callback_url`
si elle est fournie. Voir README.md pour le contrat d'entrée/sortie complet.
"""
from __future__ import annotations

import logging
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import runpod  # noqa: E402

from spark_infer.io_utils import InputError, send_callback  # noqa: E402
from spark_infer.params import parse_callback  # noqa: E402
from spark_infer.tasks import run_task  # noqa: E402

logging.basicConfig(level=os.environ.get("SPARK_LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("spark.handler")


def handler(job: dict) -> dict:
    job_id = str(job.get("id", "local"))
    inp = job.get("input") or {}
    log.info("[%s] job reçu task=%s", job_id, inp.get("task"))

    def progress(payload: dict) -> None:
        try:
            runpod.serverless.progress_update(job, payload)
        except Exception:  # noqa: BLE001 — la progression n'a jamais le droit de faire échouer un job
            pass

    try:
        result = run_task(inp, job_id, progress)
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


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
