#!/usr/bin/env python3
"""Handler RunPod Serverless — Spark GPU Inference.

Une image, deux tâches choisies par `input.task` :
  - "demucs" : séparation instrumentale (mix - voix), ensemble htdemucs_ft + MDX-Net Kim
  - "vc"     : conversion de timbre Chatterbox VC (S3Gen) avec best-of-N

Les modèles restent résidents entre deux jobs du même worker (chargés à la première demande).
Voir README.md pour le contrat d'entrée/sortie complet.
"""
from __future__ import annotations

import logging
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import runpod  # noqa: E402

from spark_infer.io_utils import InputError  # noqa: E402
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
        return result
    except InputError as e:
        log.warning("[%s] entrée refusée : %s", job_id, e)
        return {"error": str(e), "code": "bad_input", "job_id": job_id}
    except Exception as e:  # noqa: BLE001
        log.error("[%s] échec : %s\n%s", job_id, e, traceback.format_exc())
        return {"error": f"{type(e).__name__}: {e}", "code": "internal", "job_id": job_id}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
