#!/usr/bin/env python3
"""Handler RunPod Serverless — Spark GPU Inference.

Une image, deux tâches choisies par `input.task` :
  - "instrumental" : instrumental seul, BS-Roformer Leap Xe (unwa) via bs-roformer-infer
  - "vc"           : conversion de timbre Chatterbox VC (S3Gen), un seul tirage

Les modèles restent résidents entre deux jobs du même worker (chargés à la première demande).
Le traitement lui-même vit dans `spark_infer.service.process_job`, partagé avec Modal.
Voir README.md pour le contrat d'entrée/sortie complet.
"""
from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import runpod  # noqa: E402

from spark_infer.service import process_job  # noqa: E402

logging.basicConfig(level=os.environ.get("SPARK_LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def handler(job: dict) -> dict:
    job_id = str(job.get("id", "local"))

    def progress(payload: dict) -> None:
        try:
            runpod.serverless.progress_update(job, payload)
        except Exception:  # noqa: BLE001 — la progression n'a jamais le droit de faire échouer un job
            pass

    return process_job(job.get("input") or {}, job_id, progress)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
