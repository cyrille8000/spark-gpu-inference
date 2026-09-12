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

os.environ.setdefault("SPARK_PROVIDER", "runpod")  # nommé dans chaque rappel et dans le résultat

import runpod  # noqa: E402

from spark_infer.service import process_job  # noqa: E402
from spark_infer.tasks import jobs_per_gpu  # noqa: E402

logging.basicConfig(level=os.environ.get("SPARK_LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("spark.runpod")


def handler(job: dict) -> dict:
    job_id = str(job.get("id", "local"))

    def progress(payload: dict) -> None:
        try:
            runpod.serverless.progress_update(job, payload)
        except Exception:  # noqa: BLE001 — la progression n'a jamais le droit de faire échouer un job
            pass

    return process_job(job.get("input") or {}, job_id, progress)


def concurrence(_actuelle: int) -> int:
    """Combien de jobs CE worker accepte EN MEME TEMPS.

    Sans cette fonction, RunPod n'envoie qu'un job par worker : une carte de 24 Go
    qui en tient cinq en ferait tourner un seul, et un worker a plusieurs cartes
    n'en utiliserait qu'une. C'est le seul endroit ou ca se decide.

    La valeur vient de la CARTE trouvee au demarrage, et elle est calculee sur la
    tache la PLUS GOURMANDE. Un endpoint sert les deux : une separation coute
    ~4,7 Go par job, une conversion vocale ~1,3 (mesures du 2026-09-12). Dimensionner
    sur la separation garde le worker sur meme s'il recoit un melange des deux ;
    l'inverse deborderait.

    DEFAUT INCHANGE : 1 tant que ni `SPARK_JOBS_PER_GPU` ni `SPARK_JOBS_AUTO` n'est
    pose sur l'endpoint. Et c'est volontaire dans les deux sens — sans l'un des deux,
    `jobs_per_gpu` rend 1, donc le POOL de modeles vaut 1 : accepter plusieurs jobs
    ne ferait que les faire attendre DANS le worker, en donnant l'illusion d'une
    concurrence. Les deux reglages vont ensemble.
    """
    return jobs_per_gpu("bs_roformer_leap_xe")


if __name__ == "__main__":
    n = concurrence(0)
    log.info("worker RunPod : %d job(s) en parallele%s", n,
             "" if n > 1 else " (poser SPARK_JOBS_AUTO=1 sur l'endpoint pour que la carte decide)")
    runpod.serverless.start({"handler": handler, "concurrency_modifier": concurrence})
