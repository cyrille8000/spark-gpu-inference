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
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from . import registry
from .container_clock import CLOCK
from .io_utils import InputError
from .params import parse_callback, parse_heartbeat_s, parse_meta
from .tasks import places_du_lot, places_pour_taches, run_task
from .webhooks import Heartbeat, JobWebhooks, now_iso

log = logging.getLogger("spark.service")

Progress = Callable[[dict], None]


# Un lot plus gros que ça est refusé : ce n'est plus un lot, c'est une file.
MAX_SOUS_JOBS = 256


def process_lot(inp: dict, job_id: str, progress: Progress | None = None) -> dict:
    """Un LOT : une seule requête qui porte plusieurs sous-jobs, exécutés ensemble ici.

    C'est la réponse au vrai problème : la plateforme n'a que 80 places simultanées
    chez ses hébergeurs, et un job y occupait une place entière. Un lot de vingt
    séparations n'en occupe qu'UNE.

    Pourquoi ici et pas dans un réglage d'hébergeur : `concurrency_modifier` chez
    RunPod, `@modal.concurrent` chez Modal et les variables d'environnement qui vont
    avec sont trois mécanismes différents à régler et à redéployer séparément. Le lot
    est dans la charge utile : c'est l'appelant qui décide de sa taille, sans que
    personne ne reconstruise l'image.

    Chaque sous-job garde SES propres `callback_url` et `output_url` : la plateforme
    reçoit `started` / `heartbeat` / `finished` par sous-job comme avant, et le
    résultat du lot n'est qu'un récapitulatif. Un sous-job qui échoue n'emporte pas
    les autres — son erreur typée est dans sa ligne, et l'appelant le rejoue.

    Le nombre de sous-jobs qui tournent EN MÊME TEMPS vient de la carte et de la tâche
    la plus gourmande (`places_pour_taches`) ; le reste attend son tour dans le lot.
    """
    sous = inp.get("jobs")
    if not isinstance(sous, list) or not sous:
        return {"status": "error", "code": "bad_input", "job_id": job_id,
                "error": "`jobs` doit être une liste non vide de sous-jobs"}
    if len(sous) > MAX_SOUS_JOBS:
        return {"status": "error", "code": "bad_input", "job_id": job_id,
                "error": f"lot de {len(sous)} sous-jobs : maximum {MAX_SOUS_JOBS}"}
    if any(not isinstance(s, dict) for s in sous):
        return {"status": "error", "code": "bad_input", "job_id": job_id,
                "error": "chaque sous-job doit être un objet"}
    # DEUX SOUS-JOBS NE DOIVENT PAS ÉCRIRE AU MÊME ENDROIT. Le reste est déjà isolé —
    # dossier de travail unique par job (`tempfile.mkdtemp`), exemplaire de modèle
    # propre à chaque job (`registry.lease`) — mais deux `output_url` identiques se
    # recouvriraient sur R2, le second effaçant le premier, et les DEUX rendraient
    # « completed ». Une erreur silencieuse, donc refusée ici.
    sorties = [s.get("output_url") for s in sous if s.get("output_url")]
    if len(set(sorties)) != len(sorties):
        doublons = sorted({u for u in sorties if sorties.count(u) > 1})
        return {"status": "error", "code": "bad_input", "job_id": job_id,
                "error": f"{len(doublons)} `output_url` en double dans le lot : les sous-jobs "
                         f"s'écraseraient sans le dire"}

    places = min(len(sous), places_pour_taches([str(s.get("task") or "") for s in sous]))
    t0 = time.monotonic()
    log.info("[%s] LOT de %d sous-job(s), %d en parallèle sur %s", job_id, len(sous), places,
             ", ".join(registry.devices()))

    def un(i_et_job: tuple[int, dict]) -> dict:
        i, s = i_et_job
        return process_job(s, f"{job_id}-{i:03d}", progress)

    # `max_workers` borne vraiment le parallélisme : le pool de modèles ne sert alors
    # plus qu'à donner son exemplaire à chacun, pas à faire la file.
    with places_du_lot(places), ThreadPoolExecutor(max_workers=places) as ex:
        resultats = list(ex.map(un, enumerate(sous)))

    reussis = sum(1 for r in resultats if r.get("status") == "completed")
    return {
        "status": "completed", "lot": True, "job_id": job_id,
        "total": len(sous), "reussis": reussis, "echecs": len(sous) - reussis,
        "places": places, "cartes": registry.devices(),
        "gpu_name": registry.gpu_name(), "device": registry.device(),
        "elapsed_s": round(time.monotonic() - t0, 3),
        "provider": os.environ.get("SPARK_PROVIDER", ""),
        "resultats": resultats,
    }


def process_job(inp: dict, job_id: str, progress: Progress | None = None) -> dict:
    """Exécute le job et renvoie le JSON de sortie (succès ou erreur typée). Ne lève jamais."""
    # Un lot se reconnaît à sa clé `jobs` : aucun hébergeur n'a rien à savoir de plus,
    # `handler.py` et `modal_app.py` restent inchangés.
    if isinstance(inp.get("jobs"), list):
        return process_lot(inp, job_id, progress)
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
