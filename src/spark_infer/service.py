"""Le traitement d'UN job, indépendant de la plateforme qui l'héberge.

RunPod (`handler.py`) et Modal (`modal_app.py`) appellent la même fonction : parsing,
exécution, erreurs typées, rappels `started` / `heartbeat` / `finished` vers
`callback_url`. Une seule définition du comportement, deux façons d'être invoqué.
"""
from __future__ import annotations

import logging
import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from . import registry
from .container_clock import CLOCK
from .io_utils import InputError, check_url
from .params import parse_callback, parse_heartbeat_s, parse_meta
from .tasks import places_du_lot, places_pour_taches, run_task
from .webhooks import Heartbeat, JobWebhooks, now_iso

log = logging.getLogger("spark.service")

Progress = Callable[[dict], None]


# Un lot plus gros que ça est refusé : ce n'est plus un lot, c'est une file.
MAX_SOUS_JOBS = 256

# ── MODE PRISE (`claim_url`) ──────────────────────────────────────────────────
# Jobs pris PAR CARTE. Décidé par le propriétaire le 2026-09-12 : deux. La mémoire
# en permettrait davantage (5 séparations tiennent sur une carte de 24 Go), mais
# deux garde une marge confortable et rend la capacité lisible — quatre cartes
# donnent huit, trois donnent six.
JOBS_PAR_CARTE = 2
# Temps de travail que s'accorde le worker si le lancement n'en impose pas. Reste
# très en dessous des coupures des hébergeurs (600 s chez RunPod, 900 s chez Modal) :
# le worker doit rendre la main de lui-même, jamais se faire couper en pleine vague.
BUDGET_DEFAUT_S = 420.0
# En dessous de ça, inutile de redemander : on n'aurait pas le temps de finir.
PLANCHER_S = 45.0
MAX_VAGUES = 50


def avancement(lot_id: str, faits: int, total: int, places: int, t0: float) -> dict:
    """Ou en est le LOT, et combien de temps il lui reste — lisible du dehors.

    Sans ça, l'extérieur ne voit que l'avancement d'UN sous-job (`separation
    BS-Roformer, 10 %`), ce qui ne dit rien d'un lot de huit. Celui qui décide de
    lancer ou d'arrêter des workers a besoin de l'état du LOT.

    L'estimation raisonne en VAGUES et non en jobs : `places` sous-jobs tournent de
    front, donc ce qui reste à faire, ce sont des vagues entières. Estimer avec une
    moyenne par job diviserait le reste par le parallélisme et annoncerait toujours
    trop tôt.

    Le chemin de sortie existe déjà chez les trois hébergeurs : `progress_update` chez
    RunPod (lisible par `/status`), le heartbeat vers `gpu-event` chez Modal et Vast.
    """
    ecoule = max(0.0, time.monotonic() - t0)
    places = max(1, places)
    vagues_faites = max(1, -(-faits // places))
    par_vague = ecoule / vagues_faites
    vagues_restantes = -(-(total - faits) // places)
    return {
        "lot": lot_id, "total": total, "faits": faits, "restants": total - faits,
        "places": places, "percent": round(100 * faits / max(1, total)),
        "ecoule_s": round(ecoule, 1), "restant_s": round(vagues_restantes * par_vague, 1),
        "message": f"lot : {faits}/{total} sous-job(s)",
    }


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

    faits = {"n": 0}
    verrou = threading.Lock()
    dire = progress or (lambda _p: None)
    dire(avancement(job_id, 0, len(sous), places, t0))

    def un(i_et_job: tuple[int, dict]) -> dict:
        i, s = i_et_job
        out = process_job(s, f"{job_id}-{i:03d}", progress)
        # L'avancement du LOT est publié à chaque sous-job fini : c'est le seul
        # moment où le reste à faire change vraiment.
        with verrou:
            faits["n"] += 1
            dire(avancement(job_id, faits["n"], len(sous), places, t0))
        return out

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


def _resume(r: dict) -> dict:
    """Ce qu'on renvoie au serveur pour CHAQUE sous-job : de quoi libérer sa
    réservation et rejouer ce qui a échoué. Le résultat complet est déjà parti par le
    `callback_url` du sous-job — inutile de le renvoyer deux fois."""
    return {
        "job_id": r.get("job_id"), "status": r.get("status"), "task": r.get("task"),
        "error": r.get("error"), "code": r.get("code"),
        "elapsed_s": r.get("elapsed_s"), "container_s": r.get("container_s"),
        "device": r.get("device"), "bytes": r.get("bytes"), "meta": r.get("meta"),
    }


def _demander(url: str, corps: dict, essais: int = 2) -> dict:
    """Demande du travail au serveur. L'URL est SIGNÉE : elle porte son autorisation,
    le worker n'a donc aucun secret à connaître."""
    import requests
    dernier = ""
    for n in range(1, essais + 1):
        try:
            r = requests.post(url, json=corps, timeout=60)
            if r.status_code < 300:
                return r.json() if r.content else {}
            dernier = f"{r.status_code} {r.text[:200]}"
            if 400 <= r.status_code < 500 and r.status_code != 429:
                break
        except Exception as e:  # noqa: BLE001
            dernier = f"{type(e).__name__}: {e}"
        time.sleep(2 * n)
    log.error("prise de travail refusée : %s", dernier)
    return {"erreur": dernier}


def process_pull(inp: dict, job_id: str, progress: Progress | None = None) -> dict:
    """Le worker VA CHERCHER son travail au lieu qu'on le lui pousse.

    Le problème que ça résout : on demande quatre cartes à RunPod et on en reçoit
    parfois trois — mesuré le 2026-09-12, leur propre contrôle de démarrage le dit.
    Le serveur ne peut donc pas savoir combien de jobs envoyer. Le worker, lui, sait :
    il compte ses cartes au démarrage et prend `JOBS_PAR_CARTE` fois ce nombre.

    Le même mécanisme vaut pour Modal, RunPod et Vast.ai. Rien à régler chez
    l'hébergeur : la seule chose passée au lancement est `claim_url`, l'URL signée où
    aller demander du travail.

    La boucle demande, exécute la vague, et renvoie ses résultats AVEC la demande
    suivante. Elle s'arrête dès que le serveur ne donne plus rien — on ne sonde JAMAIS
    en attendant du travail, parce qu'un worker qui attend est facturé à la
    milliseconde, cartes comprises. Elle s'arrête aussi quand le budget ne permet plus
    une vague de plus : un worker coupé en pleine vague perd tout son travail.
    """
    try:
        url = check_url(inp.get("claim_url"), "claim_url")
    except InputError as e:
        return {"status": "error", "code": "bad_input", "job_id": job_id, "error": str(e)}

    cartes = registry.devices()
    par_carte = max(1, int(inp.get("jobs_par_carte") or JOBS_PAR_CARTE))
    capacite = par_carte * len(cartes)
    budget = max(PLANCHER_S, float(inp.get("budget_s") or BUDGET_DEFAUT_S))
    debut = time.monotonic()
    identite = {
        "worker": job_id, "provider": os.environ.get("SPARK_PROVIDER", ""),
        "gpu_name": registry.gpu_name(), "cartes": cartes, "capacite": capacite,
    }
    log.info("[%s] PRISE : %d carte(s) x %d = %d place(s), budget %.0f s",
             job_id, len(cartes), par_carte, capacite, budget)

    vagues: list[dict] = []
    a_renvoyer: list[dict] = []
    derniere_duree = 0.0
    raison = "file vide"
    for n in range(1, MAX_VAGUES + 1):
        restant = budget - (time.monotonic() - debut)
        # Assez de temps pour une vague de plus ? On se fie à la précédente, majorée.
        if restant < max(PLANCHER_S, derniere_duree * 1.25):
            raison = f"budget épuisé ({restant:.0f} s restantes)"
            break
        rep = _demander(url, {**identite, "vague": n, "restant_s": round(restant, 1),
                              "resultats": a_renvoyer})
        a_renvoyer = []
        if rep.get("erreur"):
            raison = f"serveur injoignable : {rep['erreur']}"
            break
        jobs = rep.get("jobs") or []
        if not jobs:
            raison = "file vide"
            break
        t = time.monotonic()
        lot = process_lot({"jobs": jobs}, f"{job_id}-v{n:02d}", progress)
        derniere_duree = time.monotonic() - t
        a_renvoyer = [_resume(r) for r in (lot.get("resultats") or [])]
        vagues.append({"vague": n, "demandes": len(jobs), "reussis": lot.get("reussis", 0),
                       "echecs": lot.get("echecs", 0), "s": round(derniere_duree, 1)})
        log.info("[%s] vague %d : %d/%d en %.1f s", job_id, n, lot.get("reussis", 0),
                 len(jobs), derniere_duree)
        if lot.get("status") != "completed":
            raison = f"lot refusé : {lot.get('error')}"
            break

    # Les résultats de la DERNIÈRE vague n'ont pas encore été renvoyés : sans ce
    # dernier envoi, le serveur garderait leurs réservations jusqu'à expiration.
    if a_renvoyer:
        _demander(url, {**identite, "vague": 0, "restant_s": 0, "fin": True,
                        "resultats": a_renvoyer})

    total = sum(v["demandes"] for v in vagues)
    reussis = sum(v["reussis"] for v in vagues)
    log.info("[%s] PRISE terminée : %d vague(s), %d/%d sous-job(s), %.0f s — %s",
             job_id, len(vagues), reussis, total, time.monotonic() - debut, raison)
    return {
        "status": "completed", "prise": True, "job_id": job_id,
        "vagues": vagues, "total": total, "reussis": reussis, "echecs": total - reussis,
        "capacite": capacite, "cartes": cartes, "gpu_name": registry.gpu_name(),
        "arret": raison, "elapsed_s": round(time.monotonic() - debut, 3),
        "provider": os.environ.get("SPARK_PROVIDER", ""),
    }


def process_job(inp: dict, job_id: str, progress: Progress | None = None) -> dict:
    """Exécute le job et renvoie le JSON de sortie (succès ou erreur typée). Ne lève jamais."""
    # Trois formes, reconnues à la charge utile — aucun hébergeur n'a rien à savoir de
    # plus, `handler.py` et `modal_app.py` restent inchangés :
    #   `claim_url` → le worker va CHERCHER son travail (lui seul connaît ses cartes) ;
    #   `jobs`      → un lot qu'on lui pousse ;
    #   sinon       → un job unique, comme depuis toujours.
    if inp.get("claim_url"):
        return process_pull(inp, job_id, progress)
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
