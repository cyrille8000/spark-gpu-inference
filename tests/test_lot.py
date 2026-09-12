"""Le LOT : une requete qui porte plusieurs sous-jobs, executes ensemble.

Ce qui compte ici n'est pas qu'ils passent, c'est qu'ils passent EN MEME TEMPS. La
plateforme n'a que 80 places simultanees chez ses hebergeurs et un job y occupait une
place entiere ; tout l'interet du lot est qu'un lot de vingt n'en occupe qu'une. Un
lot qui executerait ses sous-jobs a la file serait une regression invisible — d'ou le
test qui mesure le croisement reel.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spark_infer import registry, service, tasks  # noqa: E402


@pytest.fixture(autouse=True)
def _registre_propre():
    registry.reset()
    registry._cartes_cache = ["cpu"]
    yield
    registry.reset()


def _lot(n: int, task: str = "instrumental") -> dict:
    return {"jobs": [{"task": task, "audio_url": f"https://exemple/{i}.wav"} for i in range(n)]}


def test_le_lot_execute_ses_sous_jobs_en_meme_temps(monkeypatch):
    croisement = {"max": 0, "actuel": 0}
    verrou = threading.Lock()

    def faux_run_task(inp, job_id, progress):
        with verrou:
            croisement["actuel"] += 1
            croisement["max"] = max(croisement["max"], croisement["actuel"])
        time.sleep(0.15)
        with verrou:
            croisement["actuel"] -= 1
        return {"status": "completed", "task": inp.get("task"), "job_id": job_id}

    monkeypatch.setattr(service, "run_task", faux_run_task)
    monkeypatch.setattr(tasks.registry, "vram_total_gb", lambda: 24.0)

    out = service.process_lot(_lot(5), "lot-1")
    assert out["status"] == "completed" and out["lot"] is True
    assert out["total"] == 5 and out["reussis"] == 5 and out["echecs"] == 0
    assert len(out["resultats"]) == 5
    # 24 Go tiennent 5 separations : les cinq doivent s'etre croisees.
    assert out["places"] == 5
    assert croisement["max"] == 5, f"seulement {croisement['max']} sous-jobs croises"


def test_un_sous_job_qui_echoue_n_emporte_pas_les_autres(monkeypatch):
    def faux_run_task(inp, job_id, progress):
        if job_id.endswith("-001"):
            raise RuntimeError("carte fachee")
        return {"status": "completed", "task": inp.get("task"), "job_id": job_id}

    monkeypatch.setattr(service, "run_task", faux_run_task)
    monkeypatch.setattr(tasks.registry, "vram_total_gb", lambda: 24.0)

    out = service.process_lot(_lot(3), "lot-2")
    assert out["status"] == "completed"           # le LOT reussit : il a rendu compte
    assert out["reussis"] == 2 and out["echecs"] == 1
    rate = [r for r in out["resultats"] if r.get("status") == "error"]
    assert len(rate) == 1 and rate[0]["code"] == "internal"


def test_places_calculees_sur_la_tache_la_plus_gourmande(monkeypatch):
    monkeypatch.delenv("SPARK_JOBS_PER_GPU", raising=False)
    monkeypatch.setattr(tasks.registry, "vram_total_gb", lambda: 24.0)
    # Une carte de 24 Go tient 5 separations (mesure du 2026-09-12 sur L4) et 9
    # conversions vocales. Un lot melange doit se dimensionner sur la separation.
    assert tasks.places_pour_taches(["instrumental"]) == 5
    assert tasks.places_pour_taches(["vc"]) == 9
    assert tasks.places_pour_taches(["vc", "instrumental"]) == 5
    # Une tache inconnue est traitee en gourmande.
    assert tasks.places_pour_taches(["quoi"]) == 1


def test_places_multipliees_par_le_nombre_de_cartes(monkeypatch):
    monkeypatch.delenv("SPARK_JOBS_PER_GPU", raising=False)
    monkeypatch.setattr(tasks.registry, "vram_total_gb", lambda: 24.0)
    registry._cartes_cache = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    # 5 separations par carte x 4 cartes = 20. C'est le cas RunPod du 2026-09-12
    # (worker a 4 RTX A5000 de 25,3 Go).
    assert tasks.places_pour_taches(["instrumental"]) == 20


def test_le_lot_ouvre_les_places_au_pool_puis_les_referme(monkeypatch):
    """Sans ca, les sous-jobs s'attendraient sur une seule instance de modele : le lot
    aurait l'air de marcher et serait une file deguisee."""
    monkeypatch.delenv("SPARK_JOBS_AUTO", raising=False)
    monkeypatch.delenv("SPARK_JOBS_PER_GPU", raising=False)
    assert tasks.jobs_per_gpu("bs_roformer_leap_xe") == 1   # hors lot : inchange
    with tasks.places_du_lot(7):
        assert tasks.jobs_per_gpu("bs_roformer_leap_xe") == 7
        assert tasks.jobs_per_gpu("chatterbox_vc") == 7
    assert tasks.jobs_per_gpu("bs_roformer_leap_xe") == 1   # refermé


def test_lot_malforme_refuse_sans_rien_executer(monkeypatch):
    appels = []
    monkeypatch.setattr(service, "run_task", lambda *a, **k: appels.append(1))
    for mauvais in ({"jobs": []}, {"jobs": "trois"}, {"jobs": [1, 2]},
                    {"jobs": [{"task": "vc"}] * (service.MAX_SOUS_JOBS + 1)}):
        out = service.process_lot(mauvais, "lot-3")
        assert out["status"] == "error" and out["code"] == "bad_input"
    assert appels == []


def test_process_job_reconnait_un_lot(monkeypatch):
    """Aucun hebergeur n'a rien a savoir : `handler.py` et `modal_app.py` appellent
    `process_job` comme avant, la cle `jobs` suffit a basculer."""
    monkeypatch.setattr(service, "run_task",
                        lambda inp, job_id, progress: {"status": "completed", "job_id": job_id})
    monkeypatch.setattr(tasks.registry, "vram_total_gb", lambda: 24.0)
    out = service.process_job(_lot(2), "lot-4")
    assert out.get("lot") is True and out["total"] == 2


def test_deux_sous_jobs_ne_peuvent_pas_ecrire_au_meme_endroit(monkeypatch):
    """Le seul endroit ou un lot pourrait ecraser un resultat : deux `output_url`
    identiques. Le PUT du second effacerait le premier et les DEUX diraient
    « completed ». Le reste est deja isole (dossier de travail unique par job,
    exemplaire de modele propre a chaque job)."""
    appels = []
    monkeypatch.setattr(service, "run_task", lambda *a, **k: appels.append(1))
    lot = {"jobs": [
        {"task": "instrumental", "audio_url": "https://exemple/a.wav", "output_url": "https://r2/out.wav"},
        {"task": "instrumental", "audio_url": "https://exemple/b.wav", "output_url": "https://r2/out.wav"},
    ]}
    out = service.process_lot(lot, "lot-5")
    assert out["status"] == "error" and out["code"] == "bad_input"
    assert "ecraseraient" in out["error"].replace("é", "e") or "craseraient" in out["error"]
    assert appels == [], "aucun sous-job ne doit avoir tourne"


def test_des_sorties_distinctes_passent(monkeypatch):
    monkeypatch.setattr(service, "run_task",
                        lambda inp, job_id, progress: {"status": "completed", "job_id": job_id})
    monkeypatch.setattr(tasks.registry, "vram_total_gb", lambda: 24.0)
    lot = {"jobs": [
        {"task": "instrumental", "audio_url": "https://exemple/a.wav", "output_url": "https://r2/a.wav"},
        {"task": "instrumental", "audio_url": "https://exemple/b.wav", "output_url": "https://r2/b.wav"},
    ]}
    assert service.process_lot(lot, "lot-6")["reussis"] == 2

