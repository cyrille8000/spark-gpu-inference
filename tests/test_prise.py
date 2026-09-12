"""Le MODE PRISE : le worker va chercher son travail, et reste PLEIN.

Il resout un probleme mesure le 2026-09-12 : on demande quatre cartes a RunPod et on
en recoit parfois trois. Le serveur ne peut donc pas savoir combien de jobs envoyer ;
le worker, lui, compte ses cartes.

Ce qui doit rester vrai, et que ces tests gardent :
  - la capacite suit le NOMBRE DE CARTES REELLEMENT TROUVEES ;
  - les places sont TOUJOURS PLEINES : des qu'un sous-job finit, sa place est reprise.
    Sinon les places liberees par les jobs rapides attendent le plus lent en etant
    facturees — mesure le 2026-09-12 sur RunPod : quatre places inoccupees pendant
    66 s sur une vague de 143 s, un quart du temps paye pour rien ;
  - une prise vide fait SORTIR le worker (il ne sonde pas : un worker qui attend est
    facture a la milliseconde, cartes comprises) ;
  - il cesse de reprendre avant son budget plutot que de se faire couper ;
  - tous les resultats repartent, y compris les derniers.
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


class FauxServeur:
    """Une file de `stock` jobs. Rend au plus ce que le worker dit avoir de libre."""

    def __init__(self, stock: int, task: str = "instrumental"):
        self.stock = stock
        self.task = task
        self.demandes: list[dict] = []
        self.rendus: list[dict] = []

    def __call__(self, url: str, corps: dict, essais: int = 2) -> dict:
        self.demandes.append(corps)
        self.rendus.extend(corps.get("resultats") or [])
        if corps.get("fin"):
            return {}
        n = min(int(corps.get("libres") or 0), self.stock)
        self.stock -= n
        return {"jobs": [{"task": self.task, "audio_url": f"https://exemple/{i}.wav"}
                         for i in range(n)]}


def _prepare(monkeypatch, serveur, cartes, duree=0.0, durees_par_job=None):
    monkeypatch.setattr(service, "_demander", serveur)
    monkeypatch.setattr(tasks.registry, "vram_total_gb", lambda: 24.0)
    registry._cartes_cache = list(cartes)
    compte = {"n": 0}
    verrou = threading.Lock()

    def faux_run_task(inp, job_id, progress):
        with verrou:
            i = compte["n"]
            compte["n"] += 1
        d = durees_par_job[i % len(durees_par_job)] if durees_par_job else duree
        if d:
            time.sleep(d)
        return {"status": "completed", "task": inp.get("task"), "job_id": job_id}

    monkeypatch.setattr(service, "run_task", faux_run_task)


def test_la_capacite_suit_les_cartes_trouvees(monkeypatch):
    """Quatre cartes tiennent huit places, trois en tiennent six. C'est LE point : le
    serveur n'a pas a deviner ce que l'hebergeur a livre."""
    for cartes, attendu in ((["cuda:0", "cuda:1", "cuda:2", "cuda:3"], 8),
                            (["cuda:0", "cuda:1", "cuda:2"], 6),
                            (["cuda:0"], 2)):
        srv = FauxServeur(0)
        _prepare(monkeypatch, srv, cartes)
        out = service.process_pull({"claim_url": "https://serveur/prise?sig=x"}, "w1")
        assert out["places"] == attendu, f"{len(cartes)} cartes -> {out['places']}"
        assert srv.demandes[0]["places"] == attendu
        assert srv.demandes[0]["libres"] == attendu
        assert srv.demandes[0]["cartes"] == cartes


def test_les_places_restent_pleines(monkeypatch):
    """LE test du remplissage continu. Des durees tres inegales : sans reprise
    immediate, les places des jobs courts attendraient le plus long."""
    srv = FauxServeur(12)
    # Un job sur quatre est cinq fois plus long que les autres.
    _prepare(monkeypatch, srv, ["cuda:0", "cuda:1"],
             durees_par_job=[0.05, 0.05, 0.05, 0.25])

    out = service.process_pull({"claim_url": "https://serveur/prise?sig=x"}, "w2")

    assert out["total"] == 12 and out["reussis"] == 12
    # La premiere demande prend les 4 places d'un coup ; ensuite on ne redemande QUE
    # ce qui vient de se liberer, donc jamais les 4 a la fois.
    assert srv.demandes[0]["libres"] == 4
    assert len(srv.demandes) > 4, "il n'a pas repris place par place"
    suivantes = [int(d.get("libres") or 0) for d in srv.demandes[1:] if not d.get("fin")]
    assert suivantes and max(suivantes) < 4, \
        "il a laisse toutes ses places se vider avant de redemander"


def test_une_prise_vide_fait_sortir_le_worker(monkeypatch):
    """Il ne sonde PAS en attendant : un worker qui attend est facture."""
    srv = FauxServeur(0)
    _prepare(monkeypatch, srv, ["cuda:0", "cuda:1"])
    out = service.process_pull({"claim_url": "https://serveur/prise?sig=x"}, "w3")
    assert out["status"] == "completed" and out["total"] == 0
    assert out["arret"] == "file vide"
    assert len(srv.demandes) == 1, "une seule demande, puis on sort"


def test_il_vide_la_file_puis_sort(monkeypatch):
    srv = FauxServeur(10)
    _prepare(monkeypatch, srv, ["cuda:0", "cuda:1"])
    out = service.process_pull({"claim_url": "https://serveur/prise?sig=x"}, "w4")
    assert out["total"] == 10 and out["reussis"] == 10
    assert out["arret"] == "file vide"
    assert len(srv.rendus) == 10, f"{len(srv.rendus)} resultats rendus au lieu de 10"


def test_il_cesse_de_reprendre_avant_son_budget(monkeypatch):
    """Un job coupe en cours perd tout ce qu'il a calcule : le worker cesse de
    reprendre, laisse finir ce qui tourne, et sort."""
    srv = FauxServeur(200)
    _prepare(monkeypatch, srv, ["cuda:0"], duree=0.05)
    out = service.process_pull(
        {"claim_url": "https://serveur/prise?sig=x", "budget_s": 1}, "w5")
    assert "budget" in out["arret"]
    assert out["elapsed_s"] < 3.0, "il a depasse son budget"
    assert srv.stock > 0, "il aurait du s'arreter avant d'avoir tout pris"
    assert len(srv.rendus) == out["total"], "des resultats n'ont pas ete rendus"


def test_le_budget_par_defaut_ne_mord_pas(monkeypatch):
    """Ce qui arrete le worker, c'est la FILE VIDE. Le budget n'est qu'un garde-fou :
    avec les plafonds des hebergeurs portes a 5 heures, un worker qui sortirait au
    bout de quelques minutes alors qu'il reste du travail obligerait un autre a
    redemarrer a froid pour rien."""
    srv = FauxServeur(40)
    _prepare(monkeypatch, srv, ["cuda:0"])
    out = service.process_pull({"claim_url": "https://serveur/prise?sig=x"}, "w6")
    assert out["arret"] == "file vide", out["arret"]
    assert out["total"] == 40, "il a lache du travail alors qu'il en restait"
    assert service.BUDGET_DEFAUT_S > 4 * 3600


def test_un_serveur_injoignable_fait_sortir(monkeypatch):
    _prepare(monkeypatch, lambda url, corps, essais=2: {"erreur": "HTTP 503"},
             ["cuda:0", "cuda:1"])
    out = service.process_pull({"claim_url": "https://serveur/prise?sig=x"}, "w7")
    assert out["status"] == "completed" and out["total"] == 0
    assert "injoignable" in out["arret"]


def test_claim_url_invalide_refuse(monkeypatch):
    for mauvais in ({}, {"claim_url": ""}, {"claim_url": "pas-une-url"}):
        out = service.process_pull(mauvais, "w8")
        assert out["status"] == "error" and out["code"] == "bad_input"


def test_process_job_reconnait_la_prise(monkeypatch):
    """Aucun hebergeur n'a rien a savoir : `handler.py` et `modal_app.py` appellent
    `process_job` comme avant, la cle `claim_url` suffit a basculer."""
    srv = FauxServeur(2)
    _prepare(monkeypatch, srv, ["cuda:0"])
    out = service.process_job({"claim_url": "https://serveur/prise?sig=x"}, "w9")
    assert out.get("prise") is True and out["total"] == 2
