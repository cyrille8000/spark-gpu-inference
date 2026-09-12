"""Le MODE PRISE : le worker va chercher son travail au lieu qu'on le lui pousse.

Il resout un probleme mesure le 2026-09-12 : on demande quatre cartes a RunPod et on
en recoit parfois trois. Le serveur ne peut donc pas savoir combien de jobs envoyer ;
le worker, lui, compte ses cartes.

Ce qui doit rester vrai, et que ces tests gardent :
  - la capacite annoncee suit le NOMBRE DE CARTES REELLEMENT TROUVEES ;
  - une prise vide fait SORTIR le worker (il ne sonde pas : un worker qui attend est
    facture a la milliseconde, cartes comprises) ;
  - il s'arrete avant son budget plutot que de se faire couper en pleine vague ;
  - les resultats de la DERNIERE vague sont renvoyes, sans quoi le serveur garderait
    leurs reservations jusqu'a expiration.
"""
from __future__ import annotations

import sys
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
    """Sert des vagues preparees, et garde ce que le worker lui renvoie."""

    def __init__(self, vagues: list[int], task: str = "instrumental"):
        self.a_servir = list(vagues)
        self.task = task
        self.demandes: list[dict] = []
        self.rendus: list[dict] = []

    def __call__(self, url: str, corps: dict, essais: int = 2) -> dict:
        self.demandes.append(corps)
        self.rendus.extend(corps.get("resultats") or [])
        if corps.get("fin"):
            return {}
        if not self.a_servir:
            return {"jobs": []}
        n = self.a_servir.pop(0)
        return {"jobs": [{"task": self.task, "audio_url": f"https://exemple/{i}.wav"}
                         for i in range(n)]}


def _prepare(monkeypatch, serveur, cartes, duree=0.0):
    import time as _t
    monkeypatch.setattr(service, "_demander", serveur)
    monkeypatch.setattr(tasks.registry, "vram_total_gb", lambda: 24.0)
    registry._cartes_cache = list(cartes)

    def faux_run_task(inp, job_id, progress):
        if duree:
            _t.sleep(duree)
        return {"status": "completed", "task": inp.get("task"), "job_id": job_id}

    monkeypatch.setattr(service, "run_task", faux_run_task)


def test_la_capacite_suit_les_cartes_trouvees(monkeypatch):
    """Quatre cartes annoncent huit places, trois en annoncent six. C'est LE point :
    le serveur n'a pas a deviner ce que l'hebergeur a livre."""
    for cartes, attendu in ((["cuda:0", "cuda:1", "cuda:2", "cuda:3"], 8),
                            (["cuda:0", "cuda:1", "cuda:2"], 6),
                            (["cuda:0"], 2)):
        srv = FauxServeur([])
        _prepare(monkeypatch, srv, cartes)
        out = service.process_pull({"claim_url": "https://serveur/prise?sig=x"}, "w1")
        assert out["capacite"] == attendu, f"{len(cartes)} cartes -> {out['capacite']}"
        assert srv.demandes[0]["capacite"] == attendu
        assert srv.demandes[0]["cartes"] == cartes


def test_une_prise_vide_fait_sortir_le_worker(monkeypatch):
    """Il ne sonde PAS en attendant : un worker qui attend est facture."""
    srv = FauxServeur([])
    _prepare(monkeypatch, srv, ["cuda:0", "cuda:1"])
    out = service.process_pull({"claim_url": "https://serveur/prise?sig=x"}, "w2")
    assert out["status"] == "completed" and out["total"] == 0
    assert out["arret"] == "file vide"
    assert len(srv.demandes) == 1, "une seule demande, puis on sort"


def test_il_enchaine_les_vagues_puis_sort(monkeypatch):
    srv = FauxServeur([4, 4, 2])
    _prepare(monkeypatch, srv, ["cuda:0", "cuda:1"])
    out = service.process_pull({"claim_url": "https://serveur/prise?sig=x"}, "w3")
    assert [v["demandes"] for v in out["vagues"]] == [4, 4, 2]
    assert out["total"] == 10 and out["reussis"] == 10
    assert out["arret"] == "file vide"


def test_les_resultats_de_la_derniere_vague_repartent(monkeypatch):
    """Sans ce dernier envoi, le serveur garderait les reservations de la derniere
    vague jusqu'a leur expiration."""
    srv = FauxServeur([3])
    _prepare(monkeypatch, srv, ["cuda:0"])
    out = service.process_pull({"claim_url": "https://serveur/prise?sig=x"}, "w4")
    assert out["total"] == 3
    assert len(srv.rendus) == 3, f"{len(srv.rendus)} resultats rendus au lieu de 3"
    assert all(r["status"] == "completed" for r in srv.rendus)
    # Ils sont partis AVEC la demande suivante, pas dans un envoi separe : une seule
    # requete fait les deux, « voici ce que j'ai fait, donne m'en encore ».
    assert srv.demandes[1]["resultats"] and not srv.demandes[1].get("fin")


def test_l_envoi_final_sert_quand_la_boucle_s_arrete_d_elle_meme(monkeypatch):
    """Si le worker sort sur son budget, la derniere vague n'a pas de demande suivante
    ou se glisser : il faut un dernier envoi, sinon le serveur garderait ses
    reservations jusqu'a expiration."""
    srv = FauxServeur([2] * 20)
    _prepare(monkeypatch, srv, ["cuda:0"], duree=0.05)
    out = service.process_pull(
        {"claim_url": "https://serveur/prise?sig=x", "budget_s": 1}, "w4b")
    assert "budget" in out["arret"]
    assert srv.demandes[-1].get("fin") is True
    assert srv.demandes[-1]["resultats"], "la derniere vague n'a pas ete rendue"
    assert len(srv.rendus) == out["total"]


def test_il_s_arrete_avant_son_budget(monkeypatch):
    """Un worker coupe en pleine vague perd TOUT son travail : il doit rendre la main
    de lui-meme."""
    srv = FauxServeur([2] * 20)
    _prepare(monkeypatch, srv, ["cuda:0"], duree=0.05)
    out = service.process_pull(
        {"claim_url": "https://serveur/prise?sig=x", "budget_s": 1}, "w5")
    assert "budget" in out["arret"]
    assert out["elapsed_s"] < 3.0, "il a depasse son budget"
    assert srv.a_servir, "il aurait du s'arreter avant d'avoir tout pris"


def test_un_serveur_injoignable_fait_sortir(monkeypatch):
    def refuse(url, corps, essais=2):
        return {"erreur": "HTTP 503"}

    _prepare(monkeypatch, refuse, ["cuda:0", "cuda:1"])
    out = service.process_pull({"claim_url": "https://serveur/prise?sig=x"}, "w6")
    assert out["status"] == "completed" and out["total"] == 0
    assert "injoignable" in out["arret"]


def test_claim_url_invalide_refuse(monkeypatch):
    for mauvais in ({}, {"claim_url": ""}, {"claim_url": "pas-une-url"}):
        out = service.process_pull(mauvais, "w7")
        assert out["status"] == "error" and out["code"] == "bad_input"


def test_process_job_reconnait_la_prise(monkeypatch):
    """Aucun hebergeur n'a rien a savoir : `handler.py` et `modal_app.py` appellent
    `process_job` comme avant, la cle `claim_url` suffit a basculer."""
    srv = FauxServeur([2])
    _prepare(monkeypatch, srv, ["cuda:0"])
    out = service.process_job({"claim_url": "https://serveur/prise?sig=x"}, "w8")
    assert out.get("prise") is True and out["total"] == 2
