"""Le MODE PRISE : le worker va chercher son travail, reste PLEIN, et ne sort que sur ordre.

Il resout un probleme mesure le 2026-09-12 : on demande quatre cartes a RunPod et on
en recoit parfois trois. Le serveur ne peut donc pas savoir combien de jobs envoyer ;
le worker, lui, compte ses cartes.

Ce qui doit rester vrai, et que ces tests gardent :
  - la capacite suit le NOMBRE DE CARTES REELLEMENT TROUVEES ;
  - les places sont TOUJOURS PLEINES : des qu'un sous-job finit, sa place est reprise.
    Sinon les places liberees par les jobs rapides attendent le plus lent en etant
    facturees — mesure le 2026-09-12 sur RunPod : quatre places inoccupees pendant
    66 s sur une vague de 143 s, un quart du temps paye pour rien ;
  - FILE VIDE = on ATTEND ce que le serveur dit (`attente_s`) puis on redemande. On ne
    sort que sur l'ordre `arret` (decision du proprietaire, 2026-09-15 : c'est
    l'ordonnanceur qui monte et qui descend, les hebergeurs sont regles avec des
    coupures enormes) ;
  - `arret: net` rend d'abord ce qui est fini et la liste des jobs abandonnes, puis
    sort sans attendre ;
  - l'identite (instance, image, demarrage) et l'avancement de CHAQUE job en cours
    voyagent avec chaque demande : c'est avec ca que le serveur decide de couper ;
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
    """Une file de `stock` jobs. Rend au plus ce que le worker dit avoir de libre ;
    sur file vide, dit d'attendre `attente_s`, puis `arret: doux` a la `vides_avant_arret`-ieme
    demande vide. `net_a` : la n-ieme demande recoit `arret: net`."""

    def __init__(self, stock: int, task: str = "instrumental", attente_s: float = 0.05,
                 vides_avant_arret: int = 1, net_a: int | None = None, ids: bool = False):
        self.stock = stock
        self.task = task
        self.attente_s = attente_s
        self.vides_avant_arret = vides_avant_arret
        self.net_a = net_a
        self.ids = ids
        self.numero = 0
        self.vides = 0
        self.fins = 0
        self.demandes: list[dict] = []
        self.rendus: list[dict] = []
        self.abandonnes: list[str] = []

    def __call__(self, url: str, corps: dict, essais: int = 2) -> dict:
        self.demandes.append(corps)
        self.rendus.extend(corps.get("resultats") or [])
        self.abandonnes.extend(corps.get("abandonnes") or [])
        if corps.get("fin"):
            self.fins += 1
            return {}
        if corps.get("battement"):
            return {}
        if self.net_a is not None and len(self.demandes) >= self.net_a:
            return {"arret": "net"}
        n = min(int(corps.get("libres") or 0), self.stock)
        self.stock -= n
        if n == 0:
            self.vides += 1
            if self.vides >= self.vides_avant_arret:
                return {"arret": "doux", "attente_s": self.attente_s}
            return {"attente_s": self.attente_s}
        jobs = []
        for _ in range(n):
            self.numero += 1
            j = {"task": self.task, "audio_url": f"https://exemple/{self.numero}.wav"}
            if self.ids:
                j["id"] = f"srv-{self.numero}"
            jobs.append(j)
        return {"jobs": jobs}


class Sortie(Exception):
    def __init__(self, code: int):
        super().__init__(f"sortie {code}")
        self.code = code


def _prepare(monkeypatch, serveur, cartes, duree=0.0, durees_par_job=None, places=2):
    monkeypatch.setattr(service, "_demander", serveur)
    # Les tests de MECANIQUE (boucle, attente, arret, rendus) raisonnent sur 2 places par
    # carte ; seuls les tests de capacite passent `places=None` pour la vraie regle memoire.
    if places is not None:
        monkeypatch.setattr(service, "places_prise", lambda vram: places)
    monkeypatch.setattr(tasks.registry, "vram_total_gb", lambda: 24.0)
    monkeypatch.setattr(registry, "vram_gb", lambda c: 24.0)
    # `vast_worker.py` pose SPARK_PROVIDER=vastai a l'import, et pytest importe TOUS les
    # modules de test avant de les executer : sans ceci, l'homme-mort « se tait au lieu de
    # se tuer » de Vast s'appliquerait ici et un test d'arret ne finirait jamais.
    monkeypatch.setenv("SPARK_PROVIDER", "test")
    registry._cartes_cache = list(cartes)
    compte = {"n": 0}
    verrou = threading.Lock()

    def faux_run_task(inp, job_id, progress):
        with verrou:
            i = compte["n"]
            compte["n"] += 1
        d = durees_par_job[i % len(durees_par_job)] if durees_par_job else duree
        if d:
            progress({"task": inp.get("task"), "percent": 42, "message": "en cours"})
            time.sleep(d)
        return {"status": "completed", "task": inp.get("task"), "job_id": job_id}

    monkeypatch.setattr(service, "run_task", faux_run_task)


def test_la_capacite_suit_les_cartes_trouvees(monkeypatch):
    """Quatre cartes de 24 Go tiennent douze places, trois en tiennent neuf. C'est LE
    point : le serveur n'a pas a deviner ce que l'hebergeur a livre."""
    for cartes, attendu in ((["cuda:0", "cuda:1", "cuda:2", "cuda:3"], 12),
                            (["cuda:0", "cuda:1", "cuda:2"], 9),
                            (["cuda:0"], 3)):
        srv = FauxServeur(0)
        _prepare(monkeypatch, srv, cartes, places=None)
        out = service.process_pull({"claim_url": "https://serveur/prise?sig=x"}, "w1")
        assert out["places"] == attendu, f"{len(cartes)} cartes -> {out['places']}"
        assert srv.demandes[0]["places"] == attendu
        assert srv.demandes[0]["libres"] == attendu
        assert srv.demandes[0]["cartes"] == cartes


def test_le_worker_decide_ses_places_d_apres_chaque_carte(monkeypatch):
    """Les places suivent la memoire de CHAQUE carte (16 Go → 2, 24 Go → 3, 48 Go → 8).
    Decide par le worker, carte par carte, sans consigne du serveur — et une machine
    melangee compte juste."""
    srv = FauxServeur(0)
    _prepare(monkeypatch, srv, ["cuda:0", "cuda:1"], places=None)
    monkeypatch.setattr(registry, "vram_gb", lambda c: 17.2)
    out = service.process_pull({"claim_url": "https://serveur/prise?sig=x"}, "w1c")
    assert out["places"] == 4, "deux cartes de 16 Go = deux places chacune"
    assert srv.demandes[0]["places_par_carte"] == {"cuda:0": 2, "cuda:1": 2}
    srv = FauxServeur(0)
    _prepare(monkeypatch, srv, ["cuda:0", "cuda:1"], places=None)
    monkeypatch.setattr(registry, "vram_gb", lambda c: {"cuda:0": 17.2, "cuda:1": 50.8}[c])
    out = service.process_pull({"claim_url": "https://serveur/prise?sig=x"}, "w1d")
    assert out["places"] == 10 and srv.demandes[0]["places_par_carte"] == {"cuda:0": 2, "cuda:1": 8}
    # Le serveur ne peut que plafonner, jamais relever.
    srv = FauxServeur(0)
    _prepare(monkeypatch, srv, ["cuda:0"], places=None)
    assert service.process_pull({"claim_url": "https://serveur/prise?sig=x", "jobs_par_carte": 1}, "w1e")["places"] == 1
    assert service.process_pull({"claim_url": "https://serveur/prise?sig=x", "jobs_par_carte": 2}, "w1f")["places"] == 2
    assert service.process_pull({"claim_url": "https://serveur/prise?sig=x", "jobs_par_carte": 50}, "w1g")["places"] == 3


def test_l_identite_voyage_avec_chaque_demande(monkeypatch):
    """Instance chez l'hebergeur, tag de l'image, demarrage, temps de vie, avancement :
    tout ce qu'il faut au serveur pour retrouver ce worker par API et decider de son sort."""
    monkeypatch.setenv("SPARK_INSTANCE_ID", "inst-42")
    monkeypatch.setenv("SPARK_MACHINE_ID", "m-7")
    monkeypatch.setenv("SPARK_IMAGE_TAG", "sha-abc1234")
    srv = FauxServeur(0)
    _prepare(monkeypatch, srv, ["cuda:0"])
    out = service.process_pull({"claim_url": "https://serveur/prise?sig=x"}, "w1b")
    d = srv.demandes[0]
    assert d["instance_id"] == "inst-42" and d["machine_id"] == "m-7" and d["image_tag"] == "sha-abc1234"
    assert d["demarre_a"].endswith("Z") and d["uptime_s"] >= 0 and d["en_vol"] == []
    assert out["instance_id"] == "inst-42" and out["image_tag"] == "sha-abc1234"
    # Le dernier envoi porte `fin` et la raison, pour que le serveur libere tout.
    assert srv.demandes[-1]["fin"] is True and "arrêt" in srv.demandes[-1]["raison"]


def test_identite_hebergeur_lit_ce_que_l_hebergeur_pose(monkeypatch):
    for var in ("SPARK_INSTANCE_ID", "RUNPOD_POD_ID", "MODAL_TASK_ID", "VAST_CONTAINERLABEL", "CONTAINER_ID",
                "SPARK_MACHINE_ID", "SPARK_IMAGE_TAG", "RUNPOD_ENDPOINT_ID", "SPARK_ENDPOINT_ID"):
        monkeypatch.delenv(var, raising=False)
    assert service.identite_hebergeur()["instance_id"] is None
    monkeypatch.setenv("VAST_CONTAINERLABEL", "C.123456")
    assert service.identite_hebergeur()["instance_id"] == "123456"
    monkeypatch.setenv("RUNPOD_POD_ID", "pod-9")
    assert service.identite_hebergeur()["instance_id"] == "pod-9"      # RunPod avant Vast
    monkeypatch.setenv("SPARK_INSTANCE_ID", "force")
    assert service.identite_hebergeur()["instance_id"] == "force"      # l'ordonnanceur gagne
    assert service.identite_hebergeur()["image_tag"] is None


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
    suivantes = [int(d.get("libres") or 0) for d in srv.demandes[1:] if not d.get("fin") and d.get("libres")]
    assert suivantes and max(suivantes) < 4, \
        "il a laisse toutes ses places se vider avant de redemander"


def test_une_file_vide_fait_attendre_puis_redemander(monkeypatch):
    """Il ne sort PAS : il attend ce que le serveur dit, et redemande. Seul l'ordre
    `arret` le fait sortir."""
    srv = FauxServeur(0, attente_s=0.05, vides_avant_arret=4)
    _prepare(monkeypatch, srv, ["cuda:0", "cuda:1"])
    t0 = time.monotonic()
    out = service.process_pull({"claim_url": "https://serveur/prise?sig=x"}, "w3")
    assert out["status"] == "completed" and out["total"] == 0
    assert out["arret"] == "arrêt demandé par le serveur"
    prises = [d for d in srv.demandes if not d.get("fin") and not d.get("battement")]
    assert len(prises) == 4, f"{len(prises)} demandes : il devait redemander jusqu'a l'ordre"
    assert out["attentes"] >= 3
    assert time.monotonic() - t0 >= 3 * 0.05, "il n'a pas attendu entre deux demandes"


def test_il_vide_la_file_puis_attend_l_ordre(monkeypatch):
    srv = FauxServeur(10)
    _prepare(monkeypatch, srv, ["cuda:0", "cuda:1"])
    out = service.process_pull({"claim_url": "https://serveur/prise?sig=x"}, "w4")
    assert out["total"] == 10 and out["reussis"] == 10
    assert out["arret"] == "arrêt demandé par le serveur"
    assert len(srv.rendus) == 10, f"{len(srv.rendus)} resultats rendus au lieu de 10"
    assert srv.fins == 1


def test_un_budget_explicite_arrete_avant_la_coupure(monkeypatch):
    """Un job coupe en cours perd tout ce qu'il a calcule : avec un budget, le worker
    cesse de reprendre, laisse finir ce qui tourne, et sort."""
    srv = FauxServeur(200)
    _prepare(monkeypatch, srv, ["cuda:0"], duree=0.05)
    out = service.process_pull(
        {"claim_url": "https://serveur/prise?sig=x", "budget_s": 1}, "w5")
    assert "budget" in out["arret"]
    assert out["elapsed_s"] < 3.0, "il a depasse son budget"
    assert srv.stock > 0, "il aurait du s'arreter avant d'avoir tout pris"
    assert len(srv.rendus) == out["total"], "des resultats n'ont pas ete rendus"


def test_sans_budget_aucune_limite_de_vie(monkeypatch):
    """Par defaut le worker n'a PAS de budget : c'est le serveur qui l'arrete. Il prend
    tout ce qu'il y a, puis attend l'ordre."""
    assert service._budget({}) is None and service._budget({"budget_s": 0}) is None
    assert service._budget({"budget_s": 10}) == service.PLANCHER_S
    srv = FauxServeur(40)
    _prepare(monkeypatch, srv, ["cuda:0"])
    out = service.process_pull({"claim_url": "https://serveur/prise?sig=x"}, "w6")
    assert out["arret"] == "arrêt demandé par le serveur", out["arret"]
    assert out["total"] == 40, "il a lache du travail alors qu'il en restait"


def test_un_serveur_injoignable_fait_reessayer_puis_sortir(monkeypatch):
    """Sans serveur, aucun ordre ne peut arriver : le worker reessaie, puis, passe le
    premier seuil de silence et sans rien en cours, il sort — inutile de facturer une
    machine qui ne peut plus recevoir de travail."""
    monkeypatch.setattr(service, "SILENCE_DOUX_S", 0.15)
    monkeypatch.setattr(service, "ATTENTE_ERREUR_S", 0.05)
    demandes = {"n": 0}

    def muet(url, corps, essais=2):
        demandes["n"] += 1
        return {"erreur": "HTTP 503"}

    _prepare(monkeypatch, muet, ["cuda:0", "cuda:1"])
    out = service.process_pull({"claim_url": "https://serveur/prise?sig=x"}, "w7")
    assert out["status"] == "completed" and out["total"] == 0
    assert "muet" in out["arret"]
    assert demandes["n"] >= 3, "il devait reessayer avant de sortir"


def test_sur_vast_l_homme_mort_ne_tue_pas(monkeypatch):
    """Sur Vast, un conteneur qui sort est relance en boucle et le pod reste facture :
    passe le second seuil, le worker se tait au lieu de se tuer, et attend le balai."""
    monkeypatch.setenv("SPARK_PROVIDER", "vastai")
    monkeypatch.setattr(service, "SILENCE_DOUX_S", 0.1)
    monkeypatch.setattr(service, "SILENCE_NET_S", 0.2)
    monkeypatch.setattr(service, "ATTENTE_ERREUR_S", 0.05)
    quitte: list[int] = []
    monkeypatch.setattr(service, "_quitter", lambda code: quitte.append(code))
    bat = service.Battement("https://serveur/prise?sig=x", {}, lambda: {})
    bat._contact = time.monotonic() - 1.0    # muet depuis une seconde
    bat.silence()
    assert quitte == [] and bat.arret == "silence"


def test_un_envoi_rate_ne_perd_pas_les_resultats(monkeypatch):
    """Les resultats qu'on tenait au moment d'une panne restent en attente et repartent
    avec le premier contact reussi : rien de fini n'est perdu."""
    etat = {"n": 0}
    rendus: list[dict] = []

    def serveur_qui_tousse(url, corps, essais=2):
        etat["n"] += 1
        rendus.extend(corps.get("resultats") or [])
        if etat["n"] == 1:
            return {"jobs": [{"task": "instrumental", "audio_url": "https://exemple/1.wav", "id": "srv-1"}]}
        if etat["n"] == 2:
            return {"erreur": "HTTP 503"}       # la demande qui portait le resultat echoue
        return {"arret": "doux"}

    monkeypatch.setattr(service, "ATTENTE_ERREUR_S", 0.05)
    _prepare(monkeypatch, serveur_qui_tousse, ["cuda:0"], duree=0.05)
    out = service.process_pull({"claim_url": "https://serveur/prise?sig=x"}, "w7c")
    assert out["total"] == 1
    assert [r["job_id"] for r in rendus] == ["srv-1", "srv-1"], \
        "le resultat devait etre renvoye apres l'echec, une seule fois accepte"


def test_sur_vast_le_worker_se_tait_puis_reprend_quand_le_serveur_revient(monkeypatch):
    """Vast : passe le silence, on ne sort pas de la boucle (le conteneur serait relance et
    le pod facture) ; on sonde doucement, et on reprend des que le serveur repond."""
    monkeypatch.setenv("SPARK_PROVIDER", "vastai")
    monkeypatch.setattr(service, "SILENCE_DOUX_S", 0.1)
    monkeypatch.setattr(service, "SILENCE_NET_S", 0.2)
    monkeypatch.setattr(service, "ATTENTE_ERREUR_S", 0.05)
    monkeypatch.setattr(service, "BATTEMENT_MIN_S", 0.05)
    quitte: list[int] = []
    monkeypatch.setattr(service, "_quitter", lambda code: quitte.append(code))
    t0 = time.monotonic()
    etat = {"n": 0}

    def panne_puis_retour(url, corps, essais=2):
        etat["n"] += 1
        if time.monotonic() - t0 < 0.5:
            return {"erreur": "HTTP 503"}
        if corps.get("battement"):
            return {}
        if etat.get("donne"):
            return {"arret": "doux"}
        etat["donne"] = True
        return {"jobs": [{"task": "instrumental", "audio_url": "https://exemple/1.wav"}]}

    _prepare(monkeypatch, panne_puis_retour, ["cuda:0"], duree=0.05)
    monkeypatch.setenv("SPARK_PROVIDER", "vastai")   # APRES _prepare, qui pose un fournisseur neutre
    out = service.process_pull({"claim_url": "https://serveur/prise?sig=x", "battement_s": 0.05}, "w7d")
    assert quitte == [], "sur Vast on ne se tue pas"
    assert out["total"] == 1, "il devait reprendre du travail au retour du serveur"
    assert out["arret"] == "arrêt demandé par le serveur"


def test_un_serveur_muet_pendant_un_job_finit_par_tuer_le_worker(monkeypatch):
    """Un job en cours et plus de serveur : passe le second seuil, l'homme-mort tue le
    processus meme en plein job — l'alternative est de payer les cartes jusqu'a la
    coupure de l'hebergeur."""
    monkeypatch.setattr(service, "SILENCE_DOUX_S", 0.15)
    monkeypatch.setattr(service, "SILENCE_NET_S", 0.4)
    monkeypatch.setattr(service, "ATTENTE_ERREUR_S", 0.05)
    monkeypatch.setattr(service, "BATTEMENT_MIN_S", 0.05)
    quitte: list[int] = []
    monkeypatch.setattr(service, "_quitter", lambda code: quitte.append(code))
    etat = {"n": 0}

    def panne_apres_le_premier(url, corps, essais=2):
        etat["n"] += 1
        if etat["n"] == 1:
            return {"jobs": [{"task": "instrumental", "audio_url": "https://exemple/1.wav"}]}
        return {"erreur": "HTTP 503"}

    _prepare(monkeypatch, panne_apres_le_premier, ["cuda:0"], duree=1.2)
    out = service.process_pull({"claim_url": "https://serveur/prise?sig=x", "battement_s": 0.05}, "w7b")
    assert quitte and quitte[0] == 3, "l'homme-mort n'a pas tue le worker"
    assert "muet" in out["arret"]


def test_arret_net_rend_les_resultats_et_liste_les_abandonnes(monkeypatch):
    """`net` : ce qui est fini est rendu, ce qui tourne est declare abandonne (le serveur
    le remet en file), et on sort SANS attendre la fin des jobs en cours."""
    quitte: list[int] = []
    monkeypatch.setattr(service, "_quitter", lambda code: quitte.append(code))
    # 3 jobs, 2 places, un job court et un long : la 2e demande (a la fin du court)
    # recoit `net` pendant que le long tourne encore.
    srv = FauxServeur(3, net_a=2, ids=True)
    _prepare(monkeypatch, srv, ["cuda:0"], durees_par_job=[0.1, 0.8, 0.1])
    t0 = time.monotonic()
    out = service.process_pull({"claim_url": "https://serveur/prise?sig=x"}, "w8")
    assert quitte == [0]
    assert "NET" in out["arret"]
    assert time.monotonic() - t0 < 1.5, "il a attendu la fin du job abandonne"
    assert srv.fins == 1
    assert len(srv.rendus) == 1 and srv.rendus[0]["job_id"] == "srv-1"
    assert srv.abandonnes == ["srv-2"]


def test_les_identifiants_du_serveur_sont_repris(monkeypatch):
    """Quand le serveur numerote ses jobs (`id`), c'est SON numero qui revient dans
    `resultats` et dans `en_vol` — il n'a pas a tenir une table de correspondance."""
    srv = FauxServeur(4, ids=True)
    # Des durees inegales : quand le court finit, le long est encore en vol et doit
    # apparaitre dans la demande suivante.
    _prepare(monkeypatch, srv, ["cuda:0"], durees_par_job=[0.05, 0.4, 0.05, 0.05])
    out = service.process_pull({"claim_url": "https://serveur/prise?sig=x"}, "w9")
    assert out["total"] == 4
    assert sorted(r["job_id"] for r in srv.rendus) == ["srv-1", "srv-2", "srv-3", "srv-4"]
    en_vol = [e for d in srv.demandes for e in (d.get("en_vol") or [])]
    assert en_vol, "l'avancement des jobs en cours n'a jamais voyage"
    assert all(e["job_id"].startswith("srv-") and e["task"] == "instrumental" for e in en_vol)
    assert any(e["percent"] == 42 for e in en_vol), "le pourcentage du job n'est pas remonte"


def test_claim_url_invalide_refuse(monkeypatch):
    for mauvais in ({}, {"claim_url": ""}, {"claim_url": "pas-une-url"}):
        out = service.process_pull(mauvais, "w10")
        assert out["status"] == "error" and out["code"] == "bad_input"


def test_process_job_reconnait_la_prise(monkeypatch):
    """Aucun hebergeur n'a rien a savoir : `handler.py` et `modal_app.py` appellent
    `process_job` comme avant, la cle `claim_url` suffit a basculer."""
    srv = FauxServeur(2)
    _prepare(monkeypatch, srv, ["cuda:0"])
    out = service.process_job({"claim_url": "https://serveur/prise?sig=x"}, "w11")
    assert out.get("prise") is True and out["total"] == 2


def test_le_serveur_renouvelle_l_url_de_prise(monkeypatch):
    """Une reponse qui porte `claim_url` change l'URL de TOUTES les demandes suivantes
    (prises, battements, `fin`) : le serveur renouvelle avant l'expiration."""
    urls: list[str] = []

    class Srv(FauxServeur):
        def __call__(self, url, corps, essais=2):
            urls.append(url)
            rep = super().__call__(url, corps, essais)
            if len(self.demandes) == 1:
                rep["claim_url"] = "https://serveur/prise?sig=nouvelle"
            return rep

    srv = Srv(2, ids=True)
    _prepare(monkeypatch, srv, ["cuda:0"])
    out = service.process_pull({"claim_url": "https://serveur/prise?sig=x"}, "w9")
    assert out["total"] == 2
    assert urls[0] == "https://serveur/prise?sig=x"
    assert len(urls) >= 3 and all(u == "https://serveur/prise?sig=nouvelle" for u in urls[1:]), urls


def test_une_url_renouvelee_invalide_est_ignoree(monkeypatch):
    class Srv(FauxServeur):
        def __call__(self, url, corps, essais=2):
            rep = super().__call__(url, corps, essais)
            rep["claim_url"] = "ftp://pas-http"
            return rep

    srv = Srv(1, ids=True)
    _prepare(monkeypatch, srv, ["cuda:0"])
    bat = service.Battement("https://serveur/prise?sig=x", {}, dict, 10.0)
    bat.adopter_url({"claim_url": "ftp://pas-http"})
    assert bat.url == "https://serveur/prise?sig=x"
    bat.adopter_url({"claim_url": "https://serveur/prise?sig=ok"})
    assert bat.url == "https://serveur/prise?sig=ok"


def test_le_resume_porte_inference_depot_et_carte(monkeypatch):
    """Ce que le serveur recoit par job : l'inference SEULE (hors transferts), le depot
    verifie, les octets, la carte — de quoi apprendre le debit et facturer."""
    srv = FauxServeur(1, ids=True)
    _prepare(monkeypatch, srv, ["cuda:0"])
    monkeypatch.setattr(service, "run_task", lambda inp, job_id, progress: {
        "status": "completed", "task": inp.get("task"), "job_id": job_id,
        "timings": {"download_s": 1.0, "inference_s": 7.5, "upload_s": 0.4},
        "uploaded": True, "bytes": 123, "gpu_name": "NVIDIA L4", "meta": inp.get("meta")})
    service.process_pull({"claim_url": "https://serveur/prise?sig=x"}, "w10")
    r = srv.rendus[0]
    assert r["inference_s"] == 7.5 and r["timings"]["upload_s"] == 0.4
    assert r["uploaded"] is True and r["bytes"] == 123 and r["gpu_name"] == "NVIDIA L4"


@pytest.mark.parametrize("task", ["instrumental", "vc", "speaking_faces"])
def test_chaque_job_rend_son_temps_d_inference_seul(monkeypatch, task):
    """Chaque job, quelle que soit la tache et l'hebergeur, rend `inference_s` au premier
    niveau : le calcul du modele SEUL — le telechargement et le chargement n'y sont pas."""
    def faux(req, workdir, job_id, progress, timer):
        with timer.step("download"):
            time.sleep(0.05)
        with timer.step("model_load"):
            time.sleep(0.05)
        with timer.step("inference"):
            time.sleep(0.02)
        return {}
    for nom in ("instrumental", "vc", "speaking_faces"):
        monkeypatch.setattr(tasks, f"_run_{nom}", faux)
        monkeypatch.setattr(tasks, f"parse_{nom}", lambda inp: None)
    r = tasks.run_task({"task": task}, "j1", lambda p: None)
    assert r["status"] == "completed" and r["task"] == task
    assert r["inference_s"] == r["timings"]["inference_s"]
    assert 0.02 <= r["inference_s"] < 0.05
