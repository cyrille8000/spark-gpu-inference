import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spark_infer.container_clock import ContainerClock  # noqa: E402


def test_un_job_a_la_fois_prend_toute_la_fenetre():
    """Le cas d'hier : un seul job à la fois. Chaque job paie le boot ou l'attente
    qui le précède, plus son propre temps ; la somme = la vie du conteneur."""
    c = ContainerClock(now=100.0)
    c.enter("a", now=118.0)                          # 18 s de boot avant le job
    assert c.leave("a", now=130.5) == (30.5, True)   # 18 + 12,5
    c.enter("b", now=150.5)                          # 20 s d'attente
    assert c.leave("b", now=162.5) == (32.0, False)  # 20 + 12
    assert 30.5 + 32.0 == 162.5 - 100.0


def test_deux_jobs_en_parallele_partagent_chaque_seconde():
    """Deux jobs qui se recouvrent : la seconde partagée vaut une demi-seconde pour
    chacun. Sans ça, un conteneur à deux jobs se ferait facturer deux fois son temps."""
    c = ContainerClock(now=0.0)
    c.enter("a", now=0.0)
    c.enter("b", now=10.0)                           # a seul de 0 à 10
    part_a, first = c.leave("a", now=30.0)           # 0-10 pour a, 10-30 partagé
    assert (part_a, first) == (20.0, True)           # 10 + 20/2
    part_b, _ = c.leave("b", now=40.0)               # 10-30 partagé, 30-40 pour b seul
    assert part_b == 20.0                            # 20/2 + 10
    assert part_a + part_b == 40.0                   # la vie du conteneur, en entier


def test_trois_jobs_et_temps_mort_intermediaire():
    c = ContainerClock(now=0.0)
    c.enter("a", now=0.0)
    c.enter("b", now=0.0)
    c.enter("c", now=0.0)
    parts = [c.leave(j, now=30.0)[0] for j in ("a", "b", "c")]
    assert parts == [10.0, 10.0, 10.0]               # 30 s à trois
    c.enter("d", now=50.0)                           # 20 s de conteneur inoccupé
    assert c.leave("d", now=60.0)[0] == 30.0         # l'attente revient au suivant
    assert sum(parts) + 30.0 == 60.0


def test_horloge_sure_entre_fils():
    """Vingt jobs qui entrent et sortent depuis vingt fils : aucune part perdue."""
    c = ContainerClock(now=0.0)
    debut = threading.Barrier(20)
    parts = {}

    def job(i):
        debut.wait()
        c.enter(f"j{i}")
        parts[i] = c.leave(f"j{i}")[0]

    fils = [threading.Thread(target=job, args=(i,)) for i in range(20)]
    for f in fils:
        f.start()
    for f in fils:
        f.join()
    assert len(parts) == 20 and all(p >= 0 for p in parts.values())
    assert c.active() == 0


def test_jamais_negatif():
    c = ContainerClock(now=10.0)
    c.enter("a", now=5.0)
    assert c.leave("a", now=5.0) == (0.0, True)
