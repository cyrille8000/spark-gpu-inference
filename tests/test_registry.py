import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spark_infer import registry  # noqa: E402


@pytest.fixture(autouse=True)
def registre_propre():
    registry.reset()
    yield
    registry.reset()


def test_sondes_memoire_ne_plantent_jamais():
    """Les sondes rendent None sans CUDA, sinon deux nombres positifs — jamais une
    exception : un job ne doit pas échouer parce qu'on a voulu le mesurer."""
    registry.reset_peak_memory()   # ne doit jamais lever, GPU ou pas
    pic = registry.peak_memory_gb()
    assert pic is None or (set(pic) == {"allocated_gb", "reserved_gb"}
                           and all(isinstance(v, float) and v >= 0 for v in pic.values()))
    total = registry.vram_total_gb()
    assert total is None or total > 0


def test_une_instance_est_reutilisee_puis_rendue():
    charges = []
    for attendu_cold in (True, False, False):
        with registry.lease("m", lambda: charges.append(1) or object(), 1) as (inst, cold):
            assert inst is not None and cold is attendu_cold
            assert registry.active() == 1              # empruntée pendant le bloc
        assert registry.active() == 0                  # rendue à la sortie
    assert len(charges) == 1                           # chargée une seule fois


def test_deux_jobs_obtiennent_deux_instances_distinctes():
    """Le point qui rend le parallélisme possible : chaque job a SON modèle, donc sa
    configuration et sa voix de référence (cf. vc_engine) ne peuvent plus s'écraser."""
    with registry.lease("m", object, 2) as (a, _):
        with registry.lease("m", object, 2) as (b, _):
            assert a is not b
            assert registry.pool_state()["m"] == {"total": 2, "busy": 2, "max": 2}


def test_le_pool_plein_fait_attendre_puis_sert():
    """Pool de 1 : le second job attend que le premier rende, il ne double pas l'instance."""
    vus = []

    def prend():
        with registry.lease("m", object, 1) as (inst, _):
            vus.append(inst)
            time.sleep(0.05)

    t1 = threading.Thread(target=prend)
    t2 = threading.Thread(target=prend)
    t1.start(); t2.start(); t1.join(); t2.join()
    assert len(vus) == 2 and vus[0] is vus[1]          # la MÊME instance, l'une après l'autre
    assert registry.pool_state()["m"]["total"] == 1


def test_un_chargement_rate_rend_sa_place():
    """Sinon une panne de chargement condamnerait une place du pool pour toujours."""
    def factory_qui_echoue():
        raise RuntimeError("poids illisibles")

    with pytest.raises(RuntimeError):
        with registry.lease("m", factory_qui_echoue, 1):
            pass
    assert registry.pool_state().get("m", {"total": 0})["total"] == 0
    with registry.lease("m", object, 1) as (inst, cold):
        assert inst is not None and cold is True


def test_release_ne_vole_pas_l_instance_d_un_job_en_cours():
    """Sur OOM on libère ce qui est libre, jamais ce qu'un voisin tient : le voler le
    ferait échouer sans raison."""
    with registry.lease("m", object, 2) as (tenue, _):
        with registry.lease("m", object, 2):
            pass                                        # une instance revient au pool
        assert registry.pool_state()["m"] == {"total": 2, "busy": 1, "max": 2}
        registry.release()
        assert registry.pool_state()["m"] == {"total": 1, "busy": 1, "max": 2}
        assert tenue is not None                        # toujours valide pour son job


def test_others_loaded_excludes_the_job_model():
    with registry.lease("chatterbox_vc", object, 1):
        assert registry.others_loaded("chatterbox_vc") == []   # seul : un OOM ne se rejoue pas
        with registry.lease("bs_roformer_leap_xe", object, 1):
            assert registry.others_loaded("chatterbox_vc") == ["bs_roformer_leap_xe"]
    registry.release()
    assert registry.others_loaded(None) == []
