"""Le worker Vast.ai — ses décisions pures, celles qui protègent un pod loué à l'heure :
qui a le droit d'utiliser la carte, et quelles cartes peuvent exécuter cette image.
Le serveur lui-même (sockets, fils) se vérifie sur un vrai pod.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vast_worker import Etat, carte_supportee, jeton_valide  # noqa: E402


def test_jeton_exige_et_compare_les_deux_formes():
    """Sans jeton configuré, RIEN ne passe : un pod public sans garde offrirait sa
    carte à tout Internet."""
    assert jeton_valide("", "Bearer s3cret", "s3cret") is False
    assert jeton_valide("s3cret", "Bearer s3cret", None) is True
    assert jeton_valide("s3cret", "bearer s3cret", None) is True       # en-tête insensible à la casse
    assert jeton_valide("s3cret", None, "s3cret") is True              # ou dans le corps, comme Modal
    assert jeton_valide("s3cret", "Bearer autre", None) is False
    assert jeton_valide("s3cret", None, None) is False
    assert jeton_valide("s3cret", "s3cret", None) is False             # sans le préfixe Bearer


def test_cartes_acceptees_et_refusees():
    """Les roues torch cu128 embarquent Volta à Blackwell. Une Pascal (sm_61), encore
    courante chez Vast.ai sur les P40, ne peut rien exécuter : mieux vaut le savoir au
    démarrage que découvrir « no kernel image » après dix minutes de location."""
    arch = ["sm_70", "sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120", "compute_120"]
    for sm in ("sm_80", "sm_86", "sm_90"):          # A100, 3090/A6000, H100
        assert carte_supportee(sm, arch) is True
    assert carte_supportee("sm_61", arch) is False   # Pascal : plus ancienne que tout
    assert carte_supportee("sm_37", arch) is False   # K80
    # Compilée pour une seule architecture, plus le PTX : une carte plus récente passe
    # par compilation à la volée, une plus ancienne non.
    assert carte_supportee("sm_121", ["sm_120", "compute_120"]) is True
    assert carte_supportee("sm_89", ["sm_120", "compute_120"]) is False
    assert carte_supportee("inconnu", arch) is False


def test_etat_suit_les_jobs_et_l_inactivite():
    """L'inactivité ne compte que quand plus rien ne tourne : c'est elle qui décide
    d'éteindre le pod."""
    e = Etat()
    e.dernier = 1000.0
    assert e.inactif_depuis(now=1100.0) == 100.0
    e.debut()
    assert e.actifs == 1
    assert e.inactif_depuis(now=1e9) == 0.0          # un job tourne : jamais inactif
    e.debut()
    e.fin()
    assert (e.actifs, e.faits) == (1, 1)
    assert e.inactif_depuis(now=1e9) == 0.0          # il en reste un
    e.fin()
    assert (e.actifs, e.faits) == (0, 2)
    assert e.inactif_depuis() >= 0.0
