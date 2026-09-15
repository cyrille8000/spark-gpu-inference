"""Le worker Vast.ai — ses décisions pures, celles qui protègent un pod loué à l'heure :
qui a le droit d'utiliser la carte, et quelles cartes peuvent exécuter cette image.
Le serveur lui-même (sockets, fils) se vérifie sur un vrai pod.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vast_worker import Etat, capacite_minimale, carte_supportee, jeton_valide  # noqa: E402


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


def test_toute_carte_au_dessus_du_minimum_passe():
    """Une règle : capacité >= la plus petite compilée (7.5). Les roues cu128 n'embarquent
    pas sm_89, et pourtant le L4 de Modal (Ada, 8.9) tourne avec cette image. Le worker Vast
    exigeait le numéro exact et refusait toute carte Ada — RTX 6000 Ada louée deux fois pour
    rien le 2026-09-15, conteneur relancé toutes les 12 s et facturé."""
    reelle = ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120", "compute_120"]
    for sm in ("sm_89", "sm_87"):                    # Ada (L4, 4090, L40S, 6000 Ada), Orin
        assert carte_supportee(sm, reelle) is True
    assert carte_supportee("sm_121", reelle) is True  # Blackwell plus récente que 12.0
    assert carte_supportee("sm_130", reelle) is True  # génération future : le PTX la compile
    assert carte_supportee("sm_70", reelle) is False  # V100 : 7.0 < 7.5
    assert carte_supportee("sm_61", reelle) is False  # Pascal (P40, P100)
    assert carte_supportee("sm_89", ["sm_90"]) is False  # sous le minimum de ces roues-là


def test_capacite_minimale_lue_dans_les_roues():
    """Ce qui décide est la capacité de la CARTE, pas la version CUDA du pilote : le
    V100 refusé le 2026-09-12 tournait sur un pilote CUDA 13.0 et restait en 7.0."""
    reelle = ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120", "compute_120"]
    assert capacite_minimale(reelle) == "7.5"          # torch 2.7.1+cu128, mesuré sur le pod
    assert carte_supportee("sm_70", reelle) is False   # V100 : sous le plancher
    assert carte_supportee("sm_75", reelle) is True    # T4
    assert capacite_minimale(["sm_120", "compute_120"]) == "12.0"
    assert capacite_minimale(["compute_120"]) is None


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


def test_les_dockerfiles_sont_lisibles_par_docker():
    """Aucun caractère de contrôle, et chaque ligne continuée d'un ENV est `NOM=valeur` ou un
    commentaire : un remplacement de texte raté y avait laissé un \x01, et le build Vast est
    tombé en « can't find = » (2026-09-15)."""
    import re
    racine = Path(__file__).resolve().parents[1]
    for nom in ("Dockerfile", "Dockerfile.vast"):
        texte = (racine / nom).read_text(encoding="utf-8")
        assert not re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", texte), f"caractère de contrôle dans {nom}"
        dans_env = False
        for ligne in texte.splitlines():
            brut = ligne.strip()
            if brut.startswith("ENV "):
                dans_env = brut.endswith("\\")
                continue
            if dans_env:
                if brut.startswith("#") or brut == "":
                    continue
                assert re.match(r"^[A-Za-z_][A-Za-z0-9_]*=\S*(\s+\x5c)?$", brut), f"{nom} : ligne ENV invalide « {brut} »"
                dans_env = brut.endswith("\\")
