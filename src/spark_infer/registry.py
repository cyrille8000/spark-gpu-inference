"""Modeles residents : un POOL d'instances par (modele, carte), empruntee le temps d'un job.

Avant le 2026-09-12, UNE instance par modele servait tous les jobs — ca tenait
parce qu'un conteneur ne traitait qu'un job a la fois. Des que deux jobs se
croisent, ce partage devient FAUX pour la conversion vocale : chaque job ecrit
sa configuration et sa voix de reference DANS le modele (`vc_engine._configure`,
`_set_reference`), donc le second ecraserait la reference du premier, qui
rendrait une voix etrangere sans que rien ne le signale.

D'ou le pool : un job EMPRUNTE une instance pour lui seul (`lease`) et la rend a
la fin. Le pool en cree jusqu'a `max_instances` ; au-dela, le job attend son
tour. Chaque instance coute ses poids sur la carte : c'est la mesure `gpu_mem`
de chaque job qui dit combien on peut en tenir sur un GPU donne.

PLUSIEURS CARTES (2026-09-12). Une machine Vast.ai peut en porter deux, quatre,
douze. Mesure sur une machine a deux RTX 3090 : l'image n'en utilisait qu'une —
l'OOM disait « GPU 0 has a total capacity of 23.56 GiB » alors que 48 Go etaient
loues. La cause tenait ici : `device()` rendait la chaine « cuda », que PyTorch
resout en `cuda:0`, et les pools etaient indexes par le seul nom du modele. Ils
le sont maintenant par (modele, carte), un job va sur la carte la moins chargee,
et la capacite annoncee est la SOMME des cartes. Les moteurs n'ont rien eu a
changer : ils prenaient deja un `device` en parametre.

Le chargement d'une instance (des dizaines de secondes) se fait HORS du verrou :
sa place est reservee d'abord, pour que deux jobs simultanes n'en creent pas
trois, et rendue si le chargement echoue.
"""
from __future__ import annotations

import gc
import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterator, TypeVar

log = logging.getLogger("spark.registry")

T = TypeVar("T")


@dataclass
class _Pool:
    """Les instances d'UN modele SUR UNE CARTE : celles qui attendent, et le compte
    de celles qui existent."""

    max_size: int
    idle: list = field(default_factory=list)
    total: int = 0   # instances existantes ou en cours de chargement
    busy: int = 0    # instances actuellement empruntees


_cond = threading.Condition()
# Cle = (nom du modele, carte). Deux cartes = deux pools independants du meme modele.
_pools: dict[tuple[str, str], _Pool] = {}
# La carte sur laquelle tourne le job de CE fil : posee par `lease`, lue par les
# mesures memoire et par le resultat du job. Sans ca, un job sur `cuda:1`
# rapporterait le pic de `cuda:0`.
_courant = threading.local()
_cartes_cache: list[str] | None = None


def devices() -> list[str]:
    """Les cartes utilisables, dans l'ordre : ['cuda:0', 'cuda:1', ...] ou ['cpu'].

    Lu une seule fois : le nombre de cartes d'une machine ne change pas en cours de
    route, et `torch.cuda.device_count()` n'est pas gratuit.
    """
    global _cartes_cache
    if _cartes_cache is None:
        try:
            import torch
            n = torch.cuda.device_count() if torch.cuda.is_available() else 0
        except Exception:  # noqa: BLE001
            n = 0
        _cartes_cache = [f"cuda:{i}" for i in range(n)] or ["cpu"]
    return _cartes_cache


def limiter_cartes(cartes: list[str]) -> None:
    """Restreint les cartes utilisables — appele au demarrage apres verification.

    Une machine louee peut porter des cartes de generations differentes, ou une carte
    deja occupee par un autre locataire. Celles qui ne passent pas le controle sont
    ecartees ici plutot que de faire echouer un job au hasard plus tard.
    """
    global _cartes_cache
    if not cartes:
        raise ValueError("aucune carte utilisable")
    _cartes_cache = list(cartes)


def device() -> str:
    """La carte du job en cours sur ce fil, ou la premiere de la machine."""
    return getattr(_courant, "carte", None) or devices()[0]


def _index(carte: str | None = None) -> int | None:
    """L'indice CUDA d'une carte, ou None si on tourne sur processeur."""
    d = carte or device()
    return int(d.split(":", 1)[1]) if d.startswith("cuda:") else None


def _pool(cle: tuple[str, str], max_instances: int) -> _Pool:
    p = _pools.get(cle)
    if p is None:
        p = _pools[cle] = _Pool(max_size=max(1, int(max_instances)))
    else:
        p.max_size = max(1, int(max_instances))
    return p


def _par_carte(max_total: int) -> int:
    """La place d'UNE carte, a partir de la capacite totale du conteneur."""
    return max(1, int(max_total) // len(devices()))


@contextmanager
def lease(name: str, factory: Callable[[str], T], max_instances: int = 1) -> Iterator[tuple[T, bool]]:
    """Emprunte une instance du modele pour la duree du bloc.

    `max_instances` est la capacite TOTALE du conteneur, repartie sur les cartes.
    `factory` recoit la carte choisie (« cuda:1 »…) et doit y construire le modele.
    Rend `(instance, chargee_maintenant)`. Attend si toutes les cartes sont pleines.
    L'instance revient au pool a la sortie du bloc, meme sur exception : un job qui
    echoue ne doit pas retirer une place au suivant.
    """
    carte, inst, cold = _acquire(name, factory, max_instances)
    precedente = getattr(_courant, "carte", None)
    _courant.carte = carte
    try:
        yield inst, cold
    finally:
        _courant.carte = precedente
        _give_back(name, carte, inst)


def _choisir(name: str, par_carte: int) -> tuple[str, bool] | None:
    """La carte a prendre : d'abord une qui a une instance PRETE (pas de chargement a
    payer), sinon la plus libre. None si toutes sont pleines. Verrou tenu."""
    libre: list = []
    for carte in devices():
        p = _pool((name, carte), par_carte)
        if p.idle:
            return carte, True
        if p.total < p.max_size:
            libre.append((p.max_size - p.total, p.busy, carte))
    if not libre:
        return None
    # La plus de place restante ; a egalite, la moins occupee.
    libre.sort(key=lambda x: (-x[0], x[1]))
    return libre[0][2], False


def _acquire(name: str, factory: Callable[[str], T], max_instances: int) -> tuple[str, T, bool]:
    par_carte = _par_carte(max_instances)
    while True:
        with _cond:
            choix = _choisir(name, par_carte)
            if choix is None:
                _cond.wait(timeout=30.0)
                continue
            carte, prete = choix
            p = _pool((name, carte), par_carte)
            if prete:
                p.busy += 1
                return carte, p.idle.pop(), False
            p.total += 1   # place reservee AVANT le chargement, qui se fait hors verrou
            p.busy += 1
            break
    log.info("chargement d'une instance de << %s >> sur %s (%d/%d)", name, carte, p.total, p.max_size)
    try:
        return carte, factory(carte), True
    except BaseException:
        with _cond:
            p.total -= 1
            p.busy -= 1
            _cond.notify_all()
        raise


def _give_back(name: str, carte: str, inst: object) -> None:
    with _cond:
        p = _pools.get((name, carte))
        if p is None:
            return
        p.busy = max(0, p.busy - 1)
        p.idle.append(inst)
        _cond.notify_all()


def get(name: str, factory: Callable[[str], T]) -> T:
    """Emprunt SANS rendu — pour un appelant qui garde l'instance (tests, outils).
    Le chemin normal est `lease`."""
    _, inst, _ = _acquire(name, factory, 1)
    return inst


def active() -> int:
    """Nombre d'instances actuellement empruntees, tous modeles et toutes cartes
    confondus : le nombre de jobs qui se partagent la machine."""
    with _cond:
        return sum(p.busy for p in _pools.values())


def pool_state() -> dict:
    """Etat des pools, pour le diagnostic et le resultat d'un job. Agrege par modele ;
    le detail par carte n'apparait que si la machine en a plusieurs."""
    plusieurs = len(devices()) > 1
    with _cond:
        out: dict = {}
        for (n, carte), p in sorted(_pools.items()):
            e = out.setdefault(n, {"total": 0, "busy": 0, "max": 0})
            e["total"] += p.total
            e["busy"] += p.busy
            e["max"] += p.max_size
            if plusieurs:
                e.setdefault("cartes", {})[carte] = {"total": p.total, "busy": p.busy, "max": p.max_size}
        return out


def loaded() -> list[str]:
    """Les modeles qui ont au moins une instance (chargee ou en cours), toutes cartes."""
    with _cond:
        return sorted({n for (n, _), p in _pools.items() if p.total > 0})


def others_loaded(keep: str | None) -> list[str]:
    """Les modeles residents AUTRES que `keep` — ceux qu'une liberation rendrait."""
    return [m for m in loaded() if m != keep]


def release(name: str | None = None) -> None:
    """Libere les instances LIBRES (jamais celles qu'un job tient) et vide le cache CUDA.

    Sur OOM, on ne peut rendre que ce que personne n'utilise : voler l'instance d'un
    job concurrent le ferait echouer sans raison.
    """
    with _cond:
        for cle in [c for c in _pools if name is None or c[0] == name]:
            p = _pools.get(cle)
            if not p or not p.idle:
                continue
            log.info("liberation de %d instance(s) libre(s) de << %s >> sur %s", len(p.idle), cle[0], cle[1])
            p.total -= len(p.idle)
            p.idle.clear()
            if p.total <= 0 and p.busy == 0:
                _pools.pop(cle, None)
        _cond.notify_all()
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:  # noqa: BLE001
        pass


def reset() -> None:
    """Vide tout l'etat du registre — tests seulement."""
    global _cartes_cache
    with _cond:
        _pools.clear()
        _cond.notify_all()
    _cartes_cache = None
    _courant.carte = None


def vram_total_gb() -> float | None:
    """Memoire d'UNE carte (celle du job, ou la premiere). C'est bien par carte que se
    calcule la place : un job ne s'etale jamais sur deux GPU."""
    try:
        import torch
        i = _index()
        if i is None or not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_properties(i).total_memory / 1e9
    except Exception:  # noqa: BLE001
        return None


def vram_machine_gb() -> float | None:
    """Somme des memoires de toutes les cartes — ce que la MACHINE offre."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return sum(torch.cuda.get_device_properties(i).total_memory for i in range(len(devices()))) / 1e9
    except Exception:  # noqa: BLE001
        return None


def reset_peak_memory() -> None:
    """Remet a zero le compteur de PIC memoire de TOUTES les cartes. A n'appeler que si
    AUCUN autre job ne tourne : le compteur est global a la carte, le remettre a zero
    pendant le job d'a cote fausserait sa mesure."""
    try:
        import torch
        if not torch.cuda.is_available():
            return
        for i in range(len(devices())):
            torch.cuda.reset_peak_memory_stats(i)
    except Exception:  # noqa: BLE001
        pass


def peak_memory_gb() -> dict | None:
    """Pic memoire de la carte DU JOB depuis la derniere remise a zero (Go) :
    `allocated` = tenseurs vivants au plus haut, `reserved` = ce que l'allocateur a
    pris a la carte — c'est `reserved` qui decide si un GPU suffit. Le compteur est
    celui de la CARTE : quand plusieurs jobs se croisent dessus, le pic est celui de
    leur somme (cf. `gpu_mem.jobs`). None sans CUDA."""
    try:
        import torch
        i = _index()
        if i is None or not torch.cuda.is_available():
            return None
        return {"allocated_gb": round(torch.cuda.max_memory_allocated(i) / 1e9, 2),
                "reserved_gb": round(torch.cuda.max_memory_reserved(i) / 1e9, 2)}
    except Exception:  # noqa: BLE001
        return None


def gpu_name() -> str | None:
    """Nom de la carte du job (ou de la premiere)."""
    try:
        import torch
        i = _index()
        if i is None or not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_name(i)
    except Exception:  # noqa: BLE001
        return None
