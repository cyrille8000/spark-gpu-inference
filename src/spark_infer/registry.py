"""Modèles résidents : un POOL d'instances par modèle, empruntée le temps d'un job.

Avant le 2026-09-12, UNE instance par modèle servait tous les jobs — ça tenait
parce qu'un conteneur ne traitait qu'un job à la fois. Dès que deux jobs se
croisent, ce partage devient FAUX pour la conversion vocale : chaque job écrit
sa configuration et sa voix de référence DANS le modèle (`vc_engine._configure`,
`_set_reference`), donc le second écraserait la référence du premier, qui
rendrait une voix étrangère sans que rien ne le signale.

D'où le pool : un job EMPRUNTE une instance pour lui seul (`lease`) et la rend à
la fin. Le pool en crée jusqu'à `max_instances` ; au-delà, le job attend son
tour. Chaque instance coûte ses poids sur la carte : c'est la mesure `gpu_mem`
de chaque job qui dit combien on peut en tenir sur un GPU donné.

Le chargement d'une instance (des dizaines de secondes) se fait HORS du verrou :
sa place est réservée d'abord, pour que deux jobs simultanés n'en créent pas
trois, et rendue si le chargement échoue.
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
    """Les instances d'UN modèle : celles qui attendent, et le compte de celles qui existent."""

    max_size: int
    idle: list = field(default_factory=list)
    total: int = 0   # instances existantes ou en cours de chargement
    busy: int = 0    # instances actuellement empruntées


_cond = threading.Condition()
_pools: dict[str, _Pool] = {}


def _pool(name: str, max_instances: int) -> _Pool:
    p = _pools.get(name)
    if p is None:
        p = _pools[name] = _Pool(max_size=max(1, int(max_instances)))
    else:
        p.max_size = max(1, int(max_instances))
    return p


@contextmanager
def lease(name: str, factory: Callable[[], T], max_instances: int = 1) -> Iterator[tuple[T, bool]]:
    """Emprunte une instance du modèle pour la durée du bloc.

    Rend `(instance, chargée_maintenant)`. Attend si le pool est plein. L'instance
    revient au pool à la sortie du bloc, même sur exception : un job qui échoue ne
    doit pas retirer une place au suivant.
    """
    inst, cold = _acquire(name, factory, max_instances)
    try:
        yield inst, cold
    finally:
        _give_back(name, inst)


def _acquire(name: str, factory: Callable[[], T], max_instances: int) -> tuple[T, bool]:
    while True:
        with _cond:
            p = _pool(name, max_instances)
            if p.idle:
                p.busy += 1
                return p.idle.pop(), False
            if p.total < p.max_size:
                p.total += 1   # place réservée AVANT le chargement, qui se fait hors verrou
                p.busy += 1
                break
            _cond.wait(timeout=30.0)
    log.info("chargement d'une instance de « %s » (%d/%d)", name, p.total, p.max_size)
    try:
        return factory(), True
    except BaseException:
        with _cond:
            p.total -= 1
            p.busy -= 1
            _cond.notify_all()
        raise


def _give_back(name: str, inst: object) -> None:
    with _cond:
        p = _pools.get(name)
        if p is None:
            return
        p.busy = max(0, p.busy - 1)
        p.idle.append(inst)
        _cond.notify_all()


def get(name: str, factory: Callable[[], T]) -> T:
    """Emprunt SANS rendu — pour un appelant qui garde l'instance (tests, outils).
    Le chemin normal est `lease`."""
    inst, _ = _acquire(name, factory, 1)
    return inst


def active() -> int:
    """Nombre d'instances actuellement empruntées, tous modèles confondus : le nombre
    de jobs qui se partagent la carte."""
    with _cond:
        return sum(p.busy for p in _pools.values())


def pool_state() -> dict:
    """État des pools, pour le diagnostic et le résultat d'un job."""
    with _cond:
        return {n: {"total": p.total, "busy": p.busy, "max": p.max_size} for n, p in sorted(_pools.items())}


def loaded() -> list[str]:
    """Les modèles qui ont au moins une instance (chargée ou en cours)."""
    with _cond:
        return sorted(n for n, p in _pools.items() if p.total > 0)


def others_loaded(keep: str | None) -> list[str]:
    """Les modèles résidents AUTRES que `keep` — ceux qu'une libération rendrait."""
    return [m for m in loaded() if m != keep]


def release(name: str | None = None) -> None:
    """Libère les instances LIBRES (jamais celles qu'un job tient) et vide le cache CUDA.

    Sur OOM, on ne peut rendre que ce que personne n'utilise : voler l'instance d'un
    job concurrent le ferait échouer sans raison.
    """
    with _cond:
        for n in ([name] if name else list(_pools)):
            p = _pools.get(n)
            if not p or not p.idle:
                continue
            log.info("libération de %d instance(s) libre(s) de « %s »", len(p.idle), n)
            p.total -= len(p.idle)
            p.idle.clear()
            if p.total <= 0 and p.busy == 0:
                _pools.pop(n, None)
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
    """Vide tout l'état du registre — tests seulement."""
    with _cond:
        _pools.clear()
        _cond.notify_all()


def device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def vram_total_gb() -> float | None:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_properties(0).total_memory / 1e9
    except Exception:  # noqa: BLE001
        return None


def reset_peak_memory() -> None:
    """Remet à zéro le compteur de PIC mémoire CUDA. À n'appeler que si AUCUN autre
    job ne tourne : le compteur est global à la carte, le remettre à zéro pendant le
    job d'à côté fausserait sa mesure."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001
        pass


def peak_memory_gb() -> dict | None:
    """Pic mémoire depuis la dernière remise à zéro (Go) : `allocated` = tenseurs
    vivants au plus haut, `reserved` = ce que l'allocateur a pris à la carte — c'est
    `reserved` qui décide si un GPU suffit. Le compteur est celui de la CARTE : quand
    plusieurs jobs se croisent, le pic est celui de leur somme (cf. `gpu_mem.jobs`).
    None sans CUDA."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return {"allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                "reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2)}
    except Exception:  # noqa: BLE001
        return None


def gpu_name() -> str | None:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        return None
