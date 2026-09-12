"""Horloge du conteneur : le temps que ce processus a occupé, RÉPARTI entre les
jobs qui s'y sont croisés.

Modal et RunPod facturent le CONTENEUR, du démarrage à l'extinction, pas
seulement le temps passé dans le handler (`elapsed_s`). Tant qu'un conteneur ne
traitait qu'un job à la fois, chaque job rapportait simplement la fenêtre écoulée
depuis le rapport précédent (boot + attente + job).

Depuis le 2026-09-12 un conteneur peut traiter PLUSIEURS jobs en même temps, et
cette fenêtre n'appartient plus à un seul : chaque seconde est partagée entre les
jobs actifs à cet instant. Une seconde où trois jobs tournent vaut un tiers de
seconde pour chacun. Le temps où AUCUN job ne tourne (boot, attente entre deux
jobs) est mis de côté et revient au prochain job qui démarre. La somme des parts
de tous les jobs reste donc la vie entière du conteneur, à la queue d'inactivité
finale près (`scaledown_window` Modal, idle timeout RunPod), qu'on garde courte.

Mesure réelle, pas estimation : `time.monotonic()` du processus. Ce qui précède
le démarrage du processus (l'hébergeur qui tire l'image) n'est pas visible d'ici.
"""
from __future__ import annotations

import threading
import time


class ContainerClock:
    """Compteur par processus, sûr entre fils : plusieurs jobs peuvent entrer et sortir."""

    def __init__(self, now: float | None = None) -> None:
        t = time.monotonic() if now is None else now
        self._start = t
        self._mark = t                      # dernier instant distribué
        self._pending = 0.0                 # temps sans aucun job actif, en attente
        self._parts: dict[str, float] = {}  # part accumulée par job en cours
        self._done = 0                      # jobs ayant fermé leur part
        self._lock = threading.Lock()

    # ── lecture ──
    def is_first(self) -> bool:
        """Vrai tant qu'aucun job n'a fermé sa part : le conteneur n'a encore rien rendu."""
        return self._done == 0

    def uptime(self, now: float | None = None) -> float:
        """Secondes depuis le démarrage du processus."""
        t = time.monotonic() if now is None else now
        return max(0.0, t - self._start)

    def active(self) -> int:
        """Jobs actuellement en cours dans ce conteneur."""
        with self._lock:
            return len(self._parts)

    # ── comptage ──
    def _distribute(self, now: float) -> None:
        """Attribue le temps écoulé depuis `_mark` : aux jobs actifs, ou à l'attente."""
        delta = max(0.0, now - self._mark)
        self._mark = now
        if not self._parts:
            self._pending += delta
            return
        part = delta / len(self._parts)
        for k in self._parts:
            self._parts[k] += part

    def enter(self, job_id: str, now: float | None = None) -> None:
        """Un job commence : le temps mort accumulé (boot, attente) lui est attribué."""
        t = time.monotonic() if now is None else now
        with self._lock:
            self._distribute(t)
            self._parts[job_id] = self._parts.get(job_id, 0.0) + self._pending
            self._pending = 0.0

    def leave(self, job_id: str, now: float | None = None) -> tuple[float, bool]:
        """Un job finit : (secondes de conteneur qui lui reviennent, premier job à finir ?)."""
        t = time.monotonic() if now is None else now
        with self._lock:
            self._distribute(t)
            seconds = self._parts.pop(job_id, 0.0)
            first = self._done == 0
            self._done += 1
            return round(seconds, 3), first


CLOCK = ContainerClock()
