"""Horloge du conteneur : le temps que ce processus a occupé depuis le job
précédent, ou depuis son démarrage pour le premier job.

Modal et RunPod facturent le CONTENEUR, du démarrage à l'extinction — pas
seulement le temps passé dans le handler (`elapsed_s`). Comme un conteneur
traite ses jobs l'un après l'autre (un input à la fois, c'est le réglage ici),
il peut rapporter à la fin de chaque job la fenêtre qu'il a occupée depuis le
rapport précédent : boot du processus + attente + job. La somme sur tous les
jobs d'un conteneur = sa vie entière, à la queue d'inactivité finale près
(`scaledown_window` Modal, idle timeout RunPod), qu'on garde courte.

Mesure réelle, pas estimation : `time.monotonic()` du processus. Ce qui précède
le démarrage du processus (chargement de l'image par l'hébergeur) n'est pas
visible d'ici et reste hors compteur.
"""
from __future__ import annotations

import time


class ContainerClock:
    """Un compteur par processus. Pas de verrou : un job à la fois par conteneur."""

    def __init__(self, now: float | None = None) -> None:
        self._last = time.monotonic() if now is None else now
        self._jobs = 0

    def window(self, now: float | None = None) -> tuple[float, bool]:
        """Ferme la fenêtre courante : (secondes depuis le rapport précédent, premier job du conteneur ?)."""
        t = time.monotonic() if now is None else now
        first = self._jobs == 0
        seconds = max(0.0, t - self._last)
        self._last = t
        self._jobs += 1
        return seconds, first


CLOCK = ContainerClock()
