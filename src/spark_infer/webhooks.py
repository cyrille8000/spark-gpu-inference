"""Rappels HTTP de l'image vers la plateforme : `started`, `heartbeat`, `finished`.

La plateforme ne tient plus de connexion ouverte pendant le job : elle soumet,
puis l'IMAGE lui raconte ce qui se passe — début (GPU, conteneur neuf ou
réutilisé), battements réguliers (signe de vie + progression), fin (le résultat
complet, avec tout ce qu'il faut pour facturer : `container_s`, `timings`,
`gpu_name`, octets déposés…). Chaque rappel porte `meta`, l'objet OPAQUE que la
plateforme a mis dans l'entrée (projet, portion, tentative, compte…), renvoyé
tel quel, et `seq`, un compteur croissant par job pour trier les livraisons.

Aucun rappel ne fait jamais échouer un job : un POST raté est journalisé et
c'est tout (le résultat reste disponible côté hébergeur : statut RunPod,
`FunctionCall` Modal).
"""
from __future__ import annotations

import datetime as dt
import logging
import threading
import time
from typing import Any, Callable

from .io_utils import send_callback

log = logging.getLogger("spark.webhooks")

# (url, token, payload, retries) -> livré ?
Sender = Callable[[str, "str | None", dict, int], bool]


def now_iso() -> str:
    """Horodatage UTC ISO 8601 en millisecondes, suffixe Z."""
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class JobWebhooks:
    """Les trois rappels d'UN job, dans l'ordre. `send` est injectable (tests)."""

    def __init__(self, url: str, token: str | None, job_id: str, task: str | None,
                 meta: dict | None, provider: str, send: Sender = send_callback) -> None:
        self.url = url
        self.token = token
        self.job_id = job_id
        self.task = task
        self.meta = meta
        self.provider = provider
        self.send = send
        self.seq = 0
        self.started_at: str | None = None

    def _base(self, event: str) -> dict:
        self.seq += 1
        return {
            "event": event, "seq": self.seq, "job_id": self.job_id, "task": self.task,
            "meta": self.meta, "provider": self.provider, "sent_at": now_iso(),
        }

    def started(self, *, gpu_name: str | None, device: str, container_first_job: bool,
                container_uptime_s: float, models_loaded: list[str]) -> bool:
        self.started_at = now_iso()
        payload = {
            **self._base("started"),
            "started_at": self.started_at, "gpu_name": gpu_name, "device": device,
            "container_first_job": container_first_job,
            "container_uptime_s": round(container_uptime_s, 3),
            "models_loaded": list(models_loaded),
        }
        return self.send(self.url, self.token, payload, 3)

    def heartbeat(self, *, elapsed_s: float, progress: dict | None) -> bool:
        # Un seul essai : le battement suivant vaut mieux qu'un battement en retard.
        payload = {**self._base("heartbeat"), "elapsed_s": round(elapsed_s, 3), "progress": progress}
        return self.send(self.url, self.token, payload, 1)

    def finished(self, result: dict) -> bool:
        # Le résultat ENTIER (sans base64 : trop gros pour un rappel) + l'en-tête commun.
        payload = {k: v for k, v in result.items() if k != "audio_base64"}
        payload.update(self._base("finished"))
        payload["task"] = payload.get("task") or self.task
        return self.send(self.url, self.token, payload, 3)


class Heartbeat:
    """Un fil d'arrière-plan qui bat toutes les `interval_s` secondes pendant le job.
    `progress` est mis à jour par le job (dernier rapport de progression) et
    part avec chaque battement. Sans rappel configuré : ne fait rien."""

    def __init__(self, hooks: JobWebhooks | None, interval_s: float) -> None:
        self.hooks = hooks
        self.interval_s = max(1.0, float(interval_s))
        self.progress: dict | None = None
        self.sent = 0
        self._stop = threading.Event()
        self._t0 = time.monotonic()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self.hooks is None:
            return
        self._t0 = time.monotonic()
        self._thread = threading.Thread(target=self._run, name="spark-heartbeat", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                if self.hooks.heartbeat(elapsed_s=time.monotonic() - self._t0, progress=self.progress):
                    self.sent += 1
            except Exception as e:  # noqa: BLE001 — un battement raté n'a aucun droit sur le job
                log.warning("battement raté : %s", e)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)


def elapsed_since(t0: float) -> float:
    return time.monotonic() - t0


__all__: list[Any] = ["JobWebhooks", "Heartbeat", "now_iso", "elapsed_since"]
