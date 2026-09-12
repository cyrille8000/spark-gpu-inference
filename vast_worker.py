"""Worker Vast.ai : la MÊME inférence que Modal et RunPod, servie par un petit
serveur HTTP, avec PLUSIEURS jobs en parallèle sur la même carte.

Pourquoi un troisième point d'entrée. Modal et RunPod sont sans serveur : ils
démarrent un conteneur par job et le facturent à la seconde. Un pod Vast.ai se
LOUE À L'HEURE, carte entière : on ne rentabilise qu'en y faisant tenir plusieurs
jobs à la fois. D'où ce worker, et d'où le pool d'instances du cœur
(`spark_infer.registry.lease`) : chaque job emprunte SON exemplaire du modèle,
donc la conversion vocale ne peut plus écraser la voix de référence du voisin.

Combien de jobs à la fois : `SPARK_JOBS_PER_GPU` (défaut 1). C'est la taille des
pools ET le nombre d'entrées servies en même temps. À régler d'après la mesure
`gpu_mem` que chaque job rapporte, jamais au hasard : une instance de plus coûte
ses poids sur la carte, et le débit ne suit que si le GPU n'était pas déjà saturé.

    POST /run       exécute un job et rend son résultat (synchrone)
    POST /submit    accepte un job, l'exécute en fond, rend compte par `callback_url`
    GET  /status    carte, mémoire, jobs en cours, pools, temps de vie
    GET  /health    200 tant que le processus répond
    POST /shutdown  arrête le worker (le pod peut alors être détruit)

SÉCURITÉ. Un pod Vast.ai est sur l'Internet public : sans jeton, n'importe qui
utiliserait la carte. `SPARK_WORKER_TOKEN` est donc OBLIGATOIRE — le worker
refuse de démarrer sans. Chaque requête le porte en `Authorization: Bearer …`
ou dans le champ `api_key` du corps (comme l'endpoint web Modal).

ARRÊT AUTOMATIQUE. Un pod oublié coûte jusqu'à ce qu'on le détruise :
`SPARK_IDLE_EXIT_S` (défaut 900) arrête le worker après ce temps sans aucun job.
`0` désactive — à réserver aux mesures faites à la main.

LA CARTE DOIT POUVOIR EXÉCUTER CETTE IMAGE. Vast.ai loue des machines de toutes
générations, avec des pilotes de toutes époques ; l'image embarque torch 2.7.1
compilé pour CUDA 12.8. Deux façons d'échouer, et les deux se paient à l'heure
dès que le pod démarre :
  · pilote trop ancien — le conteneur voit la carte mais CUDA refuse de s'initialiser ;
  · architecture non compilée dans les roues (Pascal `sm_61` et avant, typiquement
    les P40 24 Go encore nombreuses là-bas) — « no kernel image is available ».
`verifier_carte()` tranche AU DÉMARRAGE, avant d'accepter le moindre job : le
worker s'arrête avec un message clair, le pod peut être détruit tout de suite.
Côté location, filtrer les offres sur `cuda_max_good >= 12.8` et la mémoire
demandée (`SPARK_MIN_VRAM_GB`, défaut 24) évite d'en arriver là.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, "/app/src")

os.environ.setdefault("SPARK_PROVIDER", "vastai")  # nommé dans chaque rappel et dans le résultat

from spark_infer import registry  # noqa: E402
from spark_infer.container_clock import CLOCK  # noqa: E402
from spark_infer.service import process_job  # noqa: E402
from spark_infer.tasks import jobs_per_gpu  # noqa: E402

log = logging.getLogger("spark.vast")

MAX_BODY = 1 << 20  # 1 Mio : un job est un petit JSON d'URL, jamais un média


def idle_exit_s() -> int:
    """Secondes sans aucun job avant que le worker s'arrête (0 = jamais)."""
    try:
        return max(0, int(os.environ.get("SPARK_IDLE_EXIT_S", "900")))
    except ValueError:
        return 900


def jeton_valide(attendu: str, entete: str | None, corps_cle: str | None) -> bool:
    """Le jeton de la requête vaut-il celui du worker ? Comparaison à temps constant
    (une comparaison naïve laisse deviner le jeton caractère par caractère). Pur."""
    import hmac

    donne = ""
    if entete and entete.lower().startswith("bearer "):
        donne = entete[7:].strip()
    elif corps_cle:
        donne = str(corps_cle)
    return bool(attendu) and hmac.compare_digest(donne, attendu)


class Etat:
    """Ce que le worker sait de lui-même : jobs en cours, dernier moment d'activité."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.actifs = 0
        self.faits = 0
        self.dernier = time.monotonic()

    def debut(self) -> None:
        with self._lock:
            self.actifs += 1
            self.dernier = time.monotonic()

    def fin(self) -> None:
        with self._lock:
            self.actifs = max(0, self.actifs - 1)
            self.faits += 1
            self.dernier = time.monotonic()

    def inactif_depuis(self, now: float | None = None) -> float:
        t = time.monotonic() if now is None else now
        with self._lock:
            return 0.0 if self.actifs > 0 else max(0.0, t - self.dernier)


ETAT = Etat()


def _cuda_info() -> dict | None:
    """Ce pour quoi l'image est compilée et ce que la carte est : à comparer en cas de doute."""
    try:
        import torch
        return {"torch": torch.__version__, "compile_pour": torch.version.cuda,
                "sm": "sm_{}{}".format(*torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
                "arch_list": list(torch.cuda.get_arch_list()) if torch.cuda.is_available() else []}
    except Exception:  # noqa: BLE001
        return None


def memoire_carte() -> dict | None:
    """Mémoire de la carte à l'instant présent (libre / totale, Go) — l'état réel,
    distinct du PIC d'un job (`gpu_mem`). None sans CUDA."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        libre, total = torch.cuda.mem_get_info()
        return {"libre_gb": round(libre / 1e9, 2), "total_gb": round(total / 1e9, 2)}
    except Exception:  # noqa: BLE001
        return None


def carte_supportee(sm: str, arch_list: list[str]) -> bool:
    """L'architecture de la carte (`sm_86`) est-elle exécutable par ces roues torch ?

    Vrai si elle est compilée telle quelle, ou si un PTX plus ancien est embarqué
    (`compute_80` pour une `sm_86`) : le pilote le compile alors au premier noyau,
    au prix de quelques secondes. Faux pour une carte plus ANCIENNE que tout ce
    qui est embarqué — là, rien ne peut s'exécuter. Pur.
    """
    if sm in arch_list:
        return True
    try:
        n = int(sm.removeprefix("sm_"))
    except ValueError:
        return False
    ptx = [int(a.removeprefix("compute_")) for a in arch_list if a.startswith("compute_")]
    return any(p <= n for p in ptx)


def min_vram_gb() -> float:
    try:
        return float(os.environ.get("SPARK_MIN_VRAM_GB", "24"))
    except ValueError:
        return 24.0


def verifier_carte() -> dict:
    """Refuse de démarrer si la machine louée ne peut pas faire tourner l'image.
    Rend la description de la carte, ou lève `SystemExit` avec la raison."""
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        log.error("torch indisponible dans l'image : %s", e)
        raise SystemExit(3) from e
    if not torch.cuda.is_available():
        log.error("aucune carte utilisable : pilote NVIDIA trop ancien pour CUDA %s, "
                  "conteneur lancé sans `--gpus`, ou machine sans GPU", torch.version.cuda)
        raise SystemExit(3)
    nom = torch.cuda.get_device_name(0)
    majeur, mineur = torch.cuda.get_device_capability(0)
    sm = f"sm_{majeur}{mineur}"
    arch = list(torch.cuda.get_arch_list())
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if not carte_supportee(sm, arch):
        log.error("%s (%s) n'est pas exécutable par ces roues torch %s+cu%s (compilées pour %s) — "
                  "louer une carte Volta ou plus récente", nom, sm, torch.__version__, torch.version.cuda, arch)
        raise SystemExit(3)
    if total_gb + 0.5 < min_vram_gb():
        log.error("%s n'a que %.1f Go : moins que les %.0f Go demandés (SPARK_MIN_VRAM_GB) — "
                  "un chunk long déborderait en plein job", nom, total_gb, min_vram_gb())
        raise SystemExit(3)
    # Un vrai calcul, pas seulement l'inventaire : c'est lui qui révèle un pilote
    # boiteux ou une carte déjà occupée par un autre locataire.
    try:
        x = torch.zeros(64, 64, device="cuda")
        torch.mm(x, x)
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001
        log.error("la carte %s refuse un calcul élémentaire : %s", nom, e)
        raise SystemExit(3) from e
    log.info("carte acceptée : %s (%s, %.1f Go), torch %s+cu%s", nom, sm, total_gb,
             torch.__version__, torch.version.cuda)
    return {"gpu_name": nom, "sm": sm, "vram_total_gb": round(total_gb, 1),
            "torch": torch.__version__, "cuda": torch.version.cuda, "arch_list": arch}


def etat_worker() -> dict:
    return {
        "ok": True,
        "provider": os.environ.get("SPARK_PROVIDER", "vastai"),
        "gpu_name": registry.gpu_name(),
        "device": registry.device(),
        "vram_total_gb": registry.vram_total_gb(),
        "vram": memoire_carte(),
        "cuda": _cuda_info(),
        "jobs_per_gpu": jobs_per_gpu(),
        "jobs_actifs": ETAT.actifs,
        "jobs_faits": ETAT.faits,
        "pools": registry.pool_state(),
        "uptime_s": round(CLOCK.uptime(), 1),
        "inactif_s": round(ETAT.inactif_depuis(), 1),
        "idle_exit_s": idle_exit_s(),
    }


def executer(inp: dict, job_id: str) -> dict:
    """Un job, du début à la fin. `process_job` ne lève jamais : il rend une erreur typée."""
    ETAT.debut()
    try:
        return process_job(inp, job_id)
    finally:
        ETAT.fin()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "spark-vast/1.0"

    # ── plomberie ──
    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        log.info("%s - %s", self.address_string(), fmt % args)

    def _rendre(self, code: int, corps: dict) -> None:
        data = json.dumps(corps).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _corps(self) -> tuple[dict | None, str | None]:
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return None, "Content-Length illisible"
        if n <= 0:
            return {}, None
        if n > MAX_BODY:
            return None, "corps trop volumineux"
        try:
            return json.loads(self.rfile.read(n).decode("utf-8")), None
        except Exception as e:  # noqa: BLE001
            return None, f"JSON invalide : {e}"

    def _autorise(self, corps: dict | None) -> bool:
        if jeton_valide(TOKEN, self.headers.get("Authorization"), (corps or {}).get("api_key")):
            return True
        self._rendre(401, {"ok": False, "error": "unauthorized"})
        return False

    # ── routes ──
    def do_GET(self) -> None:  # noqa: N802
        route = self.path.split("?", 1)[0].rstrip("/") or "/"
        if route == "/health":
            self._rendre(200, {"ok": True})
            return
        if route == "/status":
            if not self._autorise(None):
                return
            self._rendre(200, etat_worker())
            return
        self._rendre(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        route = self.path.split("?", 1)[0].rstrip("/") or "/"
        corps, err = self._corps()
        if err is not None:
            self._rendre(400, {"ok": False, "error": err})
            return
        if not self._autorise(corps):
            return
        job_id = str(corps.get("job_id") or f"vast_{uuid.uuid4().hex[:12]}")

        if route == "/run":
            self._rendre(200, executer(corps, job_id))
            return
        if route == "/submit":
            # Accepté tout de suite ; le résultat part par `callback_url` (même
            # protocole que Modal et RunPod), la plateforme n'attend aucune connexion.
            threading.Thread(target=executer, args=(corps, job_id), daemon=True).start()
            self._rendre(202, {"ok": True, "accepted": True, "job_id": job_id})
            return
        if route == "/shutdown":
            self._rendre(200, {"ok": True, "stopping": True})
            threading.Thread(target=lambda: (time.sleep(0.3), os._exit(0)), daemon=True).start()
            return
        self._rendre(404, {"ok": False, "error": "not found"})


def surveiller_inactivite(serveur: ThreadingHTTPServer) -> None:
    """Arrête le worker après `SPARK_IDLE_EXIT_S` sans job : un pod loué à l'heure
    et oublié coûte jusqu'à sa destruction."""
    limite = idle_exit_s()
    if limite <= 0:
        log.info("arrêt automatique désactivé (SPARK_IDLE_EXIT_S=0)")
        return
    while True:
        time.sleep(15)
        inactif = ETAT.inactif_depuis()
        if inactif >= limite:
            log.warning("%.0f s sans job (limite %d) — arrêt du worker", inactif, limite)
            serveur.shutdown()
            os._exit(0)


TOKEN = os.environ.get("SPARK_WORKER_TOKEN", "")


def main() -> None:
    logging.basicConfig(level=os.environ.get("SPARK_LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not TOKEN:
        log.error("SPARK_WORKER_TOKEN absent : un pod public sans jeton offrirait sa carte à tout Internet")
        raise SystemExit(2)
    carte = verifier_carte()
    port = int(os.environ.get("SPARK_WORKER_PORT", "8000"))
    serveur = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    serveur.daemon_threads = True
    threading.Thread(target=surveiller_inactivite, args=(serveur,), daemon=True).start()
    log.info("worker prêt sur :%d — %s (%s), %s Go, %d job(s) en parallèle, arrêt après %d s d'inactivité",
             port, carte["gpu_name"], carte["sm"], carte["vram_total_gb"], jobs_per_gpu(), idle_exit_s())
    serveur.serve_forever()


if __name__ == "__main__":
    main()
