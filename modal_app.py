"""Spark GPU Inference sur Modal — la MÊME image que RunPod, un endpoint web POST par compte.

ASYNCHRONE (2026-09-09) : l'endpoint web ne tient plus la requête ouverte pendant
le job. Il répond tout de suite (`submit` → `call_id`), le travail tourne dans un
conteneur GPU à part (`SparkGpu.process`, lancé par `.spawn()`), et c'est l'IMAGE
qui prévient la plateforme par rappels HTTP (`started` / `heartbeat` /
`finished`, cf. src/spark_infer/webhooks.py). Le même endpoint sert aussi de
sonde (`status`) et de coupe-circuit (`cancel`) quand un rappel ne vient pas.

Reprend le schéma de `demucs-separation` : image GHCR prise telle quelle
(`Image.from_registry`, aucun rebuild), clé partagée `MODAL_API_KEY` lue dans le
secret Modal `modal-api-key` (déjà présent sur les six comptes), un conteneur
GPU = un job à la fois, ce qui laisse le DO GpuPool compter 10 jobs simultanés
par compte.

Contrat de l'endpoint (POST JSON, `api_key` obligatoire) :
    {"action": "submit", ...même JSON que RunPod (README.md)...}  → {"status":"queued","job_id","call_id"}
    {"action": "status", "call_id": "…"}                         → {"status":"running"|"completed"|"failed"|"unknown", "result"?}
    {"action": "cancel", "call_id": "…"}                         → {"status":"cancelled"}
Sans `action`, `submit`. L'URL est inchangée : la classe web s'appelle toujours
`SparkInference` et sa méthode `run` (URL `…--spark-gpu-inference-sparkinference-run.modal.run`).

Déploiement, un compte à la fois (profils de `~/.modal.toml`) :
    MODAL_PROFILE=compte2 modal deploy modal_app.py
Image privée sur GHCR : créer sur chaque compte un secret `ghcr-pull`
(REGISTRY_USERNAME = login GitHub, REGISTRY_PASSWORD = jeton `read:packages`) et déployer avec
    GHCR_PRIVATE=1 MODAL_PROFILE=compte2 modal deploy modal_app.py
"""
from __future__ import annotations

import logging
import os
import uuid

import modal

IMAGE = os.environ.get("SPARK_IMAGE", "ghcr.io/cyrille8000/spark-gpu-inference:sha-93f1445")
GPU = os.environ.get("SPARK_MODAL_GPU", "L4")

registry_secret = modal.Secret.from_name("ghcr-pull") if os.environ.get("GHCR_PRIVATE") == "1" else None

gpu_image = (
    modal.Image.from_registry(IMAGE, secret=registry_secret)
    # Les ENV du Dockerfile, redits ici pour ne dépendre d'aucune reprise implicite par Modal.
    .env({
        "PYTHONPATH": "/app/src",
        "PYTHONUNBUFFERED": "1",
        "SPARK_MODELS_DIR": "/models",
        "BS_ROFORMER_MODELS_PATH": "/models/bsroformer",
        "HF_HOME": "/models/hf",
        "TORCH_HOME": "/models/torch",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
        # Nommé dans chaque rappel et dans le résultat (`provider`).
        "SPARK_PROVIDER": "modal",
    })
    # LE CODE VIENT DU DÉPÔT LOCAL, pas de l'image : monté PAR-DESSUS /app/src/spark_infer (le
    # chemin que PYTHONPATH fait lire en premier ; dans /root, l'image gagnait). Ce qui est déployé
    # sur Modal est donc exactement ce qui est dans le dépôt au moment du `modal deploy`, sans
    # attendre un build GHCR ; les poids et les paquets, eux, viennent de l'image.
    .add_local_dir("src/spark_infer", remote_path="/app/src/spark_infer", ignore=["**/__pycache__"])
)

# L'endpoint web n'a besoin ni de GPU ni des poids : une image minuscule, un
# conteneur CPU qui répond en millisecondes et ne coûte rien entre deux appels.
web_image = modal.Image.debian_slim(python_version="3.11").pip_install("fastapi[standard]")

app = modal.App("spark-gpu-inference")


@app.cls(
    image=gpu_image,
    gpu=GPU,
    timeout=900,
    # 10 s (était 120) : Modal facture l'inactivité APRÈS un job. L'image la
    # rapporte (`container_s`) jusqu'au rapport suivant seulement — la queue
    # finale reste hors compteur, donc on la garde courte.
    scaledown_window=10,
    secrets=[modal.Secret.from_name("modal-api-key")],
)
class SparkGpu:
    @modal.enter()
    def start(self) -> None:
        logging.basicConfig(level=os.environ.get("SPARK_LOG_LEVEL", "INFO"),
                            format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        from pathlib import Path
        for sub in ("bsroformer", "chatterbox"):
            assert (Path("/models") / sub).is_dir(), f"/models/{sub} absent de l'image"
        # Le code monté doit être celui du dépôt : `service.py` n'existe pas dans l'image de base.
        import spark_infer
        import spark_infer.service  # noqa: F401 — échoue tout de suite si le montage n'a pas pris
        print(f"[modal] image={IMAGE} gpu={GPU} code={Path(spark_infer.__file__).parent}", flush=True)

    @modal.method()
    def process(self, inp: dict, job_id: str) -> dict:
        from spark_infer.service import process_job

        return process_job(inp, job_id)


@app.cls(image=web_image, secrets=[modal.Secret.from_name("modal-api-key")])
class SparkInference:
    @modal.enter()
    def start(self) -> None:
        self.api_key = os.environ.get("MODAL_API_KEY", "")
        print(f"[modal] endpoint web prêt, api_key={'oui' if self.api_key else 'NON'}", flush=True)

    @modal.fastapi_endpoint(method="POST")
    def run(self, input_data: dict) -> dict:
        if self.api_key and input_data.get("api_key") != self.api_key:
            return {"status": "error", "error": "unauthorized", "code": "unauthorized"}
        action = str(input_data.get("action") or "submit")
        inp = {k: v for k, v in input_data.items() if k not in ("api_key", "action", "call_id")}

        if action == "submit":
            job_id = f"modal-{uuid.uuid4().hex[:16]}"
            call = SparkGpu().process.spawn(inp, job_id)
            return {"status": "queued", "job_id": job_id, "call_id": call.object_id}

        call_id = str(input_data.get("call_id") or "")
        if not call_id:
            return {"status": "error", "error": "call_id requis", "code": "bad_input"}
        try:
            fc = modal.FunctionCall.from_id(call_id)
        except Exception as e:  # noqa: BLE001
            return {"status": "error", "error": f"call_id inconnu : {e}", "code": "bad_input"}

        if action == "status":
            try:
                return {"status": "completed", "result": fc.get(timeout=0)}
            except TimeoutError:
                return {"status": "running"}
            except modal.exception.NotFoundError as e:
                # Modal ne connaît pas (ou plus) ce call_id : ce n'est pas un job échoué.
                return {"status": "unknown", "error": f"{type(e).__name__}: {e}"}
            except Exception as e:  # noqa: BLE001 — la fonction a levé (process_job ne lève jamais : conteneur mort, timeout Modal)
                return {"status": "failed", "error": f"{type(e).__name__}: {e}"}
        if action == "cancel":
            try:
                fc.cancel(terminate_containers=True)
                return {"status": "cancelled"}
            except Exception as e:  # noqa: BLE001
                return {"status": "error", "error": f"cancel : {e}", "code": "internal"}
        return {"status": "error", "error": f"action inconnue : {action}", "code": "bad_input"}
