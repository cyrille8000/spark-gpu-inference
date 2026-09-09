"""Spark GPU Inference sur Modal — la MÊME image que RunPod, un endpoint web POST par compte.

Reprend le schéma de `demucs-separation` : image GHCR prise telle quelle (`Image.from_registry`,
aucun rebuild), clé partagée `MODAL_API_KEY` lue dans le secret Modal `modal-api-key` (déjà
présent sur les cinq comptes), un conteneur = un GPU = un job à la fois, ce qui laisse le
client rond-robin de l'orchestrateur compter 10 jobs simultanés par compte.

Contrat : le MÊME JSON que RunPod (`README.md`), posté directement (pas de champ `input`),
plus `api_key`. La réponse est le même JSON de sortie ; Modal répond quand c'est fini.

Déploiement, un compte à la fois (profils de `~/.modal.toml`) :
    MODAL_PROFILE=compte2 modal deploy modal_app.py
Image privée sur GHCR : créer sur chaque compte un secret `ghcr-pull`
(REGISTRY_USERNAME = login GitHub, REGISTRY_PASSWORD = jeton `read:packages`) et déployer avec
    GHCR_PRIVATE=1 MODAL_PROFILE=compte2 modal deploy modal_app.py
"""
from __future__ import annotations

import logging
import os

import modal

IMAGE = os.environ.get("SPARK_IMAGE", "ghcr.io/cyrille8000/spark-gpu-inference:sha-93f1445")
GPU = os.environ.get("SPARK_MODAL_GPU", "L4")

registry_secret = modal.Secret.from_name("ghcr-pull") if os.environ.get("GHCR_PRIVATE") == "1" else None

image = (
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
    })
    # Seule dépendance propre à Modal : le serveur web de l'endpoint.
    .pip_install("fastapi[standard]")
    # LE CODE VIENT DU DÉPÔT LOCAL, pas de l'image : monté PAR-DESSUS /app/src/spark_infer (le
    # chemin que PYTHONPATH fait lire en premier ; dans /root, l'image gagnait). Ce qui est déployé
    # sur Modal est donc exactement ce qui est dans le dépôt au moment du `modal deploy`, sans
    # attendre un build GHCR ; les poids et les paquets, eux, viennent de l'image.
    .add_local_dir("src/spark_infer", remote_path="/app/src/spark_infer", ignore=["**/__pycache__"])
)

app = modal.App("spark-gpu-inference")


@app.cls(
    image=image,
    gpu=GPU,
    timeout=900,
    scaledown_window=120,
    secrets=[modal.Secret.from_name("modal-api-key")],
)
class SparkInference:
    @modal.enter()
    def start(self) -> None:
        logging.basicConfig(level=os.environ.get("SPARK_LOG_LEVEL", "INFO"),
                            format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        self.api_key = os.environ.get("MODAL_API_KEY", "")
        from pathlib import Path
        for sub in ("bsroformer", "chatterbox"):
            assert (Path("/models") / sub).is_dir(), f"/models/{sub} absent de l'image"
        # Le code monté doit être celui du dépôt : `service.py` n'existe pas dans l'image de base.
        import spark_infer
        import spark_infer.service  # noqa: F401 — échoue tout de suite si le montage n'a pas pris
        print(f"[modal] image={IMAGE} gpu={GPU} api_key={'oui' if self.api_key else 'NON'} "
              f"code={Path(spark_infer.__file__).parent}", flush=True)

    @modal.fastapi_endpoint(method="POST")
    def run(self, input_data: dict) -> dict:
        if self.api_key and input_data.get("api_key") != self.api_key:
            return {"status": "error", "error": "unauthorized", "code": "unauthorized"}
        from spark_infer.service import process_job

        inp = {k: v for k, v in input_data.items() if k != "api_key"}
        job_id = f"modal-{modal.current_input_id() or 'local'}"
        return process_job(inp, job_id)
