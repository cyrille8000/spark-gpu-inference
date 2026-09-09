# =============================================================================
# Spark GPU Inference — RunPod Serverless
#   task "instrumental" : BS-Roformer Leap Xe (unwa) — instrumental seul
#   task "vc"           : conversion de timbre Chatterbox VC (S3Gen), un tirage
# Tout est embarqué au build (paquets + poids) : zéro téléchargement à l'inférence.
# Base Python pure : les roues torch cu128 embarquent leurs bibliothèques CUDA/cuDNN,
# seul le pilote de l'hôte RunPod est nécessaire.
#
# BUILD :  docker build --platform linux/amd64 -t spark-gpu-inference .
# TEST  :  docker run --rm --gpus all -v $PWD/test_input.json:/app/test_input.json spark-gpu-inference
# =============================================================================
FROM python:3.11-slim-bookworm

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    SPARK_MODELS_DIR=/models \
    BS_ROFORMER_MODELS_PATH=/models/bsroformer \
    HF_HOME=/models/hf \
    TORCH_HOME=/models/torch \
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# ---------------------------------------------------------------- [1/5] système
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 curl git ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --upgrade pip wheel "setuptools<82"
# setuptools < 82 : resemble-perth (filigrane de Chatterbox) importe encore `pkg_resources`, supprimé en 82.0.0 ;
# sans lui perth.PerthImplicitWatermarker vaut None et ChatterboxVC ne se construit plus.

# ---------------------------------------------------------------- [2/5] PyTorch 2.7.1 — CUDA 12.8 (noyaux sm_50 → sm_120)
# cu128 obligatoire : le pool 24 GB de RunPod sert des RTX PRO 6000 Blackwell (MIG, sm_120), que cu124 ne sait pas
# exécuter (« no kernel image is available »). chatterbox-tts épingle torch 2.6.0 mais est installé --no-deps.
RUN pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128 \
    && python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# ---------------------------------------------------------------- [3/5] paquets Python
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt \
    && pip install --no-deps chatterbox-tts==0.1.7 \
    && python -c "import torch; assert torch.__version__.startswith('2.7.1'), torch.__version__" \
    && python -c "import setuptools, pkg_resources; assert int(setuptools.__version__.split('.')[0]) < 82, setuptools.__version__" \
    && python -c "import perth; assert perth.PerthImplicitWatermarker is not None, 'perth: filigrane indisponible'"

# Cohérence des dépendances : seul le manque de gradio (volontairement non installé) est toléré.
RUN set +e; pip check > /tmp/pipcheck.txt; set -e; cat /tmp/pipcheck.txt; \
    if grep -v '^chatterbox-tts ' /tmp/pipcheck.txt | grep -q ' requires '; then \
        echo 'Conflit de dépendances (hors gradio de chatterbox-tts)'; exit 1; fi

# ---------------------------------------------------------------- [4/5] poids : BS-Roformer Leap Xe (sha256 vérifié) + Chatterbox VC
COPY scripts/fetch_weights.py /app/scripts/fetch_weights.py
RUN python /app/scripts/fetch_weights.py \
    && rm -rf /models/hf/hub/.locks /models/chatterbox/.cache \
    && du -sh /models/*

# À partir d'ici, plus AUCUN accès réseau pour les modèles.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# ---------------------------------------------------------------- [5/5] code + vérification hors ligne (CPU)
WORKDIR /app
COPY src /app/src
COPY handler.py /app/handler.py
COPY scripts/smoke_test.py /app/scripts/smoke_test.py
RUN python /app/scripts/smoke_test.py

CMD ["python", "-u", "/app/handler.py"]
