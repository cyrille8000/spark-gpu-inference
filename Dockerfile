# =============================================================================
# Spark GPU Inference — RunPod Serverless
#   task "demucs" : instrumental (htdemucs_ft vocals + Kim_Vocal_2 + Kim_Inst)
#   task "vc"     : conversion de timbre Chatterbox VC (S3Gen) + scorer ECAPA
# Tout est embarqué au build (paquets + poids) : zéro téléchargement à l'inférence.
#
# BUILD :  docker build --platform linux/amd64 -t spark-gpu-inference .
# TEST  :  docker run --rm --gpus all -v $PWD/test_input.json:/app/test_input.json spark-gpu-inference
# =============================================================================
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    SPARK_MODELS_DIR=/models \
    HF_HOME=/models/hf \
    TORCH_HOME=/models/torch \
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# ---------------------------------------------------------------- [1/6] système
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
        ffmpeg libsndfile1 curl unzip git ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------------- [2/6] PyTorch 2.6.0 (pin de chatterbox-tts) — CUDA 12.4
RUN pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 \
    && python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# ---------------------------------------------------------------- [3/6] paquets Python
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt \
    && pip install --no-deps chatterbox-tts==0.1.7 \
    && python3 -c "import torch; assert torch.__version__.startswith('2.6.0'), torch.__version__"

# Cohérence des dépendances : seul le manque de gradio (volontairement non installé) est toléré.
RUN set +e; pip check > /tmp/pipcheck.txt; set -e; cat /tmp/pipcheck.txt; \
    if grep -v '^chatterbox-tts ' /tmp/pipcheck.txt | grep -q ' requires '; then \
        echo 'Conflit de dépendances (hors gradio de chatterbox-tts)'; exit 1; fi

# ---------------------------------------------------------------- [4/6] poids Demucs / MDX (3 fichiers du chemin --only_vocals)
# Même archive que l'image Demucs de la plateforme (files.dubbingspark.com) ; on n'en garde que les 3 poids utiles.
ARG DEMUCS_WEIGHTS_BASE_URL="https://files.dubbingspark.com/b0e526cc7578d1e1986ae652f06fd499e22360f5/d5abd690f1c69f4a889039ddd4aa88d8"
RUN mkdir -p /models/demucs /tmp/w && cd /tmp/w \
    && for part in aa ab ac; do curl -fsSL -o "models_part_$part" "$DEMUCS_WEIGHTS_BASE_URL/models_part_$part"; done \
    && cat models_part_* > models.zip \
    && unzip -j -q models.zip '*04573f0d-f3cf25b2.th' '*Kim_Vocal_2.onnx' '*Kim_Inst.onnx' -d /models/demucs \
    && cd / && rm -rf /tmp/w \
    && ls -l /models/demucs \
    && for f in 04573f0d-f3cf25b2.th Kim_Vocal_2.onnx Kim_Inst.onnx; do test -s "/models/demucs/$f" || { echo "poids manquant: $f"; exit 1; }; done

# ---------------------------------------------------------------- [5/6] poids Chatterbox VC (Hugging Face) + ECAPA (SpeechBrain)
COPY scripts/fetch_weights.py /app/scripts/fetch_weights.py
RUN python3 /app/scripts/fetch_weights.py \
    && rm -rf /models/hf/hub/.locks /models/chatterbox/.cache \
    && du -sh /models/*

# À partir d'ici, plus AUCUN accès réseau pour les modèles.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# ---------------------------------------------------------------- [6/6] code + vérification hors ligne
WORKDIR /app
COPY src /app/src
COPY handler.py /app/handler.py
COPY scripts/smoke_test.py /app/scripts/smoke_test.py
RUN python3 /app/scripts/smoke_test.py

CMD ["python3", "-u", "/app/handler.py"]
