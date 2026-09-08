#!/usr/bin/env python3
"""Télécharge au BUILD les poids Chatterbox VC et ECAPA dans l'image (les poids Demucs sont
récupérés par le Dockerfile). Rien n'est téléchargé à l'inférence : HF_HUB_OFFLINE=1 ensuite."""
from __future__ import annotations

import os
import sys
from pathlib import Path

MODELS = Path(os.environ.get("SPARK_MODELS_DIR", "/models"))
CHATTERBOX_REPO = "ResembleAI/chatterbox"
CHATTERBOX_FILES = ("s3gen.safetensors", "conds.pt")   # tout ce que ChatterboxVC.from_local lit
ECAPA_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"


def fetch_chatterbox() -> None:
    from huggingface_hub import hf_hub_download

    dest = MODELS / "chatterbox"
    dest.mkdir(parents=True, exist_ok=True)
    for name in CHATTERBOX_FILES:
        path = Path(hf_hub_download(CHATTERBOX_REPO, name, local_dir=str(dest)))
        print(f"[chatterbox] {name}: {path.stat().st_size / 1e6:.1f} MB")
    s3gen = dest / "s3gen.safetensors"
    if s3gen.stat().st_size < 900_000_000:
        sys.exit(f"s3gen.safetensors suspect ({s3gen.stat().st_size} octets)")


def fetch_ecapa() -> None:
    from speechbrain.inference.speaker import EncoderClassifier

    dest = MODELS / "ecapa"
    EncoderClassifier.from_hparams(source=ECAPA_SOURCE, savedir=str(dest), run_opts={"device": "cpu"})
    files = sorted(p.name for p in dest.iterdir())
    print(f"[ecapa] {dest}: {files}")
    if "embedding_model.ckpt" not in files:
        sys.exit("ECAPA : embedding_model.ckpt absent")


if __name__ == "__main__":
    fetch_chatterbox()
    fetch_ecapa()
    print("[fetch_weights] OK")
