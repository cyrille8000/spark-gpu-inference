#!/usr/bin/env python3
"""Télécharge au BUILD tous les poids dans l'image. Rien n'est téléchargé à l'inférence :
HF_HUB_OFFLINE=1 ensuite, et bs-roformer-infer retrouve son checkpoint dans BS_ROFORMER_MODELS_PATH."""
from __future__ import annotations

import os
import sys
from pathlib import Path

MODELS = Path(os.environ.get("SPARK_MODELS_DIR", "/models"))
CHATTERBOX_REPO = "ResembleAI/chatterbox"
CHATTERBOX_FILES = ("s3gen.safetensors", "conds.pt")   # tout ce que ChatterboxVC.from_local lit
BSROFORMER_SLUG = "roformer-model-bs-roformer-leap-xe-instrumental-by-pcunwa"


def fetch_bsroformer() -> None:
    from bs_roformer import ensure_model_assets

    dest = Path(os.environ.get("BS_ROFORMER_MODELS_PATH", str(MODELS / "bsroformer")))
    dest.mkdir(parents=True, exist_ok=True)
    ckpt, cfg = ensure_model_assets(BSROFORMER_SLUG, models_dir=str(dest))  # sha256 vérifié par le paquet
    print(f"[bsroformer] {ckpt.name}: {ckpt.stat().st_size / 1e6:.1f} MB — {cfg.name}")
    if ckpt.stat().st_size < 200_000_000:
        sys.exit(f"checkpoint BS-Roformer suspect ({ckpt.stat().st_size} octets)")


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


if __name__ == "__main__":
    fetch_bsroformer()
    fetch_chatterbox()
    print("[fetch_weights] OK")
