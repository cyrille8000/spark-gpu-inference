#!/usr/bin/env python3
"""Télécharge au BUILD tous les poids dans l'image. Rien n'est téléchargé à l'inférence :
HF_HUB_OFFLINE=1 ensuite, et bs-roformer-infer retrouve son checkpoint dans BS_ROFORMER_MODELS_PATH."""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

MODELS = Path(os.environ.get("SPARK_MODELS_DIR", "/models"))
CHATTERBOX_REPO = "ResembleAI/chatterbox"
CHATTERBOX_FILES = ("s3gen.safetensors", "conds.pt")   # tout ce que ChatterboxVC.from_local lit
BSROFORMER_SLUG = "roformer-model-bs-roformer-leap-xe-instrumental-by-pcunwa"

# --- Visages qui parlent : LR-ASD (MIT), commit épinglé du 2025-03-23 ---
# Le réseau audio-visuel est COMMITÉ dans le dépôt (3,4 Mo, pas de LFS) : on le tire du commit exact
# et on vérifie son sha256. Le détecteur S3FD, lui, n'est hébergé que sur Google Drive (l'id de
# TalkNet/LR-ASD) ; `SPARK_S3FD_URL` permet de le tirer d'un miroir HTTP à nous — recommandé :
# files.dubbingspark.com — parce que Drive rend une PAGE HTML quand son quota est atteint. Le
# sha256 et la taille ci-dessous ont été relevés sur le fichier téléchargé le 2026-09-15.
LRASD_COMMIT = "1b6dcd2d8fc2895683de6508ec6294ec47d388ca"
LRASD_ASD = ("finetuning_TalkSet.model",
             f"https://raw.githubusercontent.com/Junhua-Liao/LR-ASD/{LRASD_COMMIT}/weight/finetuning_TalkSet.model",
             "6b4ef53694e874e96cf630198dc479c78aebb3993bbf166aee3d926dfe7d9342", 3_426_337)
S3FD_DRIVE_ID = "1KafnHz7ccT-3IyddBsL5yi2xGtxAKypt"
S3FD = ("sfd_face.pth", "d54a87c2b7543b64729c9a25eafd188da15fd3f6e02f0ecec76ae1b30d86c491", 89_844_381)


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


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _verifier(path: Path, sha: str, taille: int) -> None:
    """Un poids se vérifie AU BUILD, pas à la première inférence : taille exacte, pas une page HTML
    déguisée (quota Google Drive), sha256 identique."""
    if not path.is_file():
        sys.exit(f"[lrasd] poids manquant : {path}")
    size = path.stat().st_size
    with open(path, "rb") as f:
        tete = f.read(512)
    if b"<html" in tete.lower() or b"<!doctype" in tete.lower():
        sys.exit(f"[lrasd] PAGE HTML au lieu des poids (quota Drive ?) : {path}")
    if size != taille:
        sys.exit(f"[lrasd] taille inattendue pour {path.name} : {size} octets au lieu de {taille}")
    got = _sha256(path)
    if got != sha:
        sys.exit(f"[lrasd] sha256 inattendu pour {path.name} : {got}")
    print(f"[lrasd] {path.name}: {size / 1e6:.1f} MB — sha256 OK")


def _http_download(url: str, dest: Path) -> None:
    import requests

    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)


def fetch_lrasd() -> None:
    dest = MODELS / "lrasd"
    dest.mkdir(parents=True, exist_ok=True)

    name, url, sha, taille = LRASD_ASD
    _http_download(url, dest / name)
    _verifier(dest / name, sha, taille)

    name, sha, taille = S3FD
    mirror = os.environ.get("SPARK_S3FD_URL", "").strip()
    if mirror:
        print(f"[lrasd] S3FD depuis le miroir {mirror}")
        _http_download(mirror, dest / name)
    else:
        import gdown

        print("[lrasd] S3FD depuis Google Drive (poser SPARK_S3FD_URL pour un miroir à nous)")
        gdown.download(id=S3FD_DRIVE_ID, output=str(dest / name), quiet=False)
    _verifier(dest / name, sha, taille)


if __name__ == "__main__":
    fetch_bsroformer()
    fetch_chatterbox()
    fetch_lrasd()
    print("[fetch_weights] OK")
