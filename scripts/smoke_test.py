#!/usr/bin/env python3
"""Vérification de build (CPU, HORS LIGNE) : chaque modèle se charge depuis les poids embarqués,
les points d'accroche que le moteur VC règle existent bien dans cette version de Chatterbox."""
from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

MODELS = Path(os.environ.get("SPARK_MODELS_DIR", "/models"))


def check_demucs() -> None:
    import onnxruntime as ort
    from demucs.states import load_model

    from spark_infer.demucs_engine import WEIGHTS

    d = MODELS / "demucs"
    for name in WEIGHTS:
        size = (d / name).stat().st_size
        assert size > 10_000_000, f"{name} trop petit ({size})"
        print(f"[demucs] {name}: {size / 1e6:.1f} MB")
    model = load_model(str(d / WEIGHTS[0]))
    assert list(model.sources)[3] == "vocals", model.sources
    for onnx in WEIGHTS[1:]:
        ort.InferenceSession(str(d / onnx), providers=["CPUExecutionProvider"])
    print(f"[demucs] OK — providers onnxruntime : {ort.get_available_providers()}")


def check_chatterbox() -> None:
    from chatterbox.vc import ChatterboxVC

    vc = ChatterboxVC.from_local(str(MODELS / "chatterbox"), "cpu")
    dec = vc.s3gen.flow.decoder
    assert "n_cfm_timesteps" in inspect.signature(vc.s3gen.inference).parameters
    assert "temperature" in inspect.signature(dec.forward).parameters
    print(f"[chatterbox] OK — sr={vc.sr} cfg_rate={getattr(dec, 'inference_cfg_rate', None)} "
          f"meanflow={getattr(vc.s3gen, 'meanflow', None)}")


def check_ecapa() -> None:
    from speechbrain.inference.speaker import EncoderClassifier

    EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                   savedir=str(MODELS / "ecapa"), run_opts={"device": "cpu"})
    print("[ecapa] OK")


def check_runtime() -> None:
    import runpod  # noqa: F401
    import torch

    from spark_infer import tasks  # noqa: F401 — importe la chaîne complète

    print(f"[runtime] torch {torch.__version__} cuda_build={torch.version.cuda} runpod OK")


if __name__ == "__main__":
    check_runtime()
    check_demucs()
    check_chatterbox()
    check_ecapa()
    print("[smoke_test] tout est chargeable hors ligne")
