#!/usr/bin/env python3
"""Vérification de build (CPU, HORS LIGNE) : chaque modèle se charge depuis les poids embarqués,
BS-Roformer fait une vraie passe avant, et les points d'accroche réglés par le moteur VC existent."""
from __future__ import annotations

import inspect
import os
import sys
import tempfile
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

MODELS = Path(os.environ.get("SPARK_MODELS_DIR", "/models"))


def check_bsroformer() -> None:
    import numpy as np
    import soundfile as sf

    from spark_infer.separation_engine import SAMPLE_RATE, InstrumentalSeparator
    from spark_infer.tasks import BSROFORMER_DIR

    sep = InstrumentalSeparator(BSROFORMER_DIR, device="cpu")
    # 2 s de bruit stéréo : prouve que config + checkpoint + passe avant sont cohérents
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        rng = np.random.default_rng(0)
        mix = (0.1 * rng.standard_normal((2 * SAMPLE_RATE, 2))).astype(np.float32)
        sf.write(tmp / "mix.wav", mix, SAMPLE_RATE, subtype="FLOAT")
        out = sep.separate(tmp / "mix.wav", tmp)
        y, sr = sf.read(out, dtype="float32")
        assert sr == SAMPLE_RATE and y.shape == mix.shape, (sr, y.shape, mix.shape)
        assert np.isfinite(y).all()
    print(f"[bsroformer] OK — stem={sep.stem} chunk={sep.chunk_size} overlap={sep.num_overlap}")


def check_chatterbox() -> None:
    from chatterbox.vc import ChatterboxVC

    vc = ChatterboxVC.from_local(str(MODELS / "chatterbox"), "cpu")
    dec = vc.s3gen.flow.decoder
    assert "n_cfm_timesteps" in inspect.signature(vc.s3gen.inference).parameters
    assert "temperature" in inspect.signature(dec.forward).parameters
    print(f"[chatterbox] OK — sr={vc.sr} cfg_rate={getattr(dec, 'inference_cfg_rate', None)} "
          f"meanflow={getattr(vc.s3gen, 'meanflow', None)}")


def check_runtime() -> None:
    import runpod
    import torch

    from spark_infer import tasks

    assert hasattr(runpod.serverless, "start") and hasattr(tasks, "run_task")
    print(f"[runtime] python {sys.version.split()[0]} torch {torch.__version__} cuda_build={torch.version.cuda}")


if __name__ == "__main__":
    check_runtime()
    check_bsroformer()
    check_chatterbox()
    print("[smoke_test] tout est chargeable hors ligne")
