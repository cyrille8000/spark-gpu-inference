"""Séparation instrumentale — reprise exacte du chemin `--only_vocals` de l'image Demucs actuelle.

Ensemble : htdemucs_ft (modèle « vocals », 04573f0d-f3cf25b2.th, appliqué sur +x et -x)
+ Kim_Vocal_2.onnx + Kim_Inst.onnx (MDX-Net), pondérés 12 / 8 / 3.
Instrumental = mix - vocals. Les 4 gros modèles d'ensemble (bass/drums/other) ne sont pas embarqués :
ils ne servent que pour les autres stems.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np
import onnxruntime as ort
import torch
from demucs.apply import apply_model
from demucs.states import load_model

from .mdx_net import demix_full, get_models

log = logging.getLogger("spark.demucs")

SAMPLE_RATE = 44_100
WEIGHT_DEMUCS_VOCALS = "04573f0d-f3cf25b2.th"
WEIGHT_KIM_VOCAL = "Kim_Vocal_2.onnx"
WEIGHT_KIM_INST = "Kim_Inst.onnx"
WEIGHTS = (WEIGHT_DEMUCS_VOCALS, WEIGHT_KIM_VOCAL, WEIGHT_KIM_INST)

# Valeurs de la plateforme (demucs-separate : OVERLAP_LARGE = 0.0001)
DEFAULT_OVERLAP = 0.0001
ENSEMBLE_WEIGHTS = np.array([12, 8, 3], dtype=np.float32)  # Kim_Vocal_2, Kim_Inst (inversé), Demucs


def _providers(device: str, gpu_id: int = 0):
    available = ort.get_available_providers()
    if device.startswith("cuda") and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider"], [{"device_id": gpu_id}]
    if device.startswith("cuda"):
        log.warning("CUDAExecutionProvider absent de onnxruntime (%s) — MDX sur CPU", available)
    return ["CPUExecutionProvider"], [{}]


class InstrumentalSeparator:
    def __init__(self, models_dir: Path, device: str = "cuda", chunk_size: int = 1_000_000,
                 overlap: float = DEFAULT_OVERLAP, single_onnx: bool = False):
        models_dir = Path(models_dir)
        for name in WEIGHTS:
            if not (models_dir / name).is_file():
                raise FileNotFoundError(f"poids manquant : {models_dir / name}")
        self.device = device
        self.chunk_size = int(chunk_size)
        self.overlap = min(0.99, max(0.0, float(overlap)))
        self.single_onnx = single_onnx

        self.model_vocals = load_model(str(models_dir / WEIGHT_DEMUCS_VOCALS))
        self.model_vocals.to(device).eval()

        providers, options = _providers(device)
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.mdx1 = get_models("tdf_extra", device, load=False, vocals_model_type=2)
        self.sess1 = ort.InferenceSession(str(models_dir / WEIGHT_KIM_VOCAL), so,
                                          providers=providers, provider_options=options)
        if not single_onnx:
            self.mdx2 = get_models("tdf_extra", device, load=False, vocals_model_type=2)
            self.sess2 = ort.InferenceSession(str(models_dir / WEIGHT_KIM_INST), so,
                                              providers=providers, provider_options=options)
        log.info("séparateur prêt (device=%s, chunk=%d, overlap=%g, providers=%s)",
                 device, self.chunk_size, self.overlap, providers)

    @torch.inference_mode()
    def separate(self, mix: np.ndarray, progress: Callable[[int], None] | None = None) -> np.ndarray:
        """`mix` : (samples, 2) float32 à 44,1 kHz → instrumental (samples, 2) float32."""
        assert mix.ndim == 2 and mix.shape[1] == 2, "mix attendu en (samples, 2)"
        mix = np.ascontiguousarray(mix, dtype=np.float32)
        report = progress or (lambda _p: None)

        audio = torch.from_numpy(mix.T[None]).float().to(self.device)
        vocals_demucs = 0.5 * apply_model(self.model_vocals, audio, shifts=1, overlap=self.overlap)[0][3].cpu().numpy()
        report(10)
        vocals_demucs += 0.5 * -apply_model(self.model_vocals, -audio, shifts=1, overlap=self.overlap)[0][3].cpu().numpy()
        del audio
        report(20)

        vocals_mdx1 = demix_full(mix.T, self.device, self.chunk_size, self.mdx1, self.sess1, overlap=self.overlap)[0]
        report(30)

        if not self.single_onnx:
            instrum_mdx2 = -demix_full(-mix.T, self.device, self.chunk_size, self.mdx2, self.sess2, overlap=self.overlap)[0]
            vocals_mdx2 = mix.T - instrum_mdx2
            w = ENSEMBLE_WEIGHTS
            vocals = (w[0] * vocals_mdx1.T + w[1] * vocals_mdx2.T + w[2] * vocals_demucs.T) / w.sum()
        else:
            w = np.array([6, 1], dtype=np.float32)
            vocals = (w[0] * vocals_mdx1.T + w[1] * vocals_demucs.T) / w.sum()
        report(40)

        instrumental = (mix - vocals).astype(np.float32)
        report(95)
        return instrumental
