"""Conversion de timbre avec Chatterbox VC (S3Gen), réglages fins, un seul tirage.

Port du script d'essai `vc_test.py` : prétraitement de la source, embedding de timbre
moyenné sur plusieurs clips de référence, prompt phonétique optionnel, pas/temperature/CFG
du décodeur CFM, complétion de la queue. Décisions du propriétaire (2026-09-09) : une seule
itération (pas de best-of-N, donc pas de scorer ECAPA ni WER), pas de resemble-enhance, et la
queue manquante est reconvertie puis COLLÉE bout à bout, sans recouvrement ni fondu.
"""
from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Callable

import librosa
import numpy as np
import soundfile as sf
import torch

from .audio_utils import fit_length, preprocess_source
from .params import VcParams

log = logging.getLogger("spark.vc")


class VoiceConverter:
    def __init__(self, ckpt_dir: Path, device: str = "cuda"):
        from chatterbox.vc import ChatterboxVC

        self.device = device
        self.model = ChatterboxVC.from_local(str(ckpt_dir), device)
        self.sr: int = self.model.sr
        # originaux conservés : les réglages par job sont reconstruits depuis eux (pas d'empilement de partial)
        self._orig_inference = self.model.s3gen.inference
        self._dec = self.model.s3gen.flow.decoder
        self._orig_dec_forward = self._dec.forward
        self._orig_cfg = getattr(self._dec, "inference_cfg_rate", None)
        log.info("Chatterbox VC prêt (device=%s, sr=%d, meanflow=%s, cfg=%s)",
                 device, self.sr, getattr(self.model.s3gen, "meanflow", "?"), self._orig_cfg)

    # ---------------- réglages ----------------
    def _configure(self, p: VcParams) -> list[str]:
        warnings: list[str] = []
        self.model.DEC_COND_LEN = int(p.ref_len * self.sr)
        self.model.s3gen.inference = functools.partial(self._orig_inference, n_cfm_timesteps=p.steps)
        self._dec.forward = functools.partial(self._orig_dec_forward, temperature=p.temp)
        if self._orig_cfg is not None:
            self._dec.inference_cfg_rate = p.cfg if p.cfg is not None else self._orig_cfg
        elif p.cfg is not None:
            warnings.append("le décodeur n'expose pas inference_cfg_rate : `cfg` ignoré")
        return warnings

    def _set_reference(self, ref_paths: list[str], prompt_path: str | None) -> None:
        embs = []
        for c in ref_paths:
            self.model.set_target_voice(c)
            embs.append(self.model.ref_dict["embedding"].clone())
        emb = torch.stack(embs).mean(0)
        self.model.set_target_voice(prompt_path or ref_paths[0])
        self.model.ref_dict["embedding"] = emb

    # ---------------- conversion ----------------
    def _convert(self, path: str) -> np.ndarray:
        with torch.inference_mode():
            return self.model.generate(audio=path).squeeze(0).cpu().numpy()

    def _convert_full(self, src_path: Path, y_src: np.ndarray, sr_src: int, seed: int,
                      p: VcParams, workdir: Path) -> tuple[np.ndarray, int]:
        torch.manual_seed(seed)
        src_d = len(y_src) / sr_src
        out = self._convert(str(src_path))
        k = 0
        while src_d - len(out) / self.sr > p.tail_tolerance_s and k < p.max_tail_passes:
            # la partie de la source qui n'a pas encore de sortie, reconvertie seule et collée telle quelle
            start = len(out) / self.sr
            tail_path = workdir / f"tail_{seed}_{k}.wav"
            sf.write(tail_path, y_src[int(start * sr_src):], sr_src)
            out = np.concatenate([out, self._convert(str(tail_path))])
            k += 1
        return fit_length(out, int(round(src_d * self.sr))), k

    # ---------------- entrée principale ----------------
    def run(self, source_path: Path, ref_paths: list[Path], prompt_path: Path | None, p: VcParams,
            workdir: Path, progress: Callable[[int, str], None] | None = None) -> tuple[np.ndarray, int, dict]:
        report = progress or (lambda _pct, _msg: None)
        warnings = self._configure(p)

        y_src, sr_src = librosa.load(str(source_path), sr=None, mono=True)
        if p.preproc:
            y_src = preprocess_source(y_src, sr_src)
        src_wav = workdir / "src.wav"
        sf.write(src_wav, y_src, sr_src)
        src_d = len(y_src) / sr_src

        self._set_reference([str(r) for r in ref_paths], str(prompt_path) if prompt_path else None)
        report(10, "référence prête")

        wav, tails = self._convert_full(src_wav, y_src, sr_src, p.seed, p, workdir)
        log.info("conversion seed=%d queues=%d durée=%.2fs", p.seed, tails, src_d)
        report(90, "conversion terminée")

        meta = {
            "source_duration_s": round(src_d, 3),
            "seed": p.seed, "tail_passes": tails,
            "steps": p.steps, "temp": p.temp, "cfg": p.cfg, "ref_len": p.ref_len,
            "preproc": p.preproc, "refs": len(ref_paths), "prompt": prompt_path is not None,
            "warnings": warnings,
        }
        return wav, self.sr, meta
