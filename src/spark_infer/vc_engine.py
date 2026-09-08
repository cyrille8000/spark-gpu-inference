"""Conversion de timbre avec Chatterbox VC (S3Gen), réglages fins + best-of-N.

Port du script d'essai `vc_test.py` : prétraitement de la source, embedding de timbre
moyenné sur plusieurs clips de référence, prompt phonétique optionnel, pas/temperature/CFG
du décodeur CFM, complétion de la queue par fondu enchaîné, tirages multiples départagés
par similarité de locuteur (ECAPA). Le scorer WER (Whisper) et resemble-enhance du script
ne sont pas embarqués (poids volumineux, hors périmètre de l'image).
"""
from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import librosa
import numpy as np
import soundfile as sf
import torch

from .audio_utils import crossfade_append, fit_length, preprocess_source
from .params import VcParams

log = logging.getLogger("spark.vc")

ECAPA_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"


@dataclass
class VcRun:
    index: int
    seed: int
    similarity: float | None
    tail_passes: int
    wav: np.ndarray


class VoiceConverter:
    def __init__(self, ckpt_dir: Path, device: str = "cuda", ecapa_dir: Path | None = None):
        from chatterbox.vc import ChatterboxVC

        self.device = device
        self.model = ChatterboxVC.from_local(str(ckpt_dir), device)
        self.sr: int = self.model.sr
        # originaux conservés : les réglages par job sont reconstruits depuis eux (pas d'empilement de partial)
        self._orig_inference = self.model.s3gen.inference
        self._dec = self.model.s3gen.flow.decoder
        self._orig_dec_forward = self._dec.forward
        self._orig_cfg = getattr(self._dec, "inference_cfg_rate", None)
        self.ecapa_dir = Path(ecapa_dir) if ecapa_dir else None
        self._spk = None
        self._spk_failed = False
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
            start = max(0.0, len(out) / self.sr - p.overlap)
            tail_path = workdir / f"tail_{seed}_{k}.wav"
            sf.write(tail_path, y_src[int(start * sr_src):], sr_src)
            out = crossfade_append(out, self._convert(str(tail_path)), int(p.overlap * self.sr))
            k += 1
        return fit_length(out, int(round(src_d * self.sr))), k

    # ---------------- scorer timbre (ECAPA) ----------------
    def _speaker_scorer(self) -> Callable[[np.ndarray, int], torch.Tensor] | None:
        if self._spk_failed or not self.ecapa_dir:
            return None
        if self._spk is None:
            try:
                from speechbrain.inference.speaker import EncoderClassifier
                self._spk = EncoderClassifier.from_hparams(
                    source=ECAPA_SOURCE, savedir=str(self.ecapa_dir), run_opts={"device": self.device})
            except Exception as e:  # noqa: BLE001
                self._spk_failed = True
                log.warning("scorer ECAPA indisponible (%s) — best-of-N sans départage", e)
                return None
        spk = self._spk

        def embed(wav: np.ndarray, sr: int) -> torch.Tensor:
            w = librosa.resample(wav, orig_sr=sr, target_sr=16000) if sr != 16000 else wav
            with torch.inference_mode():
                e = spk.encode_batch(torch.from_numpy(w).float().to(self.device)[None]).squeeze()
            return torch.nn.functional.normalize(e, dim=0)

        return embed

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

        scorer = self._speaker_scorer() if p.n > 1 else None
        ref_emb = None
        if scorer is not None:
            ref_emb = torch.nn.functional.normalize(
                torch.stack([scorer(*librosa.load(str(c), sr=None, mono=True)) for c in ref_paths]).mean(0), dim=0)
        elif p.n > 1:
            warnings.append("pas de scorer de timbre : le premier tirage est retenu")

        runs: list[VcRun] = []
        for i in range(p.n):
            seed = p.seed + i
            wav, tails = self._convert_full(src_wav, y_src, sr_src, seed, p, workdir)
            sim = float(torch.dot(scorer(wav, self.sr), ref_emb)) if scorer is not None else None
            runs.append(VcRun(i, seed, sim, tails, wav))
            log.info("tirage %d/%d seed=%d sim=%s queues=%d", i + 1, p.n, seed,
                     "-" if sim is None else f"{sim:.3f}", tails)
            report(10 + int(80 * (i + 1) / p.n), f"tirage {i + 1}/{p.n}")

        best = max(runs, key=lambda r: (r.similarity if r.similarity is not None else 0.0, -r.index))
        meta = {
            "source_duration_s": round(src_d, 3),
            "n": p.n,
            "best_run": best.index + 1,
            "runs": [{"run": r.index + 1, "seed": r.seed, "similarity": r.similarity,
                      "tail_passes": r.tail_passes} for r in runs],
            "steps": p.steps, "temp": p.temp, "cfg": p.cfg, "ref_len": p.ref_len,
            "preproc": p.preproc, "refs": len(ref_paths), "prompt": prompt_path is not None,
            "warnings": warnings,
        }
        return best.wav, self.sr, meta
