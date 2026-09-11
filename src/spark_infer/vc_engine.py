"""Conversion de timbre avec Chatterbox VC (S3Gen), réglages fins, un seul tirage.

Port du script d'essai `vc_test.py` : prétraitement de la source, embedding de timbre
moyenné sur plusieurs clips de référence, prompt phonétique optionnel, pas/temperature/CFG
du décodeur CFM, complétion de la queue. Décisions du propriétaire (2026-09-09) : une seule
itération (pas de best-of-N, donc pas de scorer ECAPA ni WER), pas de resemble-enhance, et la
queue manquante est reconvertie puis COLLÉE bout à bout, sans recouvrement ni fondu.
Depuis le 2026-09-11 la source est convertie PAR FENÊTRES (audio_utils.plan_windows,
60 s par défaut, coupées aux creux d'énergie) collées de même : la mémoire du décodeur
grandit avec le carré de la durée, 179 s débordaient un L4 de 22 Go.
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

from .audio_utils import fit_length, plan_windows, preprocess_source
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

    def _convert_piece(self, piece: np.ndarray, sr_src: int, nom: str, p: VcParams,
                       workdir: Path) -> tuple[np.ndarray, int]:
        """Une fenêtre : conversion, complétion de la queue (collée telle quelle), puis
        longueur EXACTE de la fenêtre source — les fenêtres concaténées redonnent la
        durée de la source, que la plateforme redécoupe à offsets fixes."""
        piece_path = workdir / f"{nom}.wav"
        sf.write(piece_path, piece, sr_src)
        src_d = len(piece) / sr_src
        out = self._convert(str(piece_path))
        k = 0
        while src_d - len(out) / self.sr > p.tail_tolerance_s and k < p.max_tail_passes:
            # la partie de la fenêtre qui n'a pas encore de sortie, reconvertie seule et collée telle quelle
            start = len(out) / self.sr
            tail_path = workdir / f"{nom}_tail_{k}.wav"
            sf.write(tail_path, piece[int(start * sr_src):], sr_src)
            out = np.concatenate([out, self._convert(str(tail_path))])
            k += 1
        return fit_length(out, int(round(src_d * self.sr))), k

    def _convert_full(self, y_src: np.ndarray, sr_src: int, seed: int, p: VcParams, workdir: Path,
                      report: Callable[[int, str], None] | None = None,
                      cuts: list[int] | None = None) -> tuple[np.ndarray, int, int]:
        """La source PAR FENÊTRES (audio_utils.plan_windows) : la mémoire du décodeur
        grandit avec le carré de la durée — 179 s = OOM sur un L4 de 22 Go, mesuré le
        2026-09-11 ; 60 s en demandent neuf fois moins. Même timbre pour toutes (la
        référence est posée une fois), fenêtres collées bout à bout, sans fondu, coupées
        sur les frontières de segments envoyées par la plateforme (`cuts`, en
        échantillons) — au creux d'énergie seulement à défaut. Rend (wav, queues, fenêtres)."""
        torch.manual_seed(seed)
        bornes = plan_windows(y_src, sr_src, p.window_s, cuts=cuts)
        outs: list[np.ndarray] = []
        tails = 0
        for i, (a, b) in enumerate(bornes):
            out, k = self._convert_piece(y_src[a:b], sr_src, f"win_{seed}_{i}", p, workdir)
            outs.append(out)
            tails += k
            if report:
                report(10 + int(80 * (i + 1) / len(bornes)), f"fenêtre {i + 1}/{len(bornes)}")
        wav = np.concatenate(outs) if outs else np.zeros(0, dtype=np.float32)
        return fit_length(wav, int(round(len(y_src) / sr_src * self.sr))), tails, len(bornes)

    # ---------------- entrée principale ----------------
    def run(self, source_path: Path, ref_paths: list[Path], prompt_path: Path | None, p: VcParams,
            workdir: Path, progress: Callable[[int, str], None] | None = None,
            cuts_s: list[float] | None = None) -> tuple[np.ndarray, int, dict]:
        report = progress or (lambda _pct, _msg: None)
        warnings = self._configure(p)

        y_src, sr_src = librosa.load(str(source_path), sr=None, mono=True)
        if p.preproc:
            y_src = preprocess_source(y_src, sr_src)
        src_d = len(y_src) / sr_src
        # Les frontières autorisées, en échantillons de LA source telle que lue (sr natif, longueur inchangée).
        cuts = [int(round(t * sr_src)) for t in (cuts_s or [])]

        self._set_reference([str(r) for r in ref_paths], str(prompt_path) if prompt_path else None)
        report(10, "référence prête")

        wav, tails, fenetres = self._convert_full(y_src, sr_src, p.seed, p, workdir, report, cuts)
        log.info("conversion seed=%d fenêtres=%d frontières=%d queues=%d durée=%.2fs",
                 p.seed, fenetres, len(cuts), tails, src_d)
        report(90, "conversion terminée")

        meta = {
            "source_duration_s": round(src_d, 3),
            "seed": p.seed, "tail_passes": tails, "windows": fenetres, "window_s": p.window_s, "cuts": len(cuts),
            "steps": p.steps, "temp": p.temp, "cfg": p.cfg, "ref_len": p.ref_len,
            "preproc": p.preproc, "refs": len(ref_paths), "prompt": prompt_path is not None,
            "warnings": warnings,
        }
        return wav, self.sr, meta
