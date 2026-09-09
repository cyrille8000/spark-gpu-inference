"""Séparation instrumentale — BS-Roformer Leap Xe (unwa), cible « instrumental » directe.

Un seul checkpoint (268 MB), 18,07 dB SDR instrumental sur le Multisong de MVSEP (juin 2026),
au-dessus des ensembles internes du site. Chargé par `bs-roformer-infer` (MIT), qui vérifie le
sha256 du checkpoint et fait le découpage fenêtré avec fondu (chunk 881 559 éch., recouvrement 2).
Aucun réseau : les poids sont dans `$BS_ROFORMER_MODELS_PATH` depuis le build.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

log = logging.getLogger("spark.separation")

MODEL_SLUG = "roformer-model-bs-roformer-leap-xe-instrumental-by-pcunwa"
SAMPLE_RATE = 44_100


class InstrumentalSeparator:
    def __init__(self, models_dir: Path, device: str = "cuda"):
        from bs_roformer import BSRoformerSession

        self.device = device
        self.session = BSRoformerSession(
            model_name=MODEL_SLUG, models_dir=str(models_dir), device=device,
            backend="torch", progress=False,
        ).load()
        from bs_roformer.backends.base import ChunkingPlan

        cfg = self.session._config  # ConfigDict chargé par load() ; lu, jamais modifié
        self.stem = cfg.training.target_instrument or "other"
        # chunk_size vit sous `audio:` dans la config Leap Xe (sous `inference:` dans d'autres) :
        # ChunkingPlan est l'unique endroit du paquet qui connaît les deux emplacements
        plan = ChunkingPlan.from_config(cfg)
        self.chunk_size = int(plan.chunk_size)
        self.num_overlap = int(plan.num_overlap)
        log.info("BS-Roformer Leap Xe prêt (device=%s, stem=%s, chunk=%d, overlap=%d)",
                 self.session.device, self.stem, self.chunk_size, self.num_overlap)

    def separate(self, wav_path: Path, workdir: Path) -> Path:
        """`wav_path` : WAV stéréo 44,1 kHz → chemin du WAV float32 instrumental."""
        in_dir = workdir / "bsr_in"
        out_dir = workdir / "bsr_out"
        shutil.rmtree(in_dir, ignore_errors=True)
        shutil.rmtree(out_dir, ignore_errors=True)
        in_dir.mkdir(parents=True)
        shutil.copyfile(wav_path, in_dir / "mix.wav")

        manifest = self.session.infer(in_dir, store_dir=out_dir, output_format="wav_float32")
        for out in manifest.outputs:
            if out.output_id == self.stem:
                return Path(out.output_path)
        raise RuntimeError(f"stem « {self.stem} » absent du manifeste : "
                           f"{[o.output_id for o in manifest.outputs]}")
