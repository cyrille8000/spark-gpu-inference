"""Visages qui parlent — LR-ASD EN INTERNE : S3FD dit OÙ sont les visages, le réseau audio-visuel
dit QUAND chacun parle. Résident sur la carte comme les deux autres moteurs (`registry.lease`).

Ce que faisait l'ancienne image (`spark-dubbing-lipsync`) : lancer `Columbia_test.py` en
sous-processus, qui ré-encodait l'extrait en MJPEG, écrivait CHAQUE image en JPEG sur le disque,
rechargeait S3FD et le réseau à chaque job, recadrait chaque visage dans une vidéo XVID puis la
relisait, et rendait deux pickles. Ici les mêmes étapes, dans le même ordre et avec les mêmes
constantes, sans fichier intermédiaire et avec les modèles chargés une fois par worker :

  1. plans (PySceneDetect, ContentDetector, sur CPU) ;
  2. PASSE 1 sur la vidéo découpée : S3FD sur chaque image (GPU), à l'échelle `facedet_scale` ;
  3. suivi par plan (`faces_geometry.track_shot`, CPU) ;
  4. PASSE 2 sur la vidéo : recadrage 224×224 de chaque visage suivi, centre 112×112 en gris,
     gardé en mémoire (12 Ko par image et par visage) ;
  5. par piste : MFCC du son de la piste + réseau sur six longueurs de fenêtre, moyennées (GPU).

Deux écarts assumés avec l'original, sans effet mesurable sur le résultat : les visages ne passent
plus par une compression XVID avant le réseau, et la vidéo est décodée deux fois (détection, puis
recadrage) au lieu d'être écrite en JPEG — décoder 150 s de 1080p prend quelques secondes, écrire
et relire 3 750 JPEG en prenait le triple.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

import numpy as np

from . import faces_geometry as geo

log = logging.getLogger("spark.faces")

MODEL_NAME = "lr_asd"
WEIGHTS_ASD = "finetuning_TalkSet.model"   # F1 96,4 % sur Columbia contre 86,1 % pour pretrain_AVA (README LR-ASD)
WEIGHTS_S3FD = "sfd_face.pth"

Progress = Callable[[int, str], None]


def detect_scenes(video_path: Path) -> list[tuple[int, int]]:
    """Les plans de l'extrait, en images `[début, fin[` : PySceneDetect avec le détecteur et le
    seuil par défaut, comme `scene_detect` de LR-ASD. Sans coupure, un seul plan sur tout l'extrait."""
    from scenedetect import ContentDetector, SceneManager, open_video

    video = open_video(str(video_path))
    manager = SceneManager()
    manager.add_detector(ContentDetector())
    manager.detect_scenes(video)
    scenes = manager.get_scene_list(start_in_scene=True)
    return [(int(s.frame_num), int(e.frame_num)) for s, e in scenes]


class SpeakingFaceDetector:
    """S3FD + LR-ASD sur UNE carte. Une instance sert un job à la fois (pool du registre)."""

    def __init__(self, models_dir: Path, device: str = "cuda"):
        import torch

        from .lrasd import ASD_Model, S3FDNet, ScoreHead, load_asd_weights

        self.torch = torch
        self.device = device
        s3fd_path, asd_path = Path(models_dir) / WEIGHTS_S3FD, Path(models_dir) / WEIGHTS_ASD
        for p in (s3fd_path, asd_path):
            if not p.is_file():
                raise FileNotFoundError(f"poids absents de l'image : {p}")
        self.s3fd = S3FDNet()
        self.s3fd.load_state_dict(torch.load(str(s3fd_path), map_location="cpu"), strict=True)
        self.s3fd.to(device).eval()
        self.asd = ASD_Model()
        self.head = ScoreHead()
        load_asd_weights(asd_path, self.asd, self.head)
        self.asd.to(device).eval()
        self.head.to(device).eval()
        log.info("LR-ASD prêt (device=%s, poids=%s)", device, WEIGHTS_ASD)

    # ------------------------------------------------------------ détection

    def detect_faces(self, image_bgr: np.ndarray, scale: float, conf_th: float = geo.CONF_TH) -> np.ndarray:
        """Les visages d'UNE image : `(n, 5)` = `[x1, y1, x2, y2, score]` en PIXELS de `image_bgr`
        (la réduction `scale` ne sert qu'à la détection, les boîtes sont remises à l'échelle de
        l'image reçue — `S3FD.detect_faces` multiplie par `[w, h, w, h]` de l'image d'origine)."""
        import cv2

        h, w = image_bgr.shape[:2]
        scaled = image_bgr if scale == 1 else cv2.resize(image_bgr, dsize=(0, 0), fx=scale, fy=scale,
                                                          interpolation=cv2.INTER_LINEAR)
        x = self.torch.from_numpy(geo.s3fd_input(scaled)).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            out = self.s3fd(x)
        d = out[0, 1].numpy()
        d = d[d[:, 0] > conf_th]
        if len(d) == 0:
            return np.empty((0, 5))
        boxes = d[:, 1:] * np.array([w, h, w, h], dtype=np.float32)
        bboxes = np.column_stack([boxes, d[:, 0]]).astype(float)
        return bboxes[geo.nms(bboxes, geo.NMS_TH)]

    # ------------------------------------------------------------ scoring

    def score_track(self, faces: np.ndarray, audio_int16: np.ndarray) -> np.ndarray:
        """Un score par image d'UNE piste : `faces` `(T, 112, 112)` uint8, `audio_int16` le son de
        la piste à 16 kHz. Six passes (fenêtres de 1 à 6 s), moyennées, arrondies au dixième."""
        import python_speech_features

        torch = self.torch
        if len(audio_int16) < 400 or len(faces) == 0:   # moins d'une trame MFCC : rien à évaluer
            return np.zeros(0, dtype=float)
        mfcc = python_speech_features.mfcc(audio_int16, geo.MFCC_PER_S * 160, numcep=13, winlen=0.025, winstep=0.010)
        n_a, n_v = geo.scoring_length(len(mfcc), len(faces))
        if n_v == 0:
            return np.zeros(0, dtype=float)
        a = np.asarray(mfcc[:n_a], dtype=np.float32)
        v = np.asarray(faces[:n_v], dtype=np.float32)
        passes: list[list[float]] = []
        with torch.no_grad():
            for d in geo.DURATIONS:
                scores: list[float] = []
                for a0, a1, v0, v1 in geo.batches(n_v, d):
                    in_a = torch.from_numpy(a[a0:a1]).unsqueeze(0).to(self.device)
                    in_v = torch.from_numpy(v[v0:v1]).unsqueeze(0).to(self.device)
                    emb_a = self.asd.forward_audio_frontend(in_a)
                    emb_v = self.asd.forward_visual_frontend(in_v)
                    out = self.asd.forward_audio_visual_backend(emb_a, emb_v)
                    scores.extend(self.head(out).detach().cpu().numpy().tolist())
                passes.append(scores)
        return geo.average_scores(passes)

    # ------------------------------------------------------------ pipeline

    def run(self, video_path: Path, audio_int16: np.ndarray, scale: float,
            progress: Progress | None = None) -> tuple[list[dict], list[np.ndarray], dict]:
        """L'analyse complète de l'extrait. Rend `(tracks, scores, info)` : `tracks` dans la forme
        de `tracks.pckl` (`{'track': {'frame', 'bbox'}}`), `scores` un tableau par piste, `info`
        les compteurs et le temps de chaque étape."""
        import cv2

        dire = progress or (lambda _p, _m: None)
        t = {"scenes": 0.0, "detect": 0.0, "track": 0.0, "crop": 0.0, "score": 0.0}

        t0 = time.monotonic()
        scenes = detect_scenes(video_path)
        t["scenes"] = time.monotonic() - t0

        # PASSE 1 — un dictionnaire par détection, `frame` ABSOLU dans l'extrait (inference_video).
        t0 = time.monotonic()
        dets: list[list[dict]] = []
        cap = cv2.VideoCapture(str(video_path))
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                fidx = len(dets)
                dets.append([{"frame": fidx, "bbox": b[:4].tolist(), "conf": float(b[4])}
                             for b in self.detect_faces(frame, scale)])
                if fidx % 100 == 0:
                    dire(10 + int(45 * fidx / max(total, fidx + 1)), f"visages : image {fidx}/{total or '?'}")
        finally:
            cap.release()
        n_frames = len(dets)
        t["detect"] = time.monotonic() - t0

        # SUIVI par plan (un plan plus court que MIN_TRACK images est ignoré, comme dans main()).
        t0 = time.monotonic()
        tracks: list[dict] = []
        for s, e in scenes:
            s, e = max(0, s), min(e, n_frames)
            if e - s >= geo.MIN_TRACK:
                tracks.extend(geo.track_shot(dets[s:e]))
        t["track"] = time.monotonic() - t0
        dire(55, f"{len(tracks)} piste(s) de visage sur {len(scenes)} plan(s)")

        # PASSE 2 — recadrage. Une image n'est décodée que si une piste la contient.
        t0 = time.monotonic()
        plans = [geo.smooth_track(tr["bbox"]) for tr in tracks]
        by_frame: dict[int, list[tuple[int, int]]] = {}
        for ti, tr in enumerate(tracks):
            for k, f in enumerate(tr["frame"].tolist()):
                by_frame.setdefault(int(f), []).append((ti, k))
        crops = [np.full((len(tr["frame"]), geo.FACE_IN, geo.FACE_IN), geo.PAD_VALUE, dtype=np.uint8)
                 for tr in tracks]
        if by_frame:
            cap = cv2.VideoCapture(str(video_path))
            try:
                idx = 0
                dernier = max(by_frame)
                while idx <= dernier:
                    if idx in by_frame:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        for ti, k in by_frame[idx]:
                            p = plans[ti]
                            rect = geo.crop_rect(float(p["x"][k]), float(p["y"][k]), float(p["s"][k]))
                            face = geo.extract(frame, rect)
                            if face.size == 0:
                                continue   # boîte dégénérée : le gris de remplissage reste
                            face = cv2.resize(face, (geo.FACE_OUT, geo.FACE_OUT))
                            crops[ti][k] = geo.center(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY))
                    elif not cap.grab():
                        break
                    idx += 1
                    if idx % 200 == 0:
                        dire(55 + int(20 * idx / max(1, dernier + 1)), f"recadrage : image {idx}/{dernier + 1}")
            finally:
                cap.release()
        t["crop"] = time.monotonic() - t0

        # SCORING par piste.
        t0 = time.monotonic()
        sr = geo.MFCC_PER_S * 160
        scores: list[np.ndarray] = []
        for ti, tr in enumerate(tracks):
            a0, a1 = geo.track_audio_span(tr["frame"])
            audio = audio_int16[int(round(a0 * sr)):int(round(a1 * sr))]
            scores.append(self.score_track(crops[ti], audio))
            dire(75 + int(20 * (ti + 1) / len(tracks)), f"parole : piste {ti + 1}/{len(tracks)}")
        t["score"] = time.monotonic() - t0

        info = {"frames": n_frames, "scenes": len(scenes), "tracks": len(tracks),
                "timings": {f"{k}_s": round(v, 3) for k, v in t.items()}}
        return [{"track": tr} for tr in tracks], scores, info
