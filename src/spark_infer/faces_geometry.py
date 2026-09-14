"""Visages qui parlent — la géométrie pure de LR-ASD : suivi des visages, recadrage, découpage
en lots pour le scoring. numpy et scipy seulement : tout ici se teste sans GPU, sans torch et sans
OpenCV, et c'est voulu — ce sont ces règles qui décident de la justesse du résultat.

Chaque constante et chaque formule est RELEVÉE dans `Columbia_test.py` de LR-ASD (commit
1b6dcd2d), pas déduite de sa documentation. Là où le port s'écarte de l'original, c'est dit.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import medfilt

# --- Détection (S3FD) ---
CONF_TH = 0.9            # confiance minimale d'une boîte (inference_video : conf_th=0.9)
NMS_TH = 0.1             # recouvrement toléré entre deux boîtes d'une même image (detect_faces : nms_(…, 0.1))
S3FD_MEAN_BGR = np.array([104.0, 117.0, 123.0], dtype=np.float32)

# --- Suivi (track_shot) ---
IOU_MIN = 0.5            # recouvrement minimal entre deux détections consécutives d'une même piste
NUM_FAILED_DET = 10      # images manquées tolérées avant de clore la piste (--numFailedDet)
MIN_TRACK = 10           # une piste de moins de 11 images est jetée (--minTrack, comparaison stricte)
MIN_FACE_SIZE = 1        # taille moyenne minimale d'une piste, en pixels (--minFaceSize)

# --- Recadrage (crop_video) ---
CROP_SCALE = 0.40        # marge autour du visage (--cropScale)
MEDFILT_K = 13           # lissage médian du centre et de la taille
PAD_VALUE = 110          # gris de remplissage hors cadre
FACE_OUT = 224           # taille du recadrage
FACE_IN = 112            # le CENTRE 112×112 est ce que le réseau voit (evaluate_network)

# --- Scoring (evaluate_network) ---
FPS = 25
MFCC_PER_S = 100
# `durationSet = {1,1,1,2,2,2,3,3,4,5,6}` dans l'original — un ENSEMBLE Python, donc {1,…,6} :
# six passes, une par longueur de fenêtre, moyennées.
DURATIONS = (1, 2, 3, 4, 5, 6)


# ============================================================ détection

def s3fd_input(image_bgr: np.ndarray) -> np.ndarray:
    """L'image telle que S3FD la reçoit : `(3, H, W)` float32, ordre RGB, moyenne VGG soustraite.

    L'original (`S3FD.detect_faces`) part du RGB, passe en BGR pour soustraire (104, 117, 123),
    puis REVIENT en RGB. Même résultat en une ligne depuis le BGR d'OpenCV : soustraire la moyenne
    BGR, puis inverser l'ordre des canaux. Le test le vérifie contre la suite d'opérations littérale."""
    x = image_bgr.astype(np.float32) - S3FD_MEAN_BGR
    x = x[:, :, ::-1]
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def nms(dets: np.ndarray, thresh: float = NMS_TH) -> np.ndarray:
    """Suppression des recouvrements sur `[x1, y1, x2, y2, score]` — celle de LR-ASD (py_cpu_nms).
    Rend les indices gardés, par score décroissant."""
    if len(dets) == 0:
        return np.zeros(0, dtype=int)
    x1, y1, x2, y2, scores = (dets[:, i] for i in range(5))
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(ovr <= thresh)[0] + 1]
    return np.array(keep, dtype=int)


# ============================================================ suivi

def iou(a, b) -> float:
    """Recouvrement de deux boîtes `[x1, y1, x2, y2]` (bb_intersection_over_union)."""
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (a[2] - a[0]) * (a[3] - a[1])
    areaB = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(areaA + areaB - inter)


def track_shot(shot_faces: list[list[dict]]) -> list[dict]:
    """Les pistes de visages d'UN plan : `[{'frame': ndarray int, 'bbox': ndarray (n, 4)}, …]`.

    `shot_faces[k]` = les détections de la k-ième image du plan, chacune `{'frame', 'bbox', 'conf'}`
    avec `frame` ABSOLU (numéro dans l'extrait entier). Port de `track_shot` : on tire une piste en
    consommant les détections qui s'enchaînent (même visage si IoU > 0,5, à moins de 10 images
    d'écart), on recommence tant qu'il reste des détections ; une piste de plus de `MIN_TRACK`
    images est INTERPOLÉE image par image (une boîte par image, sans trou) puis gardée si sa taille
    moyenne dépasse `MIN_FACE_SIZE`.

    L'entrée n'est pas modifiée (l'original consommait ses listes sur place). L'original itérait
    aussi sur une liste qu'il vidait, ce qui saute une détection après chaque retrait ; ici on
    itère sur une copie. Sans effet sur le résultat : la détection sautée est de la même image que
    celle qu'on vient de prendre, et deux boîtes d'une même image ne se recouvrent jamais assez
    (NMS à 0,1) pour rejoindre la même piste — elle est reprise à la passe suivante, comme avant."""
    faces = [list(frame_faces) for frame_faces in shot_faces]
    tracks: list[dict] = []
    while True:
        track: list[dict] = []
        for frame_faces in faces:
            for face in list(frame_faces):
                if not track:
                    track.append(face)
                    frame_faces.remove(face)
                elif face["frame"] - track[-1]["frame"] <= NUM_FAILED_DET:
                    if iou(face["bbox"], track[-1]["bbox"]) > IOU_MIN:
                        track.append(face)
                        frame_faces.remove(face)
                        continue
                else:
                    break
        if not track:
            break
        if len(track) > MIN_TRACK:
            frame_num = np.array([f["frame"] for f in track])
            bboxes = np.array([np.array(f["bbox"], dtype=float) for f in track])
            frame_i = np.arange(frame_num[0], frame_num[-1] + 1)
            bboxes_i = np.stack([interp1d(frame_num, bboxes[:, j])(frame_i) for j in range(4)], axis=1)
            if max(np.mean(bboxes_i[:, 2] - bboxes_i[:, 0]), np.mean(bboxes_i[:, 3] - bboxes_i[:, 1])) > MIN_FACE_SIZE:
                tracks.append({"frame": frame_i, "bbox": bboxes_i})
    return tracks


# ============================================================ recadrage

def smooth_track(bboxes: np.ndarray) -> dict:
    """Centre et demi-taille de chaque image d'une piste, lissés par médiane (crop_video : `dets`).
    `medfilt` complète par des zéros aux deux bouts, comme dans l'original ; à 13 ils restent
    minoritaires (6 contre 7), une piste stable garde donc ses valeurs jusqu'au bord."""
    b = np.asarray(bboxes, dtype=float)
    s = np.maximum(b[:, 3] - b[:, 1], b[:, 2] - b[:, 0]) / 2
    y = (b[:, 1] + b[:, 3]) / 2
    x = (b[:, 0] + b[:, 2]) / 2
    return {"s": medfilt(s, kernel_size=MEDFILT_K), "x": medfilt(x, kernel_size=MEDFILT_K),
            "y": medfilt(y, kernel_size=MEDFILT_K)}


def crop_rect(x: float, y: float, s: float, cs: float = CROP_SCALE) -> tuple[int, int, int, int]:
    """Le rectangle `(r0, r1, c0, c1)` à découper dans l'image NON complétée, avec exactement
    l'arithmétique de `crop_video` : l'original complète l'image de `bsi` pixels de chaque côté,
    tronque en `int` dans ces coordonnées-là, puis découpe. On refait les mêmes `int()` sur les
    mêmes nombres et on retranche `bsi` après — la troncature vers zéro ne commute pas avec la
    translation quand une borne est négative, d'où l'ordre."""
    bsi = int(s * (1 + 2 * cs))
    my = y + bsi
    mx = x + bsi
    r0, r1 = int(my - s), int(my + s * (1 + 2 * cs))
    c0, c1 = int(mx - s * (1 + cs)), int(mx + s * (1 + cs))
    return r0 - bsi, r1 - bsi, c0 - bsi, c1 - bsi


def extract(image: np.ndarray, rect: tuple[int, int, int, int], pad: int = PAD_VALUE) -> np.ndarray:
    """Découpe `rect` dans `image`, en remplissant de `pad` ce qui tombe hors cadre — l'équivalent de
    « compléter toute l'image puis découper » sans jamais copier l'image entière (une image 1080p
    par visage et par image, c'est ce que faisait l'original)."""
    r0, r1, c0, c1 = rect
    h, w = image.shape[:2]
    out = np.full((max(0, r1 - r0), max(0, c1 - c0)) + image.shape[2:], pad, dtype=image.dtype)
    rr0, rr1 = max(r0, 0), min(r1, h)
    cc0, cc1 = max(c0, 0), min(c1, w)
    if rr1 > rr0 and cc1 > cc0:
        out[rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0] = image[rr0:rr1, cc0:cc1]
    return out


def center(face224: np.ndarray) -> np.ndarray:
    """Le centre 112×112 d'un visage 224×224 (evaluate_network : `face[56:168, 56:168]`)."""
    lo, hi = FACE_OUT // 2 - FACE_IN // 2, FACE_OUT // 2 + FACE_IN // 2
    return face224[lo:hi, lo:hi]


# ============================================================ scoring

def track_audio_span(frames: np.ndarray, fps: float = FPS) -> tuple[float, float]:
    """L'intervalle audio d'une piste, en secondes DANS L'EXTRAIT : de la première image à la FIN
    de la dernière (`(frame[-1] + 1) / 25`), la convention de crop_video."""
    return float(frames[0]) / fps, float(frames[-1] + 1) / fps


def scoring_length(n_audio: int, n_video: int) -> tuple[int, int]:
    """Combien de trames MFCC et d'images entrent dans le réseau : `(n_a, n_v)` avec n_a = 4 × n_v.

    L'original prend `length = min((A - A % 4) / 100, V)` puis `A[:length*100]`, `V[:length*25]` —
    en mélangeant des SECONDES (audio) et des IMAGES (vidéo). Ça marche parce que le son d'une
    piste est toujours un peu plus court que son image (les MFCC perdent une trame et demie en
    bout). On l'écrit ici avec les deux côtés en images : même résultat dans tous les cas réels,
    et plus de plantage possible si l'audio dépasse."""
    n_v = min(n_audio // 4, n_video)
    if n_v <= 0:
        return 0, 0
    return 4 * n_v, n_v


def batches(n_video: int, duration: int) -> list[tuple[int, int, int, int]]:
    """Les lots d'une passe de `duration` secondes : `(a0, a1, v0, v1)` bornes MFCC et images,
    `ceil(length / duration)` lots comme l'original, le dernier plus court."""
    if n_video <= 0:
        return []
    nb = int(math.ceil((n_video / FPS) / duration))
    out = []
    for i in range(nb):
        v0, v1 = i * duration * FPS, min((i + 1) * duration * FPS, n_video)
        out.append((v0 * 4, v1 * 4, v0, v1))
    return out


def average_scores(passes: list[list[float]]) -> np.ndarray:
    """Moyenne des six passes, arrondie au dixième (evaluate_network) — d'où des `0.0` fréquents,
    que le seuil `>= 0` de LR-ASD classe du côté « parle »."""
    if not passes:
        return np.zeros(0, dtype=float)
    return np.round(np.mean(np.array(passes, dtype=float), axis=0), 1).astype(float)
