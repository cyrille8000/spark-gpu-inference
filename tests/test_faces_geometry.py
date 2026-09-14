"""La géométrie de LR-ASD, portée : chaque règle est comparée à l'ARITHMÉTIQUE LITTÉRALE de
`Columbia_test.py` / `S3FD.detect_faces`, pas à ce qu'on croit qu'elle fait. numpy seulement."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spark_infer import faces_geometry as geo  # noqa: E402


# ============================================================ prétraitement S3FD

def _s3fd_input_original(image_rgb):
    """La suite d'opérations de `S3FD.detect_faces`, telle quelle (entrée RGB)."""
    img_mean = np.array([104., 117., 123.])[:, np.newaxis, np.newaxis].astype('float32')
    x = np.swapaxes(image_rgb, 1, 2)
    x = np.swapaxes(x, 1, 0)
    x = x[[2, 1, 0], :, :]
    x = x.astype('float32')
    x -= img_mean
    x = x[[2, 1, 0], :, :]
    return x


def test_s3fd_input_identique_a_la_suite_d_operations_de_lrasd():
    rng = np.random.default_rng(1)
    bgr = rng.integers(0, 256, (24, 40, 3), dtype=np.uint8)
    rgb = bgr[:, :, ::-1]   # ce que faisait cv2.cvtColor(BGR2RGB) avant l'appel
    np.testing.assert_array_equal(geo.s3fd_input(bgr), _s3fd_input_original(rgb))
    assert geo.s3fd_input(bgr).flags["C_CONTIGUOUS"]


def test_nms_garde_la_meilleure_et_ecarte_ce_qui_la_recouvre():
    dets = np.array([
        [0, 0, 10, 10, 0.95],
        [1, 1, 11, 11, 0.90],      # recouvre la première à ~0,68 > 0,1 : écartée
        [50, 50, 60, 60, 0.80],    # ailleurs : gardée
    ], dtype=float)
    assert geo.nms(dets, 0.1).tolist() == [0, 2]
    assert geo.nms(np.empty((0, 5))).tolist() == []


# ============================================================ suivi

def _det(frame, x1, y1, x2, y2, conf=0.99):
    return {"frame": frame, "bbox": [x1, y1, x2, y2], "conf": conf}


def test_iou():
    assert geo.iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert geo.iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert geo.iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(50 / 150)


def test_une_piste_est_interpolee_image_par_image():
    # 15 images, une détection sur chaque sauf la 7e : la piste doit couvrir les 15 sans trou.
    shot = []
    for f in range(15):
        shot.append([] if f == 7 else [_det(f, 10 + f, 10, 50 + f, 50)])
    pistes = geo.track_shot(shot)
    assert len(pistes) == 1
    p = pistes[0]
    assert p["frame"].tolist() == list(range(15))
    assert p["bbox"].shape == (15, 4)
    assert p["bbox"][7].tolist() == pytest.approx([17, 10, 57, 50])   # interpolée entre 6 et 8
    # L'entrée n'a pas été consommée (l'original la vidait sur place).
    assert shot[0] == [_det(0, 10, 10, 50, 50)]


def test_une_piste_trop_courte_est_jetee():
    shot = [[_det(f, 10, 10, 50, 50)] for f in range(geo.MIN_TRACK + 1)]   # 11 images → gardée (> 10)
    assert len(geo.track_shot(shot)) == 1
    shot = [[_det(f, 10, 10, 50, 50)] for f in range(geo.MIN_TRACK)]       # 10 images → jetée
    assert geo.track_shot(shot) == []


def test_deux_visages_font_deux_pistes():
    shot = [[_det(f, 10, 10, 50, 50), _det(f, 200, 10, 240, 50)] for f in range(20)]
    pistes = geo.track_shot(shot)
    assert len(pistes) == 2
    assert sorted(p["bbox"][0][0] for p in pistes) == [10, 200]


def test_un_trou_trop_long_coupe_la_piste():
    # 15 images, puis 11 images sans détection (> NUM_FAILED_DET), puis 15 images.
    shot = [[_det(f, 10, 10, 50, 50)] for f in range(15)]
    shot += [[] for _ in range(11)]
    shot += [[_det(f, 10, 10, 50, 50)] for f in range(26, 41)]
    pistes = geo.track_shot(shot)
    assert len(pistes) == 2
    assert [p["frame"][0] for p in pistes] == [0, 26]


def test_un_trou_tolere_ne_coupe_pas():
    # `face.frame - track[-1].frame <= NUM_FAILED_DET` : 9 images vides font un écart de 10, toléré ;
    # 10 images vides feraient 11, et la piste se couperait (test précédent).
    shot = [[_det(f, 10, 10, 50, 50)] for f in range(15)]
    shot += [[] for _ in range(geo.NUM_FAILED_DET - 1)]
    shot += [[_det(f, 10, 10, 50, 50)] for f in range(24, 39)]
    pistes = geo.track_shot(shot)
    assert len(pistes) == 1 and pistes[0]["frame"].tolist() == list(range(39))


def test_un_visage_qui_saute_ne_rejoint_pas_la_piste():
    shot = [[_det(f, 10, 10, 50, 50)] for f in range(15)]
    shot += [[_det(f, 300, 300, 340, 340)] for f in range(15, 30)]   # même cadence, autre endroit
    assert len(geo.track_shot(shot)) == 2


def test_plan_vide():
    assert geo.track_shot([[] for _ in range(20)]) == []
    assert geo.track_shot([]) == []


# ============================================================ recadrage

def _crop_original(image, x, y, bs, cs=geo.CROP_SCALE):
    """`crop_video`, mot pour mot : compléter TOUTE l'image de `bsi`, puis découper."""
    bsi = int(bs * (1 + 2 * cs))
    frame = np.pad(image, ((bsi, bsi), (bsi, bsi), (0, 0)), 'constant', constant_values=(110, 110))
    my = y + bsi
    mx = x + bsi
    return frame[int(my - bs):int(my + bs * (1 + 2 * cs)), int(mx - bs * (1 + cs)):int(mx + bs * (1 + cs))]


def test_crop_rect_et_extract_reproduisent_crop_video():
    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
    cas = 0
    for _ in range(400):
        x = float(rng.uniform(-10, 170))
        y = float(rng.uniform(-10, 130))
        s = float(rng.uniform(0.6, 40))
        attendu = _crop_original(image, x, y, s)
        rect = geo.crop_rect(x, y, s)
        # L'original laisse numpy tronquer une découpe qui déborde de l'image COMPLÉTÉE (et un
        # indice négatif y ferait le tour) : on ne compare que là où l'original est bien défini.
        bsi = int(s * (1 + 2 * geo.CROP_SCALE))
        if rect[0] + bsi < 0 or rect[2] + bsi < 0 or rect[1] + bsi > image.shape[0] + 2 * bsi \
                or rect[3] + bsi > image.shape[1] + 2 * bsi:
            continue
        cas += 1
        np.testing.assert_array_equal(geo.extract(image, rect), attendu)
    assert cas > 300


def test_extract_remplit_de_gris_hors_cadre():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    out = geo.extract(image, (-2, 3, -1, 4))
    assert out.shape == (5, 5, 3)
    assert (out[:2] == geo.PAD_VALUE).all() and (out[:, :1] == geo.PAD_VALUE).all()
    assert (out[2:, 1:] == 0).all()
    assert geo.extract(image, (5, 5, 5, 5)).size == 0   # rectangle dégénéré : vide, pas une erreur


def test_smooth_track_lisse_par_mediane():
    b = np.array([[0, 0, 10, 20]] * 20, dtype=float)
    p = geo.smooth_track(b)
    # Demi-taille = max(h, w)/2 = 10, centre (5, 10). Aux bouts, medfilt 13 complète par des zéros
    # mais ils restent minoritaires (6 contre 7) : une piste stable garde ses valeurs jusqu'au bord.
    assert p["s"][10] == 10 and p["x"][10] == 5 and p["y"][10] == 10
    assert p["s"][0] == 10 and p["s"][-1] == 10
    # Une valeur aberrante isolée est effacée par la médiane.
    b[10] = [0, 0, 100, 200]
    assert geo.smooth_track(b)["s"][10] == 10


def test_center():
    face = np.arange(224 * 224).reshape(224, 224)
    c = geo.center(face)
    assert c.shape == (112, 112) and c[0, 0] == face[56, 56] and c[-1, -1] == face[167, 167]


# ============================================================ scoring

def test_scoring_length_quatre_trames_audio_par_image():
    # Cas réel : le son est un peu plus court que l'image (99 trames MFCC pour 25 images).
    assert geo.scoring_length(99, 25) == (96, 24)
    # L'original : length = (99 - 99 % 4) / 100 = 0,96 s → 96 trames, 24 images. Identique.
    assert geo.scoring_length(100, 25) == (100, 25)
    # Audio plus long que l'image : borné par l'image (l'original plantait dans ce cas).
    assert geo.scoring_length(1000, 25) == (100, 25)
    assert geo.scoring_length(3, 25) == (0, 0)
    assert geo.scoring_length(100, 0) == (0, 0)


def test_batches_couvrent_exactement_les_images():
    for n_v in (1, 24, 25, 26, 100, 151):
        for d in geo.DURATIONS:
            lots = geo.batches(n_v, d)
            assert lots[0][2] == 0 and lots[-1][3] == n_v
            assert all(a1 - a0 == 4 * (v1 - v0) for a0, a1, v0, v1 in lots)
            assert sum(v1 - v0 for _, _, v0, v1 in lots) == n_v
            assert len(lots) == -(-n_v // (25 * d))   # ceil(length / duration) comme l'original
    assert geo.batches(0, 1) == []


def test_average_scores_arrondit_au_dixieme():
    out = geo.average_scores([[0.04, 1.26], [-0.04, 1.34]])
    assert out.tolist() == [0.0, 1.3]
    assert geo.average_scores([]).tolist() == []


def test_track_audio_span():
    assert geo.track_audio_span(np.array([50, 51, 52])) == (2.0, 2.12)
