"""La logique de segments — des sorties de LR-ASD aux moments de parole et aux positions.

Repris de `spark-dubbing-lipsync/test_segments.py` (2026-08-15) et passé en pytest.

Ni GPU, ni torch. Mais NUMPY, si — et ce n'est pas un détail : LR-ASD rend des tableaux numpy,
pas des listes Python. Deux pannes en production l'ont rappelé : des fixtures en listes passaient
au vert sur du code qui cassait dès la vraie donnée (`bool(tableau)` lève « truth value of an array
is ambiguous »). Les fixtures d'ici imitent donc les types réels.
"""
import sys
from pathlib import Path

import numpy
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spark_infer.faces_segments import (  # noqa: E402
    MAX_POINTS, clip_to_window, merge, normalise_bbox, shift, smoothed_speaking, speaking_faces,
    speech_seconds, to_milliseconds, track_intervals,
)

# La cadence est MESURÉE en production sur le fichier découpé ; les tests en choisissent une.
FPS = 25.0


def TI(frames, speaking, fps=FPS):
    return track_intervals(frames, speaking, fps)


def SS(tracks, scores, offset=0.0, window=None, fps=FPS):
    return speech_seconds(tracks, scores, fps, offset=offset, window=window)


class FauxTrack(dict):
    """La forme EXACTE d'une piste rendue par `faces_engine.run` (= une entrée de `tracks.pckl`) :
    `{'track': {'frame': ndarray, 'bbox': ndarray (n, 4)}}`. `frame` est un tableau numpy, pas une
    liste — d'où le `.tolist()` côté production."""

    def __init__(self, frames, bbox=None):
        frames = numpy.array(frames)
        boites = numpy.zeros((len(frames), 4)) if bbox is None else numpy.array(bbox, dtype=float)
        super().__init__(track={"frame": frames, "bbox": boites})


def scores_np(valeurs):
    """Des scores comme LR-ASD les rend : un tableau numpy de flottants arrondis au dixième."""
    return numpy.round(numpy.array(valeurs, dtype=float), 1)


# ── Intervalles d'une piste ──────────────────────────────────────────────────

def test_track_intervals():
    # Images 0..4, parle sur 1,2,3 → de 1/25 à 4/25 (fin de la 3e image incluse).
    assert TI([0, 1, 2, 3, 4], [False, True, True, True, False]) == [(1 / FPS, 4 / FPS)]
    assert TI([10, 11, 12], [False, True, True]) == [(11 / FPS, 13 / FPS)]
    assert TI([0, 1, 2, 3, 4, 5], [True, False, False, True, True, False]) == [(0.0, 1 / FPS), (3 / FPS, 5 / FPS)]
    assert TI([0, 1, 2], [False, False, False]) == []
    # Les temps suivent les NUMÉROS d'image, pas leur rang.
    assert TI([100, 101, 102], [False, True, False]) == [(101 / FPS, 102 / FPS)]


def test_moins_de_scores_que_d_images():
    """LE cas qui a fait planter la première exécution réelle : LR-ASD n'évalue que la partie
    couverte par l'image ET le son, et coupe le surplus à la fin. La queue non évaluée n'est jamais
    revendiquée : sans son, on ne sait pas."""
    assert TI([0, 1, 2, 3, 4, 5, 6], [False, True, True]) == [(1 / FPS, 3 / FPS)]
    assert TI([0, 1, 2, 3, 4, 5], [False, True, True]) == [(1 / FPS, 3 / FPS)]
    assert TI([10, 11, 12, 13], [True]) == [(10 / FPS, 11 / FPS)]
    assert TI([0, 1, 2], []) == []
    assert TI([0, 1], [True, True, True, True]) == [(0.0, 2 / FPS)]


# ── Fusion, décalage, rognage ────────────────────────────────────────────────

def test_merge():
    assert merge([(0.0, 2.0), (1.0, 3.0)]) == [(0.0, 3.0)]
    assert merge([(0.0, 2.0), (2.0, 3.0)]) == [(0.0, 3.0)]
    assert merge([(0.0, 1.0), (2.0, 3.0)]) == [(0.0, 1.0), (2.0, 3.0)]
    assert merge([(5.0, 6.0), (0.0, 1.0)]) == [(0.0, 1.0), (5.0, 6.0)]
    assert merge([(0.0, 9.0), (2.0, 3.0)]) == [(0.0, 9.0)]


def test_shift():
    assert shift([(1.0, 2.0)], 300.0) == [(301.0, 302.0)]
    assert shift([(1.0, 2.0)], 0) == [(1.0, 2.0)]
    assert shift([], 300.0) == []


def test_clip_to_window():
    assert clip_to_window([(310.0, 320.0)], 300, 600) == [(310.0, 320.0)]
    assert clip_to_window([(298.0, 299.0)], 300, 600) == []
    assert clip_to_window([(299.0, 302.0)], 300, 600) == [(300.0, 302.0)]
    assert clip_to_window([(598.0, 601.0)], 300, 600) == [(598.0, 600.0)]
    assert clip_to_window([(298.0, 300.0)], 300, 600) == []
    assert clip_to_window([(200.0, 700.0)], 300, 600) == [(300.0, 600.0)]


# ── Arrondi à la milliseconde ────────────────────────────────────────────────

def test_to_milliseconds_vers_l_exterieur():
    # On ne rogne JAMAIS la parole : début en dessous, fin au-dessus.
    assert to_milliseconds([(1.60004, 4.10001)]) == [{"start": 1.6, "end": 4.101}]
    # À 25 i/s les bornes tombent sur des multiples de 40 ms : déjà sur la grille, inchangées.
    assert to_milliseconds([(3.04, 5.16)]) == [{"start": 3.04, "end": 5.16}]
    # Refusion APRÈS arrondi.
    assert to_milliseconds([(1.2, 2.4004), (2.4008, 3.1)]) == [{"start": 1.2, "end": 3.1}]
    assert to_milliseconds([(1.2, 2.4), (8.6, 9.1)]) == [{"start": 1.2, "end": 2.4}, {"start": 8.6, "end": 9.1}]
    # Un demi-silence entre deux paroles n'est plus avalé (la seconde entière le faisait).
    assert to_milliseconds([(1.2, 2.4), (2.9, 3.6)]) == [{"start": 1.2, "end": 2.4}, {"start": 2.9, "end": 3.6}]
    # Une seule image parlée reste un intervalle non vide — et fait 40 ms.
    assert to_milliseconds(TI([75], [True])) == [{"start": 3.0, "end": 3.04}]


# ── Bout en bout ─────────────────────────────────────────────────────────────

def test_speech_seconds_fusionne_les_pistes_simultanees():
    # Deux visages qui parlent en même temps → UN seul intervalle : « quelqu'un parle-t-il ? »
    out = SS([FauxTrack([25, 26, 27, 28]), FauxTrack([27, 28, 29, 30])],
             [scores_np([1, 1, 1, 1]), scores_np([1, 1, 1, 1])])
    assert out == [{"start": 1.0, "end": 1.24}]
    assert SS([FauxTrack([25, 26, 27, 28])], [scores_np([-3, -3, -3, -3])]) == []
    assert SS([], []) == []


def test_appariement_piste_score_rompu_leve():
    # `zip` tronquerait EN SILENCE : des pistes entières disparaîtraient.
    with pytest.raises(ValueError):
        SS([FauxTrack([0, 1]), FauxTrack([2, 3])], [scores_np([1, 1])])


# ── Critère « ça parle », relevé dans le code de LR-ASD ──────────────────────

def test_smoothed_speaking():
    # Le seuil est `>= 0` : les scores sont arrondis au dixième, 0.0 est COURANT.
    assert smoothed_speaking(scores_np([0.0])) == [True]
    assert smoothed_speaking(scores_np([-0.1])) == [False]
    # Lissé sur ~5 images : un creux isolé est absorbé…
    assert smoothed_speaking(scores_np([2.0, 2.0, -1.0, 2.0, 2.0])) == [True] * 5
    # …mais un vrai basculement reste visible (une image AVANT le changement brut, fenêtre centrée).
    assert smoothed_speaking(scores_np([5.0, 5.0, 5.0, -5.0, -5.0, -5.0, -5.0, -5.0])) == \
        [True, True, True, False, False, False, False, False]
    assert smoothed_speaking(scores_np([])) == []
    assert smoothed_speaking(scores_np([1.0])) == [True]


# ── Analyse par portion (le fan-out parallèle) ───────────────────────────────

def piste_parlante(t0, t1):
    """Une piste de visage qui parle de t0 à t1, en secondes DANS L'EXTRAIT."""
    frames = list(range(round(t0 * FPS), round(t1 * FPS)))
    return FauxTrack(frames), scores_np([1] * len(frames))


def test_portions_recollees_redonnent_la_parole_d_origine():
    # Une parole de 299 s à 302 s, analysée par DEUX jobs dont la coupure tombe dedans, à 300 s.
    #   job A : portion [0, 300],   marge 2 s → extrait [0, 302],   décalage 0
    #   job B : portion [300, 600], marge 2 s → extrait [298, 602], décalage 298
    tA, sA = piste_parlante(299.0, 302.0)
    a = SS([tA], [sA], offset=0.0, window=(0, 300))
    tB, sB = piste_parlante(1.0, 4.0)
    b = SS([tB], [sB], offset=298.0, window=(300, 600))
    assert a == [{"start": 299, "end": 300}]
    assert b == [{"start": 300, "end": 302}]
    assert to_milliseconds([(x["start"], x["end"]) for x in a + b]) == [{"start": 299, "end": 302}]
    # Le piège que `offset` existe pour éviter : sans lui, B se tromperait de 298 s.
    assert SS([tB], [sB]) == [{"start": 1, "end": 4}]


def test_la_marge_est_jetee():
    tM, sM = piste_parlante(298.5, 299.5)
    assert SS([tM], [sM], offset=0.0, window=(300, 600)) == []
    tF, sF = piste_parlante(300.0, 300.4)
    assert SS([tF], [sF], window=(300.5, 600)) == []       # borne NON entière : rognage exact
    assert SS([tM], [sM], window=(400, 500)) == []


# ── Positions à l'image ──────────────────────────────────────────────────────

LARGEUR, HAUTEUR = 200, 100
BOITE = [20, 10, 60, 30]     # → x=0.1, y=0.1, w=0.2, h=0.2


def SF(tracks, scores, offset=0.0, window=None, w=LARGEUR, h=HAUTEUR, fps=FPS):
    return speaking_faces(tracks, scores, fps, w, h, offset=offset, window=window)


def fixe(n, boite=None, depart=0):
    """Une piste de `n` images parlées, visage IMMOBILE, boîte constante."""
    boite = boite or BOITE
    frames = list(range(depart, depart + n))
    return FauxTrack(frames, [boite] * n), scores_np([1] * n)


def NB(bbox, w=LARGEUR, h=HAUTEUR):
    return tuple(round(v, 3) for v in normalise_bbox(bbox, w, h))


def test_normalise_bbox():
    assert NB(BOITE) == (0.1, 0.1, 0.2, 0.2)                      # fractions, pas pixels
    assert NB([-10, -5, 210, 110]) == (0.0, 0.0, 1.0, 1.0)         # rabotée, jamais négative


def test_plan_fixe_deux_points():
    t1, s1 = fixe(20)     # 0,8 s : sous la seconde, aucun point forcé par le temps
    assert SF([t1], [s1]) == [{"start": 0.0, "end": 0.8,
                               "box": [[0.0, 0.1, 0.1, 0.2, 0.2], [0.76, 0.1, 0.1, 0.2, 0.2]]}]


def test_un_deplacement_ajoute_un_point():
    bouge = [BOITE] * 10 + [[40, 10, 80, 30]] * 10
    t2 = FauxTrack(list(range(20)), bouge)
    assert [p[0] for p in SF([t2], [scores_np([1] * 20)])[0]["box"]] == [0.0, 0.4, 0.76]


def test_immobile_un_point_par_seconde():
    t3, s3 = fixe(75)
    assert [p[0] for p in SF([t3], [s3])[0]["box"]] == [0.0, 1.0, 2.0, 2.96]


def test_temps_de_la_video_d_origine_et_bornes_exactes():
    t1, s1 = fixe(20)
    assert [p[0] for p in SF([t1], [s1], offset=298.0)[0]["box"]] == [298.0, 298.76]
    t4, s4 = fixe(20, depart=5)
    assert [(f["start"], f["end"]) for f in SF([t4], [s4])] == [(0.2, 1.0)]


def test_deux_visages_simultanes_deux_suites():
    tg = FauxTrack(list(range(20)), [[10, 10, 50, 30]] * 20)
    td = FauxTrack(list(range(20)), [[120, 10, 160, 30]] * 20)
    uns = scores_np([1] * 20)
    assert [f["box"][0][1] for f in SF([tg, td], [uns, uns])] == [0.05, 0.6]


def test_fenetre_et_marge_sur_les_boites():
    tm, sm = fixe(20, depart=round(298.5 * FPS))
    assert SF([tm], [sm], window=(300, 600)) == []
    # Plan fixe de 3 s, points à 0/1/2/2,96 s : une fenêtre 1,2–1,8 s n'en contient aucun, et
    # pourtant il y a un visage à l'écran → le point le plus proche est ramené sur la bordure.
    t3, s3 = fixe(75)
    assert SF([t3], [s3], window=(1.2, 1.8)) == [{"start": 1.2, "end": 1.8, "box": [[1.2, 0.1, 0.1, 0.2, 0.2]]}]
    t1, s1 = fixe(20)
    assert SF([t1], [s1], w=0, h=0) == []          # sans dimensions, aucune boîte inventée


def test_plafond_de_points_sur_une_scene_agitee():
    derive = [[20 + 4 * (i % 40), 10, 60 + 4 * (i % 40), 30] for i in range(5000)]
    tp = FauxTrack(list(range(5000)), derive)
    points = sum(len(f["box"]) for f in SF([tp], [scores_np([1] * 5000)]))
    assert 0 < points <= MAX_POINTS
