"""La découpe de l'extrait — la ligne ffmpeg construite, jamais lancée.

Ce qui doit rester vrai : `-ss`/`-to` AVANT chaque `-i` (sinon ffmpeg décode et télécharge la vidéo
depuis son début), la cadence par le FILTRE `fps=25` et jamais `-r` (mesuré : `-r` biaise de 52 ms),
la reconnexion HTTP seulement sur les URL (sur un fichier local, ffmpeg refuse ces options).
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spark_infer.faces_clip import (  # noqa: E402
    DEFAUT_LRASD, LARGEUR_DETECTION_MIN, build_cut_command, clip_bounds, facedet_scale, parse_fps,
)

V = "https://r2.example/video_final.mp4"
A = "https://r2.example/audio_stream.m4a"


def _idx(cmd, option, n=0):
    """Position de la n-ième occurrence de `option`."""
    pos = [i for i, v in enumerate(cmd) if v == option]
    return pos[n]


def test_clip_bounds():
    assert clip_bounds((300, 600), 2) == (298.0, 602.0)
    assert clip_bounds((300, 600), 0) == (300.0, 600.0)
    assert clip_bounds((1, 300), 2) == (0.0, 302.0)      # jamais de temps négatif
    assert clip_bounds(None, 2) == (0.0, None)           # toute la vidéo


def test_deux_entrees_chacune_avec_sa_fenetre():
    cmd = build_cut_command(V, A, 298.0, 602.0, "/tmp/clip.mp4", "/tmp/audio.wav")
    assert cmd.count("-i") == 2
    # Chaque `-i` est précédé de SON `-ss` / `-to` : seek en entrée, même fenêtre pour l'image et le son.
    for n in range(2):
        i = _idx(cmd, "-i", n)
        assert cmd[i - 4:i] == ["-ss", "298.000", "-to", "602.000"]
    assert cmd[_idx(cmd, "-i", 0) + 1] == V and cmd[_idx(cmd, "-i", 1) + 1] == A
    # L'image vient de l'entrée 0, le son de l'entrée 1.
    assert cmd[_idx(cmd, "-map", 0) + 1] == "0:v:0"
    assert cmd[_idx(cmd, "-map", 1) + 1] == "1:a:0"


def test_une_seule_entree_quand_le_son_est_dans_la_video():
    for audio in (None, V):
        cmd = build_cut_command(V, audio, 0.0, 5.0, "/tmp/clip.mp4", "/tmp/audio.wav")
        assert cmd.count("-i") == 1
        assert cmd[_idx(cmd, "-map", 1) + 1] == "0:a:0"


def test_cadence_par_le_filtre_jamais_par_r():
    cmd = build_cut_command(V, A, 0.0, 5.0, "/tmp/clip.mp4", "/tmp/audio.wav")
    assert cmd[_idx(cmd, "-vf") + 1] == "fps=25"
    assert "-r" not in cmd
    assert "libx264" in cmd and "h264_nvenc" not in cmd    # CPU : marche partout (A100/H100 sans NVENC)
    # L'image sort SANS son, le son sort SANS image, en PCM 16 bits aux canaux d'origine (pas de -ac).
    assert cmd[_idx(cmd, "-an") + 1] == "/tmp/clip.mp4"
    assert "-vn" in cmd and cmd[_idx(cmd, "-c:a") + 1] == "pcm_s16le" and "-ac" not in cmd
    assert cmd[-1] == "/tmp/audio.wav"


def test_reconnexion_http_seulement_sur_les_url():
    distant = build_cut_command(V, A, 0.0, 5.0, "/tmp/c.mp4", "/tmp/a.wav")
    local = build_cut_command("/data/v.mp4", "/data/a.m4a", 0.0, 5.0, "/tmp/c.mp4", "/tmp/a.wav")
    assert distant.count("-reconnect") == 2
    assert "-reconnect" not in local


def test_sans_borne_de_fin():
    cmd = build_cut_command(V, None, 0.0, None, "/tmp/c.mp4", "/tmp/a.wav")
    assert "-to" not in cmd and cmd[_idx(cmd, "-ss") + 1] == "0.000"


def test_facedet_scale():
    # 1080p et 1280 : le défaut de LR-ASD, intact (il cherche déjà sur ≥ 320 px).
    assert facedet_scale(1920) == DEFAUT_LRASD
    assert facedet_scale(1280) == DEFAUT_LRASD
    # Source 360p (640 px) : relevée pour chercher sur 320 px.
    assert facedet_scale(640) == pytest.approx(LARGEUR_DETECTION_MIN / 640)
    # Jamais au-dessus de 1 (agrandir n'ajoute rien), jamais sous le défaut.
    assert facedet_scale(200) == 1.0
    assert all(facedet_scale(w) >= DEFAUT_LRASD for w in (320, 640, 1280, 1920, 3840))
    assert facedet_scale(None) == DEFAUT_LRASD and facedet_scale(0) == DEFAUT_LRASD


def test_parse_fps():
    assert parse_fps("25/1") == 25.0
    assert parse_fps("24000/1001") == pytest.approx(23.976, abs=1e-3)
    assert parse_fps("30") == 30.0
    assert parse_fps("0/0") is None
    assert parse_fps(None) is None and parse_fps("abc") is None
