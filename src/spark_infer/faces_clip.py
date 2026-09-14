"""Découpe de la portion à analyser, avant LR-ASD : la ligne ffmpeg (pure, testable) et les sondes.

Pourquoi découper NOUS-MÊMES plutôt que laisser LR-ASD le faire : ses `--start` / `--duration`
placent le `-ss` APRÈS le `-i`, donc ffmpeg décode la vidéo depuis son début pour la jeter. Sur une
vidéo d'une heure analysée par portions en parallèle, chaque job repaierait la totalité.

Ici le `-ss` est AVANT le `-i`, sur CHAQUE entrée. Deux conséquences, mesurées sur l'ancienne image
(`spark-dubbing-lipsync`, 2026-08) :
  - c'est frame-exact dès lors qu'on ré-encode derrière (une coupe à 12 s rend des images identiques
    une à une à la coupe de référence, sans forcer de keyframes dans la source) ;
  - ffmpeg ne tire que ce qu'il lui faut quand l'entrée est une URL qui accepte les requêtes Range
    (une URL présignée R2, la passerelle média) : le job ne télécharge pas l'heure entière.

La cadence est imposée par le FILTRE `fps=25`, jamais par `-r 25` : mesuré sur une source 30 i/s,
`-r` sort deux images de trop et retarde tout de 52 ms en moyenne, le filtre sort le compte juste à
±13 ms sans biais (il choisit vraiment l'image source la plus proche). 25 i/s est la cadence de
LR-ASD et de ses poids ; `faces_segments` convertit ensuite les images en secondes avec la cadence
MESURÉE sur le fichier produit, pas supposée.

L'image et le son sont deux fichiers dans la plateforme (`video_final.mp4` sans piste audio,
`audio_stream.m4a` à côté) : `audio_url` reçoit la MÊME fenêtre que `video_url`, sinon la
corrélation lèvres/son serait décalée. Sans son, LR-ASD ne détecte rien ET NE S'EN PLAINT PAS — il
rendrait une liste vide qu'on prendrait pour « personne ne parle ». Un extrait sans piste audio est
donc une erreur d'entrée, jamais un résultat.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

from .io_utils import InputError, _audio_filters, ffprobe_channels, run_ffmpeg

FPS_ANALYSE = 25          # cadence de LR-ASD : ses poids ont été entraînés dessus
SR_ANALYSE = 16_000       # MFCC à 16 kHz mono, comme dans Columbia_test.py
DEFAULT_MARGIN_S = 2.0    # contexte analysé puis JETÉ de chaque côté de la portion (cf. clip_to_window)

# Échelle de détection : LR-ASD réduit l'image AVANT d'y chercher des visages (`facedetScale`,
# 0,25 par défaut, calibré pour ~1280 px de large → ~320 px de recherche). En 1080p le défaut
# cherche sur 480 px, mieux que son étalon : on n'y touche pas. Mais la source d'un projet suit
# l'échelle 1080p → 720p → 360p : sur 640 px de large, 0,25 chercherait sur 160 px et perdrait
# les visages un peu éloignés, sans erreur ni log. On RELÈVE donc l'échelle sur les petites sources.
DEFAUT_LRASD = 0.25
LARGEUR_DETECTION_MIN = 320

_REMOTE = re.compile(r"^https?://", re.I)
# Options du protocole http SEULEMENT : passées sur un fichier local, ffmpeg refuse d'ouvrir l'entrée.
_RECONNECT = ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "30"]
# `-reconnect_delay_max` borne chaque tentative, pas leur total : sans plafond, une lecture HTTP figée
# retient le worker — et le facture — jusqu'à ce que l'hébergeur le tue.
TIMEOUT_DECOUPE_S = float(os.environ.get("SPARK_FACES_CUT_TIMEOUT_S", "1800"))
# CPU, toujours : les A100/H100 n'ont aucun encodeur matériel et un MIG Blackwell non plus. À
# `ultrafast`, 1080p s'encode à plusieurs centaines d'images par seconde sur quelques cœurs.
ENCODEUR_VIDEO = ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "18", "-pix_fmt", "yuv420p"]


def clip_bounds(window: tuple[float, float] | None, margin: float) -> tuple[float, float | None]:
    """Fenêtre ÉLARGIE à donner à ffmpeg. `clip_end` vaut None sans fenêtre (toute la vidéo).
    `clip_start` est aussi le décalage à ré-appliquer aux résultats : LR-ASD renumérote les
    images à partir de 0 dans l'extrait."""
    if window is None:
        return 0.0, None
    start, end = window
    return max(0.0, start - margin), end + margin


def facedet_scale(width: int | None) -> float:
    """Échelle de détection pour une image de `width` px : relevée sur les petites sources,
    jamais abaissée sous le défaut, jamais au-dessus de 1 (agrandir n'ajoute aucune information)."""
    if not width or width <= 0:
        return DEFAUT_LRASD
    return round(max(DEFAUT_LRASD, min(1.0, LARGEUR_DETECTION_MIN / width)), 3)


def parse_fps(texte: str | None) -> float | None:
    """`r_frame_rate` de ffprobe → images par seconde. C'est une FRACTION exacte ("24000/1001"),
    convertie ici sans arrondi intermédiaire. None sur "0/0" ou illisible."""
    num, _, den = (texte or "").partition("/")
    try:
        fps = float(num) / float(den or 1)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return fps if fps > 0 else None


def _entree(url: str, clip_start: float, clip_end: float | None) -> list[str]:
    args = list(_RECONNECT) if _REMOTE.match(url) else []
    # `-ss` / `-to` AVANT le `-i` : seek en ENTRÉE. `-to` y est absolu dans la source (mesuré :
    # `-ss 12 -to 17` rend bien 5 s), pas relatif au point de seek.
    args += ["-ss", f"{clip_start:.3f}"]
    if clip_end is not None:
        args += ["-to", f"{clip_end:.3f}"]
    return args + ["-i", url]


def build_cut_command(video_url: str, audio_url: str | None, clip_start: float, clip_end: float | None,
                      video_out: str | Path, audio_out: str | Path) -> list[str]:
    """La ligne ffmpeg de la découpe : UNE commande, deux sorties.

    `video_out` : l'image seule, 25 i/s (filtre `fps`), H.264 rapide, sans son.
    `audio_out` : le son seul, WAV 16 bits, canaux et fréquence d'ORIGINE — la conversion en
    16 kHz mono se fait ensuite avec une matrice explicite (`to_analysis_wav`), jamais par `-ac`.
    Le son vient de `audio_url` s'il est donné et distinct, sinon de la piste audio de `video_url`."""
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    cmd += _entree(video_url, clip_start, clip_end)
    audio_in = 0
    if audio_url and audio_url != video_url:
        cmd += _entree(audio_url, clip_start, clip_end)
        audio_in = 1
    cmd += ["-map", "0:v:0", "-vf", f"fps={FPS_ANALYSE}", *ENCODEUR_VIDEO, "-an", str(video_out)]
    cmd += ["-map", f"{audio_in}:a:0", "-vn", "-c:a", "pcm_s16le", str(audio_out)]
    return cmd


def probe_media(path: str | Path) -> dict | None:
    """Ce que ffprobe voit VRAIMENT dans le fichier produit : `{width, height, fps}`, ou None
    sans flux vidéo exploitable.

    None n'est pas qu'un repli : c'est la signature d'une fenêtre située APRÈS la fin du média.
    ffmpeg sort alors en SUCCÈS avec un conteneur vide (mesuré : 261 octets, code 0)."""
    try:
        res = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height,r_frame_rate",
             "-of", "json", str(path)],
            capture_output=True, text=True, timeout=60,
        )
        flux = json.loads(res.stdout or "{}").get("streams", [])
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    video = next((s for s in flux if s.get("codec_type") == "video"), None)
    if not video:
        return None
    fps = parse_fps(video.get("r_frame_rate"))
    if fps is None:
        return None
    return {"width": int(video.get("width") or 0), "height": int(video.get("height") or 0), "fps": fps}


def cut(video_url: str, audio_url: str | None, clip_start: float, clip_end: float | None,
        video_out: Path, audio_out: Path) -> None:
    """Exécute la découpe. Entrée impossible → `InputError` (jamais rejouée) ; réseau ou ffmpeg
    en panne → `RuntimeError` (rejouable)."""
    cmd = build_cut_command(video_url, audio_url, clip_start, clip_end, video_out, audio_out)
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_DECOUPE_S)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"découpe abandonnée après {TIMEOUT_DECOUPE_S:.0f} s — lecture de la source figée ?") from None
    if res.returncode != 0:
        err = (res.stderr or "").strip()
        if "matches no streams" in err and "a:0" in err:
            raise InputError("l'extrait n'a aucune piste audio : LR-ASD ne peut rien détecter sans le son. "
                             "Passer `audio_url` quand l'image et le son sont deux fichiers séparés.")
        if "Invalid data found" in err or "matches no streams" in err:
            raise InputError(f"découpe ffmpeg refusée : {err[-400:]}")
        raise RuntimeError(f"découpe ffmpeg échouée : {err[-1500:]}")
    if not video_out.is_file() or video_out.stat().st_size == 0:
        raise InputError(f"découpe vide à partir de {clip_start:.3f} s : fenêtre hors de la vidéo ?")


def to_analysis_wav(src: Path, dst: Path) -> Path:
    """WAV quelconque → WAV 16 kHz mono 16 bits pour les MFCC, à gain 1 (matrice `pan` explicite
    de `io_utils`, jamais `-ac` seul). Le niveau n'influence que le coefficient d'énergie des MFCC,
    mais on ne laisse pas ffmpeg choisir un gain à notre place."""
    args = ["-i", str(src), "-vn", "-map_metadata", "-1"]
    args += _audio_filters(ffprobe_channels(src), 1, SR_ANALYSE)
    args += ["-acodec", "pcm_s16le", "-f", "wav", str(dst)]
    run_ffmpeg(args)
    return dst
