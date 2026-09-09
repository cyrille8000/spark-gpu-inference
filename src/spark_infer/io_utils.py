"""Entrées/sorties : téléchargement HTTP, décodage ffmpeg, encodage, livraison (PUT présigné ou base64)."""
from __future__ import annotations

import base64
import hashlib
import logging
import os
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

log = logging.getLogger("spark.io")

MAX_DOWNLOAD_BYTES = int(os.environ.get("SPARK_MAX_DOWNLOAD_MB", "2048")) * 1024 * 1024
INLINE_LIMIT_BYTES = int(os.environ.get("SPARK_INLINE_LIMIT_MB", "10")) * 1024 * 1024
HTTP_TIMEOUT = int(os.environ.get("SPARK_HTTP_TIMEOUT_S", "180"))

CONTENT_TYPES = {"wav": "audio/wav", "mp3": "audio/mpeg"}


class InputError(ValueError):
    """Entrée invalide : renvoyée au client, jamais relancée."""


def check_url(url: object, field: str) -> str:
    if not isinstance(url, str) or not url:
        raise InputError(f"{field} manquant")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise InputError(f"{field} doit être une URL http(s)")
    return url


def _short(url: str) -> str:
    p = urlparse(url)
    return f"{p.netloc}{p.path[:80]}"


def download(url: str, dest: Path, timeout: int = HTTP_TIMEOUT) -> int:
    """Télécharge `url` vers `dest` en flux. Renvoie la taille en octets."""
    size = 0
    with requests.get(url, stream=True, timeout=timeout) as r:
        if r.status_code >= 400:
            raise InputError(f"téléchargement refusé ({r.status_code}) : {_short(url)}")
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                size += len(chunk)
                if size > MAX_DOWNLOAD_BYTES:
                    raise InputError(
                        f"fichier trop volumineux (> {MAX_DOWNLOAD_BYTES >> 20} MB) : {_short(url)}")
                f.write(chunk)
    if size == 0:
        raise InputError(f"fichier vide : {_short(url)}")
    log.info("téléchargé %s (%.1f MB)", _short(url), size / 1e6)
    return size


def run_ffmpeg(args: list[str], timeout: int = 1800) -> None:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if res.returncode != 0:
        raise InputError(f"ffmpeg a échoué : {res.stderr.strip()[-400:]}")


def ffprobe_channels(path: Path) -> int:
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=channels",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, timeout=60,
    )
    try:
        return int(res.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return 0


def _channel_filter(src_channels: int, channels: int | None) -> str | None:
    """Matrice EXPLICITE à gain 1 pour changer le nombre de canaux.

    `-ac` laisse libswresample choisir sa matrice : mono → stéréo atténue chaque canal de 0,707
    et stéréo → mono amplifie de 1,414. Selon le chemin les deux se compensent… ou pas : un WAV
    mono de la plateforme passé au modèle (stéréo) puis rendu mono ressortait 3 dB trop bas.
    Ici : dupliquer tel quel vers la stéréo, moyenner (0,5 / 0,5) vers le mono. Mesuré à 1,000."""
    if not channels or src_channels == channels:
        return None
    if channels == 2 and src_channels == 1:
        return "pan=stereo|c0=c0|c1=c0"
    if channels == 1 and src_channels == 2:
        return "pan=mono|c0=0.5*c0+0.5*c1"
    return None  # multicanal exotique : laisser ffmpeg réduire (-ac)


def _audio_filters(src_channels: int, channels: int | None, sr: int | None) -> list[str]:
    """Arguments ffmpeg communs : matrice de canaux explicite + rééchantillonnage soxr."""
    chain: list[str] = []
    args: list[str] = []
    pan = _channel_filter(src_channels, channels)
    if pan:
        chain.append(pan)
    elif channels and src_channels != channels:
        args += ["-ac", str(channels)]
    if sr:
        chain.append("aresample=resampler=soxr")
        args += ["-ar", str(sr)]
    if chain:
        args = ["-af", ",".join(chain), *args]
    return args


def decode_to_wav(src: Path, dst: Path, sr: int | None = None, channels: int | None = None) -> Path:
    """Décode n'importe quel conteneur/codec en WAV float32 (piste audio seule), à gain constant."""
    args = ["-i", str(src), "-vn", "-map_metadata", "-1"]
    args += _audio_filters(ffprobe_channels(src), channels, sr)
    args += ["-acodec", "pcm_f32le", "-f", "wav", str(dst)]
    run_ffmpeg(args)
    return dst


def ffprobe_duration(path: Path) -> float:
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, timeout=60,
    )
    try:
        return float(res.stdout.strip())
    except ValueError:
        return 0.0


def encode_output(wav_path: Path, out_path: Path, fmt: str, mono: bool, sr: int | None = None,
                  mp3_quality: int = 2) -> Path:
    """WAV float32 → WAV 16 bits ou MP3 (libmp3lame VBR, -q:a 2 comme la plateforme).
    `sr` : rééchantillonnage de sortie (soxr) ; None = fréquence d'entrée. Mixage mono à gain 1."""
    args = ["-i", str(wav_path)]
    args += _audio_filters(ffprobe_channels(wav_path), 1 if mono else None, sr)
    if fmt == "mp3":
        args += ["-codec:a", "libmp3lame", "-q:a", str(mp3_quality), str(out_path)]
    elif fmt == "wav":
        args += ["-acodec", "pcm_s16le", str(out_path)]
    else:
        raise InputError(f"output_format inconnu : {fmt}")
    run_ffmpeg(args)
    return out_path


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def upload_put(url: str, path: Path, content_type: str, retries: int = 3) -> None:
    """PUT du fichier sur une URL présignée (R2/S3). 3 tentatives, backoff 2 s / 4 s / 6 s."""
    size = path.stat().st_size
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with open(path, "rb") as f:
                r = requests.put(url, data=f, timeout=HTTP_TIMEOUT,
                                 headers={"Content-Type": content_type, "Content-Length": str(size)})
            if r.status_code < 300:
                log.info("upload OK (%d, %.1f MB)", r.status_code, size / 1e6)
                return
            last = RuntimeError(f"upload refusé ({r.status_code}) : {r.text[:200]}")
            if 400 <= r.status_code < 500 and r.status_code != 429:
                break  # une présignée expirée ne se rejoue pas
        except requests.RequestException as e:
            last = e
        time.sleep(2 * attempt)
    raise RuntimeError(f"upload échoué après {retries} tentatives : {last}")


def deliver(path: Path, output_url: str | None, fmt: str) -> dict:
    """Livre le résultat : PUT présigné si `output_url`, sinon base64 (petits fichiers seulement)."""
    size = path.stat().st_size
    out: dict = {"format": fmt, "bytes": size, "sha256": sha256_of(path), "uploaded": False}
    if output_url:
        upload_put(output_url, path, CONTENT_TYPES[fmt])
        out["uploaded"] = True
        return out
    if size > INLINE_LIMIT_BYTES:
        raise InputError(
            f"résultat de {size >> 20} MB : fournir `output_url` (PUT présigné), "
            f"le retour inline est limité à {INLINE_LIMIT_BYTES >> 20} MB")
    out["audio_base64"] = base64.b64encode(path.read_bytes()).decode("ascii")
    return out


def send_callback(url: str, token: str | None, payload: dict, retries: int = 3) -> bool:
    """POST JSON du résultat final sur le rappel du client. N'échoue jamais le job : renvoie False si KO.
    Le webhook natif RunPod (champ `webhook` de /run) reste le filet de sécurité si le worker meurt."""
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    last: str = ""
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=30)
            if r.status_code < 300:
                log.info("rappel OK (%d) → %s", r.status_code, _short(url))
                return True
            last = f"{r.status_code} {r.text[:200]}"
            if 400 <= r.status_code < 500 and r.status_code != 429:
                break
        except requests.RequestException as e:
            last = str(e)
        time.sleep(2 * attempt)
    log.error("rappel échoué après %d tentatives → %s : %s", retries, _short(url), last)
    return False
