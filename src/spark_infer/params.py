"""Validation des entrées de job (pure : ni torch, ni réseau — testable partout)."""
from __future__ import annotations

from dataclasses import dataclass, field

from .io_utils import InputError, check_url

TASKS = ("instrumental", "vc")
MAX_REFS = 8

# TOUT RÉSULTAT EST UN WAV MONO 24 kHz 16 bits (décision propriétaire, 2026-09-09) : ce n'est pas un paramètre.
OUTPUT_FORMAT = "wav"
OUTPUT_SR = 24_000
OUTPUT_MONO = True


def _bool(inp: dict, key: str, default: bool) -> bool:
    v = inp.get(key, default)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    if isinstance(v, str) and v.lower() in ("true", "false", "1", "0"):
        return v.lower() in ("true", "1")
    raise InputError(f"{key} doit être un booléen")


def _int(inp: dict, key: str, lo: int, hi: int, default: int | None) -> int | None:
    v = inp.get(key, default)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        raise InputError(f"{key} doit être un entier")
    try:
        v = int(v)
    except ValueError:
        raise InputError(f"{key} doit être un entier") from None
    if not lo <= v <= hi:
        raise InputError(f"{key} doit être entre {lo} et {hi}")
    return v


def _float(inp: dict, key: str, lo: float, hi: float, default: float | None) -> float | None:
    v = inp.get(key, default)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        raise InputError(f"{key} doit être un nombre")
    try:
        v = float(v)
    except ValueError:
        raise InputError(f"{key} doit être un nombre") from None
    if not lo <= v <= hi:
        raise InputError(f"{key} doit être entre {lo} et {hi}")
    return v


def _output_url(inp: dict) -> str | None:
    url = inp.get("output_url")
    return check_url(url, "output_url") if url else None


@dataclass
class Callback:
    url: str
    token: str | None


def parse_meta(inp: dict) -> dict | None:
    """`meta` : objet OPAQUE de l'appelant (projet, portion, tentative, compte…), renvoyé tel quel
    dans chaque rappel et dans le résultat. Tout sauf un objet est ignoré."""
    m = inp.get("meta")
    return dict(m) if isinstance(m, dict) else None


def parse_heartbeat_s(inp: dict, default: int = 30) -> int:
    """Cadence des battements (s), 5..300. Hors bornes ou illisible : le défaut."""
    try:
        v = int(inp.get("heartbeat_s", default))
    except (TypeError, ValueError):
        return default
    return v if 5 <= v <= 300 else default


def parse_callback(inp: dict) -> Callback | None:
    """Rappel de fin de job : POST JSON sur `callback_url`, `Authorization: Bearer <callback_token>` si fourni.
    Lu avec tolérance : un rappel malformé n'empêche pas le job, il est juste ignoré (et signalé)."""
    url = inp.get("callback_url")
    if not url:
        return None
    token = inp.get("callback_token")
    return Callback(check_url(url, "callback_url"), str(token) if token else None)


@dataclass
class VcParams:
    steps: int = 25            # pas du flow matching (n_cfm_timesteps) — choix propriétaire 2026-09-09
    temp: float = 0.8          # temperature du décodeur
    cfg: float | None = None   # inference_cfg_rate (None = valeur du checkpoint)
    ref_len: float = 10.0      # longueur du prompt de référence (s)
    preproc: bool = True       # passe-haut + sonie -23 LUFS sur la source
    seed: int = 1000           # graine du tirage (résultat reproductible)
    max_tail_passes: int = 5   # complétions de queue max
    tail_tolerance_s: float = 0.15


@dataclass
class InstrumentalRequest:
    audio_url: str
    output_url: str | None


@dataclass
class VcRequest:
    source_url: str
    ref_urls: list[str]
    prompt_url: str | None
    output_url: str | None
    params: VcParams = field(default_factory=VcParams)


def parse_instrumental(inp: dict) -> InstrumentalRequest:
    return InstrumentalRequest(
        audio_url=check_url(inp.get("audio_url"), "audio_url"),
        output_url=_output_url(inp),
    )


def parse_vc(inp: dict) -> VcRequest:
    refs = inp.get("ref_urls")
    if refs is None and inp.get("ref_url"):
        refs = [inp["ref_url"]]
    if not isinstance(refs, list) or not refs:
        raise InputError("ref_urls doit être une liste d'au moins une URL (clips de la voix cible)")
    if len(refs) > MAX_REFS:
        raise InputError(f"ref_urls : {MAX_REFS} clips maximum")
    refs = [check_url(u, f"ref_urls[{i}]") for i, u in enumerate(refs)]
    prompt = inp.get("prompt_url")
    params = VcParams(
        steps=_int(inp, "steps", 1, 64, 25),  # type: ignore[arg-type]
        temp=_float(inp, "temp", 0.0, 2.0, 0.8),  # type: ignore[arg-type]
        cfg=_float(inp, "cfg", 0.0, 3.0, None),
        ref_len=_float(inp, "ref_len", 1.0, 30.0, 10.0),  # type: ignore[arg-type]
        preproc=_bool(inp, "preproc", True),
        seed=_int(inp, "seed", 0, 2**31 - 1, 1000),  # type: ignore[arg-type]
    )
    return VcRequest(
        source_url=check_url(inp.get("source_url"), "source_url"),
        ref_urls=refs,
        prompt_url=check_url(prompt, "prompt_url") if prompt else None,
        output_url=_output_url(inp),
        params=params,
    )


def parse_task(inp: dict) -> str:
    task = inp.get("task")
    if task not in TASKS:
        raise InputError(f"task doit être parmi {TASKS}")
    return task
