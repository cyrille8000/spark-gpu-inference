import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spark_infer.io_utils import InputError  # noqa: E402
from spark_infer.params import (  # noqa: E402
    MAX_WINDOW_S, parse_callback, parse_instrumental, parse_speaking_faces, parse_task, parse_vc,
)

URL = "https://example.com/a.wav"


def test_task_required():
    assert parse_task({"task": "instrumental"}) == "instrumental"
    assert parse_task({"task": "vc"}) == "vc"
    assert parse_task({"task": "speaking_faces"}) == "speaking_faces"
    with pytest.raises(InputError):
        parse_task({"task": "demucs"})
    with pytest.raises(InputError):
        parse_task({})


def test_instrumental_defaults():
    r = parse_instrumental({"audio_url": URL})
    assert r.output_url is None


def test_instrumental_validation():
    with pytest.raises(InputError):
        parse_instrumental({})
    with pytest.raises(InputError):
        parse_instrumental({"audio_url": "ftp://x/y"})
    # format / mono / output_sr ne sont plus des paramètres : ignorés, la sortie est toujours un WAV mono 24 kHz
    r = parse_instrumental({"audio_url": URL, "output_url": URL, "output_format": "mp3", "mono": False, "output_sr": 44100})
    assert r.output_url == URL
    assert not hasattr(r, "output_format") and not hasattr(r, "mono") and not hasattr(r, "output_sr")


def test_callback():
    assert parse_callback({}) is None
    cb = parse_callback({"callback_url": URL, "callback_token": "abc"})
    assert cb.url == URL and cb.token == "abc"
    assert parse_callback({"callback_url": URL}).token is None
    with pytest.raises(InputError):
        parse_callback({"callback_url": "not-a-url"})


def test_vc_defaults_and_ref_url_alias():
    r = parse_vc({"source_url": URL, "ref_url": URL})
    assert r.ref_urls == [URL] and r.prompt_url is None
    p = r.params
    assert (p.steps, p.temp, p.cfg, p.ref_len, p.preproc, p.seed) == (25, 0.8, None, 10.0, True, 1000)
    assert p.window_s == 60.0  # fenêtres de conversion (2026-09-11)


def test_vc_validation():
    with pytest.raises(InputError):
        parse_vc({"source_url": URL})
    with pytest.raises(InputError):
        parse_vc({"source_url": URL, "ref_urls": []})
    with pytest.raises(InputError):
        parse_vc({"source_url": URL, "ref_urls": [URL] * 9})
    with pytest.raises(InputError):
        parse_vc({"source_url": URL, "ref_urls": [URL], "steps": "vingt"})
    r = parse_vc({"source_url": URL, "ref_urls": [URL, URL], "prompt_url": URL,
                  "steps": 30, "temp": 0.6, "cfg": 0.7, "ref_len": 8, "preproc": False})
    assert len(r.ref_urls) == 2 and r.prompt_url == URL
    assert r.params.steps == 30 and r.params.cfg == 0.7 and r.params.preproc is False
    assert parse_vc({"source_url": URL, "ref_urls": [URL], "window_s": 45}).params.window_s == 45.0
    with pytest.raises(InputError):
        parse_vc({"source_url": URL, "ref_urls": [URL], "window_s": 5})  # sous le plancher de 10 s


def test_vc_cuts_s():
    """Les frontières autorisées de la source : absentes → [], sinon triées, dédoublonnées, sans 0."""
    assert parse_vc({"source_url": URL, "ref_urls": [URL]}).cuts_s == []
    r = parse_vc({"source_url": URL, "ref_urls": [URL], "cuts_s": [7.25, 0, 3, 3.0, 61.5]})
    assert r.cuts_s == [3.0, 7.25, 61.5]
    for mauvais in ("3,7", {"a": 1}, [True], ["3"], [-1.0], [float("nan")], [float("inf")]):
        with pytest.raises(InputError):
            parse_vc({"source_url": URL, "ref_urls": [URL], "cuts_s": mauvais})
    with pytest.raises(InputError):
        parse_vc({"source_url": URL, "ref_urls": [URL], "cuts_s": [1.0] * 10_001})

def test_jobs_deduits_de_la_carte():
    """Le nombre de jobs suit la CARTE TROUVÉE et la TÂCHE, pas une constante posée au
    déploiement : Modal ne propose qu'un L4 aujourd'hui, RunPod donne ce qu'il a, Vast.ai
    tout. Coûts mesurés le 2026-09-12 (A100 80 Go, RTX PRO 4000 25 Go)."""
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "src"))
    from spark_infer.tasks import jobs_pour_vram

    # Carte de 24 Go : CALÉ SUR LA MESURE du L4 de Modal (2026-09-12), la seule où le
    # plafond a été cherché pour les deux tâches — 5 séparations tiennent (18,71 Go sur
    # 23,66), la 6e déborde ; 8 conversions vocales tiennent (17,29 Go) sans être poussées.
    assert jobs_pour_vram("bs_roformer_leap_xe", 24.0) == 5
    assert jobs_pour_vram("chatterbox_vc", 24.0) == 9
    # Une carte « 22 Go » : la mémoire décide aussi (décision du 2026-09-15, proportionnel
    # à la taille de la carte — l'ancienne règle « une place sous 23 Go » est retirée).
    assert jobs_pour_vram("bs_roformer_leap_xe", 22.5) == 4
    # Le L4 tel que torch le rapporte (23,66 Go) reste sur la mesure : 5 séparations.
    assert jobs_pour_vram("bs_roformer_leap_xe", 23.66) == 5
    assert jobs_pour_vram("chatterbox_vc", 22.5) == 8
    # Grosse carte : la mémoire décide, PLUS AUCUN plafond arbitraire. Une carte de
    # 80 Go ne doit pas être bridée comme une de 24 (décision du propriétaire).
    assert jobs_pour_vram("bs_roformer_leap_xe", 80.0) == 17
    assert jobs_pour_vram("chatterbox_vc", 80.0) == 32
    # 12 Go : deux séparations tiennent (9,4 Go mesurés à deux) — et sous 16 Go la carte
    # est de toute façon refusée en amont sur Vast (SPARK_MIN_VRAM_GB).
    assert jobs_pour_vram("bs_roformer_leap_xe", 12.0) == 2
    # Carte trop petite pour deux : jamais moins d'un job, même si le calcul dit zéro.
    assert jobs_pour_vram("bs_roformer_leap_xe", 6.0) == 1
    # 8 Go : une seule conversion vocale — la part fixe (5 Go) mange la carte. Une carte
    # aussi petite est de toute façon écartée en amont par `SPARK_MIN_VRAM_GB`.
    assert jobs_pour_vram("chatterbox_vc", 8.0) == 1
    assert jobs_pour_vram("chatterbox_vc", 5.0) == 1
    # Tâche inconnue : traitée en gourmande (12 + 6 Go), un job de moins vaut mieux qu'un OOM.
    assert jobs_pour_vram("inconnue", 24.0) == 1
    assert jobs_pour_vram("inconnue", 80.0) == 9
    # Sans carte, ou carte illisible : un seul job.
    assert jobs_pour_vram("chatterbox_vc", None) == 1
    assert jobs_pour_vram("chatterbox_vc", 0) == 1
    # `SPARK_JOBS_PER_GPU` force la valeur (mesure, incident, carte exotique).
    assert jobs_pour_vram("bs_roformer_leap_xe", 22.5, force="8") == 8
    assert jobs_pour_vram("chatterbox_vc", 80.0, force="1") == 1
    assert jobs_pour_vram("chatterbox_vc", 80.0, force="pas un nombre") == 32
    # Plafond explicite (timeout de l'hébergeur, prudence).
    assert jobs_pour_vram("chatterbox_vc", 80.0, plafond=2) == 2


def test_deduction_opt_in_modal_et_runpod_inchanges(monkeypatch):
    """La déduction ne s'active QUE si `SPARK_JOBS_AUTO` est posé.

    Ce fichier est partagé par les trois hébergeurs. Modal et RunPod tournent à un
    job par conteneur et marchent bien ainsi ; ni l'un ni l'autre ne pose
    `SPARK_JOBS_PER_GPU`, donc une déduction active par défaut les ferait passer à
    quatre sans que personne l'ait demandé. Seule l'image Vast.ai pose le drapeau.
    """
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "src"))
    from spark_infer import tasks

    monkeypatch.setattr(tasks.registry, "vram_total_gb", lambda: 80.0)
    monkeypatch.delenv("SPARK_JOBS_PER_GPU", raising=False)

    # Modal / RunPod : aucun des deux drapeaux — un job, comme aujourd'hui.
    monkeypatch.delenv("SPARK_JOBS_AUTO", raising=False)
    assert tasks.jobs_per_gpu("chatterbox_vc") == 1
    assert tasks.jobs_per_gpu("bs_roformer_leap_xe") == 1

    # Image Vast.ai : le drapeau est posé, la carte décide.
    monkeypatch.setenv("SPARK_JOBS_AUTO", "1")
    assert tasks.jobs_per_gpu("chatterbox_vc") == 32
    assert tasks.jobs_per_gpu("bs_roformer_leap_xe") == 17

    # Une valeur explicite force, drapeau ou pas — et sans interroger la carte.
    monkeypatch.delenv("SPARK_JOBS_AUTO", raising=False)
    monkeypatch.setenv("SPARK_JOBS_PER_GPU", "3")
    assert tasks.jobs_per_gpu("bs_roformer_leap_xe") == 3


def test_speaking_faces_defaults():
    r = parse_speaking_faces({"video_url": URL})
    assert r.video_url == URL and r.audio_url is None and r.window is None
    assert r.margin == 2.0 and r.output_url is None


def test_speaking_faces_window():
    r = parse_speaking_faces({"video_url": URL, "audio_url": URL, "start": 300, "end": 600, "margin": 1.5,
                              "output_url": URL})
    assert r.window == (300.0, 600.0) and r.margin == 1.5 and r.audio_url == URL and r.output_url == URL
    assert parse_speaking_faces({"video_url": URL, "start": "0", "end": "5"}).window == (0.0, 5.0)
    # `start` et `end` vont ensemble : un seul des deux ferait analyser toute la fin de la vidéo.
    for mauvais in ({"start": 300}, {"end": 600}, {"start": -1, "end": 600}, {"start": 600, "end": 600},
                    {"start": 600, "end": 300}, {"start": "abc", "end": 600}, {"start": True, "end": 600},
                    {"start": 0, "end": MAX_WINDOW_S + 1}, {"margin": -1}, {"margin": 31}):
        with pytest.raises(InputError):
            parse_speaking_faces({"video_url": URL, **mauvais})
    with pytest.raises(InputError):
        parse_speaking_faces({})
    with pytest.raises(InputError):
        parse_speaking_faces({"video_url": URL, "audio_url": "ftp://x/y"})
