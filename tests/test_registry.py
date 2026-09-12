import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spark_infer import registry  # noqa: E402


def test_sondes_memoire_ne_plantent_jamais():
    """Les sondes rendent None sans CUDA, sinon deux nombres positifs — jamais une
    exception : un job ne doit pas échouer parce qu'on a voulu le mesurer."""
    registry.reset_peak_memory()   # ne doit jamais lever, GPU ou pas
    pic = registry.peak_memory_gb()
    assert pic is None or (set(pic) == {"allocated_gb", "reserved_gb"}
                           and all(isinstance(v, float) and v >= 0 for v in pic.values()))
    total = registry.vram_total_gb()
    assert total is None or total > 0


def test_others_loaded_excludes_the_job_model(monkeypatch):
    monkeypatch.setattr(registry, "_models", {"chatterbox_vc": object()})
    assert registry.others_loaded("chatterbox_vc") == []          # seul en mémoire : un OOM ne se rejoue pas
    monkeypatch.setattr(registry, "_models", {"chatterbox_vc": object(), "bs_roformer_leap_xe": object()})
    assert registry.others_loaded("chatterbox_vc") == ["bs_roformer_leap_xe"]  # un autre occupe la carte : libérer puis rejouer
    monkeypatch.setattr(registry, "_models", {})
    assert registry.others_loaded(None) == []
