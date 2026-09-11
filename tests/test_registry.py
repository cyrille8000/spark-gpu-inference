import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spark_infer import registry  # noqa: E402


def test_others_loaded_excludes_the_job_model(monkeypatch):
    monkeypatch.setattr(registry, "_models", {"chatterbox_vc": object()})
    assert registry.others_loaded("chatterbox_vc") == []          # seul en mémoire : un OOM ne se rejoue pas
    monkeypatch.setattr(registry, "_models", {"chatterbox_vc": object(), "bs_roformer_leap_xe": object()})
    assert registry.others_loaded("chatterbox_vc") == ["bs_roformer_leap_xe"]  # un autre occupe la carte : libérer puis rejouer
    monkeypatch.setattr(registry, "_models", {})
    assert registry.others_loaded(None) == []
