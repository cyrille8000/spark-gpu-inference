import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spark_infer.webhooks import Heartbeat, JobWebhooks  # noqa: E402


def make_hooks(sent):
    def send(url, token, payload, retries):
        sent.append((url, token, payload, retries))
        return True
    return JobWebhooks("https://api.test/gpu-event?wt=x", "tok", "job-1", "instrumental",
                       {"project": "p", "portion": 3, "run": "sep_abc", "attempt": 1}, "modal", send=send)


def test_events_are_ordered_and_carry_meta():
    sent = []
    h = make_hooks(sent)
    assert h.started(gpu_name="NVIDIA L4", device="cuda", container_first_job=True,
                     container_uptime_s=18.2, models_loaded=[])
    assert h.heartbeat(elapsed_s=30.0, progress={"percent": 10, "message": "séparation"})
    assert h.finished({"status": "completed", "elapsed_s": 61.5, "container_s": 80.1,
                       "audio_base64": "AAAA", "bytes": 123})
    events = [p["event"] for _, _, p, _ in sent]
    assert events == ["started", "heartbeat", "finished"]
    assert [p["seq"] for _, _, p, _ in sent] == [1, 2, 3]
    for url, token, p, _ in sent:
        assert url == "https://api.test/gpu-event?wt=x" and token == "tok"
        assert p["meta"]["run"] == "sep_abc" and p["job_id"] == "job-1" and p["provider"] == "modal"
        assert p["sent_at"].endswith("Z")
    started, hb, fin = (p for _, _, p, _ in sent)
    assert started["gpu_name"] == "NVIDIA L4" and started["container_first_job"] is True
    assert hb["progress"]["percent"] == 10 and hb["elapsed_s"] == 30.0
    # Le résultat entier, sans le base64 ; la tâche est reportée ; 3 essais pour started/finished, 1 pour heartbeat.
    assert fin["status"] == "completed" and fin["container_s"] == 80.1 and "audio_base64" not in fin
    assert fin["task"] == "instrumental"
    assert [r for _, _, _, r in sent] == [3, 1, 3]


def test_heartbeat_thread_beats_then_stops():
    sent = []
    h = make_hooks(sent)
    hb = Heartbeat(h, interval_s=1.0)
    hb.interval_s = 0.02  # sous le plancher d'1 s pour le test seulement
    hb.progress = {"percent": 5, "message": "décodé"}
    hb.start()
    time.sleep(0.15)
    hb.stop()
    n = hb.sent
    assert n >= 2
    assert all(p["event"] == "heartbeat" and p["progress"]["percent"] == 5 for _, _, p, _ in sent)
    time.sleep(0.1)
    assert hb.sent == n  # plus rien après stop()


def test_heartbeat_without_hooks_is_a_noop():
    hb = Heartbeat(None, 30)
    hb.start()
    hb.stop()
    assert hb.sent == 0
