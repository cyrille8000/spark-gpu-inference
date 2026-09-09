import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spark_infer.container_clock import ContainerClock  # noqa: E402


def test_first_window_counts_from_process_start():
    c = ContainerClock(now=100.0)
    # boot 18 s + job 12,5 s avant le premier rapport
    assert c.window(now=130.5) == (30.5, True)
    # puis 20 s d'attente + 12 s de job : la fenêtre repart du rapport précédent
    assert c.window(now=162.5) == (32.0, False)
    # la somme des fenêtres = la vie du conteneur jusqu'ici
    assert 30.5 + 32.0 == 162.5 - 100.0


def test_window_never_negative():
    c = ContainerClock(now=10.0)
    assert c.window(now=5.0) == (0.0, True)
