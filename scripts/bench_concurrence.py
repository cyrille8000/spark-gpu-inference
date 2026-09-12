"""Banc : combien de jobs un pod a-t-il intérêt à faire tourner EN MÊME TEMPS ?

La question n'a pas de réponse théorique. Deux jobs sur une carte déjà saturée en
calcul prennent chacun deux fois plus de temps, et le débit ne bouge pas ; le gain
ne vient que du temps où le GPU dort (chargement du modèle, téléchargement,
encodage, envoi). Ce banc le MESURE sur la carte visée, au lieu de le supposer.

Il envoie la même charge utile N fois au worker Vast.ai (`vast_worker.py`), une
première fois seule pour la référence, puis par vagues de plus en plus larges.
Pour chaque vague il rapporte le temps de chaque job, le temps de la vague, le
débit (jobs par minute) et la mémoire que la carte a réellement vue.

    python scripts/bench_concurrence.py \\
        --url http://<ip>:<port> --token "$SPARK_WORKER_TOKEN" \\
        --job banc_vc.json --vagues 1,2,4

`--job` est un fichier JSON contenant exactement ce qu'on enverrait à Modal ou à
RunPod : {"task": "vc", "source_url": "…", "ref_urls": ["…"]} par exemple. Les
URL doivent être lisibles depuis le pod (R2 présigné, ou un fichier public).

Avant chaque vague, le worker doit tourner avec SPARK_JOBS_PER_GPU au moins égal
à la largeur de la vague, sinon les jobs s'attendent au lieu de se croiser : le
banc le vérifie sur /status et le dit.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def appeler(url: str, token: str, route: str, corps: dict | None, timeout: float) -> dict:
    data = json.dumps(corps).encode("utf-8") if corps is not None else None
    req = urllib.request.Request(
        url.rstrip("/") + route, data=data, method="POST" if data is not None else "GET",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return {"status": "error", "error": f"HTTP {e.code}: {e.read()[:200]!r}"}
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}


def un_job(url: str, token: str, charge: dict, timeout: float) -> dict:
    t0 = time.monotonic()
    out = appeler(url, token, "/run", charge, timeout)
    return {
        "s": round(time.monotonic() - t0, 1),
        "ok": out.get("status") == "completed",
        "erreur": out.get("error"),
        "inference_s": (out.get("timings") or {}).get("inference"),
        "model_load_s": (out.get("timings") or {}).get("model_load"),
        "container_s": out.get("container_s"),
        "gpu_mem": out.get("gpu_mem"),
        "pools": out.get("pools"),
    }


def vague(url: str, token: str, charge: dict, n: int, timeout: float) -> dict:
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=n) as ex:
        res = list(ex.map(lambda _: un_job(url, token, charge, timeout), range(n)))
    mur = round(time.monotonic() - t0, 1)
    ok = [r for r in res if r["ok"]]
    pics = [r["gpu_mem"]["reserved_gb"] for r in ok if r.get("gpu_mem")]
    return {
        "jobs": n, "reussis": len(ok), "mur_s": mur,
        "par_job_s": round(statistics.mean([r["s"] for r in ok]), 1) if ok else None,
        "debit_par_min": round(len(ok) / (mur / 60), 2) if mur > 0 and ok else 0,
        "pic_carte_gb": max(pics) if pics else None,
        "erreurs": [r["erreur"] for r in res if not r["ok"]][:3],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", required=True, help="racine du worker, ex. http://1.2.3.4:8000")
    p.add_argument("--token", required=True)
    p.add_argument("--job", required=True, help="fichier JSON de la charge utile")
    p.add_argument("--vagues", default="1,2,4", help="largeurs à mesurer, ex. 1,2,4,8")
    p.add_argument("--timeout", type=float, default=1800.0)
    a = p.parse_args()

    charge = json.loads(open(a.job, encoding="utf-8").read())
    etat = appeler(a.url, a.token, "/status", None, 30.0)
    if not etat.get("ok"):
        print(f"worker injoignable : {etat}", file=sys.stderr)
        return 1
    capacite = int(etat.get("jobs_per_gpu") or 1)
    print(f"carte {etat.get('gpu_name')} · {etat.get('vram_total_gb')} Go · "
          f"SPARK_JOBS_PER_GPU={capacite} · {etat.get('cuda', {}).get('torch')}+cu{etat.get('cuda', {}).get('compile_pour')}")
    print(f"tâche « {charge.get('task')} »\n")

    lignes = []
    for n in [int(x) for x in a.vagues.split(",") if x.strip()]:
        if n > capacite:
            print(f"vague {n} : le worker n'accepte que {capacite} job(s) à la fois — "
                  f"redémarrer le pod avec SPARK_JOBS_PER_GPU={n}, sinon les jobs s'attendent")
        r = vague(a.url, a.token, charge, n, a.timeout)
        lignes.append(r)
        print(f"vague {r['jobs']:>2} : {r['reussis']}/{r['jobs']} réussis · mur {r['mur_s']:>6} s · "
              f"par job {r['par_job_s']} s · débit {r['debit_par_min']}/min · pic carte {r['pic_carte_gb']} Go"
              + (f" · erreurs {r['erreurs']}" if r["erreurs"] else ""))

    ref = next((l for l in lignes if l["jobs"] == 1 and l["reussis"]), None)
    if ref and ref["debit_par_min"]:
        print("\ngain par rapport à un job seul :")
        for l in lignes:
            if l["debit_par_min"]:
                print(f"  {l['jobs']:>2} en parallèle → ×{round(l['debit_par_min'] / ref['debit_par_min'], 2)} de débit")
        print("\nUn gain proche de ×1 veut dire que la carte était déjà saturée : le parallélisme\n"
              "n'apporte rien et coûte de la mémoire. Un gain proche du nombre de jobs veut dire\n"
              "que le GPU dormait entre les phases, et qu'un pod loué peut en accepter d'autant plus.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
