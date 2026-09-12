"""Banc : jusqu'ou un pod peut-il faire tourner des jobs EN MEME TEMPS ?

Le banc CHERCHE le plafond au lieu qu'on le lui donne. On lui dit quelle carte on a
louee et quelle tache mesurer ; il monte jusqu'a ce qu'un job casse, resserre, et
rapporte le dernier palier qui a tenu ENTIEREMENT, avec la memoire qu'il a coute.
Aucun palier n'est choisi a la main : c'est ce qui distingue une mesure d'une
estimation (rappel du proprietaire, 2026-09-12).

    python scripts/bench_concurrence.py --url http://<ip>:<port> --token "$JETON" \\
        --job banc_vc.json

`--vagues 1,2,4` garde l'ancien mode a paliers imposes, utile pour comparer deux
cartes sur les memes largeurs.

Les jobs passent par `/submit` puis `/result` : aucune connexion n'est tenue ouverte
pendant le calcul. Le mode synchrone `/run` gardait la connexion pendant tout le job,
et au-dela de ~200 s quelque chose la coupait (`ConnectionResetError`) — des pertes
qui n'avaient rien a voir avec la carte mais qui faussaient la mesure.

`--job` est un fichier JSON contenant exactement ce qu'on enverrait a Modal ou a
RunPod. Les URL doivent etre lisibles depuis le pod, et le fichier doit etre celui de
la PRODUCTION (WAV, duree reelle) : un MP3 leger masque le telechargement, donc le
temps ou le GPU dort, donc tout le gain du parallelisme.
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


def un_job(url: str, token: str, charge: dict, timeout: float, sonde_s: float = 2.0) -> dict:
    """Soumet, puis interroge jusqu'au resultat. Le temps rapporte est arrondi a la
    periode de sonde pres — sans importance sur des jobs de dizaines de secondes."""
    t0 = time.monotonic()
    accuse = appeler(url, token, "/submit", charge, 60.0)
    job_id = accuse.get("job_id")
    if not job_id:
        return {"s": round(time.monotonic() - t0, 1), "ok": False,
                "erreur": accuse.get("error") or f"soumission refusee : {accuse}"}
    out: dict | None = None
    while time.monotonic() - t0 < timeout:
        time.sleep(sonde_s)
        rep = appeler(url, token, f"/result?job_id={job_id}", None, 60.0)
        if rep.get("pret"):
            out = rep.get("resultat") or {}
            break
    if out is None:
        return {"s": round(time.monotonic() - t0, 1), "ok": False, "erreur": "delai depasse"}
    return {
        "s": round(time.monotonic() - t0, 1),
        "ok": out.get("status") == "completed",
        "erreur": out.get("error"),
        "inference_s": (out.get("timings") or {}).get("inference"),
        "model_load_s": (out.get("timings") or {}).get("model_load"),
        "container_s": out.get("container_s"),
        "gpu_mem": out.get("gpu_mem"),
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


def dire(r: dict) -> None:
    print(f"vague {r['jobs']:>3} : {r['reussis']}/{r['jobs']} reussis - mur {r['mur_s']:>6} s - "
          f"par job {r['par_job_s']} s - {r['mur_s'] / max(1, r['reussis']):.1f} s/job en file - "
          f"debit {r['debit_par_min']}/min - pic carte {r['pic_carte_gb']} Go"
          + (f" - ECHECS {r['erreurs']}" if r["erreurs"] else ""))
    sys.stdout.flush()


def chercher_plafond(url: str, token: str, charge: dict, vram_gb: float, timeout: float,
                     cible: float, maximum: int) -> list[dict]:
    """Monte jusqu'a l'echec, puis resserre. Rapporte toutes les vagues tentees.

    La memoire par job n'est PAS constante : mesure le 2026-09-12 sur A100 80 Go, une
    separation coute 4,6 Go quand elles sont deux et 3,8 Go quand elles sont vingt
    (l'allocateur reutilise ses blocs). Toute formule ecrite d'avance se trompe donc,
    dans un sens ou dans l'autre. On s'en sert seulement pour VISER le palier suivant ;
    ce qui est rapporte est toujours ce qui a reellement tourne.
    """
    lignes: list[dict] = []
    dernier_bon = 0
    n = 1
    while n <= maximum:
        r = vague(url, token, charge, n, timeout)
        lignes.append(r)
        dire(r)
        if r["reussis"] < r["jobs"]:
            break
        dernier_bon = n
        pic = r["pic_carte_gb"] or 0
        if pic <= 0:
            n = min(maximum, n * 2)
            continue
        par_job = pic / n
        reste = vram_gb * cible - pic
        vise = n + max(1, int(reste // par_job)) if par_job > 0 else n * 2
        if vise <= n:
            print(f"-> {pic} Go sur {vram_gb:.1f} : la cible de {cible:.0%} est atteinte a {n} jobs")
            break
        # Jamais plus de x4 d'un coup : un saut trop large fait rater le vrai plafond
        # de peu, et une vague large coute cher en temps de pod.
        prochain = min(maximum, min(vise, n * 4))
        if prochain <= n:
            break
        n = prochain

    # Le plafond est entre le dernier palier tenu et le premier qui a casse.
    if dernier_bon and lignes[-1]["reussis"] < lignes[-1]["jobs"]:
        casse = lignes[-1]["jobs"]
        while casse - dernier_bon > 1:
            milieu = (dernier_bon + casse) // 2
            r = vague(url, token, charge, milieu, timeout)
            lignes.append(r)
            dire(r)
            if r["reussis"] == r["jobs"]:
                dernier_bon = milieu
            else:
                casse = milieu

    print(f"\nPLAFOND MESURE : {dernier_bon} job(s) en meme temps")
    bons = [l for l in lignes if l["jobs"] == dernier_bon and l["reussis"] == l["jobs"]]
    if bons and bons[-1]["pic_carte_gb"] and vram_gb > 0:
        pic = bons[-1]["pic_carte_gb"]
        print(f"  memoire au plafond : {pic} Go sur {vram_gb:.1f} ({pic / vram_gb:.0%} de la carte), "
              f"{pic / dernier_bon:.2f} Go par job")
    return lignes


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", required=True, help="racine du worker, ex. http://1.2.3.4:8000")
    p.add_argument("--token", required=True)
    p.add_argument("--job", required=True, help="fichier JSON de la charge utile")
    p.add_argument("--vagues", default="", help="paliers imposes, ex. 1,2,4 (defaut : chercher le plafond)")
    p.add_argument("--cible", type=float, default=0.92, help="part de la carte visee avant d'arreter de monter")
    p.add_argument("--maximum", type=int, default=64, help="garde-fou : jamais plus de tant de jobs")
    p.add_argument("--timeout", type=float, default=3600.0)
    p.add_argument("--sans-chauffe", action="store_true",
                   help="ne pas jeter un premier job d'echauffement (mesure alors biaisee)")
    a = p.parse_args()

    charge = json.loads(open(a.job, encoding="utf-8").read())
    etat = appeler(a.url, a.token, "/status", None, 30.0)
    if not etat.get("ok"):
        print(f"worker injoignable : {etat}", file=sys.stderr)
        return 1
    sonde = appeler(a.url, a.token, "/result?job_id=inexistant", None, 30.0)
    if sonde.get("error") == "not found":
        print("ce worker n'a pas la route /result : image trop ancienne. La reconstruire "
              "avant de mesurer, sinon la connexion reste ouverte pendant le job et casse.",
              file=sys.stderr)
        return 1
    mach = etat.get("machine") or {}
    capacite = int(etat.get("jobs_per_gpu") or 1)
    vram = float(etat.get("vram_total_gb") or 0)
    print(f"carte {etat.get('gpu_name')} - {vram:.1f} Go - {mach.get('gpus')} carte(s) - "
          f"{mach.get('cpu')} coeurs - {mach.get('ram_gb')} Go de RAM - "
          f"places {capacite} - {etat.get('cuda', {}).get('torch')}")
    print(f"tache << {charge.get('task')} >>\n")
    sys.stdout.flush()

    # ECHAUFFEMENT, jete. Le tout premier job charge le modele sur la carte (des
    # dizaines de secondes) : sans ca la vague de 1 le paie et les suivantes non, ce
    # qui gonfle artificiellement le gain du parallelisme (x2,44 annonce pour x1,2 reel).
    if not a.sans_chauffe:
        r = un_job(a.url, a.token, charge, a.timeout)
        print(f"echauffement (jete) : {r['s']} s, dont {r.get('model_load_s')} s de chargement du modele\n")
        sys.stdout.flush()

    if a.vagues.strip():
        lignes = []
        for n in [int(x) for x in a.vagues.split(",") if x.strip()]:
            if n > capacite:
                print(f"vague {n} : le worker n'accepte que {capacite} job(s) — les jobs vont s'attendre")
            r = vague(a.url, a.token, charge, n, a.timeout)
            lignes.append(r)
            dire(r)
    else:
        lignes = chercher_plafond(a.url, a.token, charge, vram, a.timeout, a.cible,
                                  min(a.maximum, capacite))

    ref = next((l for l in lignes if l["jobs"] == 1 and l["reussis"]), None)
    if ref and ref["debit_par_min"]:
        print("\ngain par rapport a un job seul :")
        for l in lignes:
            if l["debit_par_min"]:
                print(f"  {l['jobs']:>3} en parallele -> x{round(l['debit_par_min'] / ref['debit_par_min'], 2)} de debit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
