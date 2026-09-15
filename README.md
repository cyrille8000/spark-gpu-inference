# Spark GPU Inference — image RunPod Serverless

Une seule image, trois tâches choisies par le paramètre `task` du job :

| `task` | Ce que ça fait | Modèle embarqué |
|--------|----------------|-----------------|
| `instrumental` | Instrumental seul. **BS-Roformer Leap Xe** (unwa, juin 2026) : un seul checkpoint entraîné directement sur la cible instrumentale, 18,07 dB SDR instrumental sur le Multisong de MVSEP, au-dessus des ensembles internes du site. | `pcunwa/BS-Roformer-Leap` → `Xe/bs_leap_xe_inst.ckpt` (268 MB), chargé par [bs-roformer-infer](https://github.com/openmirlab/bs-roformer-infer) (MIT, épinglé sur le commit `b0f1386f` : la roue PyPI 0.1.5 ne connaît pas Leap) avec sha256 vérifié |
| `vc` | Conversion de timbre Chatterbox VC (S3Gen) : timbre moyenné sur 1..8 clips de référence, prompt phonétique optionnel, pas / temperature / CFG réglables, complétion de la queue collée bout à bout (sans fondu). Un seul tirage. | `ResembleAI/chatterbox` (`s3gen.safetensors`, `conds.pt`) |
| `speaking_faces` | **Visages qui parlent** : qui parle à l'image, et où. LR-ASD (Liao et al., IJCV 2025) en interne — S3FD trouve les visages, le réseau audio-visuel dit image par image si chacun parle. Sortie : `speech` (quand) + `faces` (où, boîtes en fractions de l'image). Analysable par portions, temps de la vidéo d'origine. **Code seulement au 2026-09-15** : ni build, ni déploiement, ni mesure. | LR-ASD commit `1b6dcd2d` (`finetuning_TalkSet.model`) + S3FD `sfd_face.pth`, sha256 vérifiés au build — [docs/TACHE_VISAGES_QUI_PARLENT.md](docs/TACHE_VISAGES_QUI_PARLENT.md) |

Tous les poids sont dans l'image (`/models`). À l'inférence, `HF_HUB_OFFLINE=1` et le checkpoint BS-Roformer est
résolu localement : **aucun téléchargement de modèle**. Les modèles restent résidents entre deux jobs d'un même worker.

Base `python:3.11-slim` + torch 2.7.1 cu128 (noyaux jusqu'à Blackwell sm_120 : Vast en loue beaucoup ; RunPod sert des RTX 4090 et Modal des L4, qui n'en ont pas besoin) : les roues
torch embarquent CUDA/cuDNN, seul le pilote de l'hôte est requis.

## Contrat d'entrée

Commun aux trois tâches :

| Champ | Type | Défaut | Rôle |
|-------|------|--------|------|
| `task` | `"instrumental"` \| `"vc"` \| `"speaking_faces"` | — | obligatoire |
| `output_url` | URL | — | PUT présigné (R2/S3). Sans lui, le résultat revient en `audio_base64` (≤ 10 MB) |

**Tout résultat audio est un WAV mono 24 kHz 16 bits**, sans option : c'est le format de la plateforme. Rééchantillonnage soxr,
mixage mono à gain 1 (matrices `pan` explicites, jamais `-ac`). `speaking_faces` rend du JSON (dans la réponse, et sur `output_url` si donné).
| `callback_url` | URL | — | rappels de l'image : `started`, `heartbeat`, `finished` (`POST` JSON, voir ci-dessous) |
| `callback_token` | string | — | envoyé en `Authorization: Bearer …` sur chaque rappel |
| `meta` | objet | — | OPAQUE : renvoyé tel quel dans chaque rappel et dans le résultat (projet, portion, tentative, compte…) |
| `heartbeat_s` | int | 30 | cadence des battements (5..300 s) |

### Rappels (`callback_url`)

L'image raconte le job à la plateforme, qui n'a donc aucune connexion à tenir ouverte. Chaque rappel porte `event`,
`seq` (compteur croissant par job), `job_id`, `task`, `meta`, `provider` (`modal` | `runpod`) et `sent_at` (ISO UTC).

| `event` | Quand | En plus |
|---------|-------|---------|
| `started` | avant tout travail | `started_at`, `gpu_name`, `device`, `container_first_job` (conteneur neuf ?), `container_uptime_s`, `models_loaded` |
| `heartbeat` | toutes les `heartbeat_s` secondes | `elapsed_s`, `progress` (dernier `{percent, message}`) |
| `finished` | à la fin, succès **ou** erreur | le résultat ENTIER (sans base64) : `container_s`, `timings`, `bytes`, `sha256`, `uploaded`… |

Un rappel raté (3 essais pour `started`/`finished`, 1 pour `heartbeat`) n'échoue jamais le job : le résultat reste
lisible côté hébergeur (`/status` RunPod, action `status` Modal). Ajoutez aussi le champ natif `"webhook"` dans
l'appel `/run` de RunPod : lui seul part si le worker meurt ou dépasse son timeout.

### `task: "instrumental"`

```json
{ "input": { "task": "instrumental", "audio_url": "https://…", "output_url": "https://…(PUT)",
             "callback_url": "https://api.dubbingspark.com/api/internal/gpu-complete", "callback_token": "…" } }
```

| Champ | Défaut | Rôle |
|-------|--------|------|
| `audio_url` | — | mp3, wav, m4a, mp4… (décodé et rééchantillonné en 44,1 kHz stéréo par ffmpeg pour le modèle) |

Découpage fenêtré du modèle : chunks de 881 559 échantillons (20 s), recouvrement 2, fondu aux jonctions.
Sur OOM CUDA : libération des modèles résidents puis un second essai.

### `task: "vc"`

```json
{ "input": { "task": "vc", "source_url": "https://…", "ref_urls": ["https://…", "https://…"],
             "prompt_url": "https://…", "steps": 25, "temp": 0.8, "cfg": 0.7,
             "ref_len": 10, "output_url": "https://…(PUT)" } }
```

| Champ | Défaut | Rôle |
|-------|--------|------|
| `source_url` | — | voix à convertir |
| `ref_urls` (ou `ref_url`) | — | 1..8 clips de la voix cible (timbre moyenné) |
| `prompt_url` | 1er `ref_urls` | clip dans la langue de la source (contexte phonétique) |
| `steps` | 25 | pas du flow matching (`n_cfm_timesteps`, 1..64 ; défaut interne de Chatterbox : 10) |
| `temp` | 0.8 | temperature du décodeur (0..2) |
| `cfg` | checkpoint | `inference_cfg_rate` (0..3) |
| `ref_len` | 10 | longueur du prompt de référence en secondes (1..30) |
| `preproc` | `true` | passe-haut 70 Hz + sonie −23 LUFS sur la source |
| `seed` | 1000 | graine du tirage (résultat reproductible) |
| `window_s` | 60 | longueur max d'une fenêtre de conversion (10..600 s) — la mémoire du décodeur croît avec le carré de la durée |
| `cuts_s` | `[]` | frontières AUTORISÉES, en secondes depuis le début de la source : les jonctions de segments que la plateforme a collés. Chaque fenêtre s'arrête sur la dernière frontière qui tient dans `window_s` ; sans frontière utilisable (ou sans `cuts_s`), coupe au creux d'énergie des 10 s précédant la cible |

La sortie a la durée exacte de la source. Si le modèle produit plus court, la partie de la source restée sans
sortie est reconvertie seule et **collée bout à bout, sans recouvrement ni fondu** (décision du 2026-09-09), jusqu'à 5 fois ;
`tail_passes` compte ces passes. Le filigrane Perth de Chatterbox est conservé (comportement natif de `generate`).
Un seul tirage par job (décision du 2026-09-09) : pas de best-of-N, donc ni scorer ECAPA, ni Whisper, ni `resemble-enhance`.

### `task: "speaking_faces"`

```json
{ "input": { "task": "speaking_faces", "video_url": "https://…/video_final.mp4", "audio_url": "https://…/audio_stream.m4a",
             "start": 300, "end": 600, "margin": 2, "output_url": "https://…(PUT, facultatif)" } }
```

| Champ | Défaut | Rôle |
|-------|--------|------|
| `video_url` | — | la vidéo, lue par requêtes Range (`-ss` avant `-i` : seule la portion est tirée) |
| `audio_url` | piste de `video_url` | le son, s'il vit dans un autre fichier — OBLIGATOIRE dans la plateforme (`video_final.mp4` est muet). Sans son : `bad_input`, jamais une liste vide |
| `start` / `end` | toute la vidéo | la portion, en secondes de la vidéo d'origine ; les deux ou aucun ; 3 600 s au plus |
| `margin` | 2 | contexte analysé puis jeté de chaque côté (0..30 s) |

Sortie : `speech` = `[{start, end}]` (quand quelqu'un parle, au millième, visages fusionnés) ; `faces` = `[{start, end,
box: [[t, x, y, w, h], …]}]` (où : une suite par visage et par passage, fractions de l'image, points échantillonnés) ;
`window`, `clip` (ce qui a été analysé, marge comprise, et ce que ffprobe y a mesuré), `frames`, `scenes`, `tracks`, plus les
champs communs. Les temps sont ceux de la vidéo d'origine : recoller des portions = fusionner des intervalles.
Contrat complet, poids, pièges et ce qui reste : **[docs/TACHE_VISAGES_QUI_PARLENT.md](docs/TACHE_VISAGES_QUI_PARLENT.md)**.

## Vast.ai (image à part, cœur commun)

Un pod se loue à l'heure, machine entière : on le remplit au lieu de lui donner un job à la fois.
Combien de jobs il encaisse, la mémoire de chaque tâche, les critères de location et les pièges de
mesure : **[docs/ETUDE_CAPACITE_GPU.md](docs/ETUDE_CAPACITE_GPU.md)** (mesures du 2026-09-12).
En bref : ~4 Go par séparation, donc 5 jobs sur 24 Go et 25 sur 102 Go ; la conversion vocale ne
remplit jamais la carte ; toutes les cartes de la machine sont utilisées.

```bash
# construire (quelques secondes : deux couches par-dessus l'image de production)
docker build -f Dockerfile.vast -t spark-gpu-vast:dev .

# lancer sur le pod
docker run --gpus all -p 8000:8000   -e SPARK_WORKER_TOKEN=… -e SPARK_JOBS_PER_GPU=4 -e SPARK_MIN_VRAM_GB=24   ghcr.io/cyrille8000/spark-gpu-inference-vast:sha-xxxxxxx

# mesurer ce que la carte encaisse — le banc CHERCHE le plafond, il n'est pas donné
python scripts/bench_concurrence.py --url http://<ip>:<port> --token … --job banc_instrumental.json
```

| Variable | Défaut | Rôle |
|---|---|---|
| `SPARK_WORKER_TOKEN` | — | OBLIGATOIRE : un pod est sur l'Internet public |
| `SPARK_JOBS_PER_GPU` | — | force le nombre de jobs simultanés ; vide = la carte décide |
| `SPARK_JOBS_AUTO` | 1 dans l'image Vast | déduire le nombre de jobs de la carte et de la tâche |
| `SPARK_JOBS_MAX` | 4 | plafond de la déduction |
| `SPARK_MIN_VRAM_GB` | 24 | refus au démarrage sous ce seuil |
| `SPARK_IDLE_EXIT_S` | 900 | arrêt après ce temps sans job (0 = jamais) |

`POST /run` (synchrone), `POST /submit` (rappel `callback_url`, comme Modal et RunPod),
`GET /result?job_id=` (relire un job soumis : 202 tant qu'il tourne, 200 ensuite), `GET /status`,
`GET /health`, `POST /shutdown`. Le worker vérifie TOUTES les cartes au démarrage et refuse de
démarrer si aucune ne peut exécuter l'image.

NE JAMAIS tenir une connexion ouverte pendant un job : au-delà d'environ 200 s de silence elle est
coupée (mesuré, des jobs perdus qu'on attribuait à tort à la carte). `/submit` puis `/result`.

**Quelles cartes louer.** Ces roues (torch 2.7.1+cu128) exigent une capacité de calcul **>= 7.5**, et c'est la
carte qui compte, pas la version CUDA de son pilote : un V100 sur pilote CUDA 13.0 reste en capacité 7.0, donc
refusé (vécu le 2026-09-12, conteneur en boucle de redémarrage).

| Louer | Capacité | Mémoire |
|---|---|---|
| RTX 3090 / A6000 | 8.6 | 24 / 48 Go |
| RTX 4090 / L40S | 8.9 | 24 / 48 Go |
| A100 | 8.0 | 40 / 80 Go |
| H100 | 9.0 | 80 Go |
| T4, A10 | 7.5 / 8.6 | 16 / 24 Go |

À écarter quelle que soit leur mémoire ou leur prix : **V100, P100, P40** et tout ce qui précède Turing.

## Prise : le worker va CHERCHER son travail

> Architecture complète, mesures RunPod et facturation :
> **[docs/ARCHITECTURE_PRISE.md](docs/ARCHITECTURE_PRISE.md)**.
> Combien de jobs tient une carte : **[docs/ETUDE_CAPACITE_GPU.md](docs/ETUDE_CAPACITE_GPU.md)**.

On demande quatre cartes à RunPod et on en reçoit parfois trois — leur propre contrôle
de démarrage le dit (`GPU binary test passed: 3 GPU(s) healthy`, mesuré le 2026-09-12
alors que l'endpoint est réglé sur 4). Le serveur ne peut donc pas savoir combien de
jobs envoyer. Le worker, lui, sait : il compte ses cartes.

On ne lui passe qu'une chose au lancement, une URL **signée** — elle porte son
autorisation, le worker n'a aucun secret à connaître :

```json
{"claim_url": "https://api.dubbingspark.com/api/internal/gpu-claim?sig=…"}
```

Il demande, exécute, et renvoie ses résultats AVEC la demande suivante :

```json
→ {"worker": "…", "cartes": ["cuda:0","cuda:1","cuda:2"], "capacite": 6,
   "gpu_name": "…", "vague": 2, "restant_s": 310.4, "resultats": [ … ]}
← {"jobs": [ {"task": "instrumental", "audio_url": "…", "output_url": "…"}, … ]}
```

Le serveur rend au plus `capacite` jobs. Deux règles tiennent tout le reste :

**File vide, le worker ATTEND** (décision du 2026-09-15, qui inverse celle du 12). Il dort
`attente_s` — ce que le serveur lui dit, 15 s par défaut — puis redemande. Il ne sort que
sur l'ordre `arret` du serveur : c'est l'ordonnanceur qui monte et qui descend, les
hébergeurs sont réglés avec des coupures énormes. `doux` laisse finir ; `net` rend d'abord
les résultats finis et la liste `abandonnes` des jobs en cours, puis sort sans attendre.

**Pas de budget par défaut.** `budget_s` reste possible : donné, le worker cesse de
reprendre quand il ne lui reste plus le temps d'un job, laisse finir, et sort.

**Chaque demande porte l'identité du worker** — `instance_id` (RunPod `RUNPOD_POD_ID`, Modal
`MODAL_TASK_ID`, Vast `VAST_CONTAINERLABEL`, ou `SPARK_INSTANCE_ID` posé par celui qui a lancé),
`machine_id`, `image_tag` (posé au build, `sha-…`), `demarre_a`, `uptime_s` — et l'avancement de
chaque job en cours (`en_vol[]` : id du serveur, tâche, `elapsed_s`, `percent`). Sur Vast,
`SPARK_CLAIM_URL` lance le pod directement en prise : aucun port à ouvrir, aucun jeton.

Le dernier envoi porte `"fin": true` : il ne sert que si la boucle s'arrête d'elle-même
(budget, serveur muet), car il n'y a alors plus de demande suivante où glisser les
résultats. Sans lui, le serveur garderait les réservations jusqu'à expiration.

## Avancement, lisible du dehors

Pendant qu'un lot tourne, l'image publie l'état DU LOT et pas seulement d'un sous-job :

```json
{"lot": "…", "total": 8, "faits": 3, "restants": 5, "places": 6,
 "percent": 38, "ecoule_s": 41.2, "restant_s": 27.5, "message": "lot : 3/8 sous-job(s)"}
```

L'estimation raisonne en VAGUES, pas en jobs : `places` sous-jobs tournent de front,
donc ce qui reste est un nombre de vagues entières. Le chemin de sortie existe déjà —
`progress_update` chez RunPod (lisible par `/status`), le heartbeat vers `gpu-event`
chez Modal et Vast.

## Lot : plusieurs sous-jobs dans UNE requête

La plateforme n'a que 80 places simultanées chez ses hébergeurs, et un job y occupait
une place entière. Un lot de vingt n'en occupe qu'une.

```json
{"jobs": [
  {"task": "instrumental", "audio_url": "…", "output_url": "…", "callback_url": "…"},
  {"task": "instrumental", "audio_url": "…", "output_url": "…", "callback_url": "…"}
]}
```

Chaque sous-job garde SES `callback_url` et `output_url` : la plateforme reçoit
`started` / `heartbeat` / `finished` par sous-job comme avant. La réponse du lot n'est
qu'un récapitulatif — `total`, `reussis`, `echecs`, `places`, `cartes`, et `resultats`
avec une ligne par sous-job. Un sous-job qui échoue n'emporte pas les autres.

Combien tournent EN MÊME TEMPS : déduit de la carte et de la tâche la plus gourmande
du lot, multiplié par le nombre de cartes. Rien à régler chez l'hébergeur — pas de
`concurrency_modifier`, pas de `@modal.concurrent`, pas de variable d'environnement :
**c'est l'appelant qui choisit la taille du lot, sans reconstruire l'image**. Au plus
256 sous-jobs. `SPARK_JOBS_PER_GPU` force encore la valeur si besoin.

Repères mesurés (2026-09-12) : une carte de 24 Go tient 5 séparations ou 9 conversions
vocales ; un worker RunPod à 4 cartes en tient 20 ou 36.

## Sortie

```json
{ "status": "completed", "task": "instrumental", "job_id": "…",
  "model": "roformer-model-bs-roformer-leap-xe-instrumental-by-pcunwa",
  "format": "wav", "bytes": 8640044, "sha256": "…", "uploaded": true,
  "duration_s": 180.0, "sample_rate": 24000, "channels": 1, "attempts": 1,
  "gpu_mem": { "allocated_gb": 3.1, "reserved_gb": 4.2 }, "gpu_mem_total_gb": 23.6,
  "elapsed_s": 21.4, "cold_start": true, "container_s": 27.9, "container_first_job": true,
  "started_at": "2026-09-09T20:01:02.123Z", "finished_at": "2026-09-09T20:01:24.011Z",
  "meta": { "project": "…", "portion": 3, "run": "sep_…", "attempt": 1 }, "provider": "modal", "heartbeats": 0,
  "timings": { "download_s": 0.8, "decode_s": 0.4, "model_load_s": 6.1, "inference_s": 12.9, "encode_s": 0.3, "upload_s": 0.9 },
  "device": "cuda", "gpu_name": "NVIDIA L4", "models_loaded": ["bs_roformer_leap_xe"],
  "callback_delivered": true }
```

`timings` est le temps mesuré par l'image, étape par étape. **`container_s` est ce que l'hébergeur facture** : la
fenêtre conteneur depuis le rapport précédent (ou depuis le démarrage du processus pour le premier job) = boot + attente +
ce job ; la somme sur les jobs d'un conteneur est sa vie entière, à la queue d'inactivité finale près (10 s sur Modal,
idle timeout RunPod). `executionTime` RunPod et `elapsed_s` ne couvrent que le job. `cold_start` dit si le modèle a dû être
chargé pour ce job, `container_first_job` si c'est le premier job du conteneur. En cas d'échec, le même rappel part avec `{ "status": "error", "error", "code" }`.

Pour `vc` s'ajoutent `seed`, `tail_passes`, les réglages appliqués et `warnings[]`.
Erreurs : `{ "status": "error", "error": "…", "code": "bad_input" | "internal", "job_id": "…" }`. Une `bad_input` ne doit jamais être rejouée.

## Modal (mêmes comptes que Demucs)

`modal_app.py` déploie la **même image GHCR** sur Modal, sans rebuild : un endpoint web POST par compte, même JSON
que RunPod posté directement (sans enveloppe `input`) plus `api_key` = le secret Modal `modal-api-key` déjà présent sur
les six comptes. **Asynchrone (2026-09-09)** : `{"action":"submit", …}` répond tout de suite `{"status":"queued",
"job_id", "call_id"}` et le job tourne dans un conteneur GPU à part (`SparkGpu.process`, lancé par `spawn`) ; l'image
prévient par rappels. Le même endpoint sert de sonde `{"action":"status","call_id"}` → `running` | `completed`
(+ `result`) | `failed`, et de coupe-circuit `{"action":"cancel","call_id"}`. Un conteneur = un GPU = un job.

```bash
MODAL_PROFILE=compte2 modal deploy modal_app.py            # un compte à la fois (profils de ~/.modal.toml)
SPARK_IMAGE=ghcr.io/…:sha-xxxxxxx MODAL_PROFILE=… modal deploy modal_app.py   # épingler une autre image
curl -X POST https://<workspace>--spark-gpu-inference-sparkinference-run.modal.run \n  -H 'Content-Type: application/json' -d '{"api_key":"…","task":"instrumental","audio_url":"https://…"}'
```

Côté plateforme, tout est dans Doppler **`prd_cloudflare-workers` seulement** (décision du 2026-09-09 : l'OCI n'a plus
besoin des clés Modal/RunPod ; ne pas les remonter dans `prd`, la racine propage partout) : `SPARK_GPU_MODAL_ENDPOINT_URLS` (les six URL,
séparées par des virgules), `SPARK_GPU_MODAL_API_KEY`, `SPARK_GPU_MODAL_MAX_CONCURRENT` (10 par compte), et pour RunPod
`SPARK_GPU_RUNPOD_ENDPOINT_ID` + `SPARK_GPU_RUNPOD_API_KEY` + `SPARK_GPU_RUNPOD_MAX_CONCURRENT` (20 : le plafond
d'appels simultanés côté appelant ; le « Max workers » de l'endpoint RunPod doit être au moins égal). `MODAL_ENDPOINT_URL` / `MODAL_API_KEY` restent ceux de Demucs.

Le package GHCR doit être public, ou chaque compte doit porter un secret `ghcr-pull` (`REGISTRY_USERNAME`,
`REGISTRY_PASSWORD` = jeton GitHub `read:packages`) et le déploiement se fait avec `GHCR_PRIVATE=1`.

## Build, CI, déploiement

- **CI** : `.github/workflows/docker-build.yml` — push sur `main` (ou lancement manuel) → `ghcr.io/<owner>/spark-gpu-inference:latest` et `:sha-<commit>`. Le build télécharge les poids (≈ 0,27 GB BS-Roformer + 1 GB Chatterbox) et exécute `scripts/smoke_test.py` **hors ligne sur CPU** : chargement de chaque modèle et une vraie passe avant BS-Roformer sur 2 s de bruit. L'image n'est publiée que si tout passe.
- **RunPod** : endpoint serverless, image `ghcr.io/<owner>/spark-gpu-inference:latest` (le package GHCR doit être public, ou renseigner les identifiants de registre dans RunPod), GPU 16 GB minimum (24 GB confortable pour garder les deux modèles résidents), disque conteneur ≥ 15 GB. Aucune variable d'environnement requise.
- **Local** :

```bash
docker build --platform linux/amd64 -t spark-gpu-inference .
docker run --rm --gpus all -v "$PWD/test_input.json:/app/test_input.json" spark-gpu-inference
python -m pytest tests -q          # tests unitaires (numpy/scipy seulement)
```

Variables optionnelles : `SPARK_MAX_DOWNLOAD_MB` (2048), `SPARK_INLINE_LIMIT_MB` (10), `SPARK_HTTP_TIMEOUT_S` (180), `SPARK_TMPDIR`, `SPARK_LOG_LEVEL`.

## Arborescence

```
handler.py                       # entrée RunPod (progression + start)
modal_app.py                     # entrée Modal : endpoint web POST, même image, même contrat + api_key
src/spark_infer/
├── service.py                   # process_job : parsing, exécution, erreurs typées, rappel — commun RunPod/Modal
├── params.py                    # validation des entrées (pure)
├── tasks.py                     # téléchargement → modèle → encodage → livraison, rejeu OOM
├── separation_engine.py         # InstrumentalSeparator (BS-Roformer Leap Xe via BSRoformerSession)
├── vc_engine.py                 # VoiceConverter (Chatterbox réglé, un tirage)
├── faces_engine.py              # SpeakingFaceDetector (S3FD + LR-ASD résidents : plans, détection, suivi, recadrage, scores)
├── faces_geometry.py            # la géométrie de LR-ASD, pure (suivi, recadrage, lots de scoring) — testée sans torch
├── faces_segments.py            # des pistes et scores aux moments de parole et aux boîtes (repris de spark-dubbing-lipsync)
├── faces_clip.py                # découpe ffmpeg de la portion (pure) + sondes
├── lrasd/                       # LR-ASD vendu (MIT) : S3FD (s3fd_net, s3fd_box) et le réseau audio-visuel (asd_model)
├── registry.py                  # modèles résidents, libération sur OOM
├── io_utils.py                  # HTTP, ffmpeg, PUT présigné, base64
└── audio_utils.py               # fonctions pures (prétraitement, ajustement de durée)
scripts/fetch_weights.py         # build : poids BS-Roformer + Chatterbox + LR-ASD (sha256 vérifiés)
scripts/smoke_test.py            # build : chargement hors ligne + passe avant BS-Roformer
```
