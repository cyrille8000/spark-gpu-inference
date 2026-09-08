# Spark GPU Inference — image RunPod Serverless

Une seule image, deux tâches choisies par le paramètre `task` du job :

| `task` | Ce que ça fait | Modèle embarqué |
|--------|----------------|-----------------|
| `instrumental` | Instrumental seul. **BS-Roformer Leap Xe** (unwa, juin 2026) : un seul checkpoint entraîné directement sur la cible instrumentale, 18,07 dB SDR instrumental sur le Multisong de MVSEP, au-dessus des ensembles internes du site. | `pcunwa/BS-Roformer-Leap` → `Xe/bs_leap_xe_inst.ckpt` (268 MB), chargé par [bs-roformer-infer](https://github.com/openmirlab/bs-roformer-infer) (MIT, épinglé sur le commit `b0f1386f` : la roue PyPI 0.1.5 ne connaît pas Leap) avec sha256 vérifié |
| `vc` | Conversion de timbre Chatterbox VC (S3Gen) : timbre moyenné sur 1..8 clips de référence, prompt phonétique optionnel, pas / temperature / CFG réglables, complétion de la queue. Un seul tirage. | `ResembleAI/chatterbox` (`s3gen.safetensors`, `conds.pt`) |

Tous les poids sont dans l'image (`/models`). À l'inférence, `HF_HUB_OFFLINE=1` et le checkpoint BS-Roformer est
résolu localement : **aucun téléchargement de modèle**. Les modèles restent résidents entre deux jobs d'un même worker.

Base `python:3.11-slim` + torch 2.6.0 cu124 : les roues torch embarquent CUDA/cuDNN, seul le pilote de l'hôte est requis.

## Contrat d'entrée

Commun aux deux tâches :

| Champ | Type | Défaut | Rôle |
|-------|------|--------|------|
| `task` | `"instrumental"` \| `"vc"` | — | obligatoire |
| `output_url` | URL | — | PUT présigné (R2/S3). Sans lui, le résultat revient en `audio_base64` (≤ 10 MB) |
| `output_format` | `"wav"` \| `"mp3"` | `mp3` (instrumental) / `wav` (vc) | MP3 = libmp3lame VBR `-q:a 2` (comme la plateforme) |

### `task: "instrumental"`

```json
{ "input": { "task": "instrumental", "audio_url": "https://…", "output_url": "https://…(PUT)",
             "output_format": "mp3", "mono": true } }
```

| Champ | Défaut | Rôle |
|-------|--------|------|
| `audio_url` | — | n'importe quel conteneur/codec (décodé et rééchantillonné en 44,1 kHz stéréo par ffmpeg) |
| `mono` | `false` | mixage mono en sortie |

Découpage fenêtré du modèle : chunks de 881 559 échantillons (20 s), recouvrement 2, fondu aux jonctions.
Sur OOM CUDA : libération des modèles résidents puis un second essai.

### `task: "vc"`

```json
{ "input": { "task": "vc", "source_url": "https://…", "ref_urls": ["https://…", "https://…"],
             "prompt_url": "https://…", "steps": 25, "temp": 0.8, "cfg": 0.7,
             "ref_len": 10, "output_url": "https://…(PUT)", "output_format": "wav" } }
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
| `overlap` | 1.0 | fondu de la complétion de queue (s) |
| `preproc` | `true` | passe-haut 70 Hz + sonie −23 LUFS sur la source |
| `seed` | 1000 | graine du tirage (résultat reproductible) |
| `output_sr` | 24000 | rééchantillonnage de sortie (8000..48000) |

La sortie est mono. Le filigrane Perth de Chatterbox est conservé (comportement natif de `generate`).
Un seul tirage par job (décision du 2026-09-09) : pas de best-of-N, donc ni scorer ECAPA, ni Whisper, ni `resemble-enhance`.

## Sortie

```json
{ "status": "completed", "task": "instrumental", "job_id": "…", "elapsed_s": 12.3,
  "model": "roformer-model-bs-roformer-leap-xe-instrumental-by-pcunwa",
  "format": "mp3", "bytes": 480044, "sha256": "…", "uploaded": true,
  "duration_s": 180.0, "sample_rate": 44100, "channels": 1, "attempts": 1,
  "device": "cuda", "models_loaded": ["bs_roformer_leap_xe"] }
```

Pour `vc` s'ajoutent `seed`, `tail_passes`, les réglages appliqués et `warnings[]`.
Erreurs : `{ "error": "…", "code": "bad_input" | "internal", "job_id": "…" }`. Une `bad_input` ne doit jamais être rejouée.

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
handler.py                       # entrée RunPod : dispatch par task, erreurs typées
src/spark_infer/
├── params.py                    # validation des entrées (pure)
├── tasks.py                     # téléchargement → modèle → encodage → livraison, rejeu OOM
├── separation_engine.py         # InstrumentalSeparator (BS-Roformer Leap Xe via BSRoformerSession)
├── vc_engine.py                 # VoiceConverter (Chatterbox réglé, un tirage)
├── registry.py                  # modèles résidents, libération sur OOM
├── io_utils.py                  # HTTP, ffmpeg, PUT présigné, base64
└── audio_utils.py               # fonctions pures (prétraitement, fondu)
scripts/fetch_weights.py         # build : poids BS-Roformer + Chatterbox
scripts/smoke_test.py            # build : chargement hors ligne + passe avant BS-Roformer
```
