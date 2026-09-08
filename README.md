# Spark GPU Inference — image RunPod Serverless

Une seule image, deux tâches choisies par le paramètre `task` du job :

| `task` | Ce que ça fait | Modèles embarqués |
|--------|----------------|-------------------|
| `demucs` | Instrumental seul (mix − voix). Même ensemble que la plateforme sur le chemin `--only_vocals` : htdemucs_ft (modèle « vocals ») + Kim_Vocal_2 + Kim_Inst (MDX-Net), pondérés 12/8/3. | `04573f0d-f3cf25b2.th`, `Kim_Vocal_2.onnx`, `Kim_Inst.onnx` |
| `vc` | Conversion de timbre Chatterbox VC (S3Gen) : timbre moyenné sur 1..8 clips de référence, prompt phonétique optionnel, pas / temperature / CFG réglables, complétion de la queue, best-of-N départagé par similarité de locuteur. | `ResembleAI/chatterbox` (`s3gen.safetensors`, `conds.pt`) + ECAPA `speechbrain/spkrec-ecapa-voxceleb` |

Tous les poids sont dans l'image (`/models`). À l'inférence, `HF_HUB_OFFLINE=1` : **aucun téléchargement de modèle**.
Les modèles restent résidents entre deux jobs d'un même worker (chargés à la première demande).

## Contrat d'entrée

Commun aux deux tâches :

| Champ | Type | Défaut | Rôle |
|-------|------|--------|------|
| `task` | `"demucs"` \| `"vc"` | — | obligatoire |
| `output_url` | URL | — | PUT présigné (R2/S3). Sans lui, le résultat revient en `audio_base64` (≤ 10 MB) |
| `output_format` | `"wav"` \| `"mp3"` | `mp3` (demucs) / `wav` (vc) | MP3 = libmp3lame VBR `-q:a 2` (comme la plateforme) |

### `task: "demucs"`

```json
{ "input": { "task": "demucs", "audio_url": "https://…", "output_url": "https://…(PUT)",
             "output_format": "mp3", "mono": true } }
```

| Champ | Défaut | Rôle |
|-------|--------|------|
| `audio_url` | — | n'importe quel conteneur/codec (décodé par ffmpeg) |
| `mono` | `false` | mixage mono en sortie |
| `chunk_size` | auto (VRAM) | chunk ONNX ; sinon `(VRAM − 4 GB) × 60 000 × 0,9`, borné [50 000 ; 5 000 000] |
| `vram_gb` | détectée | force la VRAM prise en compte pour `chunk_size` |
| `overlap` | `0.0001` | recouvrement (valeur de la plateforme) |
| `single_onnx` | `false` | n'utiliser que Kim_Vocal_2 (moins de VRAM, moins bon) |

Sur OOM CUDA : libération des modèles, chunk réduit de 50 000, jusqu'à 6 tentatives.

### `task: "vc"`

```json
{ "input": { "task": "vc", "source_url": "https://…", "ref_urls": ["https://…", "https://…"],
             "prompt_url": "https://…", "n": 4, "steps": 20, "temp": 0.8, "cfg": 0.7,
             "ref_len": 10, "output_url": "https://…(PUT)", "output_format": "wav" } }
```

| Champ | Défaut | Rôle |
|-------|--------|------|
| `source_url` | — | voix à convertir |
| `ref_urls` (ou `ref_url`) | — | 1..8 clips de la voix cible (timbre moyenné) |
| `prompt_url` | 1er `ref_urls` | clip dans la langue de la source (contexte phonétique) |
| `n` | 1 | tirages best-of-N (1..8), départagés par similarité ECAPA |
| `steps` | 20 | pas du flow matching (`n_cfm_timesteps`, 1..64) |
| `temp` | 0.8 | temperature du décodeur (0..2) |
| `cfg` | checkpoint | `inference_cfg_rate` (0..3) |
| `ref_len` | 10 | longueur du prompt de référence en secondes (1..30) |
| `overlap` | 1.0 | fondu de la complétion de queue (s) |
| `preproc` | `true` | passe-haut 70 Hz + sonie −23 LUFS sur la source |
| `seed` | 1000 | graine du 1er tirage (`seed + i`) |
| `output_sr` | 24000 | rééchantillonnage de sortie (8000..48000) |

La sortie est mono. Le filigrane Perth de Chatterbox est conservé (comportement natif de `generate`).
Le scorer WER (Whisper large-v3) et `resemble-enhance` du script d'essai ne sont **pas** embarqués.

## Sortie

```json
{ "status": "completed", "task": "vc", "job_id": "…", "elapsed_s": 12.3,
  "format": "wav", "bytes": 480044, "sha256": "…", "uploaded": true,
  "duration_s": 10.0, "sample_rate": 24000, "channels": 1,
  "n": 4, "best_run": 2, "runs": [{ "run": 1, "seed": 1000, "similarity": 0.71, "tail_passes": 0 }, …],
  "warnings": [], "device": "cuda", "models_loaded": ["chatterbox_vc"] }
```

Erreurs : `{ "error": "…", "code": "bad_input" | "internal", "job_id": "…" }`. Une `bad_input` ne doit jamais être rejouée.

## Build, CI, déploiement

- **CI** : `.github/workflows/docker-build.yml` — push sur `main` (ou lancement manuel) → `ghcr.io/<owner>/spark-gpu-inference:latest` et `:sha-<commit>`. Le build télécharge les poids (≈ 0,7 GB Demucs + 1 GB Chatterbox + 0,1 GB ECAPA) et exécute `scripts/smoke_test.py` **hors ligne sur CPU** : l'image n'est publiée que si chaque modèle se charge depuis les poids embarqués.
- **RunPod** : endpoint serverless, image `ghcr.io/<owner>/spark-gpu-inference:latest` (le package GHCR doit être public, ou renseigner les identifiants de registre dans RunPod), GPU 16 GB minimum (24 GB confortable pour garder Demucs et Chatterbox résidents), disque conteneur ≥ 20 GB. Aucune variable d'environnement requise.
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
├── demucs_engine.py             # InstrumentalSeparator (chemin --only_vocals de la plateforme)
├── mdx_net.py                   # MDX-Net copié tel quel de mvsep/inference_demucs.py
├── vc_engine.py                 # VoiceConverter (Chatterbox réglé + best-of-N ECAPA)
├── registry.py                  # modèles résidents, libération sur OOM
├── io_utils.py                  # HTTP, ffmpeg, PUT présigné, base64
└── audio_utils.py               # fonctions pures (chunk VRAM, prétraitement, fondu)
scripts/fetch_weights.py         # build : poids Chatterbox + ECAPA
scripts/smoke_test.py            # build : chargement hors ligne de chaque modèle
```
