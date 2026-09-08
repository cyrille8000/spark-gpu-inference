# CLAUDE.md — spark-gpu-inference

Image Docker RunPod Serverless d'inférence GPU pour Spark Dubbing. Deux tâches, choisies par `input.task` :
`instrumental` (BS-Roformer Leap Xe, instrumental seul) et `vc` (conversion de timbre Chatterbox VC).
Détails du contrat : [README.md](README.md).

## Règles

- **Zéro téléchargement à l'inférence.** Tous les poids sont posés au build sous `/models` par `scripts/fetch_weights.py` ;
  `HF_HUB_OFFLINE=1` ensuite et `BS_ROFORMER_MODELS_PATH=/models/bsroformer`. Tout nouveau modèle passe par ce script
  **et** par `scripts/smoke_test.py`, qui prouve le chargement hors ligne sur CPU (et une vraie passe avant pour BS-Roformer)
  avant publication.
- **Un seul modèle de séparation.** BS-Roformer Leap Xe (`pcunwa/BS-Roformer-Leap`, slug
  `roformer-model-bs-roformer-leap-xe-instrumental-by-pcunwa` dans bs-roformer-infer), choisi le 2026-09-09 sur le
  Multisong de MVSEP (18,07 dB instrumental) à la place de l'ensemble Demucs/MDX de la plateforme (~17,5). Le stem de
  sortie s'appelle `other` dans la config du modèle : c'est l'instrumental (`target_instrument: other`).
- **Python 3.11 obligatoire** : bs-roformer-infer importe `tomllib`. Base `python:3.11-slim`, torch cu124 par pip
  (pas de CUDA système, plus d'ONNX Runtime).
- **Chatterbox réglé par job, jamais empilé.** `VoiceConverter` garde les méthodes d'origine (`_orig_*`) et reconstruit
  les `functools.partial` à chaque job ; sinon les réglages s'accumulent d'un job à l'autre sur le modèle résident.
- **Erreurs typées.** `InputError` → `code: bad_input` (ne jamais rejouer) ; le reste → `code: internal`. Un OOM CUDA
  libère les modèles (`registry.release()`) et rejoue une fois.
- **Pins.** torch/torchaudio 2.6.0 cu124 (exigés par chatterbox-tts 0.1.7), numpy < 2, bs-roformer-infer 0.1.5.
  chatterbox-tts est installé `--no-deps` pour ne pas embarquer gradio ; ses dépendances sont listées dans
  `requirements.txt`. `pip check` du Dockerfile bloque tout autre conflit.

## Commandes

```bash
python -m pytest tests -q                                   # unitaires (numpy/scipy, sans torch)
docker build --platform linux/amd64 -t spark-gpu-inference . # build complet (télécharge ~1,4 GB de poids)
```

## CI

`.github/workflows/docker-build.yml` : push sur `main` → build + push `ghcr.io/<owner>/spark-gpu-inference:{latest,sha-…}`.
