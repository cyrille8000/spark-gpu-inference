# CLAUDE.md — spark-gpu-inference

Image Docker RunPod Serverless d'inférence GPU pour Spark Dubbing. Deux tâches, choisies par `input.task` :
`demucs` (instrumental seul) et `vc` (conversion de timbre Chatterbox VC). Détails du contrat : [README.md](README.md).

## Règles

- **Zéro téléchargement à l'inférence.** Tous les poids sont posés au build sous `/models` ; `HF_HUB_OFFLINE=1`
  ensuite. Tout nouveau modèle passe par `scripts/fetch_weights.py` (ou le Dockerfile) **et** par `scripts/smoke_test.py`,
  qui prouve le chargement hors ligne sur CPU avant publication.
- **Parité Demucs.** `demucs_engine.py` reproduit le chemin `--only_vocals` de `ffmpeg-demucs-runpod-template`
  (mêmes poids, mêmes pondérations 12/8/3, même `overlap` 0,0001, même formule de chunk). `mdx_net.py` est une copie
  verbatim : on ne retouche pas les maths sans comparer les sorties.
- **Chatterbox réglé par job, jamais empilé.** `VoiceConverter` garde les méthodes d'origine (`_orig_*`) et reconstruit
  les `functools.partial` à chaque job ; sinon les réglages s'accumulent d'un job à l'autre sur le modèle résident.
- **Erreurs typées.** `InputError` → `code: bad_input` (ne jamais rejouer) ; le reste → `code: internal`. Un OOM CUDA
  libère les modèles (`registry.release()`) et rejoue avec un chunk réduit (demucs) ou une fois (vc).
- **Pins.** torch/torchaudio 2.6.0 cu124 (exigés par chatterbox-tts 0.1.7), numpy < 2, onnxruntime-gpu 1.22 (CUDA 12,
  cuDNN 9), demucs 4.0.1. chatterbox-tts est installé `--no-deps` pour ne pas embarquer gradio ; ses dépendances sont
  listées dans `requirements.txt`. `pip check` du Dockerfile bloque tout autre conflit.

## Commandes

```bash
python -m pytest tests -q                                   # unitaires (numpy/scipy, sans torch)
docker build --platform linux/amd64 -t spark-gpu-inference . # build complet (télécharge ~1,8 GB de poids)
```

## CI

`.github/workflows/docker-build.yml` : push sur `main` → build + push `ghcr.io/<owner>/spark-gpu-inference:{latest,sha-…}`.
