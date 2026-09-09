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
- **Python 3.11 obligatoire** : bs-roformer-infer importe `tomllib`. Base `python:3.11-slim`, torch cu128 par pip
  (pas de CUDA système, plus d'ONNX Runtime).
- **VC : un seul tirage.** Pas de best-of-N (décision 2026-09-09) → ni ECAPA/speechbrain, ni Whisper, ni resemble-enhance.
- **VC : queue collée, pas de fondu.** Si la sortie est plus courte que la source, le reste est reconverti et concaténé
  tel quel (décision 2026-09-09) ; ne pas réintroduire de recouvrement/crossfade sans demande.
- **VC `steps` = 25 par défaut** (choix propriétaire 2026-09-09 ; défaut interne de Chatterbox : 10).
- **Chatterbox réglé par job, jamais empilé.** `VoiceConverter` garde les méthodes d'origine (`_orig_*`) et reconstruit
  les `functools.partial` à chaque job ; sinon les réglages s'accumulent d'un job à l'autre sur le modèle résident.
- **Fin de job = résultat RunPod + rappel client.** `callback_url` (+ `callback_token` en Bearer) reçoit le même JSON,
  succès comme erreur, sans base64 ; le webhook natif RunPod reste le filet si le worker meurt. Coût = temps seulement
  (`timings` par étape + `executionTime` RunPod), pas d'estimation en dollars (décision 2026-09-09).
- **Instrumental par défaut = WAV 24 kHz mono 16 bits**, comme l'instrumental de la plateforme.
- **Conversions de canaux à gain 1, matrices explicites.** Jamais `-ac` seul : ffmpeg atténue mono→stéréo de 0,707
  et amplifie stéréo→mono de 1,414 (un WAV mono plateforme ressortait 3 dB trop bas). `pan=stereo|c0=c0|c1=c0` et
  `pan=mono|c0=0.5*c0+0.5*c1` dans `io_utils._channel_filter`, prouvés par `tests/test_ffmpeg_levels.py`.
- **Erreurs typées.** `InputError` → `code: bad_input` (ne jamais rejouer) ; le reste → `code: internal`. Un OOM CUDA
  libère les modèles (`registry.release()`) et rejoue une fois.
- **Pins.** setuptools < 82 (resemble-perth importe `pkg_resources`, supprimé en 82 ; sinon le filigrane Chatterbox
  vaut None), torch/torchaudio 2.7.1 **cu128** (le pool 24 GB RunPod sert des Blackwell sm_120 que cu124 ne sait pas exécuter ;
  chatterbox-tts 0.1.7 épingle 2.6.0 mais est installé --no-deps), numpy < 2, bs-roformer-infer épinglé sur
  le commit GitHub `b0f1386f` (la roue PyPI 0.1.5 n'a ni l'entrée Leap ni `mlp_expansion_factor`).
  chatterbox-tts est installé `--no-deps` pour ne pas embarquer gradio ; ses dépendances sont listées dans
  `requirements.txt`. `pip check` du Dockerfile bloque tout autre conflit.

## Commandes

```bash
python -m pytest tests -q                                   # unitaires (numpy/scipy, sans torch)
docker build --platform linux/amd64 -t spark-gpu-inference . # build complet (télécharge ~1,4 GB de poids)
```

## CI

`.github/workflows/docker-build.yml` : push sur `main` → build + push `ghcr.io/<owner>/spark-gpu-inference:{latest,sha-…}`.
