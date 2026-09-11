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
- **VC : source convertie PAR FENÊTRES** (`audio_utils.plan_windows`, `window_s` = 60 s par défaut, reliquat ≤ 10 s
  absorbé, fenêtres collées sans fondu, chacune ramenée à la longueur exacte de sa source). La mémoire du décodeur
  grandit avec le carré de la durée : 176 s passaient, 179 s débordaient un L4 de 22 Go (mesuré le 2026-09-11,
  `CUDA out of memory` puis timeout 900 s).
- **VC : la coupe se pose sur les frontières envoyées par la plateforme** (`cuts_s`, secondes depuis le début de la
  source — demande du propriétaire, 2026-09-11 : c'est le worker qui a collé les segments, c'est lui qui sait où ils
  se touchent). Dernière frontière qui tient dans `window_s` ; le creux d'énergie des 10 s avant la cible n'est plus
  qu'un REPLI (segment plus long que la fenêtre, ou appelant sans `cuts_s`). Ne pas réintroduire de détection de
  silence en premier choix.
- **Chatterbox réglé par job, jamais empilé.** `VoiceConverter` garde les méthodes d'origine (`_orig_*`) et reconstruit
  les `functools.partial` à chaque job ; sinon les réglages s'accumulent d'un job à l'autre sur le modèle résident.
- **L'image raconte le job (2026-09-09).** `callback_url` (+ `callback_token` en Bearer) reçoit `started`, des
  `heartbeat` (toutes les `heartbeat_s`, défaut 30 s) et `finished` (le résultat entier sans base64), chacun avec `meta`
  (opaque, renvoyé tel quel), `seq`, `provider` — `src/spark_infer/webhooks.py`. La plateforme ne tient aucune
  connexion ouverte ; le webhook natif RunPod reste le filet si le worker meurt. Un rappel raté n'échoue jamais le job.
- **Modal asynchrone.** `modal_app.py` : `SparkInference.run` (CPU, image slim) = `submit` → `spawn` de `SparkGpu.process`
  (GPU) + `status` + `cancel` ; l'URL de l'endpoint est inchangée. Coût = temps seulement (`container_s`, `timings`,
  `executionTime` RunPod), pas d'estimation en dollars.
- **Tout résultat = WAV mono 24 kHz 16 bits**, sans option (`OUTPUT_FORMAT` / `OUTPUT_SR` / `OUTPUT_MONO` dans params.py) — décision 2026-09-09.
  Écrit `-bitexact` : en-tête canonique de 44 octets, pas de bloc LIST/INFO d'ffmpeg avant `data` (2026-09-11).
- **Conversions de canaux à gain 1, matrices explicites.** Jamais `-ac` seul : ffmpeg atténue mono→stéréo de 0,707
  et amplifie stéréo→mono de 1,414 (un WAV mono plateforme ressortait 3 dB trop bas). `pan=stereo|c0=c0|c1=c0` et
  `pan=mono|c0=0.5*c0+0.5*c1` dans `io_utils._channel_filter`, prouvés par `tests/test_ffmpeg_levels.py`.
- **Erreurs typées.** `InputError` → `code: bad_input` (ne jamais rejouer) ; le reste → `code: internal`. Un OOM CUDA
  ne rejoue UNE fois qu'après avoir libéré un AUTRE modèle résident (`registry.others_loaded`) ; le modèle du job seul
  en mémoire = pas de 2e essai (2026-09-11 : un rejeu a coûté 15 min d'L4 pour rien). Tout résultat, échec compris,
  porte `gpu_name` et `device` — la plateforme facture au vrai GPU.
- **Pins.** setuptools < 82 (resemble-perth importe `pkg_resources`, supprimé en 82 ; sinon le filigrane Chatterbox
  vaut None), torch/torchaudio 2.7.1 **cu128** (le pool 24 GB RunPod sert des Blackwell sm_120 que cu124 ne sait pas exécuter ;
  chatterbox-tts 0.1.7 épingle 2.6.0 mais est installé --no-deps), numpy < 2, bs-roformer-infer épinglé sur
  le commit GitHub `b0f1386f` (la roue PyPI 0.1.5 n'a ni l'entrée Leap ni `mlp_expansion_factor`).
  chatterbox-tts est installé `--no-deps` pour ne pas embarquer gradio ; ses dépendances sont listées dans
  `requirements.txt`. `pip check` du Dockerfile bloque tout autre conflit.

- **Temps conteneur rapporté, jamais estimé.** `container_s` (container_clock.py) = fenêtre depuis le rapport
  précédent ou le démarrage du processus : c'est ce que Modal/RunPod facturent, pas `elapsed_s`. La plateforme facture
  `container_s`. `scaledown_window` Modal = 10 s pour que la queue d'inactivité non comptée reste courte (2026-09-09).
- **Un seul traitement, deux hébergeurs.** `spark_infer/service.py::process_job` est appelé par `handler.py` (RunPod)
  et `modal_app.py` (Modal, même image GHCR via `Image.from_registry`, secret `modal-api-key`). Ne rien dupliquer.

## Commandes

```bash
python -m pytest tests -q                                   # unitaires (numpy/scipy, sans torch)
docker build --platform linux/amd64 -t spark-gpu-inference . # build complet (télécharge ~1,4 GB de poids)
```

## CI

`.github/workflows/docker-build.yml` : push sur `main` → build + push `ghcr.io/<owner>/spark-gpu-inference:{latest,sha-…}`.
