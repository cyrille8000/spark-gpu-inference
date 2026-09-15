# CLAUDE.md — spark-gpu-inference

Image Docker RunPod Serverless d'inférence GPU pour Spark Dubbing. Trois tâches, choisies par `input.task` :
`instrumental` (BS-Roformer Leap Xe, instrumental seul), `vc` (conversion de timbre Chatterbox VC) et
`speaking_faces` (visages qui parlent, LR-ASD — code seulement au 2026-09-15, ni bâti ni déployé).
Détails du contrat : [README.md](README.md).

## Règles

- **Zéro téléchargement à l'inférence.** Tous les poids sont posés au build sous `/models` par `scripts/fetch_weights.py` ;
  `HF_HUB_OFFLINE=1` ensuite et `BS_ROFORMER_MODELS_PATH=/models/bsroformer`. Tout nouveau modèle passe par ce script
  **et** par `scripts/smoke_test.py`, qui prouve le chargement hors ligne sur CPU (et une vraie passe avant pour BS-Roformer)
  avant publication.
- **Visages qui parlent = LR-ASD EN INTERNE, code vendu** (`src/spark_infer/lrasd/`, MIT, commit `1b6dcd2d`),
  jamais un sous-processus `Columbia_test.py` comme dans l'ancienne image `spark-dubbing-lipsync`. Les constantes et
  formules de `faces_geometry.py` sont RELEVÉES dans l'original, une à une, et testées sans torch : ne pas les
  « simplifier » (seuil `>= 0` après lissage sur 5 images, `NUM_FAILED_DET` = écart d'INDICE ≤ 10, medfilt 13,
  recadrage 224 puis centre 112, six passes de 1 à 6 s moyennées et arrondies au dixième). Un seul jeu de poids ASD :
  `finetuning_TalkSet` (F1 96,4 % contre 86,1 %). `sfd_face.pth` n'existe que sur Google Drive : sha256 et taille
  vérifiés au build, miroir `SPARK_S3FD_URL` à poser sur files.dubbingspark.com. `COUT_MEMOIRE_GB["lr_asd"]` est une
  ESTIMATION non mesurée. Sans `audio_url`, la plateforme ne détectera jamais rien (`video_final.mp4` est muet) : un
  extrait sans son est `bad_input`. Détail : [docs/TACHE_VISAGES_QUI_PARLENT.md](docs/TACHE_VISAGES_QUI_PARLENT.md).
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
- **Vast.ai : image À PART, cœur COMMUN** (2026-09-12). `Dockerfile.vast` part de l'image de production et
  n'ajoute que `src` (le cœur à jour) et `vast_worker.py` ; Modal et RunPod gardent leur image, inchangée tant
  qu'on ne la rebâtit pas. NE PAS toucher à `modal_app.py` ni `handler.py` : ils marchent, c'est la consigne.
- **LE WORKER VA CHERCHER SON TRAVAIL** (`claim_url`, 2026-09-12) — il compte ses cartes, ouvre ses places
  d'après la mémoire de CHACUNE (`tasks.places_prise` : < 24 Go → 1, ≥ 24 Go → 2 ; décidé par lui, 2026-09-15 ;
  `jobs_par_carte` du serveur ne peut que plafonner), les tient TOUJOURS PLEINES, et redemande dès qu'une se libère. Rien à régler chez l'hébergeur : ni
  `concurrency_modifier`, ni `@modal.concurrent`, ni variable d'environnement. Le battement va vers
  NOTRE serveur, donc sa réponse porte l'ordre d'arrêt (`{"arret":"doux"|"net"}`) — extinction à distance
  sur les trois plateformes avec le même code. Détail : [docs/ARCHITECTURE_PRISE.md](docs/ARCHITECTURE_PRISE.md).
- **FILE VIDE = IL ATTEND, IL NE SORT PLUS** (décision du propriétaire, 2026-09-15 — inverse celle du 12).
  C'est l'ORDONNANCEUR qui monte et qui descend ; les hébergeurs sont réglés avec des coupures énormes.
  Sur file vide le worker dort `attente_s` (donné par le serveur, défaut 15 s, max 300) puis redemande.
  Il ne sort que sur `arret` ; `net` rend d'abord les résultats finis et la liste `abandonnes` (le serveur
  les remet en file). Pas de budget par défaut (`budget_s` facultatif) ; l'homme-mort reste (5 min de
  silence : on cesse, 15 min : on se tue). Chaque demande porte l'IDENTITÉ (`instance_id` — RunPod
  `RUNPOD_POD_ID`, Modal `MODAL_TASK_ID`, Vast `VAST_CONTAINERLABEL`, ou `SPARK_INSTANCE_ID` posé par
  l'ordonnanceur —, `machine_id`, `image_tag` posé au build, `demarre_a`, `uptime_s`) et l'AVANCEMENT de
  chaque job en cours (`en_vol[]` : id du serveur, tâche, `elapsed_s`, `percent`) : c'est avec ça que
  l'ordonnanceur décide de couper un worker cher. Sur Vast, `SPARK_CLAIM_URL` lance le pod directement
  en prise (aucun port à ouvrir, aucun jeton de worker). À déployer EN DERNIER, après le bot.
- **RunPod ne livre pas toujours ce qu'on demande** : endpoint réglé sur 4 cartes, workers à 3 ou 4 selon
  le moment — leur propre contrôle de démarrage le dit. C'est LA raison du mode prise : le serveur ne peut
  pas deviner, le worker sait.
- **On paie TOUT le démarrage chez RunPod** (vérifié sur leur facture, `GET /v1/billing/endpoints`) :
  1 121,5 s facturées pour 464 s exécutées, ×2,42, et l'écart égale la somme des `delayTime`. **Chez Vast et
  Modal aussi** (établi le 2026-09-15) : on paie dès le démarrage, tirage de l'image compris ; chez Vast, la
  bande passante du tirage et le stockage en plus. L'ordonnanceur compte donc chaque worker depuis sa DEMANDE.
  Ne JAMAIS facturer le démarrage à l'utilisateur : deux jobs identiques auraient 8 min d'écart selon le
  hasard de l'ordonnancement. C'est un coût de plateforme.
- **Le nombre de jobs se DÉDUIT de la carte et de la tâche** (`tasks.jobs_pour_vram`), activé par
  `SPARK_JOBS_AUTO` que SEUL `Dockerfile.vast` pose : `tasks.py` est partagé, et ni Modal ni RunPod ne posent
  `SPARK_JOBS_PER_GPU` — sans ce garde-fou ils passeraient de 1 à 4 jobs sans que personne l'ait demandé.
  Mesuré : ~4 Go par séparation (5 jobs sur 24 Go, 25 sur 102 Go), et la conversion vocale ne remplit
  JAMAIS la carte. Détail complet : [docs/ETUDE_CAPACITE_GPU.md](docs/ETUDE_CAPACITE_GPU.md).
- **TOUTES les cartes de la machine sont utilisées** (2026-09-12) : pools indexés par `(modèle, carte)`,
  carte du job dans une variable de fil, capacité = somme des cartes. Avant, `device()` rendait « cuda » que
  PyTorch résout en `cuda:0` — une machine à deux RTX 3090 n'en utilisait qu'une, mesuré. Vast loue jusqu'à
  12 cartes par machine ; deux cartes donnent ×2,31 de débit, là où empiler sur UNE ne donne que ×1,1.
- **Un pod Vast.ai fait tourner PLUSIEURS jobs à la fois** — il se loue à
  l'heure, carte entière, donc on ne le rentabilise qu'en le remplissant. D'où le POOL d'instances
  (`registry.lease`) : sans lui, deux conversions vocales se volent leur voix de référence (chaque job écrit sa
  config et sa référence DANS le modèle, `vc_engine._configure` / `_set_reference`) — corruption silencieuse, pas
  un plantage. Le temps conteneur est réparti entre les jobs qui se croisent (`container_clock`), sinon un
  conteneur à trois jobs se ferait facturer trois fois son temps.
- **Vast.ai loue de tout : TOUTES les cartes sont vérifiées AU DÉMARRAGE** (`vast_worker.verifier_carte`) — CUDA
  initialisable, architecture présente dans `torch.cuda.get_arch_list()`, mémoire ≥ `SPARK_MIN_VRAM_GB`, plus un vrai
  petit calcul sur CHACUNE. Celles qui passent sont gardées, les autres écartées avec la raison : rien ne garantit
  qu'une machine louée ait des cartes identiques, ni toutes libres.
  Un pod se paie dès qu'il démarre : échouer vite et clairement vaut mieux que découvrir « no kernel image » au
  premier job — et un conteneur qui sort en erreur est RELANCÉ en boucle par Vast.ai, donc facturé.
- **CAPACITÉ DE CALCUL >= 7.5, et c'est la carte qui compte, pas le pilote.** torch 2.7.1+cu128 est compilé pour
  `['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120','compute_120']` (relevé sur un pod le 2026-09-12) : un V100
  32 Go à 0,041 $/h sur pilote CUDA 13.0 a été REFUSÉ, `cuda_max_good` élevé ou pas. À louer : T4, RTX 20xx/30xx/40xx,
  A10, A100, A6000, L40S, H100 ; jamais V100, P100, P40.
- **Combien de jobs en parallèle : le banc CHERCHE le plafond** (`scripts/bench_concurrence.py`) — il monte jusqu'à
  l'échec, resserre par dichotomie, rapporte le dernier palier tenu ENTIÈREMENT. Aucun palier choisi à la main.
  PIÈGES qui ont produit des chiffres faux et se reproduiront : deux tâches sur le même pod ne se mesurent pas
  (l'allocateur ne rend rien, la seconde relève la somme) ; un MP3 n'est pas l'entrée de la production (il masque
  le téléchargement, donc le gain du parallélisme) ; une connexion HTTP tenue pendant un job se fait couper
  au-delà de ~200 s. Les cinq pièges sont dans [docs/ETUDE_CAPACITE_GPU.md](docs/ETUDE_CAPACITE_GPU.md).
- **Chaque job rapporte son PIC de mémoire GPU** (`gpu_mem.allocated_gb` / `reserved_gb`, `gpu_mem_total_gb` —
  `registry.reset_peak_memory()` au début, `peak_memory_gb()` à la fin). C'est `reserved` qui dit si un GPU suffit ;
  avant le 2026-09-12 on ne connaissait la consommation que par l'OOM du 2026-09-11 (VC 179 s = plus de 22,5 Go).
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
  vaut None), torch/torchaudio 2.7.1 **cu128** (Vast loue des Blackwell sm_120 (RTX 5090, RTX PRO), sa famille la plus nombreuse, que cu124/cu126 ne savent pas exécuter ; RunPod ne sert plus que des RTX 4090 ;
  chatterbox-tts 0.1.7 épingle 2.6.0 mais est installé --no-deps), numpy < 2, bs-roformer-infer épinglé sur
  le commit GitHub `b0f1386f` (la roue PyPI 0.1.5 n'a ni l'entrée Leap ni `mlp_expansion_factor`).
  chatterbox-tts est installé `--no-deps` pour ne pas embarquer gradio ; ses dépendances sont listées dans
  `requirements.txt`. `pip check` du Dockerfile bloque tout autre conflit.

- **Temps d'inférence de CHAQUE job, au premier niveau du résultat** : `inference_s` (le calcul du modèle seul — ni
  téléchargement, ni décodage, ni chargement, ni encodage, ni dépôt ; détail par étape dans `timings`), pour les trois
  tâches et les trois hébergeurs, dans la réponse, le rappel `finished` et le résumé de prise (demande du propriétaire,
  2026-09-15). Pour `speaking_faces`, l'inférence couvre toute l'analyse vidéo (scènes, détection, suivi, découpe,
  score) : `timings.detect_s` et `timings.score_s` isolent les deux modèles.
- **Temps conteneur rapporté, jamais estimé.** `container_s` (container_clock.py) = fenêtre depuis le rapport
  précédent ou le démarrage du processus, pas `elapsed_s` — mais le tirage et le démarrage du conteneur AVANT
  le processus sont facturés aussi et n'y sont pas. La plateforme facture
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
