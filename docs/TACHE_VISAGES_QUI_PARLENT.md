# Visages qui parlent — tâche `speaking_faces`

## Ce que c'est

Qui parle à l'image, et où. C'est LR-ASD (*Active Speaker Detection*, Liao et al.,
IJCV 2025) : S3FD trouve les visages, un réseau audio-visuel dit, image par image,
si chacun parle. Un visage muet n'apparaît pas dans la sortie.

Troisième tâche de l'image, à côté de `instrumental` et `vc`. Même mécanique que
les deux autres : job unique, lot, prise ; modèles résidents (`registry.lease`) ;
rappels `started` / `heartbeat` / `finished` ; `container_s` facturé.

Écrit le 2026-09-15. **Code seulement** : pas de build, pas de déploiement, pas de
mesure GPU. Voir « Ce qui reste ».

---

## D'où ça vient

De l'ancienne image `spark-dubbing-lipsync` (2026-08), retirée de la plateforme
avec le lipsync le 2026-09-06 (endpoint RunPod supprimé le 2026-09-07). Elle
lançait `Columbia_test.py` de LR-ASD en sous-processus.

Repris tel quel : le contrat de sortie (`speech`, `faces`, `window`), la logique
de segments et ses tests (`faces_segments.py`), la découpe (`-ss` avant `-i`,
filtre `fps=25`), l'échelle de détection relevée sur les petites sources.

Ce qui change :

| Avant (sous-processus) | Maintenant (en interne) |
|---|---|
| S3FD et le réseau rechargés à chaque job | résidents, un exemplaire par job (`lr_asd`) |
| poids S3FD retéléchargés si le `cwd` change | poids dans `/models/lrasd`, sha256 vérifiés au build |
| chaque image écrite en JPEG, relue deux fois | vidéo décodée deux fois (détection, recadrage), rien sur le disque |
| visages recadrés dans une vidéo XVID puis relus | recadrés en mémoire (12 Ko par image et par visage) |
| boîtes a priori S3FD recalculées à chaque image (~11 000 tours de boucle) | calculées une fois par taille |
| décodage des boîtes sur GPU (centaines de petits noyaux) | sur CPU |
| son extrait par `-ac 1` | 16 kHz mono à gain 1, matrice `pan` explicite |
| `np.int` corrigé par `sed` au build | code vendu, relu, testé (`src/spark_infer/lrasd/`, MIT) |

Chaque constante et chaque formule est relevée dans `Columbia_test.py` (commit
`1b6dcd2d`), pas déduite de sa doc : `faces_geometry.py` les nomme une à une.
Un seul écart de fond corrigé : `evaluate_network` mélangeait des secondes
(audio) et des images (vidéo) dans son `min` ; ça marchait parce que le son
d'une piste est toujours un peu plus court que son image. `scoring_length`
compte des deux côtés en images.

---

## Contrat d'entrée

```json
{ "input": { "task": "speaking_faces",
             "video_url": "https://…/video_final.mp4", "audio_url": "https://…/audio_stream.m4a",
             "start": 300, "end": 600, "margin": 2,
             "output_url": "https://…(PUT, facultatif)",
             "callback_url": "https://api.dubbingspark.com/api/internal/gpu-event?kind=…", "callback_token": "…",
             "meta": { "project": "…", "portion": 2 } } }
```

| Champ | Défaut | Rôle |
|-------|--------|------|
| `video_url` | — | la vidéo, lue par requêtes Range (présignée R2 ou passerelle média) |
| `audio_url` | piste de `video_url` | le son, quand il vit dans un autre fichier. **Obligatoire dans la plateforme** : `video_final.mp4` n'a aucune piste audio. Sans son, LR-ASD ne détecte rien et ne s'en plaint pas — un extrait sans audio est refusé (`bad_input`) |
| `start` / `end` | toute la vidéo | la portion, en secondes de la vidéo d'origine. Les deux ensemble, ou aucun ; 3 600 s au plus |
| `margin` | 2 | contexte analysé de chaque côté puis jeté (0..30 s). Aux bords LR-ASD est aveugle : piste < 11 images jetée, scores sans contexte |
| `output_url` | — | dépose aussi le résultat en JSON (`application/json`). Le résultat voyage de toute façon dans la réponse et le rappel `finished` |

Les champs communs (`callback_url`, `callback_token`, `meta`, `heartbeat_s`) sont
ceux du README.

---

## Sortie

```json
{ "status": "completed", "task": "speaking_faces", "model": "lr_asd/finetuning_TalkSet.model",
  "speech": [ { "start": 302.04, "end": 310.2 }, { "start": 318.36, "end": 325.08 } ],
  "faces":  [ { "start": 302.04, "end": 310.2,
                "box": [ [302.04, 0.41, 0.22, 0.12, 0.21], [305.0, 0.44, 0.22, 0.12, 0.21] ] } ],
  "window": { "start": 300, "end": 600 },
  "clip": { "start": 298.0, "end": 602.0, "fps": 25.0, "width": 1920, "height": 1080, "frames": 7600 },
  "frames": 7600, "scenes": 5, "tracks": 12,
  "format": "json", "bytes": 4123, "sha256": "…", "uploaded": true,
  "timings": { "download_s": 21.3, "decode_s": 0.4, "model_load_s": 2.1, "inference_s": 96.0,
               "scenes_s": 4.2, "detect_s": 58.0, "track_s": 0.1, "crop_s": 14.0, "score_s": 19.0, "upload_s": 0.3 },
  "attempts": 1, "cold_start": true, "gpu_mem": { … }, "container_s": … }
```

`speech` sert à **découper du travail** : secondes au millième, arrondies vers
l'extérieur, visages fusionnés — deux personnes qui se parlent dessus font un
intervalle. La question est « parle-t-on ici ? », pas « qui ? ».

`faces` sert à **montrer** : une suite par visage et par passage, jamais fusionnée.
`box` vaut `[temps, x, y, largeur, hauteur]` en fractions de l'image (0 → 1), pour
survivre au changement de définition entre la source analysée et le proxy que le
studio joue. Les points sont échantillonnés (un quand la boîte a bougé de 1,5 %,
un par seconde au minimum, 4 000 au plus par appel) ; le client interpole.

Les temps sont **toujours ceux de la vidéo d'origine**, jamais de l'extrait.
`window` reprend la portion demandée ; `clip` dit ce qui a réellement été
analysé (marge comprise) et ce que ffprobe a mesuré dessus.

Erreurs : `bad_input` (pas de piste audio, fenêtre invalide, source illisible),
`internal` (réseau, ffmpeg, OOM après un second essai).

---

## Analyser par portions

Une vidéo d'une heure se découpe en portions confiées à autant de jobs. Chacun
reçoit sa fenêtre, ne lit que ce qui la concerne, et rend des temps déjà
exprimés dans la vidéo d'origine. Recoller = fusionner les intervalles.

Une fenêtre entièrement après la fin de la vidéo rend `speech: []`,
`faces: []`, jamais une erreur : ffmpeg sort en succès avec un conteneur vide,
et une portion de trop en fin de découpage ne doit pas faire échouer l'analyse.

Une portion de 150 s (celle de la séparation) est un bon ordre de grandeur :
l'analyse prend environ une à deux fois la durée de l'extrait, loin du délai de
900 s de Modal.

### Keyframes et bornes de portion

Le job découpe avec `-ss` AVANT le `-i` : ffmpeg saute à la keyframe qui
précède `start`, décode et jette jusqu'à l'instant exact, et ne tire par
requêtes Range que la portion. Sur un GOP de 2 s (le proxy du studio,
`proxy_320.mp4`, écrit par le conteneur `video-encoder`), le gaspillage est au
pire 2 s de décodage et la coupe reste exacte à l'image.

Convention recommandée côté plateforme : des bornes de portion **multiples de
2 s**. Avec la marge de 2 s, le début de l'extrait tombe alors pile sur une
keyframe. Rien à régler dans le job.

### Quelle vidéo envoyer

Le code accepte le proxy comme la source, l'échelle de détection s'adapte
(`facedet_scale`). Ce n'est pas le même compromis :

| | Proxy 320p, GOP 2 s | Source jusqu'à 1080p |
|---|---|---|
| Téléchargement d'une portion de 150 s | ~2 à 4 Mo | ~40 à 80 Mo (Range) |
| Décodage sur le worker (deux passes) | quelques secondes | dizaines de secondes |
| Visage d'un plan large | 30 à 50 px, lèvres de quelques pixels | 100 à 200 px |
| Fiabilité du « il parle » | bonne sur les gros plans, faible sur les petits visages | celle de LR-ASD |

L'ancienne image analysait la source pour cette raison. Par défaut : la source ;
le proxy quand la vitesse prime ou que les visages sont gros. Dans les deux cas
le son est `audio_stream.m4a`, la vidéo étant muette.

---

## Les poids

| Fichier | Origine | Taille | sha256 |
|---|---|---|---|
| `finetuning_TalkSet.model` | dépôt LR-ASD, commit `1b6dcd2d` (commité, pas de LFS) | 3 426 337 o | `6b4ef536…d9342` |
| `sfd_face.pth` | Google Drive `1KafnHz7ccT-3IyddBsL5yi2xGtxAKypt` (TalkNet / LR-ASD) | 89 844 381 o | `d54a87c2…6c491` |

Un seul jeu de poids ASD : `finetuning_TalkSet` (F1 96,4 % sur Columbia contre
86,1 % pour `pretrain_AVA`, README de LR-ASD). C'est celui que l'ancienne image
utilisait.

Google Drive rend une page HTML quand son quota est atteint : `fetch_weights.py`
vérifie la taille exacte, l'absence de HTML et le sha256, et fait échouer le
build sinon. `SPARK_S3FD_URL` (argument de build, variable de dépôt GitHub)
tire S3FD d'un miroir HTTP à nous — à poser sur `files.dubbingspark.com`.

---

## Parité vérifiée avec l'original (2026-09-15, sur CPU)

Vidéo de démo de TalkNet (20 s, 640×360, 25 i/s, un seul plan). Le MÊME extrait
découpé a été donné aux deux : ce port d'un côté, `Columbia_test.py` de LR-ASD de
l'autre (patché pour CPU, scenedetect 0.6, chemins Windows).

| | Original | Port |
|---|---|---|
| Pistes de visage | 6 | 6, exactement les mêmes images |
| Boîtes | — | écart max 0,5 à 3,7 px (JPEG relus contre flux décodé) |
| Scores par image | — | écart moyen 0,1 à 0,5 ; accord parle / parle-pas 98,2 à 100 % |
| `speech` | 4 intervalles | les mêmes, bornes à une image près (40 ms) |
| `faces` | 6 passages | 5 : un scintillement de 3 images (19,52–19,64 s) en moins |

Les écarts viennent des entrées, pas de la logique : l'original relit des JPEG et
des recadrages XVID, le port lit le flux décodé et recadre en mémoire. Sur ce
poste sans GPU, la détection a pris 19 min des 28 : c'est le CPU, pas la carte.
Scripts dans le scratchpad de la session : `parity_mine.py`, `orig_resume.py`,
`orig_score.py`, `parity_compare.py`.

---

## Ce qui reste

- **Construire l'image et la déployer** (Modal ×6, RunPod). Le smoke test fait une
  passe avant S3FD et LR-ASD sur CPU ; `libglib2.0-0` est ajouté pour `cv2`.
- **Mesurer** : `COUT_MEMOIRE_GB["lr_asd"] = (1.5, 1.0)` est une estimation. Le
  banc doit la caler avec une portion de production avant d'ouvrir plus d'une
  place par carte. La tâche est aussi bornée par le CPU (décodage, recadrage,
  MFCC) : deux jobs sur une carte se disputent les cœurs.
- **Côté plateforme, personne n'appelle cette tâche** : le lipsync est retiré.
  Quand un flux en aura besoin, une sorte `speaking_faces` dans
  `utils/gpu-job-kinds.js` et un enfant de workflow par portion, sur le modèle
  de la séparation.
- **Miroir R2 de `sfd_face.pth`**, puis `SPARK_S3FD_URL` en variable GitHub.

---

## Pièges

- **Un écart de 10 images d'indice, pas 10 images vides.** `NUM_FAILED_DET` compare
  `frame - dernière_frame <= 10` : neuf images vides passent, dix coupent la piste.
- **`medfilt` complète par des zéros** aux deux bouts, mais à 13 ils restent
  minoritaires : une piste stable garde ses valeurs jusqu'au bord.
- **Les plans de PySceneDetect** sont en images `[début, fin[`, 0-based ; un plan
  plus court que `MIN_TRACK` est ignoré, comme dans l'original.
- **Le score `0.0` parle** : les scores sont arrondis au dixième et le seuil de
  LR-ASD est `>= 0`, après lissage sur cinq images.
- **Sans `audio_url`, la plateforme ne détectera jamais rien** : `video_final.mp4`
  n'a pas de son. Le service refuse plutôt que de rendre une liste vide.
- **La vidéo est décodée deux fois** ; sur une machine à peu de cœurs c'est ce
  qui domine, pas le GPU.
