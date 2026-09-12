# Combien de jobs tient une carte — étude du 2026-09-12

## Ce qu'il faut retenir

La mémoire d'un job **ne dépend pas de la carte**. Une séparation prend 4,96 Go
sur les six cartes mesurées, d'Ampere à Blackwell, de 24 à 102 Go.

Le nombre de jobs est donc **proportionnel à la VRAM** : environ une séparation
par 4 Go. Cinq sur une carte de 24 Go, vingt-cinq sur une de 102 Go.

Empiler des jobs sur UNE carte **n'accélère rien** — chaque job s'allonge à
proportion. Ça multiplie les places, ce qui est exactement le besoin : le
plafond de 80 places simultanées.

Empiler sur **plusieurs cartes**, en revanche, accélère vraiment : ×2,31 de
débit sur une machine à deux cartes, parce que deux GPU calculent de front.

Ce qui coûte cher, c'est **la machine, pas la carte**. Une RTX 3090 à 0,22 $/h
a mis 77 s là où une Blackwell en met 24, faute de processeur et de réseau.

---

## Pourquoi cette étude

La plateforme dispose de 80 places GPU simultanées (6 comptes Modal × 10, plus
20 chez RunPod). Quand elles saturent, les jobs attendent. La question posée
était : peut-on louer des pods Vast.ai en débordement, et combien de jobs un
pod peut-il absorber ?

Contraintes posées par le propriétaire, dans l'ordre où elles sont venues.
D'abord : **Modal et RunPod ne se touchent pas** — c'est ce qui a fait construire
une image Vast.ai à part, `Dockerfile.vast`, par-dessus l'image de production.
Puis, le 2026-09-12 au soir : **Modal reste figé** — un job par worker d'une
carte, pas de concurrence, décision prise une fois su que le plan Starter plafonne
à 10 cartes par compte. **RunPod, lui, reçoit la concurrence**, pour qu'on puisse
y tester un worker à plusieurs cartes. `modal_app.py` reste donc intact.

---

## Les chiffres mesurés

Conditions : fichiers de **production** (WAV), **une seule tâche par pod**, job
d'échauffement jeté. Voir « Les pièges » pour savoir pourquoi ces trois
précautions changent tout.

### Séparation instrumentale — WAV 44,1 kHz mono, 150 s

| Carte | VRAM | Jobs tenus | Mémoire | % carte | Go/job | Casse à |
|---|---|---|---|---|---|---|
| RTX 3090 (une carte) | 25,3 | 5 | 24,42 | 96 % | 4,88 | 7 |
| 2 × RTX 3090 | 2 × 25,3 | **10** | 22,92 par carte | 91 % | 4,58 | 11 |
| A100 PCIE | 42,4 | 3 (non poussé) | 14,11 | 33 % | 4,70 | — |
| A100 SXM4 | 85,1 | 16 (non poussé) | 64,96 | 76 % | 4,06 | — |
| RTX PRO 6000 Server | 102,0 | 20 (non poussé) | 76,00 | 75 % | 3,80 | — |
| RTX PRO 6000 Workstation | 102,0 | **25** | 101,02 | 99 % | 4,04 | 28 |

« Non poussé » = la série s'est arrêtée à ce palier par choix, pas sur un échec.
Ces lignes sont des planchers, pas des plafonds.

Temps d'un job seul : 11,5 s (PRO 6000), 13,8 s (PRO 6000), 22,0 s (3090 rapide),
24,4 s (A100 PCIE), 26,9 s (PRO 4000), 94,2 s (3090 lente).

### Le L4 de Modal — mesuré le 2026-09-12 sur Modal même

Carte de 23,66 Go. Mesures faites sur une app Modal SÉPARÉE (`spark-gpu-bench`),
`max_containers=1` pour garantir un seul conteneur, image de production
`sha-a9ec11c`, `timeout` à 900 s comme en production.

| Séparations | Mur | Par job | Mémoire | % carte | Débit |
|---|---|---|---|---|---|
| 1 | 56,4 s | 56,4 s | 4,65 Go | 20 % | ×1 |
| 4 | 196,8 s | 188,2 s | 18,71 Go | 79 % | ×1,15 |
| **5** | **246,2 s** | 209,6 s | 18,71 Go | 79 % | ×1,15 |
| 6 | échec | — | 23,35 Go | OOM | — |

| Conversions vocales | Mur | Par job | Mémoire | % carte | Débit |
|---|---|---|---|---|---|
| 1 | 41,0 s | 41,0 s | 7,25 Go | 31 % | ×1 |
| 4 | 136,9 s | 114,8 s | 10,29 Go | 43 % | ×1,20 |
| 6 | 204,0 s | 158,8 s | 12,83 Go | 54 % | ×1,21 |
| **8** | 268,5 s | 216,1 s | 17,29 Go | 73 % | ×1,23 |

Huit conversions tiennent sans peine ; le plafond n'a pas été cherché plus haut.
Le délai de 900 s n'a jamais mordu : le job le plus long a pris 216 s.

Le démarrage à froid d'un conteneur L4 coûte environ **une minute** (113 s pour le
premier job contre 56 s ensuite). Un conteneur qui traite huit jobs ne la paie
qu'une fois au lieu de huit.

**Le plafond de Modal est en CARTES, pas en conteneurs.** Le plan Starter donne
10 GPU simultanés par compte (et 100 conteneurs). Mettre plusieurs cartes dans un
conteneur (`gpu="L4:4"`, supporté jusqu'à 8) n'augmente donc rien : ce sont les
mêmes dix cartes. Le seul levier chez Modal est le nombre de jobs par carte.
Chez RunPod au contraire, aucun plafond de compte n'est documenté et le nombre de
cartes par worker se configure — les deux leviers y jouent.

### Changement de voix — WAV 24 kHz mono, 120 s, 1 extrait de 13 s

| Carte | VRAM | Jobs | Mémoire | % carte |
|---|---|---|---|---|
| RTX PRO 4000 Blackwell | 25,2 | 4 | 9,73 | 39 % |
| A100 SXM4 | 85,1 | 20 | 46,19 | 54 % |

**La conversion vocale ne remplit jamais la carte.** À vingt jobs sur 85 Go,
la moitié reste libre. Sa limite est ailleurs : le délai de l'hébergeur (900 s
chez Modal) et le transport.

### Gain de débit

| Situation | Gain |
|---|---|
| Séparation, une carte, réseau rapide | ×1,09 à ×1,33 |
| Séparation, une carte, réseau lent | ×1,78 |
| Séparation, **deux cartes** | **×2,31** |
| Conversion vocale, une carte | ×1,02 à ×1,62 |

Un gain proche de ×1 veut dire que le GPU était déjà saturé. Un gain élevé sur
une seule carte ne signale pas une bonne carte : il signale que le GPU dormait
pendant le téléchargement, donc une machine au réseau lent.

### Coût par job

| Tâche | Vast.ai (pod plein) | Modal L4 | Rapport |
|---|---|---|---|
| Séparation | 0,0024 à 0,0029 $ | 0,0149 $ | 5 à 6× |
| Conversion vocale | 0,0050 $ | 0,0233 $ | 4,6× |

Ces chiffres supposent le pod **plein**. Un pod se paie à l'heure même à vide :
le seuil de rentabilité est d'environ **16 % d'occupation** pour la séparation
et **22 %** pour la conversion vocale. En dessous, Modal coûte moins cher.

---

## Ce qui borne, et ce qui ne borne pas

**La VRAM borne la séparation.** 4 à 4,9 Go par job, et le coût par job *baisse*
quand on empile — 4,88 Go à cinq jobs, 3,80 à vingt. L'allocateur de PyTorch
réutilise ses blocs de mieux en mieux. Une carte se remplit jusqu'à 91–99 %
sans incident.

**La VRAM ne borne pas la conversion vocale.** Elle s'arrête bien avant, sur le
temps de traitement et sur le transport.

**Le processeur peut border avant la carte.** Non observé jusqu'à 25 jobs sur
une machine à 32 cœurs, mais une machine faible le montre autrement : la RTX
3090 à 0,22 $/h mettait 77 s par conversion contre 24 s ailleurs, avec un gain
de parallélisme de ×2,27 — preuve que le GPU dormait.

**Le réseau borne aussi.** Sur une machine lente, trois téléchargements
simultanés de 13 Mo ont échoué (`IncompleteRead`).

**Une seule carte est utilisée si on ne fait rien.** Voir plus bas.

---

## Les pièges de mesure

Quatre erreurs ont produit des chiffres faux avant d'être trouvées. Elles sont
documentées parce qu'elles se reproduiront.

### 1. Deux tâches sur le même pod ne se mesurent pas

L'allocateur de PyTorch ne rend **jamais** ce qu'il a réservé, et
`reset_peak_memory_stats` remet le compteur au réservé *courant*, pas à zéro.
Enchaîner conversion puis séparation sur un même pod fait donc relever à la
seconde la somme des deux.

Conséquence vécue : une séparation semblait coûter 11 Go sur une carte, 14 sur
une autre, 17 sur une troisième — et on en concluait que le coût dépendait de la
carte. Le chiffre vrai est **4,96 Go partout**. L'écart était le pool de
conversion vocale resté résident.

**Règle : une tâche par pod, toujours.** `mesure_parallele.sh` prend un
paramètre `vc | sep | both`.

### 2. Un MP3 n'est pas l'entrée de la production

La production envoie des WAV : chunk de conversion de 120 s à 24 kHz mono
(5,8 Mo), tranche de séparation de 150 s à la fréquence d'origine (13 Mo). Un
MP3 de 1,2 Mo masque le téléchargement, donc le temps où le GPU dort, donc tout
le gain du parallélisme.

Mesuré : avec le vrai WAV, le parallélisme rend ×1,78 à trois séparations au
lieu de ×1,11.

### 3. Une connexion HTTP tenue ouverte se fait couper

Le banc appelait `/run` en synchrone et gardait la connexion pendant tout le
job. Au-delà d'environ 200 s de silence, elle est coupée — 4 jobs perdus sur 24,
puis 2 sur 20, avec des `ConnectionResetError`. Ces pertes n'avaient rien à voir
avec la carte, mais elles contaminaient la mesure.

Corrigé : le banc passe par `/submit` puis `/result` (route ajoutée pour ça), et
la file d'accueil du serveur est passée de 5 — le défaut de la bibliothèque
standard Python — à 128.

**La production n'a jamais eu ce problème** : Modal rend un `call_id`, RunPod
passe par un webhook, l'image raconte son travail à `gpu-event`. C'est un
argument de plus pour ne jamais tenir une connexion pendant un job.

### 4. Le pic d'une carte divisé par les jobs de toutes

Sur une machine à deux cartes, le banc annonçait 2,29 Go par job. Faux : le pic
est celui d'**une** carte alors que les dix jobs se répartissent sur **deux**.
Le coût réel est 4,58 Go, cohérent avec la mesure sur carte unique.

### 5. Le job d'échauffement

Le tout premier job d'un worker charge le modèle (des dizaines de secondes).
Sans le jeter, la vague de 1 le paie et les suivantes non, ce qui gonfle le gain
du parallélisme — ×2,44 annoncé pour ×1,2 réel.

### Et le principe qui en découle

**Aucun palier n'est choisi à la main.** `scripts/bench_concurrence.py` monte
jusqu'à ce qu'une vague casse, resserre par dichotomie, et rapporte le dernier
palier qui a tenu entièrement. La prédiction par la mémoire ne sert qu'à viser
le palier suivant, jamais à conclure — parce qu'elle se trompe : le coût par job
n'est pas constant.

---

## Louer une machine

### Ce qui disqualifie une carte

**La capacité de calcul, pas la version CUDA du pilote.** torch 2.7.1+cu128 est
compilé pour `['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120','compute_120']`.
Un V100 32 Go à 0,041 $/h, sur un pilote CUDA 13.0, a été refusé : il est en
capacité 7.0. À louer : T4, RTX 20xx/30xx/40xx/50xx, A10, A100, A6000, L40S,
H100, RTX PRO 6000. Jamais V100, P100, P40.

### Les filtres retenus

| Critère | Valeur | Pourquoi |
|---|---|---|
| `compute_cap` | ≥ 750 | roues torch |
| `gpu_ram` | ≥ 24 000 Mo par carte | `SPARK_MIN_VRAM_GB` |
| `cpu_cores_effective` | ≥ 8 | le CPU plafonne avant la carte sur les machines maigres |
| `inet_down` | ≥ 2000 Mbps | l'entrée fait 13 Mo par job |
| `reliability2` | ≥ 0,97 | |
| `verified`, `rentable` | vrai | |

`VAST_GPUS` choisit le nombre de cartes par machine.

**Le filtre `gpu_name` de l'API est ignoré** : elle rend d'autres modèles. Il
faut filtrer côté client sur le nom rendu.

### Ce qu'il faut savoir d'autre

Un pod met **jusqu'à neuf minutes** à être joignable, presque tout en
téléchargement des 5,3 Go de l'image. Une machine qui l'a déjà la ressert en
quelques secondes. Un pod ne répondra donc jamais à un pic de charge.

Une offre listée peut être prise dans la minute (`no_such_ask`) : il faut
essayer les offres en séquence.

**Un conteneur qui sort en erreur est relancé en boucle par Vast.ai, toutes les
12 secondes, et facturé jusqu'à la destruction du pod.** L'arrêt sur inactivité
de l'image (`SPARK_IDLE_EXIT_S`) ne suffit donc pas : seule la destruction par
l'API arrête les frais.

### L'inventaire du 2026-09-12

457 offres, 79 configurations. De 24 à 179 Go par carte ; jusqu'à 12 cartes par
machine (288 Go) et 4 × RTX PRO 6000 (384 Go). Prix planchers observés :
24 Go à 0,125 $/h, 48 Go à 0,255, 80 Go à 0,951, 96 Go à 1,202, 140 Go à 3,789,
179 Go (B200) à 5,315.

---

## Ce que l'image fait maintenant

### Le nombre de jobs se déduit de la carte et de la tâche

`tasks.jobs_pour_vram(modèle, vram, force, plafond)` — pure, testable sans GPU.
Le calcul est **par tâche** : une séparation coûte 4,8 Go de plus par job, une
conversion 1,3.

**La déduction est optionnelle, activée par `SPARK_JOBS_AUTO`**, que seul
`Dockerfile.vast` pose. `tasks.py` est partagé par les trois hébergeurs, et ni
Modal ni RunPod ne posent `SPARK_JOBS_PER_GPU` : sans ce garde-fou ils
seraient passés de 1 à 4 jobs sans que personne l'ait demandé. Un test verrouille
ce point.

### Toutes les cartes de la machine sont utilisées

Avant le correctif, une machine à deux RTX 3090 n'en utilisait qu'une :
`device()` rendait la chaîne `"cuda"`, que PyTorch résout en `cuda:0`, et les
pools étaient indexés par le seul nom du modèle.

Depuis : les pools sont indexés par `(modèle, carte)` ; `lease()` passe la carte
choisie à la fabrique et préfère une carte qui a déjà une instance prête ; la
carte du job vit dans une variable de fil, donc le pic mémoire rapporté est celui
de *sa* carte ; la capacité annoncée est la somme des cartes.

`verifier_carte()` contrôle **toutes** les cartes au démarrage — capacité de
calcul, mémoire, et un vrai `torch.mm` — garde celles qui passent, écarte les
autres avec la raison, et refuse de démarrer s'il n'en reste aucune. Rien ne
garantit qu'une machine louée ait des cartes identiques ni toutes libres.

Les moteurs n'ont rien eu à changer : `VoiceConverter` et `InstrumentalSeparator`
prenaient déjà un `device` en paramètre.

### Le pool d'instances

Indispensable, et pas seulement pour la mémoire : `vc_engine._configure` et
`_set_reference` écrivent la configuration **et la voix de référence** dans le
modèle. Deux conversions simultanées sur une instance partagée se voleraient leur
voix, **en silence**. Chaque job emprunte son exemplaire.

### La facturation

`container_clock` répartit le temps conteneur entre les jobs qui se croisent :
une seconde à trois jobs vaut 0,33 s chacun, et le temps mort revient au suivant.
Sans ça, un conteneur à trois jobs se ferait facturer trois fois son temps.

---

## Ce qui reste à faire

### Mesures manquantes

La batterie n'est pas complète. Manquent : 32, 48, 64, 140 et 179 Go ; la
conversion vocale sur plusieurs tailles ; les machines à plus de deux cartes.
Coût estimé pour compléter : 6 à 8 $ de location. Crédit restant : 3,75 $.

### Calibrage

`COUT_MEMOIRE_GB` vaut aujourd'hui `(0,6 ; 4,8)` pour la séparation et
`(5,0 ; 1,3)` pour la conversion. Les mesures donnent 3,8 à 4,9 Go par
séparation selon l'échelle, et 1,8 à 2,3 pour la conversion. La table est donc
prudente pour l'une et optimiste pour l'autre.

**Une table figée se trompera toujours.** L'image mesure déjà `gpu_mem` à chaque
job : elle a de quoi se calibrer elle-même — annoncer une capacité prudente au
premier lot, puis l'ajuster d'après ce qu'elle a réellement consommé.

`PLAFOND_DEFAUT` vaut 4. Il protégeait le temps d'un job, ce qui n'est pas
l'objectif : le but est de paralléliser au maximum de ce que la VRAM permet. À
relever ou à supprimer.

### Intégration en production

Le point dur n'est pas la mesure, c'est l'unité que compte le `GpuPool` : il
compte des **conteneurs** et traite un conteneur comme un job. Tant que c'est
vrai, une carte de 102 Go vaut une place, comme un L4.

Trois étages à construire :

1. **L'image annonce sa capacité** — fait. `/status` rend `jobs_par_tache`, la
   machine (cœurs, RAM, cartes) et la VRAM totale.
2. **Un conteneur accepte N jobs.** Vast.ai le fait déjà (le worker HTTP accepte
   N requêtes). RunPod non : `runpod.serverless.start({"handler": handler})` sans
   `concurrency_modifier`. Modal non plus : `@app.cls(...)` sans décorateur de
   concurrence. Une ligne chacun — mais ce sont les fichiers que le propriétaire
   a demandé de ne pas toucher, donc c'est sa décision.
3. **Le pool compte des places.** Un pod de 25 séparations vaut 25 places.

Et le modèle visé par le propriétaire : le conteneur annonce sa capacité, reçoit
une **liste** de jobs à traiter en parallèle, rend les résultats un par un, et
les jobs échoués repartent à la file pour être rejoués. Points à trancher avant
d'écrire : lot d'une seule tâche ou mélangé, un lot puis mort ou redemande,
durée du **bail** (un conteneur qui meurt avec 25 jobs en emporte 25), et la
portée de l'authentification qui s'élargit.

### Sécurité et coût

La destruction des pods doit passer par l'API et être pilotée par un cron, avec
une durée de vie maximale en garde-fou. C'est le même piège que les conteneurs
Cloudflare zombies, qui avaient représenté 97 % d'une facture.

---

## Outils

| Outil | Rôle |
|---|---|
| `scripts/bench_concurrence.py` | cherche le plafond, rapporte mémoire et débit |
| `vast_worker.py` | serveur du pod : `/run`, `/submit`, `/result`, `/status`, `/health`, `/shutdown` |
| `Dockerfile.vast` | image Vast : part de l'image de production, ajoute `src` et le worker |
| `.github/workflows/vast-build.yml` | construit `ghcr.io/<dépôt>-vast` |
| scratchpad `vast.mjs` | inventaire, location, logs, destruction |
| scratchpad `mesure_parallele.sh` | loue, mesure une tâche, détruit |

Fichiers de banc, publiés sur `files.dubbingspark.com/tests/gpu/` :
`vc_source_120s.wav`, `vc_ref_13s.wav`, `sep_input_150s.wav`.

Piège de construction : le workflow Vast a besoin de l'étape « Free disk space »,
sinon le runner meurt **sans log** pendant le tirage des 5,3 Go de l'image de base.
