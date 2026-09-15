# L'ordonnanceur GPU — comment ça marche, en clair

Tranché le 2026-09-15 et **écrit le même jour** : le bot vit dans le worker Cloudflare
`spark-dubbing-api` (Durable Object `GpuScheduler`, dossier `src/gpu-scheduler/`), testé
par 41 tests et un simulateur ; rien n'est déployé (voir « Où on en est »).

---

## En une phrase

Une **file** de petits jobs, des **workers** qui viennent y piocher, et un **bot**
qui décide quand allumer une machine, laquelle, et quand l'éteindre — toujours en
comparant des dollars.

---

## Les trois acteurs

```
 workflows (séparation, voix, visages)
        │  posent des jobs
        ▼
   ┌─────────┐   « donne-moi du travail »    ┌──────────────┐
   │  FILE   │ ◄──────────────────────────── │   WORKERS    │  Modal, RunPod, Vast
   │  (D1)   │ ────────────────────────────► │  (l'image)   │
   └─────────┘   jobs + ordre d'attendre     └──────────────┘
        ▲            ou de s'arrêter                ▲
        │                                           │ allume / éteint
   ┌─────────┐                                      │ par l'API de l'hébergeur
   │   BOT   │ ─────────────────────────────────────┘
   │  (DO)   │   regarde la file, les workers, les prix
   └─────────┘
```

| Acteur | Où il vit | Ce qu'il fait |
|---|---|---|
| La file | table D1 `gpu_jobs` + Durable Object | garde chaque job, son état, qui l'a pris, son résultat |
| Les workers | l'image `spark-gpu-inference`, chez Modal, RunPod ou Vast | piochent, calculent, rendent, attendent |
| Le bot | un Durable Object Cloudflare (alarme) | monte, descend, coupe, achète sur Vast |

---

## Un job, de bout en bout

1. Un workflow pose un job dans la file : tâche, URLs, projet, durée du média.
2. Un worker demande du travail. La file lui réserve des jobs, autant qu'il a de places libres.
3. Le worker calcule. Toutes les 30 s il dit où il en est (pourcentage, temps écoulé).
4. Le job fini, le worker envoie le résultat, et redemande dans la même requête.
5. La file marque le job fini, prévient le workflow, écrit la consommation.
6. Plus de jobs ? Le worker attend le temps que la file lui dit, puis redemande.

Un job dure au plus **deux minutes d'inférence GPU réelle** — hors téléchargement,
décodage, encodage et envoi, qui ne comptent pas (tranché le 2026-09-15). C'est ce qui
rend tout le reste simple : couper un worker coûte au pire un job. Le worker rend pour
chaque job son temps d'inférence à part : c'est lui qui nourrit le débit par carte.

---

## Ce que fait un worker, et rien d'autre

- Il compte ses cartes et lit la mémoire de chacune : **moins de 24 Go, une place ; 24 Go
  et plus, deux**. C'est lui qui décide, pas le serveur, et la règle est la même chez les
  trois hébergeurs. Il garde ses places pleines.
- Il **ne sort jamais de lui-même**. File vide : il attend `attente_s` et redemande.
- Il obéit à `arret` : `doux` = finir ce qui tourne puis sortir ; `net` = rendre ce
  qui est fini, déclarer ce qu'il abandonne, sortir tout de suite.
- Sur Modal et RunPod, l'hébergeur le coupe à **5 h** (réglage choisi). Le bot lui passe
  donc `budget_s` = 5 h moins 5 min : à 4 h 55 il cesse de prendre, finit, sort. C'est un
  relais planifié, pas une coupe, et le worker se protège seul même sans serveur.
- À chaque demande il se présente : hébergeur, identifiant d'instance, image, démarré
  à, cartes, et l'avancement de chacun de ses jobs. C'est avec ça que le bot décide.
- Homme-mort, par sa propre horloge : 5 min sans serveur, il cesse de prendre ; 15 min, il
  se tue sur Modal et RunPod (le conteneur finit, la facturation aussi). Sur Vast il ne se
  tue pas — un conteneur qui sort y est relancé en boucle, facturé — il se tait, et c'est le
  balai du serveur qui détruit le pod par l'API dès que le serveur est de retour.

---

## Le rythme : 10 minutes au réveil, puis au fil de l'eau

Le 10 minutes n'est pas un rythme, c'est un **démarrage à froid**.

1. La file est vide depuis un moment : tout dort, aucun worker payant.
2. Un premier job arrive. On laisse la file se remplir **jusqu'à 10 minutes** avant de
   démarrer, pour ne pas payer un démarrage de machine pour un seul job. Si la file
   devient grosse avant — plus de N minutes de GPU accumulées, réglage Doppler — le bot
   démarre sans attendre la fin des 10 minutes (tranché le 2026-09-15).
3. On démarre. À partir de là, **tout ce qui arrive est traité au fil de l'eau** : les
   workers piochent en continu, le bot ajuste la capacité en continu.
4. Dès que la file d'attente est vide, tout worker qui n'a rien en main s'arrête. Quand le
   dernier s'éteint, on dort ; le 10 minutes se réarme au prochain job.

Le réveil vaut pour **toutes les tâches** : séparation, changement de voix lancé depuis
le studio, visages. Seule la voie express y échappe (tranché le 2026-09-15).

En croisière, la règle du bot est de **garder la file courte au meilleur coût**
(proposition, question 1) :

- **Modal, tant qu'il a du crédit** : la file doit rester à zéro. Le bot ouvre autant de
  conteneurs qu'il faut pour que ce qui attend soit vidé dans le temps d'un démarrage après
  leur arrivée (≈ 1 min, jamais moins qu'un job), et **jamais plus de places libres que de jobs
  en file** : 1 job → 1 conteneur, 4 jobs → 2. Le crédit est dépensé dès qu'il y a du travail,
  sans réserve.
- **RunPod et Vast** : on démarre une machine de plus seulement si, **à son arrivée
  prévue**, il lui restera au moins `max(5 min, 3 × son démarrage facturé)` de travail.
  Le 5 min est la file tolérée (Doppler, même seuil que l'express) ; le ×3 garantit qu'un
  démarrage payé ne dépasse jamais un quart de ce que la machine produit. Une machine
  RunPod froide (8 min facturés) exige donc 24 min de travail devant elle ; une machine
  Vast qui a l'image (20 s) part pour 5 min de file.
- **Couper** : dès que le vidage prévu repasse sous la cible sans elle, la machine la plus
  chère par place passe en grâce (ci-dessous), puis s'éteint.

Ordre de grandeur mesuré : une heure de vidéo = 24 séparations de 150 s ; sur un L4 ça fait
≈ 20 min de carte (56 s par job, le parallélisme n'y rend que ×1,15), sur les 10 cartes d'un
compte Modal ≈ 2 min.

**Voie express** : un projet dont le média total fait moins de 5 minutes (seuil Doppler)
n'attend pas les 10 minutes du réveil. Ses jobs sont prenables tout de suite et servis
AVANT les autres. Le critère est la durée du projet, pas celle des jobs découpés.

- un worker allumé a une place libre, **quel que soit l'hébergeur** → il prend l'express
  dans la seconde, ce qui est allumé ne coûte rien de plus ;
- aucun worker allumé → le bot en démarre un **chez Modal seulement**, s'il a du crédit ;
- ni worker allumé ni crédit Modal → l'express attend le réveil comme les autres. On ne
  démarre jamais une machine RunPod ou Vast pour aller vite.

---

## Ce que décide le bot

Trois décisions, une seule règle : **des dollars contre des dollars**.

**Allumer.** Quand la file ne se videra pas assez vite avec ce qui tourne, il ajoute
des workers, autant qu'il en faut, chez le moins cher qui a de la place. Une machine
en cours de démarrage compte déjà, avec son heure d'arrivée prévue. Si la file se
vide avant qu'elle arrive, il l'annule tant que c'est gratuit.

**Éteindre.** Un worker inactif coûte, à la seconde près, chez les trois. Sur Vast c'est
la machine entière, toutes ses cartes, tant que le pod existe. Sur Vast, éteindre = détruire
par l'API, sinon le disque se paie encore.

- **File d'attente vide : tout s'arrête, tout de suite** (tranché le 2026-09-15). Tout worker
  qui n'a rien en main reçoit l'arrêt, le dernier compris, même si d'autres finissent encore
  leurs jobs ; ceux-là s'arrêtent à leur tour en finissant. Un démarrage en route est annulé.
  Mesuré au banc : le dernier worker s'éteint 20 s (un tick) après le dernier job.
- **Une seule exception** : un arrêt net en cours. Ses jobs abandonnés vont revenir en file,
  la cible de la coupe les attend.
- **La grâce** (le prix d'un redémarrage observé, 1 à 10 min) ne joue plus que dans un cas
  rare : la file ne contient que des jobs que ce worker a déjà ratés et ne peut pas reprendre.
- **Conséquence assumée** : sur un flux clairsemé (un job toutes les 3 min), chaque job trouve
  tout éteint et repasse par le réveil. Mesuré : attente moyenne ≈ 7 min contre ≈ 2 min sur un
  flux d'un job par minute. Le réglage du réveil (`SPARK_GPU_REVEIL_S`) est le curseur.
- **Le sommeil n'a pas d'horloge à lui** : dormir, c'est n'avoir plus aucun worker et une
  file vide ; le prochain job réarme les 10 min.

**Couper.** Un worker cher qui n'a plus qu'un job, et un worker moins cher déjà
allumé et libre ? Garder coûte `prix × temps restant` ; déplacer coûte
`prix_cible × temps complet`. Si garder coûte plus, il coupe en `net` et le job
repart ailleurs. Le progrès du job est dans le calcul : à 90 % on garde, à 30 %
on coupe. Un job coupé une fois ne l'est plus jamais.

---

## Le prix d'une place, par fournisseur

`coût par seconde de job = prix horaire ÷ 3600 × (1 + surcoût de démarrage) ÷ (places × débit)`

| | Modal | RunPod | Vast.ai |
|---|---|---|---|
| Prix | **0 tant qu'il reste du crédit** (6 comptes × 30 $/mois, renouvelé), dépensé dès qu'il y a du travail, sans réserve (tranché) | réel, à la seconde | prix de l'offre, à la seconde, machine entière |
| Capacité | 10 cartes par compte, 1 carte par worker | 20 workers × 4 cartes | sans plafond ; machines de 1 à 12 cartes et plus, **aucune taille exclue** |
| Démarrage | ~1 min | 20-30 s à chaud, jusqu'à 8 min à froid, **facturé** | 20 s si l'image est en cache, sinon minutes ; tirage gratuit |
| Éteindre | `cancel` | `cancel` ; sinon idle timeout | `destroy` par l'API |
| Ce que le bot surveille | crédit restant du mois | **solde lu par l'API**, temps de démarrage observés | **solde lu par l'API**, machines connues avec l'image en cache |

Conséquence : Modal est le plancher gratuit, toujours rempli en premier. RunPod et
Vast ne servent que les pics et les longues charges. Sur Vast, le bot ne se limite à
aucun nombre de cartes : il classe les offres au **débit par dollar** — ce qui traite le
plus pour le moins cher (une 3090 bon marché a coûté plus cher à l'usage qu'une
Blackwell ; deux cartes rendent ×2,3) — préfère une machine qui a déjà l'image, et ne
prend une grosse machine que s'il a de quoi la remplir.

### Choisir une machine Vast

**Compatibilité, pas une liste de cartes.** Le filtre garde toute machine que l'image
peut exécuter, quel que soit le nom de la carte :

| Critère | Seuil | Pourquoi |
|---|---|---|
| capacité de calcul | ≥ 7,5 | les roues torch 2.7.1 + CUDA 12.8 (un V100 est refusé, même sur pilote récent) |
| pilote `cuda_max_good` | ≥ 12.8 | l'image embarque CUDA 12.8 |
| VRAM par carte | ≥ 16 Go | de 16 à 22 Go la carte ne tient qu'**une place** (tranché le 2026-09-15) ; à partir de 24 Go la mémoire décide du nombre de places |
| cœurs CPU | plancher 4, puis facteur de coût | le décodage et l'encodage tournent sur CPU ; en dessous la carte dort |
| débit réseau | plancher 500 Mb/s, puis facteur de coût | 13 Mo par job (0,2 s à 500 Mb/s) ; le tirage de l'image passe de 20 s à 85 s, c'est du temps de démarrage, pas de la qualité |
| disque | ≥ 15 Go | l'image fait 5,3 Go compressés |
| `verified` | oui | tranché |
| fiabilité | plancher 0,80, puis facteur de coût | une machine moins fiable est classée un peu plus chère (risque de refaire des jobs), pas exclue |

Mesuré le 2026-09-15 sur l'API : 514 machines compatibles à 24 Go ; avec des seuils stricts
(CPU 8, 2 Gb/s, fiabilité 0,90) il n'en restait que 268, presque tout perdu sur le réseau ;
avec les planchers souples, 477 ; en acceptant 16 Go, **608**. La version CUDA, elle, n'écarte
que 76 machines. Trop de filtre, plus de machine — c'est la formule qui départage.

Le filtre `gpu_name` de l'API Vast est ignoré par Vast : on trie côté client. Les
machines qui ont refusé l'image ou rendu un résultat invalide vont en liste noire.

**Le coût complet d'une machine pour un lot**, pas seulement son prix horaire :

`coût = durée prévue × (prix horaire total + disque alloué × prix stockage) + octets entrants × prix entrée + octets sortants × prix sortie`

- la durée prévue vient du **débit** de la machine : mesuré si on la connaît (mémoire des
  machines), sinon celui de sa classe de carte, corrigé par le CPU et le réseau ;
- les octets viennent des jobs du lot (13 Mo par séparation, 6 par voix, plus pour les
  visages) ;
- le tirage de l'image n'est pas facturé en temps ; sa bande passante, à vérifier.

**Dans l'ordre** : l'appel à l'API part avec les filtres ci-dessus et rend la liste **triée par
prix**, le moins cher en premier. Sur les meilleures offres de cette liste, le bot recalcule
le **coût par job** avec la formule, et retient la première machine qu'on peut remplir. La
**bande passante** y pèse deux fois : comme filtre, et comme facteur de débit et de coût de
transfert — à prix égal, la machine au meilleur réseau gagne. C'est ce qui fait qu'une offre
à 0,12 $/h peut perdre contre une à 0,60.

---

## Ce que le bot sait à chaque instant

| Donnée | D'où elle vient |
|---|---|
| Profondeur de la file, par tâche et par projet | la file |
| Workers vivants, places libres, jobs en vol et leur avancement | chaque demande de prise |
| Durée moyenne d'un job par tâche et par carte | l'historique des jobs finis |
| Temps de démarrage par fournisseur et par état (chaud, froid, cache) | chaque démarrage observé |
| Prix, crédits, soldes | Doppler + API des hébergeurs |
| Machines Vast connues : carte, prix vu, image en cache, débit mesuré | la mémoire des machines (D1) |

Le bot **apprend** : chaque démarrage, chaque job, chaque coupe nourrit la décision
suivante. Rien n'est une constante.

---

## Un job, un seul worker

- La réservation se fait dans le Durable Object, qui traite ses requêtes une par une :
  deux workers ne reçoivent jamais le même job.
- Chaque réservation porte un jeton. Un résultat n'est accepté que du worker qui la
  tient ; un résultat en retard d'un worker déclaré mort est refusé.
- Remise en file seulement sur `abandonnes` ou après 15 min sans battement — les mêmes
  15 min que l'homme-mort du worker, exprès : quand le serveur requeue, le worker a déjà
  lâché. Un job ne tourne deux fois que si son worker est mort en plein calcul ; les
  sorties sont idempotentes (même clé R2).
- Panne du serveur : le worker garde ses résultats en attente et les livre au premier
  contact réussi ; un résultat arrivé après le requeue est refusé.

---

## Sécurité : des machines qui ne sont pas à nous

Principe : **une machine louée est un inconnu**. Elle ne reçoit que ce qu'il faut pour
son travail, jamais de quoi en faire un autre, et tout ce qu'elle rend est vérifié.

**Ce que le worker reçoit**

| Ce qu'il a | Ce que ça permet | Ce que ça ne permet pas |
|---|---|---|
| une URL de prise signée (HMAC), liée à SON identité, valable quelques heures, renouvelée par le serveur à chaque réponse | demander ses jobs, rendre ses résultats | prendre les jobs d'un autre worker, parler à autre chose que la route de prise |
| par job, un ticket de lecture du média et un PUT présigné de sortie, courte durée | lire cette entrée, écrire cette sortie | lire le reste du projet, lister R2, écraser autre chose |
| rien d'autre | | pas de clé R2, Doppler, Modal, RunPod, Vast, Discord dans l'image |

**Ce que le serveur exige**

- Signature vérifiée sur chaque demande ; identité du worker (instance, image) figée à
  la première prise, une divergence coupe le worker.
- Un résultat n'est accepté que du worker qui tient la réservation, une seule fois
  (jeton de réservation : pas de rejeu, pas de résultat en retard d'un worker mort).
- Limite de requêtes par worker ; une URL révocable par identifiant à tout moment.
- Ce qui revient est contrôlé avant d'être utilisé : taille annoncée, en-tête WAV ou
  JSON, sha256, durée plausible par rapport au média d'entrée.

**Ce que la machine ne peut pas faire**

- Rien n'entre : sur Vast en prise, aucun port ouvert, aucun jeton de worker ; le pod ne
  fait que sortir vers notre API. Sur Modal et RunPod, pareil.
- L'image est désignée par empreinte (`@sha256`) : un hôte ne peut pas en substituer une.
  L'image est publique et ne contient aucun secret : la lire n'apprend rien.
- Les clés des hébergeurs ne vivent que dans le bot (Doppler), avec un journal de chaque
  démarrage, arrêt et location.

**Choix des machines Vast** (tranché le 2026-09-15) : **toutes les tâches** peuvent y aller,
sur des machines de confiance (`verified`) et compatibles avec l'image (capacité de calcul et
version CUDA du pilote, cf. « Choisir une machine Vast ») pour ne pas planter. La fiabilité
n'est pas une barrière — trop stricte, on ne trouverait plus de machine — mais un facteur du
classement au coût. Liste noire des machines qui ont refusé l'image ou rendu un résultat
invalide.

**Limite qu'aucun code ne lève** : sur Vast, l'opérateur de la machine peut lire ce que
le conteneur traite (l'audio et la vidéo de l'utilisateur) et pourrait rendre un
résultat corrompu — le contrôle de plausibilité attrape le grossier, pas le subtil.
Modal et RunPod sont des datacenters ; Vast, un particulier ou une petite société.
Le propriétaire accepte ce risque avec les filtres ci-dessus.

---

## Les garde-fous

- **Pas de plafond par jour : le solde est la limite** (tranché le 2026-09-15). RunPod et
  Vast sont prépayés et leur API donne le crédit restant ; le bot le lit avant chaque
  démarrage, n'engage jamais plus qu'il ne reste, et alerte sous un seuil. Budget Modal par
  compte, plafond de workers par fournisseur et coupe-circuit restent dans Doppler.
- Balai toutes les 3 minutes : toute instance Vast inconnue du registre est détruite.
- Un worker coupé rend ses résultats avant de sortir ; rien de fini n'est perdu.
- Un job abandonné repart en tête de file et ne peut plus être coupé.
- Alertes Discord dédoublonnées sur toute anomalie (solde bas, worker muet, machine refusée).

---

## Comment on prouve que c'est solide

Un **simulateur** dans les tests du bot : des scénarios (rafale de 100 portions,
charge continue, worker qui se tait, fournisseur en panne, prix qui bougent) et,
pour chacun, le coût en dollars de la décision du bot contre celui d'une règle
naïve. Une règle qui ne bat pas la naïve ne rentre pas.

Puis un **faux worker** en script, qui pioche et rend avec des durées réalistes :
le bot se teste de bout en bout sans allumer une seule carte.

---

## Où on en est

| | État |
|---|---|
| Image : prise, attente, `arret` doux/net, identité, avancement par job, `SPARK_CLAIM_URL` sur Vast, `inference_s` et dépôt dans chaque résumé, URL de prise renouvelée | **écrit et testé** (24 tests de prise), non déployé (commit local) |
| Bot : file, prise signée, registre, prévision, décision, actions Modal/RunPod/Vast, comptes, balai, statut | **écrit et testé** (41 tests + simulateur), non déployé (commit local) |
| Branchement des workflows (séparation, changement de voix) | **écrit**, derrière `SPARK_GPU_SCHEDULER_ENABLED` (faux par défaut : l'ancien chemin GpuPool reste actif) |
| Visages qui parlent | la file l'accepte (`speaking_faces`) ; aucun workflow ne la pose encore |
| Où ça tourne | **tout sur Cloudflare** (Workers, Durable Objects, D1, cron) ; l'OCI ne fait plus partie de l'architecture (tranché le 2026-09-15) |
| Réglages à changer au déploiement | coupures Modal (`modal_app.py`, 900 s) et RunPod (`executionTimeout` 900 s côté backend) à porter à **5 h** ; le bot passe `budget_s` = 5 h − 5 min à chaque worker |

---

## Où c'est, comment ça se règle, comment on le teste

**Le code** (`backend/cloudflare/workers/spark-dubbing-api`) :

| Fichier | Rôle |
|---|---|
| `src/durable-objects/GpuScheduler.js` | le bot : `/poser`, `/claim`, `/job`, `/status`, l'alarme (un tick toutes les 20 s tant qu'il y a un job ou un worker) |
| `src/gpu-scheduler/file.js` | la file : réservation avec jeton, résultats, abandons, relances (3 au plus), remise en file |
| `src/gpu-scheduler/workers.js` | le registre : identité figée au premier contact, ordres, inactivité |
| `src/gpu-scheduler/prevision.js` | débits appris (inférence ÷ média, par carte), démarrages appris, temps de vidage |
| `src/gpu-scheduler/decision.js` | les règles, pures : réveil, Modal d'abord, payant au coût par job, annuler, couper, grâce |
| `src/gpu-scheduler/vast-offres.js` | filtre et coût complet d'une offre Vast, mémoire et liste noire des machines |
| `src/gpu-scheduler/actions.js` | démarrer / annuler / tuer par les API, comptes (secondes, dollars, usage Modal), soldes, balai |
| `src/gpu-scheduler/fournisseurs/*.js` | Modal (`submit`/`cancel`), RunPod (`/run`, `/cancel`, solde GraphQL), Vast (offres, louer, détruire, solde) |
| `src/gpu-scheduler/enfant.js` | le chemin des enfants de workflow par la file (poser → attendre → clore et compter) |
| `src/routes/gpu-claim.js` | `POST /api/internal/gpu-claim` (URL signée, limite par worker) et `/api/internal/gpu-scheduler/<status|tick|balai|arreter|degeler|poser>` |
| `scripts/faux-worker-gpu.mjs` (backend) | un worker sans carte qui parle le protocole de prise, pour tester de bout en bout |

**Les réglages** (Doppler `prd_cloudflare-workers`, lus à chaque tick, défauts entre parenthèses) :
`SPARK_GPU_SCHEDULER_ENABLED` (false — bascule les enfants sur la file), `SPARK_GPU_SCHEDULER_PAUSE`,
`SPARK_GPU_CLAIM_SECRET` (sinon `UPLOAD_JWT_SECRET`), `SPARK_GPU_REVEIL_S` (600),
`SPARK_GPU_REVEIL_ANTICIPE_GPU_S` (1800), `SPARK_GPU_FILE_CIBLE_S` (300), `SPARK_GPU_FACTEUR_DEMARRAGE` (3),
`SPARK_GPU_EXPRESS_MAX_S` (300), `SPARK_GPU_GRACE_MIN_S`/`MAX_S` (60/600), `SPARK_GPU_BUDGET_S` (17 700),
`SPARK_GPU_MORT_S` (900), `SPARK_GPU_SOLDE_MIN_USD` (2) ; Modal : `SPARK_GPU_MODAL_ENDPOINT_URLS`, `_API_KEY`,
`_MAX_CONCURRENT` (10), `_BUDGET_USD` (29) ; RunPod : `SPARK_GPU_RUNPOD_API_KEY`, `_PRISE_ENDPOINT_ID`
(sinon `_ENDPOINT_ID`), `_MAX_WORKERS` (20), `_CARTES` (4), `_PRICE_PER_HOUR` (2,76) ; Vast :
`SPARK_GPU_VAST_API_KEY`, `_IMAGE` (par empreinte), `_MAX_WORKERS` (10), `_DISK_GB` (20), `_VRAM_MIN_MB` (16 000),
`_CUDA_MIN` (12.8), `_CPU_MIN` (4), `_INET_MIN_MBPS` (500), `_FIABILITE_MIN` (0,8) ; `SPARK_GPU_PRICES` (table existante).

**Les tests** : `npm test` dans `spark-dubbing-api` (vitest) — la file (jeton, doublons, relances,
abandons, mort), la décision (réveil, express, Modal d'abord, payant, Vast, annuler, grâce, couper,
mort), les offres, la signature, et le **simulateur** (`tests/gpu-scheduler/simulateur.js`) qui fait
tourner la vraie file, le vrai registre et la vraie décision contre de faux hébergeurs, bot contre
règle naïve (une machine RunPod par 40 jobs, arrêt à 15 min) :

| Scénario | Bot | Naïf |
|---|---|---|
| 100 portions, crédit Modal | 0 $ payé (1,16 $ de crédit), fini en 4 min | 3,66 $ |
| 300 portions, RunPod seul (démarrage 8 min) | 1,79 $, une machine, 34 min | 9,87 $, huit machines, 12 min |
| 300 portions, Vast (image en cache) + RunPod | 0,25 $, une machine Vast, 25 min | 9,87 $ |

Le bot est toujours moins cher ; sur du payant à démarrage lent il est plus lent (une seule
machine, la règle des 3 × démarrage) — `SPARK_GPU_FACTEUR_DEMARRAGE` règle ce curseur.

**De bout en bout sans carte** : poser un job à la main (`POST /api/internal/gpu-scheduler/poser`),
lire l'URL de prise du worker que le bot a créé dans `/status` (journal), et lancer
`node scripts/faux-worker-gpu.mjs "<claim_url>"` : il prend, calcule, rend, obéit.

## Ce que je te demande de trancher

Rien : la croisière est acceptée telle quelle ; pour la grâce, le propriétaire a tranché
plus simple le 2026-09-15 — file d'attente vide, tout s'arrête tout de suite, plus de
« dernier worker gardé 5 min ». Reste à faire, dans l'ordre : mettre les clés et réglages dans
Doppler, déployer le backend (le bot dort tant que la file est vide), tester avec le faux
worker, déployer les images (Modal `timeout` 5 h, endpoint RunPod de prise à 5 h, image Vast
par empreinte), puis passer `SPARK_GPU_SCHEDULER_ENABLED` à true.

Tranché le 2026-09-15 : un job = 2 min d'inférence GPU réelle au plus, hors transferts ; le
réveil démarre avant 10 min si la file est déjà grosse ; pas de plafond Vast par jour, le
solde lu par l'API (Vast et RunPod) est la limite ; le crédit Modal se dépense dès qu'il y
a du travail, sans réserve ; tout vit sur Cloudflare, plus d'OCI.
