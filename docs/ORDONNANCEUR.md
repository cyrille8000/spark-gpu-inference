# L'ordonnanceur GPU — comment ça marche, en clair

Proposition au 2026-09-15, à valider avant d'écrire. Rien de ce qui suit n'existe
encore côté serveur. Côté image, les workers savent déjà tout faire de ce que ce
document leur demande (voir « Où on en est »).

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

Un job dure au plus deux minutes de GPU. C'est ce qui rend tout le reste simple :
couper un worker coûte au pire un job.

---

## Ce que fait un worker, et rien d'autre

- Il compte ses cartes et ouvre deux places par carte. Il les garde pleines.
- Il **ne sort jamais de lui-même**. File vide : il attend `attente_s` et redemande.
- Il obéit à `arret` : `doux` = finir ce qui tourne puis sortir ; `net` = rendre ce
  qui est fini, déclarer ce qu'il abandonne, sortir tout de suite.
- Sur Modal et RunPod, l'hébergeur le coupe à **5 h** (réglage choisi). Le bot lui passe
  donc `budget_s` = 5 h moins 5 min : à 4 h 55 il cesse de prendre, finit, sort. C'est un
  relais planifié, pas une coupe, et le worker se protège seul même sans serveur.
- À chaque demande il se présente : hébergeur, identifiant d'instance, image, démarré
  à, cartes, et l'avancement de chacun de ses jobs. C'est avec ça que le bot décide.
- Homme-mort : 5 min sans serveur, il cesse de prendre ; 15 min, il se tue.

---

## Le rythme : des lots de 10 minutes

Les demandes des utilisateurs ne partent pas une à une. Quand un utilisateur demande,
ses jobs sont créés et posés dans la file. **Toutes les 10 minutes**, le bot prend tout
ce qui s'est accumulé et lance le traitement du lot.

Ce que ça donne :

- le bot **planifie un lot entier** : il connaît le nombre de jobs, les tâches, les
  durées de média, et achète d'un coup la capacité pour finir le lot **avant le lot
  suivant** — c'est la cible d'attente ;
- en cours de lot il ne fait que corriger : rajouter si ça traîne, couper si ça va
  plus vite que prévu ;
- entre deux lots il décide qui reste chaud : Modal oui (crédit gratuit), RunPod et
  Vast seulement si 10 min d'attente coûtent moins qu'un redémarrage ;
- un job posé pendant un lot attend le tick suivant.

---

## Ce que décide le bot

Trois décisions, une seule règle : **des dollars contre des dollars**.

**Allumer.** Quand la file ne se videra pas assez vite avec ce qui tourne, il ajoute
des workers, autant qu'il en faut, chez le moins cher qui a de la place. Une machine
en cours de démarrage compte déjà, avec son heure d'arrivée prévue. Si la file se
vide avant qu'elle arrive, il l'annule tant que c'est gratuit.

**Éteindre.** Un worker inactif coûte (à la milliseconde sur Modal et RunPod, à
l'heure sur Vast). Le bot le garde chaud tant que ça coûte moins qu'un redémarrage,
puis l'éteint. Sur Vast, éteindre = détruire par l'API, sinon le disque se paie.

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
| Prix | **0 tant qu'il reste du crédit** (6 comptes × 30 $/mois, renouvelé) | réel | prix de l'offre |
| Capacité | 10 cartes par compte, 1 carte par worker | 20 workers × 4 cartes | sans plafond, machines à 1..12 cartes |
| Démarrage | ~1 min | 20-30 s à chaud, jusqu'à 8 min à froid, **facturé** | 20 s si l'image est en cache, sinon minutes ; tirage gratuit |
| Éteindre | `cancel` | `cancel` ; sinon idle timeout | `destroy` par l'API |
| Ce que le bot surveille | crédit restant du mois | solde, temps de démarrage observés | solde, machines connues avec l'image en cache |

Conséquence : Modal est le plancher gratuit, toujours rempli en premier. RunPod et
Vast ne servent que les pics et les longues charges. Sur Vast, le bot choisit
l'offre au **coût par job** (une 3090 bon marché a coûté plus cher à l'usage qu'une
Blackwell), préfère une machine qui a déjà l'image, et retient la taille de machine
d'après la part de travail qu'il lui donne.

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
- Remise en file seulement sur `abandonnes` ou après expiration sans battement. Un job
  ne tourne deux fois que si son worker est mort en plein calcul ; les sorties sont
  idempotentes (même clé R2).

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
- Les clés des hébergeurs ne vivent que dans le bot (Doppler), avec un plafond de
  dépense par jour et un journal de chaque démarrage, arrêt et location.

**Choix des machines Vast** : `verified` et fiabilité ≥ 0,97 seulement, liste noire des
machines qui ont refusé l'image ou rendu un résultat invalide.

**Limite qu'aucun code ne lève** : sur Vast, l'opérateur de la machine peut lire ce que
le conteneur traite (l'audio et la vidéo de l'utilisateur) et pourrait rendre un
résultat corrompu — le contrôle de plausibilité attrape le grossier, pas le subtil.
Modal et RunPod sont des datacenters ; Vast, un particulier ou une petite société.
Voir la question 6.

---

## Les garde-fous

- Plafond de dépense Vast par jour, plafond de workers par fournisseur, budget Modal
  par compte — dans Doppler, coupe-circuit compris.
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
| Image : prise, attente, `arret` doux/net, identité, avancement par job, `SPARK_CLAIM_URL` sur Vast | **écrit et testé**, non déployé (commit local) |
| File, registre des workers, prédiction, décision, actions par API, branchement des workflows | **à écrire** |
| Référence pour les appels API Vast, RunPod, Modal | le vieil orchestrateur Python de l'OCI (`runpod-ytdlp-manifest`) : à recopier, pas à réveiller |
| Réglages à changer au déploiement | coupures Modal (`modal_app.py`, 900 s) et RunPod (`executionTimeout` 900 s côté backend) à porter à **5 h** ; le bot passe `budget_s` = 5 h − 5 min à chaque worker |

---

## Ce que je te demande de trancher

1. **Cible d'attente** : un lot doit-il toujours finir dans ses 10 minutes, ou le coût peut-il primer et laisser déborder ?
2. **Période de grâce** d'un worker inactif entre deux lots : fixée par fournisseur, ou calculée par le bot d'après les lots précédents ?
7. **Le lot vaut pour toutes les tâches** (y compris le changement de voix lancé depuis le studio), ou certaines partent tout de suite ?
8. **Les 10 minutes sont fixes** (cron), ou le bot peut déclencher plus tôt quand le lot est déjà gros ?
3. **Durée max d'un job** : 2 minutes de GPU te va ?
4. **Crédit Modal** : on le dépense dès qu'il y a du travail, sans le garder en réserve ?
5. **Plafond Vast par jour**, en dollars.
6. **Vast et la confidentialité** : on y envoie toutes les tâches en filtrant sur `verified`
   et la fiabilité, ou seulement certaines ?
