# Le worker va chercher son travail — architecture et état au 2026-09-15

> **Mise à jour du 2026-09-15.** Le propriétaire a inversé la règle « file vide, on sort » :
> le worker ATTEND désormais (`attente_s` donné par le serveur) et ne sort que sur `arret`.
> C'est l'ordonnanceur — à écrire, voir [ORDONNANCEUR.md](ORDONNANCEUR.md) — qui monte et
> qui descend ; les coupures des hébergeurs seront réglées très hautes. Chaque demande porte
> l'identité du worker (`instance_id`, `machine_id`, `image_tag`, `demarre_a`, `uptime_s`) et
> l'avancement de chaque job en cours (`en_vol[]`). `net` rend d'abord ce qui est fini et
> liste `abandonnes`. Les paragraphes « Deux sorties, jamais une attente » et « Budget
> épuisé » ci-dessous décrivent l'ancien comportement.

## Où on en est

L'image sait faire trois choses : un job, un lot poussé, ou **aller chercher son
travail toute seule**. Le troisième mode est celui qu'on garde.

Tout est écrit et testé — 64 tests. **Rien n'est branché en production** : il manque
la route de prise côté backend, et c'est le prochain chantier.

`modal_app.py` n'est pas touché. `handler.py` a gagné une fonction de concurrence qui
ne sert plus vraiment depuis le mode prise, mais qui ne gêne pas.

---

## Le problème qu'on résout

La plateforme a 80 places GPU simultanées, et un job occupait une place entière. Un
lot de vingt n'en occupe qu'une.

Et surtout : **on demande quatre cartes à RunPod et on en reçoit parfois trois.** Leur
propre contrôle de démarrage le dit — `GPU binary test passed: 3 GPU(s) healthy`, le
2026-09-12, alors que l'endpoint est réglé sur quatre. Le même worker en a vu quatre à
19 h 34 et trois à 20 h 06. Le serveur ne peut donc pas savoir combien de jobs envoyer.

Le worker, lui, sait : il compte ses cartes.

---

## Ce que l'image fait

Trois formes, reconnues à la charge utile. Aucun hébergeur n'a rien à savoir de plus :
`handler.py` et `modal_app.py` appellent `process_job` comme avant.

| Charge utile | Mode |
|---|---|
| `{"claim_url": "…"}` | **prise** — le worker va chercher |
| `{"jobs": [ … ]}` | lot poussé |
| `{"task": "…"}` | un job, comme depuis toujours |

### La prise, en détail

On ne passe qu'une chose au lancement : une **URL signée** — elle porte son
autorisation, le worker n'a aucun secret à connaître.

```json
{"claim_url": "https://api.dubbingspark.com/api/internal/gpu-claim?sig=…",
 "budget_s": 17400, "jobs_par_carte": 2, "battement_s": 30}
```

Le worker compte ses cartes et lit la mémoire de chacune — **une place sous 24 Go, deux à
partir de 24 Go** (2026-09-15 : décidé par lui, `tasks.places_prise`, même règle chez les trois
hébergeurs ; `jobs_par_carte` du serveur ne peut que plafonner) — et les tient **pleines** :

```
→ {"worker":"…", "cartes":["cuda:0","cuda:1","cuda:2"], "places":6,
   "libres":6, "restant_s":17280, "resultats":[ … ]}
← {"jobs":[ {"task":"instrumental","audio_url":"…","output_url":"…"}, … ]}
```

Dès qu'**un seul** sous-job finit, sa place est reprise. On ne travaille pas par
vagues qu'on attendrait en entier : sinon les places libérées par les jobs rapides
attendent le plus lent **en étant facturées**. Mesuré le 2026-09-12 sur RunPod —
quatre places inoccupées pendant 66 s sur une vague de 143 s, un quart du temps payé
pour rien.

Les résultats voyagent **avec** la demande de remplissage : une seule requête dit
« voici ce qui vient de finir, donne-m'en autant ». Le tout dernier envoi, qui n'a
plus de demande où se glisser, part avec `"fin": true` — sans lui le serveur garderait
ces réservations jusqu'à expiration.

### Deux sorties, jamais une attente

**File vide** : le worker laisse finir ce qui tourne et sort. Il ne sonde jamais — un
worker qui attend est facturé à la milliseconde, cartes comprises, alors qu'un
redémarrage sur une machine qui a déjà l'image coûte une vingtaine de secondes
(mesuré : 501 s la première fois, 31, 21 et 16 s ensuite).

**Budget épuisé** : il cesse de reprendre, laisse finir, sort. Il ne reprend un job
que s'il a le temps de le finir, estimé sur le plus long déjà vu, majoré de 25 %. Un
job coupé en cours perd tout ce qu'il a calculé.

### Le battement, et l'arrêt à distance

Le battement va vers l'URL de prise, donc vers **notre** serveur, donc **sa réponse
peut porter un ordre**. C'est ce qui donne l'extinction à distance sur les trois
plateformes avec le même code, sans toucher à l'API de Modal, de RunPod ni de Vast.

| Réponse du serveur | Effet |
|---|---|
| `{"arret": "doux"}` | cesse de reprendre, laisse finir, sort |
| `{"arret": "net"}` | sort tout de suite, abandonne ce qui tourne |

L'ordre est lu dans **n'importe quelle** réponse, prise comme battement : on n'attend
jamais un battement pour obéir.

**Et le battement n'ajoute aucune requête tant que les prises parlent.** Il se tait si
une prise a eu lieu depuis moins que sa période. Un worker à huit places sur des jobs
de 80 s demande déjà du travail toutes les dix secondes ; le battement ne sert qu'au
worker silencieux, celui dont toutes les places sont prises par des jobs longs.

### L'homme-mort

Si c'est **nous** que le worker n'atteint plus, aucun ordre ne lui parviendra jamais,
par définition. Il doit donc décider seul :

| Silence | Effet |
|---|---|
| 5 min | cesse de reprendre |
| 15 min | **se tue**, même en plein job |

On perd le calcul en cours. L'alternative est de payer les cartes jusqu'au bout des
cinq heures de plafond.

---

## Les mesures RunPod du 2026-09-12

Endpoint `xysqfw4qcp35qu` : quatre cartes par worker, 13 workers au plus, idle timeout
5 s, coupure d'exécution 600 s, `flashboot: false`, dimensionnement sur `QUEUE_DELAY`
à 4 s, 60 Go de disque facturés.

### Le lot fonctionne

Huit sous-jobs, une seule requête, 8/8 réussis, huit fichiers distincts sur R2 de
7 198 892 octets chacun. Les logs prouvent la répartition — deux instances par carte :

```
chargement de bs_roformer_leap_xe sur cuda:0 (1/2), cuda:1 (1/2),
                                     cuda:2 (1/2), cuda:3 (1/2), puis (2/2) sur chacune
```

| Worker | Cartes | Mur du lot |
|---|---|---|
| 4 cartes | 8 places | **85 s** |
| 3 cartes | 6 places (bug d'arrondi) | 143 s |

Huit séparations en 85 s, là où elles prendraient 448 s à la file.

### Ce qu'on paie vraiment

Lu dans la facture RunPod elle-même, `GET /v1/billing/endpoints` :

| | |
|---|---|
| Facturé | **1 121,5 s** |
| Exécuté | 464,0 s |
| Écart | 657,5 s — **×2,42** |

Et la somme des `delayTime` de ces mêmes jobs : **612 s**. L'écart facturé **est** le
délai, à quarante-cinq secondes près — les idle timeouts et l'imprécision de mesure.

**On paie donc tout le démarrage** : tirage de l'image, initialisation du conteneur,
chargement des modèles, idle timeout. Seule échappe l'attente pure avant qu'un worker
soit attribué, et avec `scalerValue: 4` elle dure quatre secondes.

Coût réel de cet endpoint (pool 24 Go, à l'époque) : **2,76 $/h** pour quatre cartes. Depuis le
2026-09-15, l'endpoint de prise ne sert que des machines à **4 × RTX 4090, 4,40 $/h la machine**. Un démarrage à froid de
huit minutes y revient à 0,37 $ ; sur une configuration à huit cartes à 16 $/h, à
2,13 $. Et chaque **nouveau tag invalide tous les caches** : réchauffer 13 workers
coûte une fois le prix du tirage. L'image pèse 5,34 Go compressés, dont 1,17 Go de
modèles embarqués — les sortir vers un volume réseau les ferait sortir du tirage.

### Les trois plateformes ne facturent pas pareil

| | Tirage de l'image | Démarrage | À l'arrêt |
|---|---|---|---|
| RunPod | **payé** (vérifié sur facture) | payé | idle timeout, puis rien |
| Vast.ai | **payé** (établi le 2026-09-15 ; on lisait « You are not charged when it says Loading »), plus sa bande passante | payé | stockage tant que le pod existe |
| Modal | **payé** (établi le 2026-09-15) | payé | scaledown, puis rien |

Chez Vast, seule la **destruction par l'API** arrête les frais : le stockage court
« for every single second your instance exists », instance arrêtée comprise.

### Contrôle à distance, par plateforme

| | Tuer un job | Tuer un worker |
|---|---|---|
| Modal | `FunctionCall.cancel()` | `cancel(terminate_containers=True)` — déjà exposé |
| RunPod | `POST /v2/{ep}/cancel/{jobId}` — vérifié | **aucune API worker** ; `workersMax: 0` sur tout l'endpoint |
| Vast.ai | — | `DELETE /instances/{id}` |

**En mode prise, « un job » est toute la session du worker.** L'annuler tue le worker
et tout ce qu'il tient. Pour annuler un seul sous-job, c'est le serveur de prise qui
décide — et l'ordre `arret` dans sa réponse suffit, sans API d'hébergeur.

---

## Décisions du propriétaire

**Modal reste figé** : un job par worker d'une carte, pas de concurrence. Le plan
Starter y plafonne à **10 cartes par compte** (100 conteneurs, mais 10 cartes), donc ni
la concurrence ni le multi-GPU n'y augmenteraient la capacité. `modal_app.py` n'est pas
touché.

**RunPod** : quatre cartes par worker, lots de **8 au maximum**.

**Deux jobs par carte.** La mémoire en permettrait cinq (mesuré), mais deux garde une
marge et rend la capacité lisible.

**Coupures portées à cinq heures partout.** Ce n'est donc plus le timeout qui arrête un
worker, c'est la file vide — et l'ordre d'arrêt. Le budget interne (4 h 50) n'est qu'un
garde-fou contre un blocage.

**File vide, on sort.** Jamais de sondage.

**Toujours plein.** On reprend une place dès qu'elle se libère.

---

## Ce qui reste à faire

**La route de prise, côté backend.** C'est le seul vrai manque. Elle doit rendre au
plus `libres` jobs, réserver les lignes de façon atomique — deux workers ne doivent
jamais recevoir le même job —, et libérer une réservation dont le worker s'est tu. Vos
tables D1 ont déjà le statut et le compteur de tentatives.

**La décision de démarrer des workers**, et combien. Sujet non abordé.

**Vast.ai** : location et destruction automatiques. Le mode prise y marchera sans rien
changer, seule la carte diffère.

**Réduire le trafic.** Sur les 34 requêtes par minute d'un worker à huit places, **16
sont les battements par sous-job** vers leurs `callback_url`. En mode prise ils sont
redondants : les résultats reviennent déjà par les prises. Si le serveur ne met pas de
`callback_url` sur les jobs qu'il distribue, le trafic est divisé par trois.

**Réconcilier la facture.** `GET /v1/billing/endpoints?bucketSize=hour` donne le temps
facturé heure par heure. Le rapport avec la somme des `container_s` est l'indicateur à
suivre : 2,42 le 2026-09-12. Il faut une clé RunPod avec permission **Read au niveau du
compte** — les deux clés de Doppler sont restreintes au runtime, et `RUNPOD_API_KEY`
est à l'ancien format, n'est plus honorée, et ne sert plus que de secret partagé à
`pod-monitor`.

---

## Ce qu'on facture à l'utilisateur

`container_s`, et rien d'autre. Il colle à ce que RunPod facture pour l'exécution —
84,989 s contre 85,088 mesurées sur le lot de 8, soit 99 millisecondes d'écart. Il
contient déjà le démarrage du processus : le temps mort avant le premier job revient à
ce premier job, et quand plusieurs jobs se croisent chaque seconde est divisée entre
eux.

**Le tirage de l'image ne doit PAS être facturé à l'utilisateur.** Deux jobs
identiques, l'un sur un worker froid et l'autre sur un chaud, ont huit minutes d'écart
pour un travail rigoureusement identique. C'est un coût de plateforme, à réconcilier
mensuellement comme pour Cloudflare.

---

## Pièges rencontrés

**La carte du job lue trop tard.** Le résultat s'assemble après la fermeture du bail,
donc `device()` retombait sur la première carte : un lot de huit rapportait « cuda:0 »
huit fois. Les cartes étaient pourtant bien utilisées — 9,81 Go de pic, soit deux jobs,
là où huit sur une carte de 24 Go auraient débordé. Corrigé.

**La division entière.** Huit places sur trois cartes donnaient deux par carte, donc
six instances pour huit sous-jobs. Arrondi au-dessus : 3, 3 et 2.

**Arrêter le sondeur local n'annule rien.** Un lot de 20 soumis puis « annulé » de mon
côté a tourné 87 secondes sur quatre cartes et n'a produit **aucun** fichier — coupé
avant que le premier sous-job finisse. Un lot est tout ou rien.

**Un `try/except` n'attrape pas un blocage.** Tous les appels réseau et tous les
`ffmpeg` ont un délai, donc les cas courants sont couverts. Restent le pilote CUDA, qui
peut ne jamais revenir, et `registry._acquire` qui boucle sur `while True` : un job figé
qui tient un exemplaire bloque tout le lot. C'est à ça que sert le garde-fou.

**Un battement de plus, c'est du bruit.** Un worker occupé émet déjà 34 requêtes par
minute. D'où le battement qui se tait quand les prises parlent.
