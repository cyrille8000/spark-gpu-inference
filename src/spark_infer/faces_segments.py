"""Des sorties brutes de LR-ASD aux MOMENTS DE PAROLE (à la milliseconde) et aux POSITIONS du visage.

Repris tel quel de `spark-dubbing-lipsync/segments.py` (l'ancienne image, 2026-08-15), avec ses
tests : c'est la seule logique métier de la tâche, et elle se teste sans GPU, sans torch, sans
modèle. Seule différence : `tracks` n'est plus un pickle relu mais la liste rendue par
`faces_engine` — même forme, `{'track': {'frame': ndarray, 'bbox': ndarray}}`.

Rien n'y est supposé. La cadence des images est MESURÉE sur le fichier découpé et passée en
paramètre. Le critère « ça parle » est celui de LR-ASD lui-même, lissage compris, relevé dans son
code et non dans sa documentation.

Six étapes, dans cet ordre — l'ordre compte :
  1. le score de chaque image est LISSÉ, puis comparé au seuil ;
  2. par piste, les suites d'images « ça parle » deviennent des intervalles ;
  3. on REMET sur la timeline d'origine : quand on analyse une portion découpée,
     LR-ASD renumérote ses images à partir de 0 et ne sait rien de l'extrait ;
  4. toutes pistes confondues, on FUSIONNE ce qui se chevauche : la sortie dit
     « quelqu'un parle à l'image », pas « qui » ;
  5. on ROGNE la marge de contexte (voir `clip_to_window`) ;
  6. on arrondit à la MILLISECONDE, VERS L'EXTÉRIEUR, puis on refusionne.

L'ordre des étapes 5 et 6 n'est pas indifférent : on rogne sur des flottants, donc exactement,
quelles que soient les bornes de la portion. Rogner après l'arrondi ne redonnerait le bon résultat
que dans le cas particulier où ces bornes tombent, elles aussi, sur la grille de l'arrondi.
"""

from math import floor, ceil

# Fenêtre de lissage du score, en images, reprise telle quelle de LR-ASD :
#
#     s = score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)]
#
# soit deux images avant et trois après (borne haute exclue).
LISSAGE_AVANT = 2
LISSAGE_APRES = 3

# Seuil de décision. C'est `>= 0` et non `> 0` — relevé dans le code de LR-ASD,
# `colorDict[int((face['score'] >= 0))]`. La nuance n'est pas théorique : les
# scores sont arrondis au dixième (`numpy.round(…, 1)`), donc 0.0 est une valeur
# courante, et la trancher du mauvais côté perd de vraies images de parole.
SEUIL_PAROLE = 0.0


def smoothed_speaking(scores):
    """Parle-t-on, image par image, selon le critère de LR-ASD lui-même.

    Le score brut n'est jamais comparé directement au seuil : LR-ASD le lisse
    d'abord sur une fenêtre glissante d'environ cinq images. Sans ce lissage, un
    score qui vacille d'une image à l'autre découpe la parole en confettis.

    Un écart assumé avec l'original : sa borne haute est `len(score) - 1`, ce qui
    exclut la toute dernière image de sa propre fenêtre. On prend `len(score)`.
    L'effet se limite aux deux dernières images — 80 ms, absorbés par l'arrondi.

    `scores` arrive en TABLEAU NUMPY, pas en liste : LR-ASD le construit avec
    `numpy.round(numpy.mean(...))`. D'où le test sur `len()` et non sur la vérité
    du tableau — `bool(tableau)` lève « truth value of an array with more than
    one element is ambiguous ». Et le retour est converti en `bool` Python, pour
    que la sortie ne traîne pas de types numpy jusqu'au JSON.
    """
    n = len(scores)
    out = []
    for i in range(n):
        fenetre = scores[max(i - LISSAGE_AVANT, 0):min(i + LISSAGE_APRES, n)]
        if len(fenetre) == 0:
            out.append(False)
        else:
            out.append(bool(sum(fenetre) / len(fenetre) >= SEUIL_PAROLE))
    return out


def speaking_spans(speaking, n):
    """Suites d'images consécutives où l'on parle, en INDICES `(i0, i1)` inclus.

    Sépare le PARCOURS des images de ce qu'on en tire. Les moments de parole en
    ont besoin (`track_intervals`) et les positions à l'image aussi
    (`speaking_faces`) : les deux doivent découper exactement aux mêmes endroits,
    sans quoi une boîte s'afficherait sur une seconde annoncée muette.
    """
    out = []
    debut = None
    for i in range(n):
        if speaking[i]:
            if debut is None:
                debut = i
        elif debut is not None:
            out.append((debut, i - 1))
            debut = None
    if debut is not None:
        out.append((debut, n - 1))
    return out


def track_intervals(frames, speaking, fps):
    """Intervalles de parole (secondes, pleine précision) d'UNE piste.

    `frames` : numéros d'image de la piste ; `speaking` : un booléen par image ;
    `fps` : la cadence RÉELLE du fichier analysé, mesurée, jamais supposée.

    La fin d'un intervalle est la fin de la DERNIÈRE image parlée, donc
    `(image + 1) / fps` et non `image / fps`. Sans ce `+1`, chaque intervalle
    perdait une image à son extrémité droite. C'est la convention de LR-ASD
    lui-même, qui calcule ses propres bornes audio ainsi.

    `speaking` PEUT ÊTRE PLUS COURT que `frames`, et ce n'est pas une anomalie :
    LR-ASD n'évalue que la partie couverte par les deux modalités, et coupe le
    surplus à la FIN —

        length = min((mfcc - mfcc % 4) / 100, videoFeature.shape[0])
        videoFeature = videoFeature[:int(round(length * 25)),:,:]

    — si bien que l'audio, presque toujours un peu plus court que l'image, fixe
    la limite. Les images non évaluées sont ignorées : sans son, on ne peut rien
    affirmer, et mieux vaut taire une parole que d'en inventer une.
    """
    n = min(len(frames), len(speaking))
    return [
        (frames[i0] / fps, (frames[i1] + 1) / fps)
        for i0, i1 in speaking_spans(speaking, n)
    ]


def merge(intervals):
    """Fusionne les intervalles qui se chevauchent ou se touchent, triés."""
    out = []
    for start, end in sorted(intervals):
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


def shift(intervals, offset):
    """Ramène des intervalles d'extrait sur la timeline de la vidéo d'origine.

    LR-ASD numérote ses images à partir de 0 dans l'extrait qu'on lui donne
    (`{'frame': fidx}`, un `enumerate` sur les jpg). Il ignore donc totalement
    d'où vient cet extrait : sans ce décalage, une parole à 302 s d'une portion
    commençant à 300 s serait annoncée à 2 s. Aucune erreur, aucun log — juste
    un résultat faux.
    """
    if not offset:
        return list(intervals)
    return [(s + offset, e + offset) for s, e in intervals]


def clip_to_window(intervals, start, end):
    """Ne garde que ce qui tombe dans [start, end], en tronquant les débordements.

    C'est ici qu'on jette la MARGE. On donne à LR-ASD un peu plus que la portion
    demandée, parce qu'à ses bords il est aveugle : une piste de visage de moins
    de 11 images est purement jetée (`--minTrack`), et le modèle score d'autant
    moins bien qu'il lui manque du contexte temporel. La marge encaisse ces
    effets de bord, et disparaît ici.

    Rogner, pas jeter : une parole à cheval sur la bordure est TRONQUÉE. Elle
    reste entière une fois les portions recollées, puisque la portion voisine
    en tient l'autre moitié.
    """
    out = []
    for s, e in intervals:
        s2, e2 = max(s, start), min(e, end)
        if e2 > s2:
            out.append((s2, e2))
    return out


def to_milliseconds(intervals):
    """Arrondit à la MILLISECONDE vers l'extérieur : début en dessous, fin au-dessus.

    Vers l'extérieur, et pas au plus proche : l'intervalle doit CONTENIR la
    parole. À l'arrondi au plus proche, une parole commençant à 1,6006 s serait
    annoncée à 1,601 s — six dixièmes de milliseconde de voix hors de
    l'intervalle. Le principe vaut à toutes les échelles ; c'est l'échelle qui
    a changé.

    Le débordement est désormais de 1 ms au pire, contre 2 s à la seconde
    entière. C'est tout l'objet du changement : ces secondes de silence étaient
    fabriquées, puis facturées, alors que personne ne parlait dedans.

    Les nombres restent des SECONDES flottantes, avec trois décimales — c'est
    l'unité de `speech.json` et de tout ce qui le relit, en aval comme dans
    l'éditeur de zones. Milliseconde désigne la finesse, pas l'unité.

    On refusionne APRÈS l'arrondi : deux intervalles séparés d'une fraction de
    milliseconde se retrouvent collés une fois élargis, et une union qui
    contiendrait des intervalles superposés se contredirait elle-même.
    """
    arrondis = [(floor(s * 1000) / 1000, ceil(e * 1000) / 1000) for s, e in intervals]
    return [{"start": s, "end": e} for s, e in merge(arrondis)]


def speech_seconds(tracks, scores, fps, offset=0.0, window=None):
    """Sortie du service : les moments de parole, en secondes au millième, fusionnés.

    Le nom dit l'UNITÉ (des secondes), pas la finesse : les bornes portent trois
    décimales depuis le 2026-08-15 — voir `to_milliseconds`.

    `tracks` / `scores` sont les deux pickles rendus par LR-ASD.

    `fps` : la cadence MESURÉE sur le fichier que LR-ASD a produit. Elle n'a pas
    de valeur par défaut, volontairement — une cadence supposée fausse décale
    tous les temps sans qu'aucune erreur ne se déclenche.

    `offset` : instant, dans la vidéo d'origine, de la première image de
    l'extrait analysé. `window` : `(start, end)` de la portion réellement
    demandée, marge exclue. Les deux valent zéro / None quand on a analysé la
    vidéo entière.
    """
    # Structure relevée dans le code de LR-ASD, pas devinée :
    #   tracks.pckl = liste de {'track': {'frame': ndarray, 'bbox': ndarray},
    #                           'proc_track': {...}}
    #   scores.pckl = liste de ndarray float64, un par piste
    #
    # L'appariement des deux tient à un `glob` trié sur des fichiers nommés
    # `%05d` — donc solide, mais rien ne le garantit ici. Un `zip` tronquerait
    # EN SILENCE si les longueurs divergeaient, et on perdrait des pistes
    # entières sans rien voir.
    if len(tracks) != len(scores):
        raise ValueError(
            f"{len(tracks)} pistes pour {len(scores)} series de scores : "
            "l'appariement piste/score est rompu"
        )

    bruts = []
    for track, score in zip(tracks, scores):
        frames = track["track"]["frame"].tolist()
        bruts.extend(track_intervals(frames, smoothed_speaking(score), fps))

    bruts = merge(shift(bruts, offset))
    if window is not None:
        bruts = clip_to_window(bruts, window[0], window[1])
    return to_milliseconds(bruts)


# ── OÙ, à l'image ────────────────────────────────────────────────────────────
#
# `speech` dit QUAND quelqu'un parle. Ce qui suit dit OÙ : la boîte du visage
# qui parle, pour que le navigateur la pose sur la vidéo.
#
# LR-ASD les a déjà. Trois points RELEVÉS DANS SON CODE, pas supposés — ils
# décident de la justesse des coordonnées, et se tromper dessus donnerait une
# boîte à côté du visage sans qu'aucune erreur ne se déclenche :
#
#   1. `tracks.pckl` est la liste des retours de `crop_video`, c'est-à-dire
#      `{'track': {'frame', 'bbox'}, 'proc_track': {…}}` (Columbia_test.py,
#      `return {'track':track, 'proc_track':dets}` puis `pickle.dump(vidTracks)`).
#      `proc_track` est un LISSAGE médian du centre et de la taille, servant au
#      recadrage 224×224 — pas la position du visage. C'est `track.bbox` qu'on
#      lit ;
#   2. `bbox` vaut `[x1, y1, x2, y2]` en PIXELS de l'image extraite. S3FD rend
#      des coordonnées relatives qu'il multiplie par `torch.Tensor([w, h, w, h])`
#      où `w, h` sont ceux de l'image REÇUE — pas de l'image réduite par
#      `facedetScale`, qui ne sert qu'à la détection (`s3fd/__init__.py`) ;
#   3. rien n'est redimensionné entre le fichier et la détection : `video.avi`
#      est produit sans `-vf scale`, et les jpg avec un simple `-f image2`. Les
#      pixels des boîtes sont donc ceux de `video.avi` — le fichier même que le
#      handler sonde pour sa cadence.
#
# `bbox` est INTERPOLÉ image par image (`interp1d` sur `arange(première,
# dernière+1)`), donc `bbox[i]` correspond exactement à `frame[i]`, sans trou.
#
# Tout le travail restant est de les rendre TRANSPORTABLES : les donner toutes
# ferait 25 boîtes par seconde et par visage, soit des mégaoctets pour une heure
# de vidéo, à faire passer par un webhook, un événement de workflow, R2, puis le
# réseau de l'utilisateur.

#: Un point n'est retenu que si la boîte a bougé d'au moins ça, en fraction de
#: l'image. En dessous, l'écart est invisible : 1,5 % d'une image 1080p font
#: 29 px de large et 16 px de haut, et le navigateur interpole entre deux points.
#: C'est ce qui rend un plan fixe — le cas courant d'une interview — presque
#: gratuit : deux points pour dix secondes de parole.
BOUGE_MIN = 0.015

#: … mais jamais plus d'une seconde sans point, même immobile. Une dérive lente
#: peut rester sous le seuil des dizaines de secondes durant, et le cumul, lui,
#: se voit.
PAUSE_MAX_S = 1.0

#: Plafond de points par appel. Il n'est pas là pour le cas courant mais pour la
#: scène pathologique — caméra à l'épaule, six visages — qui ferait gonfler la
#: charge utile d'un événement de workflow. Dépassé, le seuil est relâché et on
#: recommence : mieux vaut une boîte moins fidèle qu'une réponse trop lourde.
MAX_POINTS = 4000

#: Décimales gardées. 3 sur les coordonnées = un millième d'image, soit 2 px en
#: 1080p, invisible. 2 sur le temps = 10 ms, le quart d'une image à 25 i/s.
DEC_COORD = 3
DEC_TEMPS = 2


def normalise_bbox(bbox, width, height):
    """`(x1, y1, x2, y2)` en pixels → `(x, y, w, h)` en fraction de l'image.

    NORMALISÉ, et c'est essentiel : la vidéo analysée est la source (jusqu'à
    1080p), celle que le navigateur joue est la déclinaison HLS 320p. Des pixels
    ne voudraient rien dire de l'une à l'autre — une fraction, si.

    Les coordonnées sont RABOTÉES dans [0, 1] : S3FD rend volontiers une boîte
    qui dépasse un peu du cadre sur un visage au bord, et une largeur négative
    dessinerait un rectangle à l'envers.
    """
    if not (width > 0 and height > 0):
        return None
    x1, y1, x2, y2 = (float(v) for v in list(bbox)[:4])
    gx = max(0.0, min(1.0, x1 / width))
    gy = max(0.0, min(1.0, y1 / height))
    dx = max(0.0, min(1.0, x2 / width))
    dy = max(0.0, min(1.0, y2 / height))
    return (gx, gy, max(0.0, dx - gx), max(0.0, dy - gy))


def _ecart(a, b):
    """Distance entre deux boîtes : le plus grand écart de leurs quatre nombres.

    Le maximum, pas la moyenne : un visage qui ne fait que grossir bouge autant
    qu'un visage qui se déplace, et une moyenne diluerait le changement dans les
    trois composantes restées immobiles.
    """
    return max(abs(a[i] - b[i]) for i in range(4))


def _points(frames, bboxes, i0, i1, fps, width, height, offset, seuil):
    """Les points retenus d'UNE suite d'images parlées, dans l'ordre du temps."""
    out = []
    dernier = None
    for i in range(i0, i1 + 1):
        boite = normalise_bbox(bboxes[i], width, height)
        if boite is None:
            continue
        t = offset + frames[i] / fps
        # Le premier et le dernier sont toujours pris : ils tiennent les deux
        # bouts de l'interpolation. Entre eux, seul ce qui a bougé.
        if (dernier is None or i == i1
                or _ecart(boite, dernier[1]) >= seuil
                or t - dernier[0] >= PAUSE_MAX_S):
            out.append((t, boite))
            dernier = (t, boite)
    return out


def _clip_points(points, debut, fin):
    """Ne garde que les points de [debut, fin], sans jamais vider la boîte.

    Une suite peut chevaucher la bordure de la fenêtre et n'avoir aucun point
    dedans (plan fixe : deux points, l'un avant, l'autre après). La jeter
    laisserait un intervalle de parole sans visage — on ramène alors le point le
    plus proche sur la bordure. La position est bonne : elle n'a pas bougé,
    c'est précisément pour ça qu'aucun point n'a été retenu entre les deux.
    """
    gardes = [p for p in points if debut <= p[0] <= fin]
    if gardes:
        return gardes
    avant = [p for p in points if p[0] <= debut]
    proche = avant[-1] if avant else points[0]
    return [(debut, proche[1])]


def _arrondi(points):
    """`[(t, (x, y, w, h))]` → `[[t, x, y, w, h]]`, décimales coupées."""
    return [
        [round(t, DEC_TEMPS)] + [round(v, DEC_COORD) for v in boite]
        for t, boite in points
    ]


def speaking_faces(tracks, scores, fps, width, height, offset=0.0, window=None):
    """Où, à l'image, se trouve le visage qui parle — et quand.

    Rend une liste de suites de parole, une par visage et par passage :

        [{"start": 302.04, "end": 310.2, "box": [[t, x, y, w, h], …]}, …]

    Mêmes temps que `speech` — ceux de la vidéo d'origine, marge rognée — au
    centième près (`DEC_TEMPS`) contre le millième pour `speech`. L'écart n'a
    aucune portée : une boîte sert à viser un visage entre deux points
    interpolés, et 10 ms valent le quart d'une image.

    Les suites NE SONT PAS fusionnées entre visages, contrairement à `speech` :
    deux personnes qui parlent en même temps, ce sont deux boîtes à l'écran, et
    les confondre en une seule reviendrait à encadrer le vide entre les deux.

    Aucune identité de visage n'est rendue. Les pistes de LR-ASD sont numérotées
    par extrait : la « piste 2 » d'une fenêtre n'est pas celle de la suivante, et
    exposer ce numéro inviterait à lui prêter une continuité qu'il n'a pas.
    """
    if len(tracks) != len(scores):
        raise ValueError(
            f"{len(tracks)} pistes pour {len(scores)} series de scores : "
            "l'appariement piste/score est rompu"
        )
    if not (width > 0 and height > 0):
        # Sans les dimensions de l'image, aucune fraction n'a de sens. On rend
        # une liste vide plutôt qu'une boîte fausse : le repérage temporel, lui,
        # reste bon, et l'utilisateur perd l'incrustation, pas le résultat.
        return []

    # Lissage et découpe faits UNE fois : seule la sélection des points dépend
    # du seuil, et c'est elle seule qu'on rejoue si la charge est trop lourde.
    prepares = []
    for track, score in zip(tracks, scores):
        frames = track["track"]["frame"].tolist()
        bboxes = list(track["track"]["bbox"])
        speaking = smoothed_speaking(score)
        n = min(len(frames), len(speaking), len(bboxes))
        prepares.append((frames, bboxes, speaking_spans(speaking, n)))

    seuil = BOUGE_MIN
    suites = []
    for _ in range(4):
        suites = []
        for frames, bboxes, spans in prepares:
            for i0, i1 in spans:
                debut = offset + frames[i0] / fps
                fin = offset + (frames[i1] + 1) / fps
                if window is not None:
                    debut, fin = max(debut, window[0]), min(fin, window[1])
                    if fin <= debut:
                        continue
                points = _points(frames, bboxes, i0, i1, fps, width, height, offset, seuil)
                if not points:
                    continue
                if window is not None:
                    points = _clip_points(points, debut, fin)
                suites.append({
                    "start": round(debut, DEC_TEMPS),
                    "end": round(fin, DEC_TEMPS),
                    "box": _arrondi(points),
                })
        if sum(len(s["box"]) for s in suites) <= MAX_POINTS:
            break
        # Trop de points : on double la tolérance et on recommence. Déterministe,
        # borné, et la dégradation est visuelle — la boîte suit d'un peu moins
        # près — jamais une perte de repérage.
        seuil *= 2

    suites.sort(key=lambda s: (s["start"], s["end"]))
    return suites
