"""LR-ASD, vendu tel quel — le réseau de détection du locuteur actif et son détecteur de visages S3FD.

Origine : https://github.com/Junhua-Liao/LR-ASD (Liao et al., IJCV 2025 / CVPR 2023), commit
`1b6dcd2d8fc2895683de6508ec6294ec47d388ca` du 2025-03-23, licence MIT (reproduite dans
LICENSE à côté de ce fichier). Trois fichiers, réunis depuis `model/Model.py`, `model/Encoder.py`,
`model/Classifier.py`, `loss.py` et `model/faceDetector/s3fd/{nets,box_utils}.py`.

Pourquoi vendre le code plutôt que cloner le dépôt au build : l'image ne doit dépendre d'aucun
téléchargement à l'inférence, et l'ancien handler (`spark-dubbing-lipsync`) devait réécrire à coups
de `sed` le chemin des poids et les `np.int` supprimés de NumPy. Ici le code est relu, corrigé, et
c'est lui qui est testé.

Ce qui a changé par rapport à l'original, et rien d'autre :
  - S3FD : plus de téléchargement Google Drive à l'import (les poids sont passés au constructeur),
    plus d'import torchvision (jamais utilisé), `np.int` → `int` ; le décodage des boîtes (`Detect`)
    tourne sur CPU — sa boucle de suppression des recouvrements fait des centaines de petites
    opérations, qui coûtent bien plus en lancements CUDA qu'en calcul ;
  - ASD : le modèle et sa tête de score (`lossAV.FC`) prennent un `device` au lieu d'appeler
    `.cuda()` en dur ; l'optimiseur, le scheduler et les chemins d'entraînement sont retirés.
    La tête de score rend le LOGIT de la classe « parle » — c'est ce que `lossAV.forward(x, None)`
    rendait, et le seuil de décision de LR-ASD est `>= 0` dessus.
"""
from .asd_model import ASD_Model, ScoreHead, load_asd_weights
from .s3fd_net import S3FDNet

__all__ = ["ASD_Model", "ScoreHead", "load_asd_weights", "S3FDNet"]
