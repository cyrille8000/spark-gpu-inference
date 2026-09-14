"""S3FD — le réseau de détection de visages (VGG16 + têtes multi-échelles).

Port de `model/faceDetector/s3fd/nets.py` de LR-ASD (MIT). Deux changements :

  - le décodage des boîtes (`Detect`) tourne sur CPU. Sa boucle de suppression des recouvrements
    enchaîne des centaines de toutes petites opérations par image ; sur GPU ce sont autant de
    lancements de noyaux, bien plus chers que le calcul lui-même. Le réseau, lui, reste sur la carte.
  - les boîtes a priori (`PriorBox`) ne dépendent que de la taille de l'image : elles sont calculées
    une fois par taille et gardées, au lieu d'être reconstruites — en Python pur, ~11 000 tours de
    boucle — à CHAQUE image. Sur 3 750 images d'une portion, c'est ~40 s de CPU économisés.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .s3fd_box import Detect, PriorBox

# Une entrée par taille d'image vue. Une portion n'a qu'une taille ; on borne quand même.
_PRIORS_CACHE: dict = {}
_PRIORS_CACHE_MAX = 8


def _priors(size, feature_maps):
    key = (tuple(int(v) for v in size), tuple(tuple(int(v) for v in fm) for fm in feature_maps))
    p = _PRIORS_CACHE.get(key)
    if p is None:
        if len(_PRIORS_CACHE) >= _PRIORS_CACHE_MAX:
            _PRIORS_CACHE.clear()
        p = _PRIORS_CACHE[key] = PriorBox(size, feature_maps).forward()
    return p


class L2Norm(nn.Module):

    def __init__(self, n_channels, scale):
        super().__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


class S3FDNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.vgg = nn.ModuleList([
            nn.Conv2d(3, 64, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, ceil_mode=True),

            nn.Conv2d(256, 512, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 512, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 1024, 3, 1, padding=6, dilation=6),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 1, 1),
            nn.ReLU(inplace=True),
        ])

        self.L2Norm3_3 = L2Norm(256, 10)
        self.L2Norm4_3 = L2Norm(512, 8)
        self.L2Norm5_3 = L2Norm(512, 5)

        self.extras = nn.ModuleList([
            nn.Conv2d(1024, 256, 1, 1),
            nn.Conv2d(256, 512, 3, 2, padding=1),
            nn.Conv2d(512, 128, 1, 1),
            nn.Conv2d(128, 256, 3, 2, padding=1),
        ])

        self.loc = nn.ModuleList([
            nn.Conv2d(256, 4, 3, 1, padding=1),
            nn.Conv2d(512, 4, 3, 1, padding=1),
            nn.Conv2d(512, 4, 3, 1, padding=1),
            nn.Conv2d(1024, 4, 3, 1, padding=1),
            nn.Conv2d(512, 4, 3, 1, padding=1),
            nn.Conv2d(256, 4, 3, 1, padding=1),
        ])

        self.conf = nn.ModuleList([
            nn.Conv2d(256, 4, 3, 1, padding=1),
            nn.Conv2d(512, 2, 3, 1, padding=1),
            nn.Conv2d(512, 2, 3, 1, padding=1),
            nn.Conv2d(1024, 2, 3, 1, padding=1),
            nn.Conv2d(512, 2, 3, 1, padding=1),
            nn.Conv2d(256, 2, 3, 1, padding=1),
        ])

        self.softmax = nn.Softmax(dim=-1)
        self.detect = Detect()

    def forward(self, x):
        """`x` : (N, 3, H, W) float32, prétraité comme `faces_geometry.s3fd_input` le fait.
        Rend un tenseur CPU (N, 2, 750, 5) = `[score, x1, y1, x2, y2]`, coordonnées relatives,
        classe 1 = visage, lignes triées par score décroissant puis zéros."""
        size = x.size()[2:]
        sources = []
        loc = []
        conf = []

        for k in range(16):
            x = self.vgg[k](x)
        s = self.L2Norm3_3(x)
        sources.append(s)

        for k in range(16, 23):
            x = self.vgg[k](x)
        s = self.L2Norm4_3(x)
        sources.append(s)

        for k in range(23, 30):
            x = self.vgg[k](x)
        s = self.L2Norm5_3(x)
        sources.append(s)

        for k in range(30, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        loc_x = self.loc[0](sources[0])
        conf_x = self.conf[0](sources[0])

        max_conf, _ = torch.max(conf_x[:, 0:3, :, :], dim=1, keepdim=True)
        conf_x = torch.cat((max_conf, conf_x[:, 3:, :, :]), dim=1)

        loc.append(loc_x.permute(0, 2, 3, 1).contiguous())
        conf.append(conf_x.permute(0, 2, 3, 1).contiguous())

        for i in range(1, len(sources)):
            x = sources[i]
            conf.append(self.conf[i](x).permute(0, 2, 3, 1).contiguous())
            loc.append(self.loc[i](x).permute(0, 2, 3, 1).contiguous())

        features_maps = []
        for i in range(len(loc)):
            features_maps.append([loc[i].size(1), loc[i].size(2)])

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        priors = _priors(size, features_maps)
        # Décodage sur CPU (voir l'en-tête du module).
        return self.detect.forward(
            loc.view(loc.size(0), -1, 4).detach().cpu(),
            self.softmax(conf.view(conf.size(0), -1, 2)).detach().cpu(),
            priors,
        )
