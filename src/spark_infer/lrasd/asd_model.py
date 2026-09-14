"""LR-ASD — le réseau audio-visuel qui dit, image par image, si le visage parle.

Port de `model/Model.py`, `model/Encoder.py`, `model/Classifier.py` et de la tête `lossAV` de
`loss.py` (LR-ASD, MIT), sans l'entraînement : ni optimiseur, ni scheduler, ni `.cuda()` en dur.

Entrées, telles que `Columbia_test.py` les construit :
  - audio : MFCC `python_speech_features.mfcc(int16, 16000, numcep=13, winlen=0.025, winstep=0.010)`,
    soit 100 trames par seconde, `(1, T_audio, 13)` ;
  - vidéo : le CENTRE 112×112 du visage recadré en 224×224, en niveaux de gris, `(1, T_video, 112, 112)`
    en 0..255 (la normalisation `(x/255 - 0.4161)/0.1688` est faite ici, dans `forward_visual_frontend`) ;
  - T_audio = 4 × T_video (25 images par seconde contre 100 trames MFCC).

Sortie de `ScoreHead` : un LOGIT par image pour la classe « parle ». Le critère de LR-ASD est
`score >= 0`, appliqué après lissage sur cinq images (cf. `faces_segments.smoothed_speaking`).
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class Audio_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_1, kernel_2):
        super().__init__()

        self.relu = nn.ReLU()
        self.padding_1 = int((kernel_1 - 1) / 2)
        self.padding_2 = int((kernel_2 - 1) / 2)

        self.m_1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=(kernel_1, 1), padding=(self.padding_1, 0), bias=False)
        self.m_norm_1 = nn.BatchNorm2d(out_channels // 2, momentum=0.01, eps=0.001)
        self.m_2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=(kernel_2, 1), padding=(self.padding_2, 0), bias=False)
        self.m_norm_2 = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)

        self.t_1 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, kernel_1), padding=(0, self.padding_1), bias=False)
        self.t_norm_1 = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)
        self.t_2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, kernel_2), padding=(0, self.padding_2), bias=False)
        self.t_norm_2 = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)

    def forward(self, x):
        x = self.relu(self.m_norm_1(self.m_1(x)))
        x = self.relu(self.m_norm_2(self.m_2(x)))

        x = self.relu(self.t_norm_1(self.t_1(x)))
        x = self.relu(self.t_norm_2(self.t_2(x)))
        return x


class Visual_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_1, kernel_2, is_down=False):
        super().__init__()

        self.relu = nn.ReLU()
        self.padding_1 = int((kernel_1 - 1) / 2)
        self.padding_2 = int((kernel_2 - 1) / 2)

        if is_down:
            self.s_1 = nn.Conv3d(in_channels, out_channels // 2, kernel_size=(1, kernel_1, kernel_1), stride=(1, 2, 2), padding=(0, self.padding_1, self.padding_1), bias=False)
        else:
            self.s_1 = nn.Conv3d(in_channels, out_channels // 2, kernel_size=(1, kernel_1, kernel_1), padding=(0, self.padding_1, self.padding_1), bias=False)

        self.s_norm_1 = nn.BatchNorm3d(out_channels // 2, momentum=0.01, eps=0.001)

        self.s_2 = nn.Conv3d(out_channels // 2, out_channels, kernel_size=(1, kernel_2, kernel_2), padding=(0, self.padding_2, self.padding_2), bias=False)
        self.s_norm_2 = nn.BatchNorm3d(out_channels, momentum=0.01, eps=0.001)

        self.t_1 = nn.Conv3d(out_channels, out_channels, kernel_size=(kernel_1, 1, 1), padding=(self.padding_1, 0, 0), bias=False)
        self.t_norm_1 = nn.BatchNorm3d(out_channels, momentum=0.01, eps=0.001)
        self.t_2 = nn.Conv3d(out_channels, out_channels, kernel_size=(kernel_2, 1, 1), padding=(self.padding_2, 0, 0), bias=False)
        self.t_norm_2 = nn.BatchNorm3d(out_channels, momentum=0.01, eps=0.001)

    def forward(self, x):
        x = self.relu(self.s_norm_1(self.s_1(x)))
        x = self.relu(self.s_norm_2(self.s_2(x)))

        x = self.relu(self.t_norm_1(self.t_1(x)))
        x = self.relu(self.t_norm_2(self.t_2(x)))
        return x


class visual_encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = Visual_Block(1, 32, 5, 3, is_down=True)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.block2 = Visual_Block(32, 64, 5, 3)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.block3 = Visual_Block(64, 128, 5, 3)

        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

        self.__init_weight()

    def forward(self, x):
        x = self.block1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.pool2(x)

        x = self.block3(x)
        x = x.transpose(1, 2)
        B, T, C, W, H = x.shape
        x = x.reshape(B * T, C, W, H)

        x = self.maxpool(x)

        x = x.view(B, T, C)
        return x

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class audio_encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = Audio_Block(1, 32, 5, 3)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 1, 3), stride=(1, 1, 2), padding=(0, 0, 1))

        self.block2 = Audio_Block(32, 64, 5, 3)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 1, 3), stride=(1, 1, 2), padding=(0, 0, 1))

        self.block3 = Audio_Block(64, 128, 5, 3)

        self.__init_weight()

    def forward(self, x):
        x = self.block1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.pool2(x)

        x = self.block3(x)

        x = torch.mean(x, dim=2, keepdim=True)
        x = x.squeeze(2).transpose(1, 2)
        return x

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Fusion(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.attention = nn.Conv1d(channel, channel, kernel_size=1, padding=0, bias=False)
        self.bn = nn.BatchNorm1d(channel, momentum=0.01, eps=0.001)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 2)
        identity = x.transpose(1, 2)
        w = self.sigmoid(self.bn(self.attention(identity)))
        x = (identity * w).transpose(1, 2)
        return x


class Detector(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.gru_forward = nn.GRU(input_size=channel, hidden_size=channel // 4, num_layers=1, bidirectional=False, bias=True, batch_first=True)
        self.gru_backward = nn.GRU(input_size=channel, hidden_size=channel // 4, num_layers=1, bidirectional=False, bias=True, batch_first=True)
        self.drop = nn.Dropout(0.5)
        self.attention = Fusion(channel // 2)
        self.__init_weight()

    def forward(self, x):
        x1, _ = self.gru_forward(self.drop(x))
        x = torch.flip(x, dims=[1])
        x2, _ = self.gru_backward(self.drop(x))
        x2 = torch.flip(x2, dims=[1])
        x = self.attention(x1, x2)
        return x

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.GRU):
                torch.nn.init.kaiming_normal_(m.weight_ih_l0)
                torch.nn.init.kaiming_normal_(m.weight_hh_l0)
                m.bias_ih_l0.data.zero_()
                m.bias_hh_l0.data.zero_()


class ASD_Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.visualEncoder = visual_encoder()
        self.audioEncoder = audio_encoder()
        self.fusion = Fusion(256)
        self.detector = Detector(256)

    def forward_visual_frontend(self, x):
        B, T, W, H = x.shape
        x = x.view(B, 1, T, W, H)
        x = (x / 255 - 0.4161) / 0.1688
        x = self.visualEncoder(x)
        return x

    def forward_audio_frontend(self, x):
        x = x.unsqueeze(1).transpose(2, 3)
        x = self.audioEncoder(x)
        return x

    def forward_audio_visual_backend(self, x1, x2):
        x = self.fusion(x1, x2)
        x = self.detector(x)
        x = torch.reshape(x, (-1, 128))
        return x

    def forward_visual_backend(self, x):
        x = torch.reshape(x, (-1, 128))
        return x

    def forward(self, audioFeature, visualFeature):
        audioEmbed = self.forward_audio_frontend(audioFeature)
        visualEmbed = self.forward_visual_frontend(visualFeature)
        outsAV = self.forward_audio_visual_backend(audioEmbed, visualEmbed)
        outsV = self.forward_visual_backend(visualEmbed)
        return outsAV, outsV


class ScoreHead(nn.Module):
    """La tête `lossAV` de LR-ASD réduite à l'inférence : le logit de la classe « parle »,
    exactement ce que `lossAV.forward(x, labels=None)` rendait (`x[:, 1]`)."""

    def __init__(self):
        super().__init__()
        self.FC = nn.Linear(128, 2)

    def forward(self, x):
        x = x.squeeze(1)
        x = self.FC(x)
        return x[:, 1]


def load_asd_weights(path: Path, model: ASD_Model, head: ScoreHead) -> None:
    """Charge un `.model` de LR-ASD (l'état complet de sa classe `ASD` : `model.*`, `lossAV.*`,
    `lossV.*`) dans le réseau et la tête de score. `lossV` ne sert qu'à l'entraînement : ignoré.
    STRICT dans les deux sens : une clé manquante ou inconnue arrête tout, un poids qui ne
    se charge pas en silence est un modèle qui se trompe en silence."""
    state = torch.load(str(path), map_location="cpu")
    model_sd, head_sd, inconnues = {}, {}, []
    for k, v in state.items():
        k2 = k[len("module."):] if k.startswith("module.") else k
        if k2.startswith("model."):
            model_sd[k2[len("model."):]] = v
        elif k2.startswith("lossAV."):
            head_sd[k2[len("lossAV."):]] = v
        elif k2.startswith("lossV."):
            continue
        else:
            inconnues.append(k)
    if inconnues:
        raise RuntimeError(f"{path.name} : clés inattendues {inconnues[:5]}")
    model.load_state_dict(model_sd, strict=True)
    head.load_state_dict(head_sd, strict=True)
