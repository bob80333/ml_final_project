import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


# will use convs / resblocks + downsampling
class ConvModel(nn.Module):
    pass


# will use a patching operation to reduce the sequence length from 16384 to 256, 64x patching, then use a transformer
class TransformerModel(nn.Module):
    def __init__(self, patching=64, width=384, layers=6):
        super().__init__()
        self.patching = patching
        self.width = width
        self.layers = layers

        # batch, 1 channels, 16384 samples -> batch, 64 channels, 256 patches
        self.patcher = Rearrange("b c (n p) -> b (c p) n", p=patching)

        self.patch_in = nn.Linear(patching, width)

        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=width // 64,
            dim_feedforward=width * 4,
            dropout=0.0,
            activation="gelu",
            layer_norm_eps=1e-6,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)

        self.embed_out = nn.Linear(width, width)

    def forward(self, x):
        # x: batch, 1, 16384
        x = self.patcher(x)
        # swap seq and channel dim, convert from patch size to model width
        x = self.patch_in(x.permute(0, 2, 1))
        # x: batch, 64, 256
        x = self.transformer(x)
        # x: batch, 64, 256
        # grab last item in sequence as embedding
        x = x[:, -1, :]
        return self.embed_out(x)
