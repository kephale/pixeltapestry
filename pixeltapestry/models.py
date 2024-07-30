import torch
import torch.nn as nn
import numpy as np
from typing import Sequence
from monai.networks.nets import SwinTransformer, UnetrBasicBlock, UnetrUpBlock, UnetOutBlock
from monai.utils import ensure_tuple_rep, look_up_option
from monai.networks.nets import SwinUNETR


class BinaryEmbeddingSwinUNETR(SwinUNETR):
    """Swin UNETR with a modified forward that returns the embedding."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def binarize(self, x):
        """Binarize the pixel embeddings for inference."""
        return (x > 0).float()

    def sigmoid_binarize(self, x):
        """Differentiable binarization using sigmoid."""
        return torch.sigmoid(x)

    def forward(self, x_in):
        if self.training:
            return self.training_forward(x_in)
        else:
            return self.inference_forward(x_in)

    def inference_forward(self, x_in):
        if not torch.jit.is_scripting():
            self._check_input_size(x_in.shape[2:])
        hidden_states_out = self.swinViT(x_in, self.normalize)
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])
        dec3 = self.decoder5(dec4, hidden_states_out[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        pixel_embedding = self.decoder1(dec0, enc0)
        binary_pixel_embedding = self.binarize(pixel_embedding)
        return binary_pixel_embedding

    def training_forward(self, x_in):
        """During training, return the embedding and segmentation logits.

        Returns
        -------
        sigmoid_pixel_embedding : torch.Tensor
            The sigmoid binarized pixel embedding.
        logits : torch.Tensor
            The segmentation logits.
        """
        if not torch.jit.is_scripting():
            self._check_input_size(x_in.shape[2:])
        hidden_states_out = self.swinViT(x_in, self.normalize)
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])
        dec3 = self.decoder5(dec4, hidden_states_out[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        pixel_embedding = self.decoder1(dec0, enc0)
        sigmoid_pixel_embedding = self.sigmoid_binarize(pixel_embedding)
        logits = self.out(sigmoid_pixel_embedding)
        return sigmoid_pixel_embedding, logits
