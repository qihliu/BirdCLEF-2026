import timm
import torch
import torch.nn as nn

from config import CFG


class BirdModel(nn.Module):
    """
    EfficientNet-B4 NoisyStudent backbone with a fresh classification head.

    Input:  (B, 3, IMG_SIZE, IMG_SIZE)  float32 ImageNet-normalised
    Output: (B, NUM_CLASSES)            raw logits (no sigmoid)
    """

    def __init__(self, cfg: CFG):
        super().__init__()
        self.backbone = timm.create_model(
            cfg.MODEL_NAME,
            pretrained=cfg.PRETRAINED,
            in_chans=cfg.IN_CHANNELS,
            num_classes=0,          # strip the pretrained classifier
            global_pool="avg",
            drop_rate=cfg.DROP_RATE,
        )
        num_features = self.backbone.num_features
        self.classifier = nn.Linear(num_features, cfg.NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)          # (B, num_features)
        logits   = self.classifier(features) # (B, NUM_CLASSES)
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return backbone embeddings (useful for future feature-based work)."""
        return self.backbone(x)
