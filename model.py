import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model

class Model(nn.Module):
    def __init__(self, pretrained=True, freeze_backbone=True, dropout=0.3):
        super(Model, self).__init__()

        # === ViT-Base Patch16 224 ===
        self.vit = create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=0)  # 768-dim
        # === Swin-Base Patch4 Window7 224 ===
        self.swin = create_model('swin_base_patch4_window7_224', pretrained=pretrained, num_classes=0)  # 1024-dim

        # Freeze backbones (recommended for AIGC detection with limited data)
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False
            for param in self.swin.parameters():
                param.requires_grad = False

        # Fusion MLP: 768 + 1024 = 1792 → 512 → 128 → 2
        self.fusion = nn.Sequential(
            nn.Linear(768 + 1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(128, 2)  # Exactly 2 classes: real vs synthetic
        )

    def forward(self, x):
        # Extract features
        vit_feat = self.vit(x)           # [B, 768]
        swin_feat = self.swin(x)         # [B, 1024]

        # For Swin, forward_features returns [B, H*W, C] → global avg pool if needed
        if len(swin_feat.shape) == 3:
            swin_feat = swin_feat.mean(1)  # [B, 1024]

        # Concatenate
        combined = torch.cat([vit_feat, swin_feat], dim=1)  # [B, 1792]

        # Final classification
        out = self.fusion(combined)
        return out