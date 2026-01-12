"""
DEIMv2: Real-Time Object Detection Meets DINOv3
Copyright (c) 2025 The DEIMv2 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DINOv3 (https://github.com/facebookresearch/dinov3)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import register
from .vit_tiny import VisionTransformer
from .dinov3 import DinoVisionTransformer


# ================================
# Large Kernel Spatial Prior Module
# ================================
class LargeKernelSpatialPriorModule(nn.Module):
    def __init__(self, inplanes=16):
        super().__init__()

        # 1/4
        self.stem = nn.Sequential(
            nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # 1/8
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                inplanes,
                inplanes,
                kernel_size=7,
                stride=2,
                padding=3,
                groups=inplanes,
                bias=False,
            ),
            nn.Conv2d(inplanes, 2 * inplanes, kernel_size=1, bias=False),
            nn.SyncBatchNorm(2 * inplanes),
        )

        # 1/16
        self.conv3 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(
                2 * inplanes,
                4 * inplanes,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.SyncBatchNorm(4 * inplanes),
        )

        # 1/32
        self.conv4 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(
                4 * inplanes,
                4 * inplanes,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.SyncBatchNorm(4 * inplanes),
        )

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.conv2(c1)   # 1/8
        c3 = self.conv3(c2)   # 1/16
        c4 = self.conv4(c3)   # 1/32
        return c2, c3, c4


# ================================
# Backbone + Residual Concat Fusion
# ================================
@register()
class DINOv3STAs(nn.Module):
    def __init__(
        self,
        name=None,
        weights_path=None,
        interaction_indexes=[],
        finetune=True,
        embed_dim=192,
        num_heads=3,
        patch_size=16,
        use_sta=True,
        conv_inplane=16,
        hidden_dim=None,
    ):
        super().__init__()

        # -------- Backbone --------
        if 'dinov3' in name:
            self.dinov3 = DinoVisionTransformer(name=name)
            if weights_path and os.path.exists(weights_path):
                self.dinov3.load_state_dict(torch.load(weights_path))
        else:
            self.dinov3 = VisionTransformer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                return_layers=interaction_indexes
            )
            if weights_path and os.path.exists(weights_path):
                self.dinov3._model.load_state_dict(torch.load(weights_path))

        self.embed_dim = self.dinov3.embed_dim
        self.interaction_indexes = interaction_indexes
        self.patch_size = patch_size
        self.use_sta = use_sta

        if not finetune:
            self.dinov3.eval()
            self.dinov3.requires_grad_(False)

        # -------- Spatial Prior --------
        if use_sta:
            self.sta = LargeKernelSpatialPriorModule(inplanes=conv_inplane)

        # -------- Projection layers (for residual branch) --------
        hidden_dim = hidden_dim if hidden_dim is not None else self.embed_dim

        self.convs = nn.ModuleList([
            nn.Conv2d(self.embed_dim + conv_inplane * 2, hidden_dim, kernel_size=1, bias=False),
            nn.Conv2d(self.embed_dim + conv_inplane * 4, hidden_dim, kernel_size=1, bias=False),
            nn.Conv2d(self.embed_dim + conv_inplane * 4, hidden_dim, kernel_size=1, bias=False),
        ])

        self.norms = nn.ModuleList([
            nn.SyncBatchNorm(hidden_dim),
            nn.SyncBatchNorm(hidden_dim),
            nn.SyncBatchNorm(hidden_dim),
        ])

        # ⭐ 残差缩放系数（关键）
        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        bs = x.shape[0]
        H_c, W_c = x.shape[2] // 16, x.shape[3] // 16

        # -------- Transformer semantic features --------
        if len(self.interaction_indexes) > 0 and not isinstance(self.dinov3, VisionTransformer):
            all_layers = self.dinov3.get_intermediate_layers(
                x, n=self.interaction_indexes, return_class_token=True
            )
        else:
            all_layers = self.dinov3(x)

        if len(all_layers) == 1:
            all_layers = [all_layers[0], all_layers[0], all_layers[0]]

        sem_feats = []
        num_scales = len(all_layers) - 2

        for i, (feat, _) in enumerate(all_layers):
            f = feat.transpose(1, 2).view(bs, -1, H_c, W_c).contiguous()
            resize_H = int(H_c * 2 ** (num_scales - i))
            resize_W = int(W_c * 2 ** (num_scales - i))
            f = F.interpolate(f, size=(resize_H, resize_W),
                              mode="bilinear", align_corners=False)
            sem_feats.append(f)

        # -------- Residual Concat Fusion --------
        if self.use_sta:
            detail_feats = self.sta(x)
            out_feats = []

            for i, (sem_feat, detail_feat) in enumerate(zip(sem_feats, detail_feats)):
                # 对齐尺寸（保险）
                detail_feat = F.interpolate(
                    detail_feat,
                    size=sem_feat.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

                fused = torch.cat([sem_feat, detail_feat], dim=1)
                res = self.norms[i](self.convs[i](fused))

                # ⭐ 残差式 concat
                out_feats.append(sem_feat + self.gamma * res)
        else:
            out_feats = sem_feats

        return out_feats[0], out_feats[1], out_feats[2]
