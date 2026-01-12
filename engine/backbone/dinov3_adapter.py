"""
DEIMv2: Real-Time Object Detection Meets DINOv3
Copyright (c) 2025 The DEIMv2 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DINOv3 (https://github.com/facebookresearch/dinov3)

Copyright (c) Meta Platforms, Inc. and affiliates.

This software may be used and distributed in accordance with
the terms of the DINOv3 License Agreement.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from functools import partial
from ..core import register
from .vit_tiny import VisionTransformer
from .dinov3 import DinoVisionTransformer


# 1. 空间先验模块 (CNN 分支)
class LargeKernelSpatialPriorModule(nn.Module):
    def __init__(self, inplanes=16):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inplanes, inplanes, kernel_size=7, stride=2, padding=3, groups=inplanes, bias=False),
            nn.Conv2d(inplanes, 2 * inplanes, kernel_size=1, bias=False),
            nn.SyncBatchNorm(2 * inplanes),
        )
        self.conv3 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
        )
        self.conv4 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
        )

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.conv2(c1)  # 1/8
        c3 = self.conv3(c2)  # 1/16
        c4 = self.conv4(c3)  # 1/32
        return c2, c3, c4


# 2. 空间引导模块 (融合转换器)
class SpatialGuidanceModule(nn.Module):
    def __init__(self, detail_dim, sem_dim):
        super().__init__()
        self.guidance = nn.Sequential(
            nn.Conv2d(detail_dim, detail_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.SyncBatchNorm(detail_dim // 2),
            nn.GELU(),
            nn.Conv2d(detail_dim // 2, sem_dim, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, sem_feat, detail_feat):
        weight = self.guidance(detail_feat)
        return sem_feat * (1 + weight)  # 残差加权增强


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
        super(DINOv3STAs, self).__init__()

        # 初始化 Backbone
        if 'dinov3' in name:
            self.dinov3 = DinoVisionTransformer(name=name)
            if weights_path and os.path.exists(weights_path):
                self.dinov3.load_state_dict(torch.load(weights_path))
        else:
            self.dinov3 = VisionTransformer(embed_dim=embed_dim, num_heads=num_heads, return_layers=interaction_indexes)
            if weights_path and os.path.exists(weights_path):
                self.dinov3._model.load_state_dict(torch.load(weights_path))

        self.embed_dim = self.dinov3.embed_dim
        self.interaction_indexes = interaction_indexes
        self.patch_size = patch_size
        self.use_sta = use_sta

        if not finetune:
            self.dinov3.eval()
            self.dinov3.requires_grad_(False)

        # 初始化 STA 增强模块
        if use_sta:
            print(f"Using Large Kernel Spatial Prior + Spatial Guidance")
            self.sta = LargeKernelSpatialPriorModule(inplanes=conv_inplane)
            self.guidance2 = SpatialGuidanceModule(conv_inplane * 2, self.embed_dim)
            self.guidance3 = SpatialGuidanceModule(conv_inplane * 4, self.embed_dim)
            self.guidance4 = SpatialGuidanceModule(conv_inplane * 4, self.embed_dim)

        # 3. 投影层：注意！必须放在 if/else 外面，且输入通道始终是 embed_dim
        hidden_dim = hidden_dim if hidden_dim is not None else self.embed_dim
        self.convs = nn.ModuleList([
            nn.Conv2d(self.embed_dim, hidden_dim, kernel_size=1, bias=False) for _ in range(3)
        ])
        self.norms = nn.ModuleList([
            nn.SyncBatchNorm(hidden_dim) for _ in range(3)
        ])

    def forward(self, x):
        bs = x.shape[0]
        H_c, W_c = x.shape[2] // 16, x.shape[3] // 16

        # 提取语义特征
        if len(self.interaction_indexes) > 0 and not isinstance(self.dinov3, VisionTransformer):
            all_layers = self.dinov3.get_intermediate_layers(x, n=self.interaction_indexes, return_class_token=True)
        else:
            all_layers = self.dinov3(x)

        if len(all_layers) == 1:
            all_layers = [all_layers[0], all_layers[0], all_layers[0]]

        sem_feats = []
        num_scales = len(all_layers) - 2
        for i, (feat, _) in enumerate(all_layers):
            f = feat.transpose(1, 2).view(bs, -1, H_c, W_c).contiguous()
            resize_H, resize_W = int(H_c * 2 ** (num_scales - i)), int(W_c * 2 ** (num_scales - i))
            f = F.interpolate(f, size=[resize_H, resize_W], mode="bilinear", align_corners=False)
            sem_feats.append(f)

        # 特征增强融合
        if self.use_sta:
            detail_feats = self.sta(x)
            out2 = self.guidance2(sem_feats[0], detail_feats[0])
            out3 = self.guidance3(sem_feats[1], detail_feats[1])
            out4 = self.guidance4(sem_feats[2], detail_feats[2])
        else:
            out2, out3, out4 = sem_feats

        # 最终投影输出
        c2 = self.norms[0](self.convs[0](out2))
        c3 = self.norms[1](self.convs[1](out3))
        c4 = self.norms[2](self.convs[2](out4))

        return c2, c3, c4