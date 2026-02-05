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
from torchvision.ops import deform_conv2d

from functools import partial
from ..core import register
from .vit_tiny import VisionTransformer
from .dinov3 import DinoVisionTransformer

class SpatialPriorModulev2(nn.Module):
    def __init__(self, inplanes=16):
        super().__init__()

        # 1/4
        self.stem = nn.Sequential(
            *[
                nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(inplanes),
                nn.GELU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ]
        )
        # 1/8
        self.conv2 = nn.Sequential(
            *[
                nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(2 * inplanes),
            ]
        )
        # 1/16
        self.conv3 = nn.Sequential(
            *[
                nn.GELU(),
                nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(4 * inplanes),
            ]
        )
        # 1/32
        self.conv4 = nn.Sequential(
            *[
                nn.GELU(),
                nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(4 * inplanes),
            ]
        )

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.conv2(c1)     # 1/8
        c3 = self.conv3(c2)     # 1/16
        c4 = self.conv4(c3)     # 1/32

        return c2, c3, c4
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    return x.view(batchsize, -1, height, width)


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

        # --- 恢复完整的权重加载逻辑 ---
        if 'dinov3' in name:
            self.dinov3 = DinoVisionTransformer(name=name)
            if weights_path is not None and os.path.exists(weights_path):
                print(f'Loading DINOv3 ckpt from {weights_path}...')
                self.dinov3.load_state_dict(torch.load(weights_path))
            else:
                print('Training DINOv3 from scratch...')
        else:
            self.dinov3 = VisionTransformer(embed_dim=embed_dim, num_heads=num_heads, return_layers=interaction_indexes)
            if weights_path is not None and os.path.exists(weights_path):
                print(f'Loading ViT-Tiny ckpt from {weights_path}...')
                # 注意这里原代码中的 ._model 访问
                self.dinov3._model.load_state_dict(torch.load(weights_path))
            else:
                print('Training ViT-Tiny from scratch...')

        self.interaction_indexes = interaction_indexes
        self.patch_size = patch_size
        embed_dim = self.dinov3.embed_dim

        if not finetune:
            self.dinov3.eval()
            self.dinov3.requires_grad_(False)

        # --- 空间先验模块 (STA) ---
        self.use_sta = use_sta
        if use_sta:
            print(f"Using Lite Spatial Prior with AFS Fusion, inplanes={conv_inplane}")
            self.sta = SpatialPriorModulev2(inplanes=conv_inplane)

            # 计算融合后的通道数 (ViT + CNN)
            c2_in = embed_dim + conv_inplane * 2
            c3_in = embed_dim + conv_inplane * 4
            c4_in = embed_dim + conv_inplane * 4
        else:
            c2_in = c3_in = c4_in = embed_dim

        # --- 投影层与 Norm ---
        hidden_dim = hidden_dim if hidden_dim is not None else embed_dim
        self.convs = nn.ModuleList([
            nn.Conv2d(c2_in, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(c3_in, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(c4_in, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
        ])

        self.norms = nn.ModuleList([
            nn.SyncBatchNorm(hidden_dim),
            nn.SyncBatchNorm(hidden_dim),
            nn.SyncBatchNorm(hidden_dim)
        ])

    def forward(self, x):
        bs, C, h, w = x.shape
        H_c, W_c = h // 16, w // 16

        # 提取 Vision Transformer 特征
        if len(self.interaction_indexes) > 0 and not isinstance(self.dinov3, VisionTransformer):
            all_layers = self.dinov3.get_intermediate_layers(x, n=self.interaction_indexes, return_class_token=True)
        else:
            all_layers = self.dinov3(x)

        if len(all_layers) == 1:
            all_layers = [all_layers[0]] * 3

        sem_feats = []
        num_scales = len(all_layers) - 2
        for i, sem_feat in enumerate(all_layers):
            feat, _ = sem_feat
            # 还原空间维度 [B, D, H, W]
            sem_feat = feat.transpose(1, 2).view(bs, -1, H_c, W_c).contiguous()
            # 缩放至目标尺寸
            target_H, target_W = int(H_c * 2 ** (num_scales - i)), int(W_c * 2 ** (num_scales - i))
            sem_feat = F.interpolate(sem_feat, size=[target_H, target_W], mode="bilinear", align_corners=False)
            sem_feats.append(sem_feat)

        # --- 创新：不对称洗牌融合 (AFS) ---
        if self.use_sta:
            detail_feats = self.sta(x)  # 得到 c2, c3, c4

            # 依次处理三层特征
            c2_fused = torch.cat([sem_feats[0], detail_feats[0]], dim=1)
            c2_fused = channel_shuffle(c2_fused, groups=2)
            c2 = self.norms[0](self.convs[0](c2_fused))

            c3_fused = torch.cat([sem_feats[1], detail_feats[1]], dim=1)
            c3_fused = channel_shuffle(c3_fused, groups=2)
            c3 = self.norms[1](self.convs[1](c3_fused))

            c4_fused = torch.cat([sem_feats[2], detail_feats[2]], dim=1)
            c4_fused = channel_shuffle(c4_fused, groups=2)
            c4 = self.norms[2](self.convs[2](c4_fused))
        else:
            c2 = self.norms[0](self.convs[0](sem_feats[0]))
            c3 = self.norms[1](self.convs[1](sem_feats[1]))
            c4 = self.norms[2](self.convs[2](sem_feats[2]))

        return c2, c3, c4