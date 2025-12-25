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


# class SpatialPriorModulev2(nn.Module):
#     def __init__(self, inplanes=16):
#         super().__init__()
#
#         # 1/4
#         self.stem = nn.Sequential(
#             *[
#                 nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
#                 nn.SyncBatchNorm(inplanes),
#                 nn.GELU(),
#                 nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
#             ]
#         )
#         # 1/8
#         self.conv2 = nn.Sequential(
#             *[
#                 nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
#                 nn.SyncBatchNorm(2 * inplanes),
#             ]
#         )
#         # 1/16
#         self.conv3 = nn.Sequential(
#             *[
#                 nn.GELU(),
#                 nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
#                 nn.SyncBatchNorm(4 * inplanes),
#             ]
#         )
#         ## 1/8
#         # self.conv2 = nn.Sequential(
#         #     nn.Conv2d(inplanes, inplanes, kernel_size=7, stride=2, padding=3, groups=inplanes, bias=False),
#         #     # Depthwise conv
#         #     nn.Conv2d(inplanes, 2 * inplanes, kernel_size=1, bias=False),  # Pointwise conv
#         #     nn.SyncBatchNorm(2 * inplanes),
#         # )
#         #
#         # ## 1/16
#         # self.conv3 = nn.Sequential(
#         #     nn.GELU(),
#         #     nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=7, stride=2, padding=3, bias=False),
#         #     nn.SyncBatchNorm(4 * inplanes),
#         # )
#         # 1/32
#         self.conv4 = nn.Sequential(
#             *[
#                 nn.GELU(),
#                 nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
#                 nn.SyncBatchNorm(4 * inplanes),
#             ]
#         )
#
#     def forward(self, x):
#         c1 = self.stem(x)
#         c2 = self.conv2(c1)     # 1/8
#         c3 = self.conv3(c2)     # 1/16
#         c4 = self.conv4(c3)     # 1/32
#
#         return c2, c3, c4

class DeepSpatialPrior(nn.Module):
    def __init__(self, inplanes=16):
        super().__init__()
        # 输入下采样 1/4
        self.stem = nn.Sequential(
            nn.Conv2d(3, inplanes, 7, stride=2, padding=3, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.GELU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        # 1/8
        self.conv2 = nn.Sequential(
            nn.Conv2d(inplanes, 2*inplanes, 5, stride=2, padding=2, bias=False),
            nn.SyncBatchNorm(2*inplanes),
            nn.GELU(),
        )

        # 1/16 多感受野
        self.conv3 = nn.ModuleList([
            nn.Conv2d(2*inplanes, 4*inplanes, 3, stride=2, padding=1, dilation=1, bias=False),
            nn.Conv2d(2*inplanes, 4*inplanes, 3, stride=2, padding=2, dilation=2, bias=False),
            nn.Conv2d(2*inplanes, 4*inplanes, 3, stride=2, padding=3, dilation=3, bias=False),
        ])
        self.bn3 = nn.SyncBatchNorm(4*inplanes)
        self.gelu3 = nn.GELU()

        # 1/32
        self.conv4 = nn.Sequential(
            nn.Conv2d(4*inplanes, 4*inplanes, 3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4*inplanes),
            nn.GELU(),
        )

        # 通道注意力
        self.ca2 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(2*inplanes, 2*inplanes, 1), nn.Sigmoid())
        self.ca3 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(4*inplanes, 4*inplanes, 1), nn.Sigmoid())
        self.ca4 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(4*inplanes, 4*inplanes, 1), nn.Sigmoid())

        # 对齐 c2 通道数到 4*inplanes
        self.align_c2 = nn.Conv2d(2*inplanes, 4*inplanes, 1, bias=False)

    def forward(self, x):
        c1 = self.stem(x)           # 1/4
        c2 = self.conv2(c1)         # 1/8
        c2 = c2 * self.ca2(c2)      # 通道注意力增强

        # 多感受野卷积融合
        c3 = sum(conv(c2) for conv in self.conv3)
        c3 = self.bn3(c3)
        c3 = self.gelu3(c3)
        c3 = c3 * self.ca3(c3)      # 通道注意力增强

        c4 = self.conv4(c3)         # 1/32
        c4 = c4 * self.ca4(c4)      # 通道注意力增强

        # 跨尺度融合
        c2_aligned = self.align_c2(c2)   # 对齐通道数
        c3_up = F.interpolate(c3, size=c2.shape[2:], mode='bilinear', align_corners=False)
        c4_up = F.interpolate(c4, size=c2.shape[2:], mode='bilinear', align_corners=False)
        fused = c2_aligned + c3_up + c4_up

        return c2, c3, c4, fused



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
        if 'dinov3' in name:
            self.dinov3 = DinoVisionTransformer(name=name)
            if weights_path is not None and os.path.exists(weights_path):
                print(f'Loading ckpt from {weights_path}...')
                self.dinov3.load_state_dict(torch.load(weights_path))
            else:
                print('Training DINOv3 from scratch...')
        else:
            self.dinov3 =  VisionTransformer(embed_dim=embed_dim, num_heads=num_heads, return_layers=interaction_indexes)
            if weights_path is not None and os.path.exists(weights_path):
                print(f'Loading ckpt from {weights_path}...')
                self.dinov3._model.load_state_dict(torch.load(weights_path))
            else:
                print('Training ViT-Tiny from scratch...')

        embed_dim = self.dinov3.embed_dim
        self.interaction_indexes = interaction_indexes
        self.patch_size = patch_size

        if not finetune:
            self.dinov3.eval()
            self.dinov3.requires_grad_(False)

        # init the feature pyramid
        self.use_sta = use_sta
        if use_sta:
            print(f"Using Lite Spatial Prior Module with inplanes={conv_inplane}")
            # self.sta = SpatialPriorModulev2(inplanes=conv_inplane)
            self.sta = DeepSpatialPrior(inplanes=conv_inplane)
        else:
            conv_inplane = 0

        # linear projection
        hidden_dim = hidden_dim if hidden_dim is not None else embed_dim
        self.convs = nn.ModuleList([
            nn.Conv2d(embed_dim + conv_inplane*2, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(embed_dim + conv_inplane*4, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(embed_dim + conv_inplane*4, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
        ])
        # norm
        self.norms = nn.ModuleList([
            nn.SyncBatchNorm(hidden_dim),
            nn.SyncBatchNorm(hidden_dim),
            nn.SyncBatchNorm(hidden_dim)
        ])

    def forward(self, x):
        # Code for matching with oss
        H_c, W_c = x.shape[2] // 16, x.shape[3] // 16
        H_toks, W_toks = x.shape[2] // self.patch_size, x.shape[3] // self.patch_size
        bs, C, h, w = x.shape

        if len(self.interaction_indexes) > 0 and not isinstance(self.dinov3, VisionTransformer):
            all_layers = self.dinov3.get_intermediate_layers(
                x, n=self.interaction_indexes, return_class_token=True
            )
        else:
            all_layers = self.dinov3(x)

        if len(all_layers) == 1:    # repeat the same layer for all the three scales
            all_layers = [all_layers[0], all_layers[0], all_layers[0]]

        sem_feats = []
        num_scales = len(all_layers) - 2
        for i, sem_feat in enumerate(all_layers):
            feat, _ = sem_feat
            sem_feat = feat.transpose(1, 2).view(bs, -1, H_c, W_c).contiguous()  # [B, D, H, W]
            resize_H, resize_W = int(H_c * 2**(num_scales-i)), int(W_c * 2**(num_scales-i))
            sem_feat = F.interpolate(sem_feat, size=[resize_H, resize_W], mode="bilinear", align_corners=False)
            sem_feats.append(sem_feat)

        # fusion
        fused_feats = []
        if self.use_sta:
            detail_feats = self.sta(x)
            for sem_feat, detail_feat in zip(sem_feats, detail_feats):
                fused_feats.append(torch.cat([sem_feat, detail_feat], dim=1))
        else:
            fused_feats = sem_feats

        c2 = self.norms[0](self.convs[0](fused_feats[0]))
        c3 = self.norms[1](self.convs[1](fused_feats[1]))
        c4 = self.norms[2](self.convs[2](fused_feats[2]))

        return c2, c3, c4