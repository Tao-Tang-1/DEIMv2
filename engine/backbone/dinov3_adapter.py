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
class LargeKernelSpatialPriorModule(nn.Module):
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
        # self.conv2 = nn.Sequential(
        #     *[
        #         nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
        #         nn.SyncBatchNorm(2 * inplanes),
        #     ]
        # )
        # # 1/16
        # self.conv3 = nn.Sequential(
        #     *[
        #         nn.GELU(),
        #         nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
        #         nn.SyncBatchNorm(4 * inplanes),
        #     ]
        # )
        ## 1/8
        self.conv2 = nn.Sequential(
            nn.Conv2d(inplanes, inplanes, kernel_size=7, stride=2, padding=3, groups=inplanes, bias=False),
            # Depthwise conv
            nn.Conv2d(inplanes, 2 * inplanes, kernel_size=1, bias=False),  # Pointwise conv
            nn.SyncBatchNorm(2 * inplanes),
        )

        ## 1/16
        self.conv3 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
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

class FeatureAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 针对融合后的总通道数进行加权
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 计算每个通道的权重并作用于输入
        # 将原有的 x * self.fc(...) 改为残差形式
        # 这能确保基准特征流（Identity）不被破坏
        return x + x * self.fc(self.gap(x))

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 基于通道维度的平均池化和最大池化
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        res = torch.cat([avg_out, max_out], dim=1)
        res = self.sigmoid(self.conv(res))
        return x * res

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
            self.sta = LargeKernelSpatialPriorModule(inplanes=conv_inplane)
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
        # 在初始化卷积层的地方同时初始化注意力层
        # 通道数对应 fused_feats 的维度：
        # scale 0: embed_dim + conv_inplane*2
        # scale 1: embed_dim + conv_inplane*4
        # scale 2: embed_dim + conv_inplane*4
        self.attns = nn.ModuleList([
            FeatureAttention(embed_dim + conv_inplane * 2),
            FeatureAttention(embed_dim + conv_inplane * 4),
            FeatureAttention(embed_dim + conv_inplane * 4)
        ])

        self.spatials = nn.ModuleList([
            SpatialAttention(kernel_size=7) for _ in range(3)
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
        # --- 融合部分 ---
        fused_feats = []
        if self.use_sta:
            detail_feats = self.sta(x)
            for i, (sem_feat, detail_feat) in enumerate(zip(sem_feats, detail_feats)):
                # 1. 拼接
                f = torch.cat([sem_feat, detail_feat], dim=1)
                # 2. 加入注意力进行筛选 (新增)
                f = self.attns[i](f)  # 通道注意力
                f = self.spatials[i](f)  # 空间注意力 (可选尝试)
                fused_feats.append(f)
        else:
            fused_feats = sem_feats

        # --- 降维投影部分 ---
        c2 = self.norms[0](self.convs[0](fused_feats[0]))
        c3 = self.norms[1](self.convs[1](fused_feats[1]))
        c4 = self.norms[2](self.convs[2](fused_feats[2]))

        return c2, c3, c4