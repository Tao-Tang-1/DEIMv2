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

# class LargeKernelSpatialPriorModule(nn.Module):
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
# class LargeKernelSpatialPriorModule(nn.Module):
#     def __init__(self, inplanes=16):
#         super().__init__()
#
#         # 1/4 - 保持不变
#         self.stem = nn.Sequential(
#             *[
#                 nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
#                 nn.SyncBatchNorm(inplanes),
#                 nn.GELU(),
#                 nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
#             ]
#         )
#
#         # 1/8 - 保持 Large Kernel (7x7)
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(inplanes, inplanes, kernel_size=7, stride=2, padding=3, groups=inplanes, bias=False),
#             nn.Conv2d(inplanes, 2 * inplanes, kernel_size=1, bias=False),
#             nn.SyncBatchNorm(2 * inplanes),
#         )
#
#         # 1/16 - 【Exp 2 修改点】移除 Dilation
#         # 原 padding=6 (dilation=2), 修改为 padding=3 (dilation=1)
#         self.conv3 = nn.Sequential(
#             nn.GELU(),
#             nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=7, stride=2,
#                       padding=3, dilation=1, bias=False), # 此处移除空洞
#             nn.SyncBatchNorm(4 * inplanes),
#         )
#
#         # 1/32 - 【Exp 2 修改点】移除 Dilation
#         # 原 padding=2 (dilation=2), 修改为 padding=1 (dilation=1)
#         self.conv4 = nn.Sequential(
#             nn.GELU(),
#             nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2,
#                       padding=1, dilation=1, bias=False), # 此处移除空洞
#             nn.SyncBatchNorm(4 * inplanes),
#         )
#
#     def forward(self, x):
#         c1 = self.stem(x)
#         c2 = self.conv2(c1)
#         c3 = self.conv3(c2)
#         c4 = self.conv4(c3)
#         return c2, c3, c4


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

        # 【新增改进：异构特征校准因子】
        # 为每个尺度的拼接特征定义一个可学习的权重向量，初始化为 0.1 以稳定训练
        self.gamma2 = nn.Parameter(torch.ones(embed_dim + conv_inplane * 2) * 0.1)
        self.gamma3 = nn.Parameter(torch.ones(embed_dim + conv_inplane * 4) * 0.1)
        self.gamma4 = nn.Parameter(torch.ones(embed_dim + conv_inplane * 4) * 0.1)

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
        # if self.use_sta:
        #     detail_feats = self.sta(x)
        #     for sem_feat, detail_feat in zip(sem_feats, detail_feats):
        #         fused_feats.append(torch.cat([sem_feat, detail_feat], dim=1))
        # else:
        #     fused_feats = sem_feats
        #
        # c2 = self.norms[0](self.convs[0](fused_feats[0]))
        # c3 = self.norms[1](self.convs[1](fused_feats[1]))
        # c4 = self.norms[2](self.convs[2](fused_feats[2]))
        #
        # return c2, c3, c4

        # fusion 逻辑修改
        fused_feats = []
        if self.use_sta:
            detail_feats = self.sta(x)  # 得到 c2, c3, c4

            # 尺度 1 (1/8): 拼接并校准
            f2 = torch.cat([sem_feats[0], detail_feats[0]], dim=1)
            f2 = f2 * self.gamma2.view(1, -1, 1, 1)

            # 尺度 2 (1/16): 拼接并校准
            f3 = torch.cat([sem_feats[1], detail_feats[1]], dim=1)
            f3 = f3 * self.gamma3.view(1, -1, 1, 1)

            # 尺度 3 (1/32): 拼接并校准
            f4 = torch.cat([sem_feats[2], detail_feats[2]], dim=1)
            f4 = f4 * self.gamma4.view(1, -1, 1, 1)

            # 映射并归一化
            c2 = self.norms[0](self.convs[0](f2))
            c3 = self.norms[1](self.convs[1](f3))
            c4 = self.norms[2](self.convs[2](f4))
        else:
            # 如果不使用 sta，直接处理 sem_feats
            c2 = self.norms[0](self.convs[0](sem_feats[0]))
            c3 = self.norms[1](self.convs[1](sem_feats[1]))
            c4 = self.norms[2](self.convs[2](sem_feats[2]))

        return c2, c3, c4