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
from ..core import register
from .vit_tiny import VisionTransformer
from .dinov3 import DinoVisionTransformer

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    return x.view(batchsize, -1, height, width)

def depthwise_separable_conv(in_ch, out_ch, stride=1):
    return nn.Sequential(
        # Depthwise
        nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False),
        nn.SyncBatchNorm(in_ch),
        nn.GELU(),
        # Pointwise
        nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
        nn.SyncBatchNorm(out_ch),
    )

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
        self.conv2 = depthwise_separable_conv(inplanes, 2 * inplanes, stride=2)
        # 1/16
        self.conv3 = nn.Sequential(
            nn.GELU(),
            depthwise_separable_conv(2 * inplanes, 4 * inplanes, stride=2)
        )
        # 1/32
        self.conv4 = nn.Sequential(
            nn.GELU(),
            depthwise_separable_conv(4 * inplanes, 4 * inplanes, stride=2)
        )

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.conv2(c1)     # 1/8
        c3 = self.conv3(c2)     # 1/16
        c4 = self.conv4(c3)     # 1/32

        return c2, c3, c4


@register()
class CG_AFS_DINOv3STAs(nn.Module):
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
        super(CG_AFS_DINOv3STAs, self).__init__()

        # 1. 加载 Backbone 逻辑
        if 'dinov3' in name:
            self.dinov3 = DinoVisionTransformer(name=name)
            if weights_path is not None and os.path.exists(weights_path):
                print(f'Loading DINOv3 ckpt from {weights_path}...')
                self.dinov3.load_state_dict(torch.load(weights_path))
        else:
            self.dinov3 = VisionTransformer(embed_dim=embed_dim, num_heads=num_heads, return_layers=interaction_indexes)
            if weights_path is not None and os.path.exists(weights_path):
                print(f'Loading ViT-Tiny ckpt from {weights_path}...')
                self.dinov3._model.load_state_dict(torch.load(weights_path))

        self.interaction_indexes = interaction_indexes
        self.patch_size = patch_size
        embed_dim = self.dinov3.embed_dim

        if not finetune:
            self.dinov3.eval()
            self.dinov3.requires_grad_(False)

        # 2. 初始化 STA
        self.use_sta = use_sta
        if use_sta:
            self.sta = SpatialPriorModulev2(inplanes=conv_inplane)
            c2_in = embed_dim + conv_inplane * 2
            c3_in = embed_dim + conv_inplane * 4
            c4_in = embed_dim + conv_inplane * 4
        else:
            c2_in = c3_in = c4_in = embed_dim

        # 3. 投影层
        hidden_dim = hidden_dim if hidden_dim is not None else embed_dim
        self.convs = nn.ModuleList([
            nn.Conv2d(c2_in, hidden_dim, kernel_size=1, bias=False),
            nn.Conv2d(c3_in, hidden_dim, kernel_size=1, bias=False),
            nn.Conv2d(c4_in, hidden_dim, kernel_size=1, bias=False)
        ])
        self.norms = nn.ModuleList([
            nn.SyncBatchNorm(hidden_dim),
            nn.SyncBatchNorm(hidden_dim),
            nn.SyncBatchNorm(hidden_dim)
        ])

    def forward(self, x):
        bs, _, h, w = x.shape
        H_c, W_c = h // 16, w // 16

        # 提取 ViT 特征
        if len(self.interaction_indexes) > 0 and not isinstance(self.dinov3, VisionTransformer):
            all_layers = self.dinov3.get_intermediate_layers(x, n=self.interaction_indexes, return_class_token=True)
        else:
            all_layers = self.dinov3(x)

        if len(all_layers) == 1:
            all_layers = [all_layers[0]] * 3

        sem_feats = []
        num_scales = len(all_layers) - 2
        for i, (feat, _) in enumerate(all_layers):
            s_feat = feat.transpose(1, 2).view(bs, -1, H_c, W_c).contiguous()
            target_H, target_W = int(H_c * 2 ** (num_scales - i)), int(W_c * 2 ** (num_scales - i))
            s_feat = F.interpolate(s_feat, size=[target_H, target_W], mode="bilinear", align_corners=False)
            sem_feats.append(s_feat)

        if self.use_sta:
            detail_feats = self.sta(x)
            final_outs = []
            for i, (sem, det) in enumerate(zip(sem_feats, detail_feats)):
                # 改进：引入 Soft-Center Gating
                # 相比 max 或 mean，使用自适应注意力权重能让大目标的特征更“纯”
                # 计算通道注意力 (Squeeze-and-Excitation 思想)
                avg_pool = torch.mean(sem, dim=(2, 3), keepdim=True)
                channel_attn = torch.sigmoid(avg_pool)
                sem = sem * channel_attn

                # 空间门控保持聚焦
                spatial_mask = torch.sigmoid(torch.mean(sem, dim=1, keepdim=True))
                det_guided = det * spatial_mask

                # 融合并洗牌
                fused = torch.cat([sem, det_guided], dim=1)
                fused = channel_shuffle(fused, groups=2)

                out = self.norms[i](self.convs[i](fused))
                final_outs.append(out)
            return tuple(final_outs)

        return [self.norms[i](self.convs[i](f)) for i, f in enumerate(sem_feats)]

# @register()
# class DINOv3STAs(nn.Module):
#     def __init__(
#             self,
#             name=None,
#             weights_path=None,
#             interaction_indexes=[],
#             finetune=True,
#             embed_dim=192,
#             num_heads=3,
#             patch_size=16,
#             use_sta=True,
#             conv_inplane=16,
#             hidden_dim=None,
#     ):
#         super(DINOv3STAs, self).__init__()
#         if 'dinov3' in name:
#             self.dinov3 = torch.hub.load('./dinov3', name, source='local', weights=weights_path)
#             while len(self.dinov3.blocks) != (interaction_indexes[-1] + 1):
#                 del self.dinov3.blocks[-1]
#             del self.dinov3.head
#         else:
#             self.dinov3 = VisionTransformer(embed_dim=embed_dim, num_heads=num_heads)
#             if weights_path is not None:
#                 print(f'Loading ckpt from {weights_path}...')
#                 checkpoint = torch.load(weights_path)
#                 self.dinov3._model.load_state_dict(checkpoint)
#             else:
#                 print('Training ViT-Tiny from scratch!')
#
#         embed_dim = self.dinov3.embed_dim
#         self.interaction_indexes = interaction_indexes
#         self.patch_size = patch_size
#
#         if not finetune:
#             self.dinov3.eval()
#             self.dinov3.requires_grad_(False)
#
#         # init the feature pyramid
#         self.use_sta = use_sta
#         if use_sta:
#             print(f"Using Lite Spatial Prior Module with inplanes={conv_inplane}")
#             self.sta = SpatialPriorModulev2(inplanes=conv_inplane)
#         else:
#             conv_inplane = 0
#
#         # linear projection
#         hidden_dim = hidden_dim if hidden_dim is not None else embed_dim
#         self.convs = nn.ModuleList([
#             nn.Conv2d(embed_dim + conv_inplane * 2, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.Conv2d(embed_dim + conv_inplane * 4, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.Conv2d(embed_dim + conv_inplane * 4, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
#         ])
#         # norm
#         self.norms = nn.ModuleList([
#             nn.SyncBatchNorm(hidden_dim),
#             nn.SyncBatchNorm(hidden_dim),
#             nn.SyncBatchNorm(hidden_dim)
#         ])
#
#     def forward(self, x):
#         # Code for matching with oss
#         H_c, W_c = x.shape[2] // 16, x.shape[3] // 16
#         H_toks, W_toks = x.shape[2] // self.patch_size, x.shape[3] // self.patch_size
#         bs, C, h, w = x.shape
#
#         if len(self.interaction_indexes) > 0 and not isinstance(self.dinov3, VisionTransformer):
#             all_layers = self.dinov3.get_intermediate_layers(
#                 x, n=self.interaction_indexes, return_class_token=True
#             )
#         else:
#             all_layers = self.dinov3(x)
#
#         if len(all_layers) == 1:  # repeat the same layer for all the three scales
#             all_layers = [all_layers[0], all_layers[0], all_layers[0]]
#
#         sem_feats = []
#         num_scales = len(all_layers) - 2
#         for i, sem_feat in enumerate(all_layers):
#             feat, _ = sem_feat
#             sem_feat = feat.transpose(1, 2).view(bs, -1, H_c, W_c).contiguous()  # [B, D, H, W]
#             resize_H, resize_W = int(H_c * 2 ** (num_scales - i)), int(W_c * 2 ** (num_scales - i))
#             sem_feat = F.interpolate(sem_feat, size=[resize_H, resize_W], mode="bilinear", align_corners=False)
#             sem_feats.append(sem_feat)
#
#         # fusion
#         fused_feats = []
#         if self.use_sta:
#             detail_feats = self.sta(x)
#             for sem_feat, detail_feat in zip(sem_feats, detail_feats):
#                 fused_feats.append(torch.cat([sem_feat, detail_feat], dim=1))
#         else:
#             fused_feats = sem_feats
#
#         c2 = self.norms[0](self.convs[0](fused_feats[0]))
#         c3 = self.norms[1](self.convs[1](fused_feats[1]))
#         c4 = self.norms[2](self.convs[2](fused_feats[2]))
#
#         return c2, c3, c4