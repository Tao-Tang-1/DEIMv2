"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import copy
from calflops import calculate_flops
from typing import Tuple
import calflops.pytorch_ops as ops

def conv_flops_patch(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, **kwargs):
    B, C_in, H_in, W_in = input.shape
    C_out, _, kH, kW = weight.shape

    # 如果是 int，转换为 tuple
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    H_out = (H_in + 2*padding[0] - dilation[0]*(kH-1) - 1)//stride[0] + 1
    W_out = (W_in + 2*padding[1] - dilation[1]*(kW-1) - 1)//stride[1] + 1

    flops = B * H_out * W_out * kH * kW * (C_in//groups) * C_out
    if bias is not None:
        flops += B * H_out * W_out * C_out
    macs = flops // 2
    return flops, macs

ops._conv_flops_compute = conv_flops_patch

def stats(
    cfg,
    input_shape: Tuple=(1, 3, 640, 640), ) -> Tuple[int, dict]:

    base_size = cfg.train_dataloader.collate_fn.base_size
    input_shape = (1, 3, base_size, base_size)

    model_for_info = copy.deepcopy(cfg.model).deploy()

    flops, macs, _ = calculate_flops(model=model_for_info,
                                        input_shape=input_shape,
                                        output_as_string=True,
                                        output_precision=4,
                                        print_detailed=False)
    params = sum(p.numel() for p in model_for_info.parameters())
    del model_for_info

    return params, {"Model FLOPs:%s   MACs:%s   Params:%s" %(flops, macs, params)}
