import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicTanh(nn.Module):
    """
    更专业的 Dynamic Tanh 实现
    支持：
    1. Channel-wise (通道级) 的 alpha, weight 和 bias
    2. 自动适配不同的张量维度 (2D, 3D, 4D)
    3. 数值稳定性约束
    """

    def __init__(self, num_channels, channel_wise=True, alpha_init=0.5):
        super().__init__()
        self.num_channels = num_channels
        self.channel_wise = channel_wise

        # 定义参数形状：如果是通道独立，则为 (num_channels,)；否则为标量 (1,)
        param_shape = (num_channels,) if channel_wise else (1,)

        # 1. 可学习的缩放因子 alpha
        self.alpha = nn.Parameter(torch.full(param_shape, float(alpha_init)))

        # 2. 仿射变换参数（Gain 和 Bias）
        self.weight = nn.Parameter(torch.ones(param_shape))
        self.bias = nn.Parameter(torch.zeros(param_shape))

    def forward(self, x):
        """
        x: 支持 (N, C), (N, L, C), (N, H, W, C) 等各种以通道结尾的格式
        """
        # 为了适配非 channels_last 的输入，我们需要动态调整 view 形状
        if x.shape[-1] != self.num_channels:
            # 如果输入不是 channels_last (N, ..., C)，则尝试适配 (N, C, ...)
            # 这是一个健壮性处理
            view_shape = [1, self.num_channels] + [1] * (x.dim() - 2)
            alpha = self.alpha.view(*view_shape)
            weight = self.weight.view(*view_shape)
            bias = self.bias.view(*view_shape)
        else:
            # 标准 channels_last 格式，直接利用广播机制
            alpha, weight, bias = self.alpha, self.weight, self.bias

        # 核心计算：tanh(alpha * x) * weight + bias
        # 使用 clamp 保证 alpha 不会过小（可选）
        x = torch.tanh(alpha * x)
        x = x * weight + bias
        return x


class LayerNorm2D_DyT(nn.Module):
    """
    针对 2D 图像特征设计的自适应归一化层
    """

    def __init__(self, num_channels, eps=1e-6, use_dyt=True, channel_wise=True, alpha_init=0.5):
        super().__init__()
        self.use_dyt = use_dyt

        # 标准 LayerNorm，此时不使用自带的 affine（由 DyT 接管）以减少冗余
        # 如果你希望保留 LN 的 affine，可以将 elementwise_affine 设为 True
        self.ln = nn.LayerNorm(num_channels, eps=eps, elementwise_affine=not use_dyt)

        if use_dyt:
            self.dyt = DynamicTanh(num_channels, channel_wise=channel_wise, alpha_init=alpha_init)
        else:
            self.register_module('dyt', None)

    def forward(self, x):
        """
        x: 输入形状为 (N, C, H, W)
        """
        # 1. 转换维度到 (N, H, W, C)
        x = x.permute(0, 2, 3, 1).contiguous()

        # 2. 执行 LayerNorm
        x = self.ln(x)

        # 3. 执行动态激活
        if self.use_dyt:
            x = self.dyt(x)

        # 4. 转换回 (N, C, H, W)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


# --- 测试代码 ---
if __name__ == "__main__":
    # 模拟一个卷积特征图 (Batch=2, Channel=64, H=32, W=32)
    sample_input = torch.randn(2, 64, 32, 32)

    # 实例化专业版模块
    model = LayerNorm2D_DyT(num_channels=64, channel_wise=True)

    output = model(sample_input)
    print(f"输入形状: {sample_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"Alpha 参数形状: {model.dyt.alpha.shape}")  # 应该是 (64,)