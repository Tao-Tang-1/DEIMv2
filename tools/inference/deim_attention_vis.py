import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import cv2
import numpy as np
from PIL import Image

# --- 环境配置 ---
project_root = "/zhangcc/tq/code/DEIMv2"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from engine.core import YAMLConfig


class DeformableRawHook:
    def __init__(self):
        self.query = None
        self.ref_points = None

    def __call__(self, module, input, output):
        self.query = input[0].detach()
        self.ref_points = input[1].detach()


def visualize_top_tier_blue(model, config, img_path, output_path):
    size = config.yaml_cfg["eval_spatial_size"]
    im_pil = Image.open(img_path).convert('RGB')
    w, h = im_pil.size

    img_tensor = T.Compose([T.Resize(size), T.ToTensor()])(im_pil).unsqueeze(0).cuda()

    hook = DeformableRawHook()
    target_layer = model.decoder.decoder.layers[3].cross_attn
    handle = target_layer.register_forward_hook(hook)

    with torch.no_grad():
        outputs = model(img_tensor)
    handle.remove()

    # --- 参数解析 ---
    num_heads = target_layer.num_heads
    num_levels = target_layer.num_levels
    num_points_list = target_layer.num_points_list
    offset_scale = target_layer.offset_scale
    num_points_scale = target_layer.num_points_scale.to(img_tensor.dtype).unsqueeze(-1)

    logits = outputs['pred_logits'].sigmoid()[0]
    query_idx = logits.max(-1)[0].argmax().item()

    q = hook.query[:, query_idx:query_idx + 1]
    ref = hook.ref_points[:, query_idx:query_idx + 1]

    with torch.no_grad():
        sampling_offsets = target_layer.sampling_offsets(q).reshape(1, 1, num_heads, sum(num_points_list), 2)
        attn_weights = target_layer.attention_weights(q).reshape(1, 1, num_heads, sum(num_points_list))
        attn_weights = F.softmax(attn_weights, dim=-1)

    # 坐标计算
    ref_xy = ref[:, :, None, :, :2]
    ref_wh = ref[:, :, None, :, 2:]
    offset = sampling_offsets * num_points_scale * ref_wh * offset_scale
    sampling_locations = ref_xy + offset

    # --- 绘图逻辑 (深蓝背景风格) ---
    locs = sampling_locations[0, 0].cpu().numpy()
    weights = attn_weights[0, 0].cpu().numpy()
    ori_cv = cv2.cvtColor(np.array(im_pil), cv2.COLOR_RGB2BGR)

    # 1. 创建深蓝色遮罩背景
    # 创建一个纯深蓝色的层 (BGR: 40, 0, 0 左右是深蓝)
    blue_mask = np.zeros_like(ori_cv)
    blue_mask[:] = [45, 10, 5]  # 极深蓝色底
    # 将原图调暗并与蓝色底融合，模仿热力图背景
    canvas = cv2.addWeighted(ori_cv, 0.3, blue_mask, 0.7, 0)

    flat_locs = locs.reshape(-1, 2)
    flat_weights = weights.reshape(-1)

    # 2. 指数增强：让权重差异更明显
    enhanced_weights = np.power(flat_weights, 2)
    norm_weights = (enhanced_weights / (enhanced_weights.max() + 1e-8) * 255).astype(np.uint8)
    heatmap_colors = cv2.applyColorMap(np.arange(256).astype(np.uint8), cv2.COLORMAP_JET)

    for i in range(len(flat_locs)):
        px, py = int(flat_locs[i, 0] * w), int(flat_locs[i, 1] * h)
        idx = norm_weights[i]
        color = tuple(map(int, heatmap_colors[idx][0]))

        if 0 <= px < w and 0 <= py < h:
            # 过滤掉权重极低的点，保持画面干净
            if idx > 25:
                # 半径动态调整
                r = max(2, int(np.sqrt(flat_weights[i] * 600)))
                # 画发光点
                cv2.circle(canvas, (px, py), r, color, -1)
                # 核心高亮
                if idx > 200:
                    cv2.circle(canvas, (px, py), r, (255, 255, 255), 1)

    # 3. 绘制“弱化版”引导框 (Journal Style)
    box = outputs['pred_boxes'][0, query_idx].cpu().numpy()
    cx, cy, bw, bh = box[0] * w, box[1] * h, box[2] * w, box[3] * h
    x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
    x2, y2 = int(cx + bw / 2), int(cy + bh / 2)

    # 使用极细的灰色半透明框，仅作为位置参考
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (200, 200, 200), 1)
    canvas = cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0)

    cv2.imwrite(output_path, canvas)
    print(f"【顶刊级蓝调对比图】已保存: {output_path}")


def main():
    config_path = "/zhangcc/tq/code/DEIMv2/configs/deimv2/ablation_experiments/deimv2_dinov3_s_wheat_ABC_132.yml"
    resume_path = "/zhangcc/tq/code/DEIMv2/outputs/ablation_experiments/deimv2_dinov3_s_ablation_ABC_132_12/best_stg2.pth"
    # img_path = "/zhangcc/tq/DataSet/WheatzazhuCoCo/images_v1/images/test/IMG_20250429_094610.jpg"
    # img_path = "/zhangcc/tq/DataSet/WheatzazhuCoCo/images_v1/images/test/IMG_20250429_094921.jpg"
    img_path = "/zhangcc/tq/DataSet/WheatzazhuCoCo/images_v1/images/test/IMG_20250429_095025.jpg"

    cfg = YAMLConfig(config_path, resume=resume_path)
    model = cfg.model.eval().cuda()

    checkpoint = torch.load(resume_path, map_location='cpu')
    state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
    model.load_state_dict(state)

    visualize_top_tier_blue(model, cfg, img_path, "deim_final_blue_contrast095025.jpg")


if __name__ == '__main__':
    main()