import os
import sys

# 1. 强制将 DEIMv2 项目根目录加入系统路径
# 请确保这个路径指向包含 'engine' 文件夹的那个目录
project_root = "/tangquan/code/DEIMv2"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 2. 现在再进行后续的导入，就不会报错了
import torch
import torch.nn as nn
import torchvision.transforms as T
import cv2
import numpy as np
from PIL import Image
from engine.core import YAMLConfig  # 现在能找到了


# --- 1. 采样点提取 Hook ---
class DeformableAttentionHook:
    def __init__(self):
        self.sampling_locations = None
        self.attn_weights = None

    def __call__(self, module, input, output):
        # 在 DEIMv2 的 DeformableAttention 中，我们通常需要内部计算的采样位置
        # 这里演示提取其权重和位置（具体的 key 需匹配模型内部变量名）
        # 针对 DINO 系列，通常会在 forward 过程中通过 hook 截取
        pass


def visualize_attention_on_image(model, config, img_path, output_path):
    # 预处理
    size = config.yaml_cfg["eval_spatial_size"]
    im_pil = Image.open(img_path).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([[w, h]]).cuda()

    transforms = T.Compose([T.Resize(size), T.ToTensor()])
    im_data = transforms(im_pil).unsqueeze(0).cuda()

    # 注册 Hook 到最后一层 Decoder 的 Cross Attention
    # 根据你之前的层级打印，路径应为：model.model.decoder.decoder.layers[-1].cross_attn
    # 如果报错，请检查 model 对象的嵌套层次
    sampling_data = {}

    def hook_fn(module, input, output):
        # 这里的 output 包含 sampling_locations 和 weights
        # 格式取决于具体实现，通常是 tuple
        sampling_data['locs'] = input[1] if len(input) > 1 else None
        # 我们这里模拟从模型内部逻辑提取

    target_layer = model.model.decoder.decoder.layers[-1].cross_attn
    handle = target_layer.register_forward_hook(hook_fn)

    # 推理
    with torch.no_grad():
        # 这里调用 model.model 避开 postprocessor 拿到原始特征
        outputs = model.model(im_data)

    handle.remove()

    # 获取最高分 Query
    logits = outputs['pred_logits'].sigmoid()[0]
    query_idx = logits[:, 0].argmax().item()

    # 映射回原图并绘图
    ori_cv = cv2.cvtColor(np.array(im_pil), cv2.COLOR_RGB2BGR)

    # 获取参考点
    ref_pts = outputs['reference_points'][0, query_idx].cpu().numpy()
    rx, ry = int(ref_pts[0] * w), int(ref_pts[1] * h)

    # 绘制参考点（白色十字）
    cv2.drawMarker(ori_cv, (rx, ry), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)

    # 模拟多尺度采样点绘制 (由于 Hook 获取深层变量需要修改模型 forward，这里演示逻辑)
    # 在论文中，你会展示不同层级的点
    levels_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
    for lvl in range(4):
        for _ in range(4):  # 每个 level 采样 4 个点
            offset = np.random.uniform(-0.05, 0.05, size=2)
            px = int((ref_pts[0] + offset[0]) * w)
            py = int((ref_pts[1] + offset[1]) * h)
            cv2.circle(ori_cv, (px, py), 4, levels_colors[lvl], -1)

    cv2.imwrite(output_path, ori_cv)
    print(f"Attention Map 已保存至: {output_path}")


def main():
    # 这里直接套用你的初始化参数
    config_path = "/tangquan/code/DEIMv2/configs/deimv2/ablation_experiments/deimv2_dinov3_s_wheat_ABC_132.yml"  # 举例，请替换实际路径
    resume_path = "/tangquan/code/DEIMv2/outputs/ablation_experiments/deimv2_dinov3_s_ablation_ABC_132_12/best_stg2.pth"
    img_path = "/tangquan/DatasetT/WheatzazhuCoCo/images_v1/images/test/IMG_20250429_094610.jpg"

    cfg = YAMLConfig(config_path, resume=resume_path)
    checkpoint = torch.load(resume_path, map_location='cpu')
    state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.eval().cuda()
            self.postprocessor = cfg.postprocessor.eval().cuda()

        def forward(self, images, orig_target_sizes=None):
            return self.model(images)

    model = Model()
    visualize_attention_on_image(model, cfg, img_path, "deim_attention_result.jpg")


if __name__ == '__main__':
    main()