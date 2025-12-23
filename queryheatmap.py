import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from engine.core import YAMLConfig

# -----------------------------
# 绘制检测框函数（去掉 NMS）
# -----------------------------
def draw(images, labels, boxes, scores, thrh=0.4, save_path=None):
    for i, im in enumerate(images):
        draw_obj = ImageDraw.Draw(im)

        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scr[scr > thrh]

        for j, b in enumerate(box):
            draw_obj.rectangle(list(b), outline='red', width=3)
            draw_obj.text((b[0], b[1]), text=f"{lab[j].item()} {round(scrs[j].item(), 2)}", fill='blue')

        if save_path is not None:
            im.save(save_path)

# -----------------------------
# 生成注意力热力图
# -----------------------------
def generate_attention_map(model, device, im_pil, size=(640, 640), save_dir="./heatmap"):
    os.makedirs(save_dir, exist_ok=True)

    # 1. 修正路径：根据打印结构，layers 在 decoder.decoder 里面
    all_layers = model.model.decoder.decoder.layers
    num_layers = len(all_layers)

    # 选取第 0 层（初始）、第 2 层（演进）、第 5 层（最终决策）
    target_indices = {0, 2, num_layers - 1}
    target_indices = sorted([i for i in target_indices if i >= 0])

    print(f"Decoder 总层数: {num_layers}, 正在处理层: {target_indices}")

    for idx in target_indices:
        activations = []

        def hook_fn(module, input, output):
            # output 通常是 [B, Num_Queries, C], 例如 [1, 300, 256]
            activations.append(output.detach().cpu())

        target_layer = all_layers[idx]
        handle = target_layer.register_forward_hook(hook_fn)

        try:
            w, h = im_pil.size
            orig_size = torch.tensor([[w, h]]).to(device)

            transforms = T.Compose([
                T.Resize(size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            im_data = transforms(im_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                model(im_data, orig_size)

            if len(activations) > 0:
                feat = activations[0]  # [1, 300, 256]

                # 计算每个 Query 的平均激活强度
                # B, N, C = feat.shape -> 我们取 N 维度的信息
                query_info = torch.mean(feat, dim=-1).squeeze(0)  # [300]

                # 关键：将 300 个 Query 的响应重塑为 2D 布局
                # 对于 RT-DETR/DEIM，Query 通常在空间上是有序排列的，或者是全局的
                # 我们将其重塑为最接近的平方矩阵，例如 300 -> 17x17 (约289)
                num_queries = query_info.shape[0]
                side = int(num_queries ** 0.5)
                heatmap_data = query_info[:side * side].reshape(side, side)

                # 归一化
                heatmap_data = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min() + 1e-8)
                heatmap_data = heatmap_data.numpy()

                # 绘图
                plt.figure(figsize=(10, 10))
                plt.imshow(im_pil)
                # 使用 jet 颜色，通过 bilinear 插值将 17x17 放大到原图尺寸，形成平滑热力图
                plt.imshow(heatmap_data, cmap='jet', alpha=0.5, extent=(0, w, h, 0), interpolation='bilinear')
                plt.axis('off')
                plt.title(f"Decoder Layer {idx} Query Activation")

                save_path = os.path.join(save_dir, f"decoder_layer_{idx}_heatmap.jpg")
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
                plt.close()
                print(f"成功保存: {save_path}")

        finally:
            handle.remove()

# -----------------------------
# 处理单张图片
# -----------------------------
# -----------------------------
# 处理单张图片
# -----------------------------
def process_image(model, device, file_path, size=(640, 640), output_dir="./heatmap"):
    os.makedirs(output_dir, exist_ok=True)
    im_pil = Image.open(file_path).convert('RGB')

    # 【关键修改】：创建两个副本，一个用于热力图，一个用于画检测框
    im_for_heatmap = im_pil.copy()
    im_for_detection = im_pil.copy()

    w, h = im_pil.size
    orig_size = torch.tensor([[w, h]]).to(device)

    transforms = T.Compose([
        T.Resize(size),
        T.ToTensor(),
    ])
    im_data = transforms(im_pil).unsqueeze(0).to(device)

    # 前向得到检测结果
    labels, boxes, scores = model(im_data, orig_size)

    # 1. 保存纯净的热力图（使用没画框的副本）
    generate_attention_map(model, device, im_for_heatmap, size=size,
                           save_dir=os.path.join(output_dir, "heatmap"))

    # 2. 保存带检测框的图（使用另一个副本）
    draw([im_for_detection], labels, boxes, scores, save_path=os.path.join(output_dir, "det.jpg"))
# -----------------------------
# 主函数
# -----------------------------
def main(args):
    cfg = YAMLConfig(args.config, resume=args.resume)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
    else:
        raise AttributeError("Must provide resume checkpoint.")

    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()  # deploy 模式
            self.postprocessor = cfg.postprocessor.deploy()
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    device = args.device
    model = Model().to(device)
    # --- 在这里添加打印语句 ---
    # print("-" * 30)
    # print("Decoder Structure:")
    # print(model.model.decoder)
    # print("-" * 30)
    # -----------------------
    file_path = args.input
    img_size = cfg.yaml_cfg["eval_spatial_size"]

    if os.path.splitext(file_path)[-1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
        process_image(model, device, file_path, img_size, output_dir="./heatmap")
        print("Image processing complete.")
    else:
        raise NotImplementedError("Video processing not included in this demo.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--resume', type=str, required=True)
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-d', '--device', type=str, default='cpu')
    args = parser.parse_args()
    main(args)
