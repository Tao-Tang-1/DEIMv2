"""
DEIMv2 Interactive Inference - Image Selection
Copyright (c) 2025 The DEIMv2 Authors. All Rights Reserved.
"""

import os
import sys
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tkinter as tk
from tkinter import filedialog

# 保持项目路径兼容
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from engine.core import YAMLConfig

# 只检测小麦
label_map = {0: 'wheat'}
COLOR_MAP = {0: (255, 0, 0)}


# -----------------------------
# 绘制函数
# -----------------------------
def draw(image, labels, boxes, scores, thrh=0.3):
    draw_obj = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    labels, boxes, scores = labels[scores > thrh], boxes[scores > thrh], scores[scores > thrh]

    for j, box in enumerate(boxes):
        category = labels[j].item()
        color = COLOR_MAP.get(category, (255, 255, 255))
        box = list(map(int, box))

        # 画边框
        draw_obj.rectangle(box, outline=color, width=2)

        # 添加标签和置信度
        text = f"{label_map[category]} {scores[j].item():.2f}"
        text_bbox = draw_obj.textbbox((0, 0), text, font=font)
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

        # 文本背景
        text_background = [box[0], box[1] - text_height - 2, box[0] + text_width + 4, box[1]]
        draw_obj.rectangle(text_background, fill=color)
        draw_obj.text((box[0] + 2, box[1] - text_height - 2), text, fill="black", font=font)

    return image


# -----------------------------
# 系统选择图片
# -----------------------------
def select_images():
    root = tk.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(
        title="选择麦穗图片进行推理",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    return file_paths


# -----------------------------
# 主推理函数
# -----------------------------
def main():
    # -----------------------------
    # 1. 配置与模型加载
    # -----------------------------
    config_path = r"E:\jaas\code\DEIMv2\configs\deimv2\wheat\deimv2_dinov3_s_wheat_train54.yml"
    checkpoint_path = r"E:\jaas\code\DEIMv2\outputs\wheat_train\deimv2_dinov3_s_wheat_train54\best_stg2.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载 YAML 配置和 checkpoint
    cfg = YAMLConfig(config_path, resume=checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
    cfg.model.load_state_dict(state)

    # 定义部署模型
    class DeployModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_target_sizes)

    model = DeployModel().to(device).eval()
    img_size = cfg.yaml_cfg.get("eval_spatial_size", (640, 640))

    transforms = T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # -----------------------------
    # 2. 选择图片
    # -----------------------------
    image_paths = select_images()
    if not image_paths:
        print("未选择任何图片，程序退出。")
        return
    print(f"Selected {len(image_paths)} images")

    # -----------------------------
    # 3. 循环推理与展示
    # -----------------------------
    with torch.no_grad():
        for path in image_paths:
            pil_img = Image.open(path).convert('RGB')
            w, h = pil_img.size
            orig_size = torch.tensor([[w, h]], dtype=torch.float32, device=device)

            img_tensor = transforms(pil_img).unsqueeze(0).to(device)
            output = model(img_tensor, orig_size)
            labels, boxes, scores = output

            # NMS 后处理
            thrh = 0.3
            keep_idx = scores[0] > thrh
            s_boxes, s_labels, s_scores = boxes[0][keep_idx], labels[0][keep_idx], scores[0][keep_idx]

            if len(s_boxes) > 0:
                keep = torch.ops.torchvision.nms(s_boxes, s_scores, iou_threshold=0.3)
                s_boxes, s_labels, s_scores = s_boxes[keep], s_labels[keep], s_scores[keep]

            # 绘制检测图
            vis_img = draw(pil_img.copy(), s_labels, s_boxes, s_scores, thrh)

            # -----------------------------
            # 4. 左右拼接显示 + 标题 + 间隙
            # -----------------------------
            ori_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            vis_bgr = cv2.cvtColor(np.array(vis_img), cv2.COLOR_RGB2BGR)

            gap = 30
            title_height = 40
            h_canvas = max(ori_bgr.shape[0], vis_bgr.shape[0]) + title_height
            w_canvas = ori_bgr.shape[1] + vis_bgr.shape[1] + gap

            canvas = np.ones((h_canvas, w_canvas, 3), dtype=np.uint8) * 255
            canvas[title_height:title_height + ori_bgr.shape[0], 0:ori_bgr.shape[1]] = ori_bgr
            canvas[title_height:title_height + vis_bgr.shape[0], ori_bgr.shape[1] + gap: w_canvas] = vis_bgr

            # 标题
            title_text = "Original      Detection"
            font_scale = 1.0
            font_thickness = 2
            text_size, _ = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            text_x = (w_canvas - text_size[0]) // 2
            text_y = (title_height + text_size[1]) // 2
            cv2.putText(canvas, title_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)

            # 自适应框厚度
            box_thickness = max(1, round(min(vis_bgr.shape[0], vis_bgr.shape[1]) / 200))

            # Resize if too tall
            max_display_h = 800
            if canvas.shape[0] > max_display_h:
                scale = max_display_h / canvas.shape[0]
                canvas = cv2.resize(canvas, None, fx=scale, fy=scale)

            cv2.imshow(f"DEIMv2 Inference - {os.path.basename(path)}", canvas)
            print(f"{os.path.basename(path)}: Detected {len(s_boxes)} objects")
            print("Press any key for next, ESC to exit...")

            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == 27:
                break


if __name__ == '__main__':
    main()
