# -*- coding: utf-8 -*-
import os
import sys
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import torch
import torch.nn as nn
import torchvision.transforms as T
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import nms

project_root = r"E:\jaas\code\DEIMv2"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from engine.core import YAMLConfig


# =========================
# SAHI: sliding window slice
# =========================
def slice_image(image_pil, tile_size=640, overlap_ratio=0.2):
    """
    SAHI-style slicing with overlap ratio.
    Returns list of (tile_pil, (x_offset, y_offset)).
    """
    w, h = image_pil.size
    overlap = int(tile_size * overlap_ratio)
    stride = tile_size - overlap
    tiles = []

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            x1 = min(x + tile_size, w)
            y1 = min(y + tile_size, h)
            x0 = max(0, x1 - tile_size)
            y0 = max(0, y1 - tile_size)
            tile = image_pil.crop((x0, y0, x1, y1))
            tiles.append((tile, (x0, y0)))

    return tiles


# =========================
# SAHI inference on one image
# =========================
@torch.no_grad()
def sahi_inference(model_wrapper, image_pil, eval_size=(640, 640),
                   tile_size=640, overlap_ratio=0.2, score_thr=0.3):
    tiles = slice_image(image_pil, tile_size, overlap_ratio)
    print(f"  Image size {image_pil.size}, split into {len(tiles)} tiles (tile={tile_size}, overlap_ratio={overlap_ratio})")

    all_boxes, all_scores, all_labels = [], [], []

    for idx, (tile_pil, (ox, oy)) in enumerate(tiles):
        tw, th = tile_pil.size
        orig_size = torch.tensor([[tw, th]], device='cuda')

        transform = T.Compose([T.Resize(eval_size), T.ToTensor()])
        im = transform(tile_pil).unsqueeze(0).cuda()

        outputs = model_wrapper(im, orig_size)[0]

        boxes = outputs["boxes"]
        scores = outputs["scores"]
        labels = outputs["labels"]

        keep = scores > score_thr
        if keep.sum() == 0:
            continue

        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        boxes[:, [0, 2]] += ox
        boxes[:, [1, 3]] += oy

        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)

        print(f"  Tile [{idx+1}/{len(tiles)}] offset=({ox},{oy}) size=({tw}x{th}) -> {keep.sum().item()} detections")

    if len(all_boxes) == 0:
        return None, None, None

    return torch.cat(all_labels), torch.cat(all_boxes), torch.cat(all_scores)


def global_nms(labels, boxes, scores, iou_thr=0.5):
    keep = nms(boxes, scores, iou_thr)
    return labels[keep], boxes[keep], scores[keep]


# =========================
# Visualization
# =========================
def visualize(image_pil, labels, boxes, scores, output_path,
              class_names=None, conf_threshold=0.3):
    if class_names is None:
        class_names = ["abnormal"]

    W, H = image_pil.size
    img_pil = image_pil.copy()
    draw = ImageDraw.Draw(img_pil)

    font_path = r"C:\Users\15199\Downloads\DejaVuSans-Bold.ttf"
    font_size = 80

    if os.path.exists(font_path):
        font = ImageFont.truetype(font_path, size=font_size)
    else:
        linux_fonts = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
        ]
        loaded = False
        for lf in linux_fonts:
            if os.path.exists(lf):
                font = ImageFont.truetype(lf, size=font_size)
                loaded = True
                print(f"[WARN] Fallback font: {lf}")
                break
        if not loaded:
            print("[WARN] Using PIL default font")
            font = ImageFont.load_default()

    line_width = 10
    drawn = 0

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].cpu().numpy().astype(int)
        conf = scores[i].item()
        cls_id = labels[i].item()

        if conf < conf_threshold:
            continue

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        cls_name = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"

        if conf >= 0.7:
            color = (0, 255, 220)
        else:
            color = (0, 0, 255)

        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        label = f"{cls_name} {conf:.3f}"
        text_x = x1
        text_y = max(0, y1 - font_size - 5)
        draw.text((text_x, text_y), label, fill=color, font=font)

        drawn += 1

    print(f"[INFO] Drew {drawn} boxes on image")

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    result_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, result_cv)
    print(f"[OK] Saved: {output_path}")


# =========================
# Main
# =========================
def main():
    config_path = r"E:\jaas\code\DEIMv2\configs\deimv2\ablation_experiments\deimv2_dinov3_s_offtype.yml"
    resume_path = r"E:\jaas\code\DEIMv2\outputs\ablation_experiments\deimv2_dinov3_s_offtype_CG_AFS_SCD_QS_MAL_ASL5\best_stg2.pth"

    img_path = r"E:\毕业材料\大论文\数据集\OffType\split\test\images\BIGNAN_2024_LITERAL_BLE_157539_17Fumi-PE_Session 2024-06-24 14-47-57_l1_rgb_images_Plot206_Camera2_6.jpg"

    # img_path = r"E:\毕业材料\大论文\数据集\OffType\split\test\images\Xinxiang-20230524-7#10-rep0-DSC-RX0-1684890119568.JPG"

    # img_path = r"E:\毕业材料\大论文\数据集\OffType\split\test\images\Xinxiang-20230524-14#11-rep0-DSC-RX0-1684893187305.JPG"

    output_dir = "./output_predictimg/deimv2"
    os.makedirs(output_dir, exist_ok=True)
    img_name = os.path.basename(img_path).replace('.jpg', '_sahi_draw.jpg')
    output_path = os.path.join(output_dir, img_name)

    # --- SAHI params ---
    tile_size = 640
    overlap_ratio = 0.2
    score_thr = 0.5
    nms_iou = 0.9

    print(f"[INFO] Config: {config_path}")
    print(f"[INFO] Checkpoint: {resume_path}")

    cfg = YAMLConfig(config_path, resume=resume_path)

    checkpoint = torch.load(resume_path, map_location='cpu', weights_only=False)
    state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
    cfg.model.load_state_dict(state)

    eval_size = cfg.yaml_cfg["eval_spatial_size"]

    class ModelWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.eval().cuda()
            self.post = cfg.postprocessor.eval().cuda()

        def forward(self, images, orig_sizes):
            out = self.model(images)
            return self.post(out, orig_sizes)

    model = ModelWrapper()

    im_pil = Image.open(img_path).convert('RGB')
    W, H = im_pil.size
    print(f"[INFO] Image: {img_path}")
    print(f"[INFO] Image size: {W}x{H}, eval_size: {eval_size}")

    print(f"[INFO] Running SAHI inference (tile={tile_size}, overlap_ratio={overlap_ratio}, score_thr={score_thr})")
    labels, boxes, scores = sahi_inference(
        model, im_pil,
        eval_size=eval_size,
        tile_size=tile_size,
        overlap_ratio=overlap_ratio,
        score_thr=score_thr,
    )

    if labels is None:
        print("[WARN] No detections found. Saving original image.")
        im_pil.save(output_path)
        return

    print(f"[INFO] Before NMS: {len(boxes)} detections")
    labels, boxes, scores = global_nms(labels, boxes, scores, iou_thr=nms_iou)
    print(f"[INFO] After NMS:  {len(boxes)} detections")

    if len(boxes) == 0:
        print("[WARN] All detections removed after NMS.")
        im_pil.save(output_path)
        return

    print(f"[INFO] Score range: [{scores.min().item():.4f}, {scores.max().item():.4f}]")

    class_names = ["abnormal"]

    visualize(im_pil, labels, boxes, scores, output_path,
              class_names=class_names, conf_threshold=score_thr)


if __name__ == '__main__':
    main()
