"""
DEIMv2: Real-Time Object Detection Meets DINOv3
Copyright (c) 2025 The DEIMv2 Authors.
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import nms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from engine.core import YAMLConfig

# =========================
# Label & Color
# =========================
label_map = {0: 'wheat'}
COLOR_MAP = {0: (255, 0, 0)}

# =========================
# Draw
# =========================
def draw(image, labels, boxes, scores, thrh=0.5):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    keep = scores > thrh
    labels, boxes, scores = labels[keep], boxes[keep], scores[keep]

    for i, box in enumerate(boxes):
        cls = labels[i].item()
        color = COLOR_MAP.get(cls, (255, 255, 255))
        box = list(map(int, box.tolist()))

        draw.rectangle(box, outline=color, width=3)
        text = f"{label_map[cls]} {scores[i]:.2f}"

        tw, th = draw.textbbox((0, 0), text, font=font)[2:]
        bg = [box[0], box[1] - th - 4, box[0] + tw + 4, box[1]]
        draw.rectangle(bg, fill=color)
        draw.text((box[0] + 2, box[1] - th - 2), text, fill="black", font=font)

    return image

# =========================
# Tile utils
# =========================
def tile_image(image, tile_size=1024, overlap=200):
    w, h = image.size
    stride = tile_size - overlap
    tiles = []

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            x1 = min(x + tile_size, w)
            y1 = min(y + tile_size, h)
            x0 = max(0, x1 - tile_size)
            y0 = max(0, y1 - tile_size)
            tile = image.crop((x0, y0, x1, y1))
            tiles.append((tile, (x0, y0)))
    return tiles

# =========================
# Tile inference
# =========================
@torch.no_grad()
def infer_large_image_with_tiles(
    model,
    image_pil,
    eval_size=(640, 640),
    tile_size=1024,
    overlap=200,
    score_thr=0.5
):
    device = "cuda"
    transform = T.Compose([
        T.Resize(eval_size),
        T.ToTensor()
    ])

    all_boxes, all_scores, all_labels = [], [], []

    tiles = tile_image(image_pil, tile_size, overlap)

    for tile_pil, (ox, oy) in tiles:
        tw, th = tile_pil.size
        orig_size = torch.tensor([[tw, th]], device=device)

        im = transform(tile_pil).unsqueeze(0).to(device)
        outputs = model(im, orig_size)[0]

        boxes = outputs["boxes"]
        scores = outputs["scores"]
        labels = outputs["labels"]

        keep = scores > score_thr
        if keep.sum() == 0:
            continue

        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        # 映射回大图
        boxes[:, [0, 2]] += ox
        boxes[:, [1, 3]] += oy

        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)

    if len(all_boxes) == 0:
        return None, None, None

    return (
        torch.cat(all_labels),
        torch.cat(all_boxes),
        torch.cat(all_scores)
    )

def global_nms(labels, boxes, scores, iou_thr=0.6):
    keep = nms(boxes, scores, iou_thr)
    return labels[keep], boxes[keep], scores[keep]

# =========================
# Dataset processing
# =========================
def process_dataset(
    model,
    dataset_path,
    output_path,
    thrh=0.5,
    eval_size=(640, 640),
    tile_size=1024,
    overlap=200
):
    os.makedirs(output_path, exist_ok=True)
    images = [f for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.png'))]

    print(f"Found {len(images)} images")

    for idx, name in enumerate(images):
        img_path = os.path.join(dataset_path, name)
        image = Image.open(img_path).convert("RGB")

        labels, boxes, scores = infer_large_image_with_tiles(
            model,
            image,
            eval_size=eval_size,
            tile_size=tile_size,
            overlap=overlap,
            score_thr=thrh
        )

        if labels is None:
            continue

        labels, boxes, scores = global_nms(labels, boxes, scores, iou_thr=0.6)

        vis = draw(image.copy(), labels, boxes, scores, thrh)
        vis.save(os.path.join(output_path, f"vis_{name}"))

        if idx % 100 == 0:
            print(f"[{idx}/{len(images)}] processed")

    print("Done.")

# =========================
# Main
# =========================
def main(args):
    cfg = YAMLConfig(args.config, resume=args.resume)

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        state = ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"]
    else:
        raise RuntimeError("Resume checkpoint required")

    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.eval().cuda()
            self.post = cfg.postprocessor.eval().cuda()

        def forward(self, images, orig_sizes):
            out = self.model(images)
            return self.post(out, orig_sizes)

    model = Model()
    eval_size = cfg.yaml_cfg["eval_spatial_size"]

    process_dataset(
        model,
        args.dataset,
        args.output,
        thrh=0.5,
        eval_size=eval_size,
        tile_size=1024,
        overlap=200
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-r", "--resume", required=True)
    parser.add_argument("-d", "--dataset", required=True)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()
    main(args)
