import json
import numpy as np

ann_file = "/root/autodl-tmp/dataset/tiles/annotations/annotations_train.json"

data = json.load(open(ann_file, "r"))

areas = []
wh_list = []

for ann in data["annotations"]:
    x, y, w, h = ann["bbox"]
    area = w * h
    areas.append(area)
    wh_list.append((w, h))

areas = np.array(areas)

print("========== DATASET ANALYSIS ==========")
print("Total objects:", len(areas))
print("Min area:", areas.min())
print("Max area:", areas.max())
print("Median area:", np.median(areas))

# COCO标准划分
small = areas < 32 * 32
medium = (areas >= 32 * 32) & (areas < 96 * 96)
large = areas >= 96 * 96

print("\n========== SCALE DISTRIBUTION ==========")
print("Small (<32^2):", small.mean(), f"({small.sum()})")
print("Medium (32^2~96^2):", medium.mean(), f"({medium.sum()})")
print("Large (>96^2):", large.mean(), f"({large.sum()})")