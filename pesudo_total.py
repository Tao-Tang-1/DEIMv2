import json

with open('/root/autodl-tmp/pseudo/annotations/annotations_train.json') as f:
    data = json.load(f)

total = len(data['images'])
bg = sum(1 for img in data['images'] if '_bg_' in img['file_name'])
obj = total - bg

# 统计有多少图片有标注
img_ids_with_ann = set(ann['image_id'] for ann in data['annotations'])
imgs_with_ann = sum(1 for img in data['images'] if img['id'] in img_ids_with_ann)

print(f'总图片数: {total}')
print(f'背景图(_bg_): {bg} ({bg / total * 100:.1f}%)')
print(f'目标图(非_bg_): {obj} ({obj / total * 100:.1f}%)')
print(f'有标注的图片数: {imgs_with_ann}')
ann_count = len(data["annotations"])
print(f'标注框总数: {ann_count}')