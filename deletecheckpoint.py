import os
import glob

# 根目录
root_dir = "/tangquan/code/DEIMv2/outputs/wheat_train"

# 需要保留、不删除的目录（绝对路径）
exclude_dir = os.path.join(
    root_dir,
    "deimv2_dinov3_s_wheat_train46"
)

# 匹配所有训练子目录
train_dirs = glob.glob(os.path.join(root_dir, "deimv2_dinov3_s_wheat_train*"))

total_deleted = 0

for train_dir in train_dirs:
    # 跳过指定目录
    if os.path.abspath(train_dir) == os.path.abspath(exclude_dir):
        print(f"⏭️ Skip: {train_dir}")
        continue

    ckpts = glob.glob(os.path.join(train_dir, "checkpoint*.pth"))
    if not ckpts:
        continue

    for ckpt in ckpts:
        os.remove(ckpt)
        total_deleted += 1
        print(f"Deleted: {ckpt}")

print(f"\n✅ Done. Total deleted checkpoints: {total_deleted}")
