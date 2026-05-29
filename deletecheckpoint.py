import os
import glob

root_dir = "/root/autodl-tmp/code/DEIMv2/outputs/ablation_experiments"

exclude_dir = os.path.join(
    root_dir,
    "deimv2_dinov3_s_wheat_train46"
)

train_dirs = glob.glob(os.path.join(root_dir, "deimv2_dinov3*"))

print("找到的训练目录：")
for d in train_dirs:
    print(d)

print("\n开始检查 checkpoint 文件...\n")

total_deleted = 0

for train_dir in train_dirs:

    if os.path.abspath(train_dir) == os.path.abspath(exclude_dir):
        print(f"⏭️ Skip: {train_dir}")
        continue

    # 看看里面到底有什么
    files = os.listdir(train_dir)
    print(f"\n📂 {train_dir}")
    print(files)

    ckpts = glob.glob(os.path.join(train_dir, "checkpoint*.pth"))

    print("匹配到的 checkpoint:")
    print(ckpts)

    for ckpt in ckpts:
        os.remove(ckpt)
        total_deleted += 1
        print(f"Deleted: {ckpt}")

print(f"\n✅ Done. Total deleted checkpoints: {total_deleted}")