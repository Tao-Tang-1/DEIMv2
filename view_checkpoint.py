import torch

ckpt_path = "/tangquan/code/DEIMv2/outputs/wheat_train/deimv2_dinov3_x_wheat_train5/last.pth"
ckpt = torch.load(ckpt_path, map_location="cpu",weights_only=False)

print("🔍 Checkpoint 键列表:")
for key in ckpt.keys():
    print(f" - {key}")

# 如果包含 epoch 信息
if "last_epoch" in ckpt:
    print(f"\n🕒 Last epoch: {ckpt['last_epoch']}")
elif "epoch" in ckpt:
    print(f"\n🕒 Epoch: {ckpt['epoch']}")

# 打印包含的 state_dict 结构
for key in ["model", "ema", "criterion", "postprocessor"]:
    if key in ckpt:
        if isinstance(ckpt[key], dict):
            if "state_dict" in ckpt[key]:
                print(f"✅ Load {key}.state_dict ({len(ckpt[key]['state_dict'])} tensors)")
            elif "module" in ckpt[key]:
                print(f"✅ Load {key} (module with {len(ckpt[key]['module'])} tensors)")
            else:
                print(f"✅ Load {key} ({len(ckpt[key])} tensors)")
        else:
            print(f"ℹ️ {key} 类型: {type(ckpt[key])}")
    else:
        print(f"❌ 未找到 {key}")

# 统计模型参数量
if "ema" in ckpt and "module" in ckpt["ema"]:
    state_dict = ckpt["ema"]["module"]
elif "model" in ckpt:
    state_dict = ckpt["model"]
else:
    state_dict = ckpt

total_params = sum(p.numel() for p in state_dict.values() if torch.is_tensor(p))
total_bytes = sum(p.numel() * p.element_size() for p in state_dict.values() if torch.is_tensor(p))

print(f"\n📊 参数张量数量: {len(state_dict)}")
print(f"🧩 总参数量: {total_params:,d} 个")
print(f"💾 约占内存: {total_bytes / (1024 ** 2):.2f} MB")
