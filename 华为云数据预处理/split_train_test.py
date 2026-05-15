import os
import random
import json
from pathlib import Path

# 根目录（修改这里）
root = Path(os.environ["OCTPS_DATASET_DIR"]) / "fisheye_data_aug"

# 获取所有子文件夹名
all_folders = [
    name for name in os.listdir(root)
    if os.path.isdir(os.path.join(root, name))
]

# 打乱
random.shuffle(all_folders)

#train:val =  4:1 划分
n_total = len(all_folders)
n_train = int(n_total * (4/5))

train_list = all_folders[:n_train]
val_list = all_folders[n_train:]

# 组织为 JSON
data = {
    "train": train_list,
    "val": val_list
}

script_root = Path(os.environ["TARGET_RESULT_DIR"]) / "scripts"
script_root.mkdir(parents=True, exist_ok=True)
train_val_split_path = script_root / "train_val_split.json"
# 保存 json
with open(train_val_split_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print("总数:", n_total)
print("训练集:", len(train_list))
print("验证集:", len(val_list))
print("已生成 train_val_split.json")