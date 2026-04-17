import os
import random
import json

# 根目录（修改这里）
root = r"/data9/xieying/mono3d/dataset_fisheye/fisheye_data_2w_frame"

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

# 保存 json
with open("/data9/xieying/mono3d/脚本/train_val_split.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print("总数:", n_total)
print("训练集:", len(train_list))
print("验证集:", len(val_list))
print("已生成 train_val_split.json")