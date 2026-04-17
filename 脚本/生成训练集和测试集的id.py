import os
import json

# =========================
# 路径配置
# =========================
dataset_root = '/data9/xieying/mono3d/dataset_fisheye/fisheye_data_2w_frame'

train_val_split_path = "/data9/xieying/mono3d/脚本/train_val_split.json"

all_frame_txt = "/data9/xieying/mono3d/脚本/all_frame_ids.txt"

output_train_txt = "/data9/xieying/mono3d/脚本/train_ids.txt"
output_val_txt = "/data9/xieying/mono3d/脚本/val_ids.txt"

# =========================
# 读取有效帧
# =========================
with open(all_frame_txt, 'r') as f:
    valid_frame_ids = set(
        line.strip() for line in f if line.strip()
    )

print("Valid frames:", len(valid_frame_ids))

# =========================
# 读取 train / val 划分
# =========================
with open(train_val_split_path, 'r') as f:
    scene_split = json.load(f)

train_scenes = scene_split["train"]

train_ids = []

# =========================
# 遍历 train scene
# =========================
for scene in train_scenes:

    pc_dir = os.path.join(dataset_root, scene, 'point_cloud')

    if not os.path.isdir(pc_dir):
        continue

    file_list = sorted([
        f for f in os.listdir(pc_dir)
        if f.endswith('.pcd')
    ])

    for fname in file_list:

        frame_id = os.path.splitext(fname)[0]

        if frame_id in valid_frame_ids:
            train_ids.append(frame_id)

# =========================
# 计算 val_ids
# =========================
train_ids_set = set(train_ids)

val_ids = sorted(list(valid_frame_ids - train_ids_set))

# =========================
# 写入 train_ids.txt
# =========================
with open(output_train_txt, 'w') as f:
    for fid in train_ids:
        f.write(fid + '\n')

# =========================
# 写入 val_ids.txt
# =========================
with open(output_val_txt, 'w') as f:
    for fid in val_ids:
        f.write(fid + '\n')

# =========================
# 打印统计
# =========================
print("Train ids:", len(train_ids))
print("Val ids:", len(val_ids))
print("Saved train ids to:", output_train_txt)
print("Saved val ids to:", output_val_txt)