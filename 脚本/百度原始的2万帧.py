import os

root = "/data9/xieying/mono3d/dataset_fisheye/fisheye_data_2w_frame"

frame_ids = []

for subdir, _, files in os.walk(root):
    # 只遍历 image0 文件夹
    if subdir.endswith("/image0"):
        for f in files:
            if f.endswith(".jpeg"):
                frame_id = os.path.splitext(f)[0]  # 去掉 .jpeg
                frame_ids.append(frame_id)

# 写入 txt
save_path = "/data9/xieying/mono3d/脚本/2w_frame_ids.txt"
with open(save_path, "w") as txt:
    for fid in frame_ids:
        txt.write(fid + "\n")

print(f"总帧数：{len(frame_ids)}")
print(f"已写入：{save_path}")
