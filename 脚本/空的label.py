import os

folder = "/media/whut-user/350b5acb-7342-4f97-b1a7-57e9622bff8b/mono3d/data_camera_cyl/demo_data/trainval/labels"

frame_ids = []
empty_frames = []

for filename in os.listdir(folder):
    if filename.endswith(".txt"):
        # 提取 frame_id（去掉 .txt）
        frame_id = os.path.splitext(filename)[0]
        frame_ids.append(frame_id)

        # 读取文件内容判断空
        full_path = os.path.join(folder, filename)
        with open(full_path, "r") as f:
            content = f.read().strip()

        if content == "":
            empty_frames.append(frame_id)

# 输出统计
print("总帧数:", len(frame_ids))
print("空的txt数量:", len(empty_frames))
print("空的帧ID列表:", empty_frames)

# 如需写入到 txt
with open("/media/whut-user/350b5acb-7342-4f97-b1a7-57e9622bff8b/mono3d/脚本/all_frame_ids.txt", "w") as f:
    for fid in frame_ids:
        f.write(fid + "\n")

with open("/media/whut-user/350b5acb-7342-4f97-b1a7-57e9622bff8b/mono3d/脚本/empty_frame_ids.txt", "w") as f:
    for fid in empty_frames:
        f.write(fid + "\n")
