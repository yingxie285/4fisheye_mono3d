import os

# 图片文件夹路径
img_dir = "/data9/xieying/mono3d/data/demo_data/trainval/image_front"

# 输出txt路径
out_txt = "/data9/xieying/mono3d/data/demo_data/trainval/train.txt"

# 遍历文件夹，提取名字
names = []
for fname in sorted(os.listdir(img_dir)):
    if fname.endswith(".jpg"):
        base = os.path.splitext(fname)[0]   # 去掉扩展名
        names.append(base)

# 写入txt
with open(out_txt, "w") as f:
    for n in names:
        f.write(n + "\n")

print(f"共提取{len(names)}个文件名，已写入 {out_txt}")
