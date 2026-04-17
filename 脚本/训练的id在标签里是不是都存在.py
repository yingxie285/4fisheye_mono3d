import os

base_dir = "/data9/xieying/mono3d/data_camera/demo_data/trainval"

train_txt = "/data9/xieying/mono3d/data/demo_data/trainval/val.txt"

calib_dir = os.path.join(base_dir, "calibs")
img_dir = os.path.join(base_dir, "images")
label_dir = os.path.join(base_dir, "labels")

missing = []

with open(train_txt, "r") as f:
    ids = [line.strip() for line in f]

for id_ in ids:
    
    calib_path = os.path.join(calib_dir, id_ + ".txt")
    img_path = os.path.join(img_dir, id_ + ".jpg")
    label_path = os.path.join(label_dir, id_ + ".txt")

    ok = True
    
    if not os.path.exists(calib_path):
        print(f"missing calib: {id_}")
        ok = False

    if not os.path.exists(img_path):
        print(f"missing image: {id_}")
        ok = False

    if not os.path.exists(label_path):
        print(f"missing label: {id_}")
        ok = False

    if not ok:
        missing.append(id_)

print("\n====================")
print("总ID数量:", len(ids))
print("缺失数量:", len(missing))
print("====================")

if missing:
    print("缺失ID示例:", missing[:10])