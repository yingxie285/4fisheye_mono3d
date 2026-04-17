import os

base = "/data9/xieying/mono3d/data_camera/demo_data/trainval"

for folder in ["calibs", "images", "labels"]:
    path = os.path.join(base, folder)
    num = len(os.listdir(path))
    print(folder, num)