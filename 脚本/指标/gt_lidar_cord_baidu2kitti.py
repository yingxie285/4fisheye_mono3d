import numpy as np
import cv2
import json
import numpy as np
import os
from natsort import natsorted
import re
import torch
import shutil
import math

img2cam = {"image0": "front", "image1": "right", "image2": "left", "image3": "back"}
cam2img = {"front": "image0", "right": "image1", "left": "image2", "back": "image3"}
cam2idx = {"front": 0, "right": 1, "left": 2, "back": 3}

category_info = [
    {"name": "car", "description": "普通轿车, 敞篷轿车, SUV, MPV,面包车,皮卡,警车,救护车,房车,其他机动车,未知机动车,未知机动车车轮,未知机动车车灯"},
    {"name": "truck", "description": "房车,普通小型货车,箱式小型货车,普通大型货车,轿运车"},
    {"name": "bus", "description": "客车,校车, 普通公交车, 铰链公交车, 有轨电车,无轨电车"},
    {"name": "construction_vehicle", "description": "消防车,清洁车,工程车,拖车,拖拉机,叉车,油罐车"},
    {"name": "pedestrian", "description": "成人,儿童,交警,环卫工人,道路施工人员"},
    {"name": "motor", "description": "两轮摩托,三轮摩托"},
    {"name": "bicycle", "description": "两轮电动车,三轮电动车,两轮自行车,三轮自行车,滑板车,平衡车,婴儿车,轮椅,平板小推车,超市购物车,手推车,其他非机动车,未知非机动车,非机动车组"},
    {"name": "animal", "description": "小型动物,大型动物"},
    {"name": "traffic_cone", "description": "锥桶"},
    {"name": "barrier", "description": "防撞桶,路桩,石墩,水马,柱子,石块,树枝/树杈,空中漂浮物,路坑/水洼,其他静态障碍物"},
    {"name": "stopper", "description": "车位停止器,车位锁,减速带"},
    {"name": "trash_bin", "description": "垃圾桶,灭火器,箱子"},
    {"name": "sign", "description": "A字牌,三角牌,施工警示牌,道闸杆"}
]

# 构建中文子类到英文类别的映射
mapping = {}
for cat in category_info:
    for sub in cat["description"].split(","):
        mapping[sub.strip()] = cat["name"]

def map_class(chinese_class: str) -> str:
    return mapping.get(chinese_class, "unknown")

def load_camera_params(sample_dir, image_path):
    """
    sample_dir: /.../20240112-145620_20240112-145636/
    返回 dict:
    cam_name -> {"K":.., "dist":.., "lidar2cam":.., "image_path":..}
    """
    with open(sample_dir, "r") as f:
        cfg_list = json.load(f)
    cam_params = {}
    cam_name_mapping ={
        0: "front",
        1: "right",
        2: "left",
        3: "back",
    }
    for index, cam_cfg in enumerate(cfg_list):
        cam_name = cam_name_mapping[index]
        # K = np.array(cam_cfg["camera_internal"], dtype=np.float64)
        K = np.array([
                [cam_cfg["camera_internal"]["fx"], 0, cam_cfg["camera_internal"]["cx"]],
                [0, cam_cfg["camera_internal"]["fy"], cam_cfg["camera_internal"]["cy"]],
                [0, 0, 1]
            ], dtype=np.float64)
        dist = [cam_cfg["distortion"]['k1'],cam_cfg["distortion"]['k2'],cam_cfg["distortion"]['k3'],cam_cfg["distortion"]['k4']]
        lidar2cam = np.array(cam_cfg["camera_external"], dtype=np.float64).reshape(-1, 4).T #这里已经将行主序转成列主序了，最后一行是0 0 0 1
        r_velo2cam = lidar2cam[:3,:3]
        t_velo2cam = lidar2cam[:3, 3].reshape(3, 1)
        Tr_velo_to_cam = np.hstack((r_velo2cam, t_velo2cam))
        img_path = image_path[cam_name]
        cam_params[cam_name] = {
            "K": K,
            "dist": dist,
            "lidar2cam": lidar2cam,
            "r_velo2cam": r_velo2cam,
            "t_velo2cam": t_velo2cam,
            "Tr_velo_to_cam": Tr_velo_to_cam,
            "img_path": img_path
        }
    return cam_params

def normalize_angle(angle):
    # make angle in range [-0.5pi, 1.5pi]
    alpha_tan = np.tan(angle)
    alpha_arctan = np.arctan(alpha_tan)
    if np.cos(angle) < 0:
        alpha_arctan = alpha_arctan + math.pi
    return alpha_arctan

def convert_point(point, matrix):
    return matrix @ point

def load_annotation(anno_path, cam_params):
    """
    返回 annotations: 每个 (目标, 相机) 组合一条记录
    """
    frame_id = os.path.splitext(os.path.basename(anno_path))[0]
    with open(anno_path, "r") as f:
        anno_list = json.load(f)
    annotation_list = anno_list['objects']
    annotations = []

    for annotation in annotation_list:
        hwl = [annotation['contour']['size3D']['z'], 
               annotation['contour']['size3D']['y'], 
               annotation['contour']['size3D']['x']]
        center = [annotation['contour']['center3D']['x'], 
                      annotation['contour']['center3D']['y'], 
                      annotation['contour']['center3D']['z']]
        yaw_lidar = annotation['contour']['rotation3D']['z']
        className = annotation['className']
        
        # truncation / occlusion
        truncation = occlusion = None
        for cv in annotation.get('classValues', []):
            if cv.get('name') == 'truncation':
                truncation = cv.get('value')
            elif cv.get('name') == 'occlusion':
                occlusion = cv.get('value')
        
        # 2D bbox per camera
        bbox_2d_all = {}
        if '2D_bbox' in annotation:
            for cam_name, box in annotation['2D_bbox'].items():
                bbox_2d_all[cam_name] = box

        # 为每个出现的相机生成一条独立记录
        for cam_key in bbox_2d_all.keys():  # cam_key: "image0", "image1", ...
            cam_name = img2cam[cam_key]     # "front", "right", ...
            
            # 添加记录：每个 (目标, 相机) 一条
            annotations.append({
                "class": className,
                "hwl": hwl,
                "center": center,
                "yaw": yaw_lidar,
                "frame_id": frame_id,
                "truncation": truncation,
                "occlusion": occlusion,
                "bbox_2d_all": bbox_2d_all,           # 保留所有视角（可选）
                "camera": cam_name,                   # 新增：标记属于哪个相机
                "image_key": cam_key,                 # 如 "image0"
            })

    return annotations


def get_3d_bbox_corners(center, size, yaw):
    h, w, l = size  #Kitti是hwl
    x_c, y_c, z_c = center
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    z_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2] #这里是构建以原点（0,0,0）为原点，构建一个长宽高为l,w,h的box
    corners = np.vstack([x_corners, y_corners, z_corners])
    R = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])
    corners = R @ corners #乘yaw旋转矩阵  #真实的yaw角
    corners = corners + np.array(center).reshape(3,1)  #平移到真实的位置
    return corners.T  # (8,3)  #这里没有lidar2cam，得到的角点是lidar坐标系下的

def project_points_fisheye(points_3d, K, dist,lidar2cam):
    # 齐次坐标变换
    points_3d_h = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
    cam_points = (lidar2cam @ points_3d_h.T).T[:, :3].astype(np.float64)

    # 若8个角点都在相机后方 → 过滤
    if np.all(cam_points[:, 2] <= 0):
        return np.empty((0, 2))
    
    # 若正z点 ≤ 2 → 过滤掉
    num_front = np.sum(cam_points[:, 2] > 0)
    if num_front <= 2:
        return np.empty((0, 2))
    
    K = np.array(K, dtype=np.float64)
    D = np.array(dist, dtype=np.float64)

    img_points = np.zeros((cam_points.shape[0], 2), dtype=np.float64)

    # 前方点：用cv2.fisheye.projectPoints
    front_mask = cam_points[:, 2] > 0
    if np.any(front_mask):
        img_front, _ = cv2.fisheye.projectPoints(
            cam_points[front_mask].reshape(-1, 1, 3),
            rvec=np.zeros((3, 1), dtype=np.float64),
            tvec=np.zeros((3, 1), dtype=np.float64),
            K=K,
            D=D
        )
        img_points[front_mask] = img_front.reshape(-1, 2)

    # 后方点：用极坐标投影
    back_mask = ~front_mask
    if np.any(back_mask):
        X = cam_points[back_mask, 0]
        Y = cam_points[back_mask, 1]
        Z = cam_points[back_mask, 2]

        r = np.sqrt(X**2 + Y**2)
        theta = np.arctan2(r, Z + 1e-8)
        phi = np.arctan2(Y, X)

        k1, k2, k3, k4 = D.flatten()
        theta_d = theta * (1 + k1*theta**2 + k2*theta**4 + k3*theta**6 + k4*theta**8)
        
        fx, fy = K[0,0], K[1,1]
        cx, cy = K[0,2], K[1,2]

        u = cx + fx * theta_d * np.cos(phi)
        v = cy + fy * theta_d * np.sin(phi)

        img_points[back_mask] = np.stack([u, v], axis=1)
        
        # # 若不是8个角点（例如部分NaN或无效），则丢弃
        # if img_points.shape[0] != 8 or np.any(np.isnan(img_points)):
        #     return np.empty((0, 2))

    return img_points

def draw_bbox(img, idx, corners_2d, color=(0,255,0), text_color=(0,0,255),thickness=2):
    """
    绘制 2D bbox 或投影后的 3D bbox。
    
    corners_2d: np.ndarray, shape (N,2)，投影后的角点坐标
    """
    if corners_2d is None or len(corners_2d) == 0:
        return img  # 没有点直接返回

    corners_2d = corners_2d.astype(int)

    if corners_2d.ndim == 1 and corners_2d.shape[0] == 2:
        corners_2d = corners_2d[None, :]  # shape (1,2)
    
    n = len(corners_2d)
    corner_colors = [
        (0,0,255),      # red
        (0,128,255),    # orange
        (0,255,255),    # yellow
        (0,255,0),      # green
        (255,255,0),    # cyan
        (255,0,0),      # blue
        (255,0,255),    # magenta
        (128,0,255)     # purple
    ]
    if n == 1:
        # 单个点，画圆点
        pt = tuple(corners_2d[0])
        cv2.circle(img, pt, 2, color, thickness)
    elif n == 2:
        # 只有两点，画一条线
        pt1, pt2 = tuple(corners_2d[0]), tuple(corners_2d[1])
        cv2.line(img, pt1, pt2, color, thickness)
    elif n == 4:
        # 画矩形
        for i in range(4):
            pt1, pt2 = tuple(corners_2d[i]), tuple(corners_2d[(i+1)%4])
            cv2.line(img, pt1, pt2, color, thickness)
    elif n >= 8:
        # 原始 3D bbox 边
        edges = [
            (0,1),(1,2),(2,3),(3,0),
            (4,5),(5,6),(6,7),(7,4),
            (0,4),(1,5),(2,6),(3,7)
        ]
        for i,j in edges:
            pt1, pt2 = tuple(corners_2d[i]), tuple(corners_2d[j])
            cv2.line(img, pt1, pt2, color, thickness)
    else:
        # 其他点数，顺序连线
        for i in range(n-1):
            pt1, pt2 = tuple(corners_2d[i]), tuple(corners_2d[i+1])
            cv2.line(img, pt1, pt2, color, thickness)
            
    # 在每个角点旁标注id
    for cid, pt in enumerate(corners_2d):
        x, y = int(pt[0]), int(pt[1])
        c = corner_colors[cid % len(corner_colors)]   # 循环使用颜色
        cv2.circle(img, (x, y), 5, c, -1)             # 更明显的点
        cv2.putText(
            img, str(cid), (x+5, y-5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA
        )

    return img

def vis3d(annotations, image_path,cam_params, save_dir):
    cams = ["front", "right","left", "back"]

    for cam in cams:
        frame_id = annotations[0]['frame_id']
        vis_save_path = os.path.join(save_dir, f"{frame_id}_{cam}_vis.jpg")
        img = cv2.imread(image_path[cam])
        for idx, gt in enumerate(annotations):
            # 只画当前相机中存在的目标
            if gt.get("camera") != cam:
                continue

            K= cam_params[cam]['K']
            D= cam_params[cam]['dist']
            lidar2cam= cam_params[cam]['lidar2cam']
            corners_3d = get_3d_bbox_corners(gt["center"], gt["hwl"], gt["yaw"]) # (8,3)
            corners_2d = project_points_fisheye(corners_3d, K, D , lidar2cam)
            if len(corners_2d) != 0:
                    img = draw_bbox(img, idx, corners_2d)
                
        cv2.imwrite(vis_save_path,  img)

def label_save(annotations, save_dir):
    """
    将 annotations 保存为 KITTI 格式的 label 文件。
    
    输出结构:
        save_dir/
        ├── label_front/
        │   └── {frame_id}_front_label.txt
        ├── label_right/
        │   └── {frame_id}_right_label.txt
        ├── label_left/
        │   └── {frame_id}_left_label.txt
        └── label_back/
            └── {frame_id}_back_label.txt
    """
    # 相机名到 label 子目录名的映射
    cam_to_label_dir = {
        "front": "label_front",
        "right": "label_right",
        "left": "label_left",
        "back": "label_back"
    }

    # 按 (frame_id, camera) 分组
    frame_cam_annotations = {}
    for ann in annotations:
        frame_id = ann["frame_id"]
        cam = ann["camera"]
        key = (frame_id, cam)
        if key not in frame_cam_annotations:
            frame_cam_annotations[key] = []
        frame_cam_annotations[key].append(ann)

    # 保存每个 (frame, camera)
    for (frame_id, cam), anns in frame_cam_annotations.items():
        # 确定输出子目录
        label_subdir = cam_to_label_dir[cam]
        label_dir_path = os.path.join(save_dir, label_subdir)
        os.makedirs(label_dir_path, exist_ok=True)

        # 文件名: {frame_id}_{camera}_label.txt
        # label_filename = f"{frame_id}_{cam}_label.txt"
        label_filename = f"{frame_id}.txt"
        label_path = os.path.join(label_dir_path, label_filename)

        with open(label_path, "w") as f:
            for ann in anns:
                cls_chinese = ann["class"]
                cls = map_class(cls_chinese)   # 把中文类别转成统一类别
                # ==== 新增：根据摄像头名取对应方向的truncation ====
                if "truncation" in ann and ann["truncation"] is not None:
                    trunc_list = ann["truncation"]
                    if isinstance(trunc_list, list) and len(trunc_list) == 4:
                        tr_idx = cam2idx[ann['camera']]
                        truncation = int(trunc_list[tr_idx])
                    else:
                        truncation = -1
                else:
                    truncation = -1
                # ==== 新增：根据摄像头名取对应方向的occlusion ====
                if "occlusion" in ann and ann["occlusion"] is not None:
                    occ_list = ann["occlusion"]
                    if isinstance(occ_list, list) and len(occ_list) == 4:
                        occ_idx = cam2idx[ann['camera']]
                        occlusion = int(occ_list[occ_idx])  # 取对应方向
                    else:
                        occlusion = -1  # 异常情况
                else:
                    occlusion = -1

                alpha = 0

                # 获取 2D bbox
                image_key = ann["image_key"]
                if image_key not in ann["bbox_2d_all"]:
                    continue  # 理论上不会发生
                x1, y1, x2, y2 = ann["bbox_2d_all"][image_key]

                h, w, l = ann["hwl"]
                x, y, z = ann["center"]  #几何中心
                yaw = ann["yaw"]

                # 写入 KITTI 格式行
                line = f"{cls} {truncation} {occlusion} {alpha} " \
                       f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} " \
                       f"{h:.2f} {w:.2f} {l:.2f} {x:.2f} {y:.2f} {z:.2f} {yaw:.2f}\n"
                f.write(line)

def calib_texts(cam_params, annotations, save_dir):
    """
    为每个 (frame_id, camera) 生成对应的 calib 文件（只生成 label 被写入的那些）
    """
    cam_order = ["front", "right", "left", "back"]
    
    # 收集所有需要生成 calib 的 (frame_id, camera)
    frame_cam_set = set()
    for ann in annotations:
        frame_cam_set.add((ann["frame_id"], ann["camera"]))

    for frame_id, target_cam in frame_cam_set:
        calib_dir = os.path.join(save_dir, f"calib_{target_cam}")
        os.makedirs(calib_dir, exist_ok=True)
        # calib_path = os.path.join(calib_dir, f"{frame_id}_{target_cam}_calib.txt")
        calib_path = os.path.join(calib_dir, f"{frame_id}.txt")

        # P0-P3 (全部4个相机)
        P_lines = []
        for cam in cam_order:
            K = cam_params[cam]["K"]
            # P = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
            P_str = ' '.join([f"{x:.12e}" for x in K.flatten()])
            P_lines.append(P_str)

        R0_str = ' '.join([f"{x:.12e}" for x in np.eye(3).flatten()])
        Tr = cam_params[target_cam]["Tr_velo_to_cam"]
        Tr_str = ' '.join([f"{x:.12e}" for x in Tr.flatten()])
        Tr_imu_str = ' '.join([f"{x:.12e}" for x in np.hstack([np.eye(3), np.zeros((3, 1))]).flatten()])
        dist_str = ' '.join([f"{x:.12e}" for x in cam_params[target_cam]["dist"]])

        with open(calib_path, "w") as f:
            f.write(f"P0: {P_lines[0]}\n")
            f.write(f"P1: {P_lines[1]}\n")
            f.write(f"P2: {P_lines[2]}\n")
            f.write(f"P3: {P_lines[3]}\n")
            f.write(f"R0_rect: {R0_str}\n")
            f.write(f"Tr_velo_to_cam: {Tr_str}\n")
            f.write(f"Tr_imu_to_velo: {Tr_imu_str}\n")
            f.write(f"dist: {dist_str}\n")

def image_save(cam_params, annotations, save_dir):
    """
    为每个 (frame_id, camera) 保存对应的 image（只保存 label 被写入的那些）
    """
    # 收集所有需要保存 image 的 (frame_id, camera)
    frame_cam_set = set()
    for ann in annotations:
        frame_cam_set.add((ann["frame_id"], ann["camera"]))

    for frame_id, cam_name in frame_cam_set:
        cam_dir = os.path.join(save_dir, f"image_{cam_name}")
        os.makedirs(cam_dir, exist_ok=True)
        src = cam_params[cam_name]["img_path"]
        # dst = os.path.join(cam_dir, f"{frame_id}_{cam_name}.jpg")
        dst = os.path.join(cam_dir, f"{frame_id}.jpg")
        if os.path.exists(src):  # 安全检查
            shutil.copyfile(src, dst)


if __name__ == "__main__":
    dataset_dir = "/media/whut-user/350b5acb-7342-4f97-b1a7-57e9622bff8b/mono3d/fisheye_data_2w_frame"
    scene_list = [d for d in natsorted(os.listdir(dataset_dir)) 
              if os.path.isdir(os.path.join(dataset_dir, d))] #只遍历目录，过滤掉文件
    # scene_list = ["20231127-155718_20231127-155731"]  # 只处理这一条
    for scene_index in scene_list:
        scene_dir = os.path.join(dataset_dir, scene_index)
        save_dir = os.path.join("/media/whut-user/350b5acb-7342-4f97-b1a7-57e9622bff8b/mono3d/gt_lidar_cord")
        os.makedirs(save_dir, exist_ok=True)
        camera_config_dir = os.path.join(scene_dir, "camera_config")
        image_dir = {
            "front_dir": os.path.join(scene_dir, "image0"),
            "right_dir": os.path.join(scene_dir, "image1"),
            "left_dir": os.path.join(scene_dir, "image2"),
            "back_dir": os.path.join(scene_dir, "image3"),
        }
        annotation_dir = os.path.join(scene_dir, "result")
        point_cloud_dir = os.path.join(scene_dir, "point_cloud")
        
        cam_cfg_list = natsorted(os.listdir(camera_config_dir))
        anno_list    = natsorted(os.listdir(annotation_dir))
        front_list   = natsorted(os.listdir(image_dir["front_dir"]))
        right_list   = natsorted(os.listdir(image_dir["right_dir"]))
        left_list    = natsorted(os.listdir(image_dir["left_dir"]))
        back_list    = natsorted(os.listdir(image_dir["back_dir"]))
        point_cloud_list    = natsorted(os.listdir(point_cloud_dir))
        for cam_cfg_json, anno_json, front_json, right_json, left_json, back_json, pcd_file in zip(cam_cfg_list, anno_list, front_list, 
                                                                                        right_list, left_list, back_list, point_cloud_list):
            cam_cfg_json_path = os.path.join(camera_config_dir, cam_cfg_json)
            anno_json_path = os.path.join(annotation_dir, anno_json)
            image_path ={
                        "front": os.path.join(image_dir["front_dir"], front_json),
                        "right": os.path.join(image_dir["right_dir"], right_json),
                        "left": os.path.join(image_dir["left_dir"], left_json),
                        "back": os.path.join(image_dir["back_dir"], back_json),
            }
            
            cam_params = load_camera_params(cam_cfg_json_path, image_path)
            annotations = load_annotation(anno_json_path,cam_params)
            # vis3d(annotations, image_path, cam_params, save_dir)
            label_save(annotations,save_dir)
            calib_texts(cam_params, annotations, save_dir)   # 已修改
            image_save(cam_params, annotations, save_dir)    # 已修改
            #这样是直接从百度的可见的bbox中读取数据，生成有目标的帧的label、calib和image
            #没有目标的帧就不生成label、calib和image。比如，前视图像，2w帧，只生成了18852帧有目标的
    print('转换完成')