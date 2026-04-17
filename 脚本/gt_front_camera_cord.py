import json
import cv2
import numpy as np
import os
from natsort import natsorted
import math
import shutil
from scipy.spatial.transform import Rotation as R

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

# ------------------ 新增：解析 label 行 ------------------
def parse_label_line(line):
    items = line.strip().split()
    cls = items[0]
    truncation = float(items[1])
    occlusion = int(float(items[2]))
    alpha = float(items[3])
    bbox = list(map(float, items[4:8]))  # x1,y1,x2,y2
    h, w, l = map(float, items[8:11])
    x, y, z = map(float, items[11:14])
    yaw, pitch, roll = map(float, items[14:17])
    return cls, truncation, occlusion, alpha, bbox, h, w, l, x, y, z, yaw, pitch, roll

# ------------------ 保存高精度 label ------------------
def save_label_file(annotations, save_dir, frame_id):
    label_dir = os.path.join(save_dir, "labels")
    os.makedirs(label_dir, exist_ok=True)
    label_path = os.path.join(label_dir, f"{frame_id}.txt")
    
    with open(label_path, "w") as f:
        for ann in annotations:
            ann["class"] = map_class(ann["class"])  # 把中文类别转成统一类别
            # 使用高精度 .12f
            line = f"{ann['class']} {ann['truncation']} {ann['occlusion']} {ann['alpha']:.12f} " \
                   f"{ann['bbox_2d'][0]:.12f} {ann['bbox_2d'][1]:.12f} {ann['bbox_2d'][2]:.12f} {ann['bbox_2d'][3]:.12f} " \
                   f"{ann['hwl'][0]:.12f} {ann['hwl'][1]:.12f} {ann['hwl'][2]:.12f} " \
                   f"{ann['center'][0]:.12f} {ann['center'][1]:.12f} {ann['center'][2]:.12f} " \
                   f"{ann['yaw']:.12f} {ann['pitch']:.12f} {ann['roll']:.12f}\n"
            f.write(line)
    return label_path

def get_3d_bbox_corners(center, size, yaw, pitch, roll):
    h, w, l = size
    x_c, y_c, z_c = center
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    # z_corners = [0, 0, 0, 0, h, h, h, h]
    z_corners = [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2]  #这才是几何中心
    corners = np.vstack([x_corners, y_corners, z_corners])

    """
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch),  np.cos(pitch)]
    ])

    R_y = np.array([
        [ np.cos(yaw), 0, -np.sin(yaw)],
        [ 0,           1,  0          ],
        [np.sin(yaw), 0, np.cos(yaw)]
    ])

    R_z = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll),  np.cos(roll), 0],
        [0,              0,             1]
    ])
    R = R_y @ R_x @ R_z
    """
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])

    R_y = np.array([
        [ np.cos(pitch), 0, -np.sin(pitch)],
        [ 0,           1,  0          ],
        [np.sin(pitch), 0, np.cos(pitch)]
    ])

    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0,              0,             1]
    ])
    R_ = R_x @ R_y @ R_z

    corners_R = R_ @ corners #乘3个欧拉角耦合的旋转矩阵
    corners_3d = corners_R.T + np.array(center).reshape(-1,3)
    # r = R.from_euler('YXZ', [yaw, pitch, roll], degrees=False)
    # rotation_matrix = r.as_matrix() # 3x3
    # corners_rotated = np.dot(rotation_matrix, corners)
    # corners_3d = (corners_rotated + np.array([x_c, y_c, z_c]).reshape(3,1)).T
    return corners_3d  # (8,3)

# ------------------ 从 label.txt 可视化 ------------------
def vis_from_label(label_path, image_path, K, D, ego2cam):
    img = cv2.imread(image_path)
    if img is None:
        return None
        
    with open(label_path, "r") as f:
        lines = f.readlines()
        
    for line in lines:
        if not line.strip():
            continue
        cls, truncation, occlusion, alpha, bbox, h, w, l, x, y, z, yaw, pitch, roll = parse_label_line(line)
        corners_3d = get_3d_bbox_corners([x, y, z], [h, w, l], yaw, pitch, roll)
        corners_2d = project_points_fisheye(corners_3d, K, D, ego2cam)
        if len(corners_2d) > 0:
            # 这里不需要 idx，所以简化 draw_bbox
            img = draw_bbox_simple(img, corners_2d)
    return img

def project_points_fisheye(points_3d, K, dist, lidar2cam):
    # 齐次坐标变换
    # points_3d_h = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
    # cam_points = (lidar2cam @ points_3d_h.T).T[:, :3].astype(np.float64)
    cam_points = np.array(points_3d)

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
        
        # 若不是8个角点（例如部分NaN或无效），则丢弃
        if img_points.shape[0] != 8 or np.any(np.isnan(img_points)):
            return np.empty((0, 2))

    return img_points

def draw_bbox_simple(img, corners_2d, color=(0,255,0), thickness=2):
    """简化版 draw_bbox，不画角点 ID"""
    if len(corners_2d) == 0:
        return img
    corners_2d = corners_2d.astype(int)
    n = len(corners_2d)
    if n >= 8:
        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        for i,j in edges:
            cv2.line(img, tuple(corners_2d[i]), tuple(corners_2d[j]), color, thickness)
    return img

def normalize_angle(angle):
    # make angle in range [-0.5pi, 1.5pi]
    alpha_tan = np.tan(angle)
    alpha_arctan = np.arctan(alpha_tan)
    if np.cos(angle) < 0:
        alpha_arctan = alpha_arctan + math.pi
    return alpha_arctan

def get_camera_3d_8points(obj_size, yaw_lidar, center_lidar, center_in_cam, r_velo2cam, t_velo2cam):
    liadr_r = np.matrix(
        [[math.cos(yaw_lidar), -math.sin(yaw_lidar), 0], [math.sin(yaw_lidar), math.cos(yaw_lidar), 0], [0, 0, 1]]
    )
    l, w, h = obj_size
    corners_3d_lidar = np.matrix(
        [
            [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [0, 0, 0, 0, h, h, h, h],
        ]
    )
    corners_3d_lidar = liadr_r * corners_3d_lidar + np.matrix(center_lidar).T
    corners_3d_cam = r_velo2cam * corners_3d_lidar + t_velo2cam

    x0, z0 = corners_3d_cam[0, 0], corners_3d_cam[2, 0]
    x3, z3 = corners_3d_cam[0, 3], corners_3d_cam[2, 3]
    dx, dz = x0 - x3, z0 - z3
    # yaw_cam = math.atan2(-dz, dx)  #这是X朝前
    yaw_cam = math.atan2(dx, dz)   #Z朝前

    alpha = yaw_cam - math.atan2(center_in_cam[0], center_in_cam[2])

    if alpha > math.pi:
        alpha = alpha - 2.0 * math.pi
    if alpha <= (-1 * math.pi):
        alpha = alpha + 2.0 * math.pi

    # alpha_arctan = normalize_angle(alpha)

    rt_matrix1 = np.eye(4)
    rt_matrix1[:3, 3] = center_lidar
    rt_matrix1[:3, :3] = liadr_r
    rt_matrix2 = np.eye(4)
    rt_matrix2[:3, 3] = t_velo2cam.flatten()
    rt_matrix2[:3, :3] = r_velo2cam

    rt_matrix = rt_matrix2 @ rt_matrix1
    r_obj_cam = rt_matrix[:3, :3]
    rotation_obj_cam = R.from_matrix(r_obj_cam)
    euler_angles = rotation_obj_cam.as_euler('XYZ', degrees=False)
    roll_cam = euler_angles[0]
    pitch_cam = euler_angles[1]
    yaw_cam = euler_angles[2]
    return alpha, yaw_cam, pitch_cam, roll_cam, corners_3d_cam


def calib_texts(save_dir, basename, K, Tr_velo_to_cam,D):
    calib_dir = os.path.join(save_dir, "calibs")
    os.makedirs(calib_dir, exist_ok=True)
    calib_path = os.path.join(calib_dir, f"{basename}.txt")

    # P0: 3x4 projection matrix = K @ [I | 0]
    # P0 = np.zeros((3, 4))
    P0 = K  # K is 3x3

    # Tr_velo_to_cam: 3x4 [R | t]
    Tr = Tr_velo_to_cam  # shape (3, 4)
    T = np.eye(3, 4).reshape(-1)
    dist = D

    with open(calib_path, 'w') as f:
        f.write(f"P0: {' '.join(map(lambda x: f'{x:.12e}', P0.flatten()))}\n")
        f.write(f"P1: {' '.join(map(str, np.zeros(12)))}\n")
        f.write(f"P2: {' '.join(map(str, np.zeros(12)))}\n")
        f.write(f"P3: {' '.join(map(str, np.zeros(12)))}\n")
        f.write(f"R0_rect: {' '.join(map(str, np.eye(3).flatten()))}\n")
        f.write(f"Tr_velo_to_cam: {' '.join(map(lambda x: f'{x:.12e}', Tr.flatten()))}\n")
        f.write(f"Tr_imu_to_velo: {' '.join(f'{v:.12e}' for v in T)}\n")
        f.write(f"dist: {' '.join(map(lambda x: f'{x:.12e}', dist.flatten()))}\n")
        
    return calib_path


def image_save(image_path, save_dir, basename):
    image_dir = os.path.join(save_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    # 强制使用 .jpg 后缀，无论原始是 .jpeg 还是 .png
    dst_path = os.path.join(image_dir, f"{basename}.jpg")
    shutil.copy(image_path, dst_path)
    return dst_path


def convert_point(point, matrix):
    return matrix @ point

# ------------------ 主函数 ------------------
if __name__ == "__main__":
    dataset_dir = "/media/whut-user/350b5acb-7342-4f97-b1a7-57e9622bff8b/mono3d/fisheye_data_2w_frame"
    scene_list = natsorted(os.listdir(dataset_dir))
    for scene_index in scene_list:
        scene_dir = os.path.join(dataset_dir, scene_index)
        save_dir = os.path.join("/media/whut-user/350b5acb-7342-4f97-b1a7-57e9622bff8b/mono3d/data_camera/demo_data/trainval")
        os.makedirs(save_dir, exist_ok=True)
        camera_config_dir = os.path.join(scene_dir, "camera_config")
        label_dir = os.path.join(scene_dir, "result")
        image_dir = os.path.join(scene_dir, "image0")
        
        if not all(os.path.exists(d) for d in [camera_config_dir, label_dir, image_dir]):
            continue
        
        # 获取公共 basename
        cam_basenames = {os.path.splitext(f)[0] for f in os.listdir(camera_config_dir)}
        label_basenames = {os.path.splitext(f)[0] for f in os.listdir(label_dir)}
        image_basenames = {os.path.splitext(f)[0] for f in os.listdir(image_dir)}
        common_basenames = natsorted(list(cam_basenames & label_basenames & image_basenames))
        
        print(f"Scene {scene_index}: found {len(common_basenames)} frames")
        
        for basename in common_basenames:
            def find_file(dir_path, basename):
                for f in os.listdir(dir_path):
                    if os.path.splitext(f)[0] == basename:
                        return os.path.join(dir_path, f)
                return None
            
            cam_cfg_path = find_file(camera_config_dir, basename)
            anno_path = find_file(label_dir, basename)
            image_path = find_file(image_dir, basename)

            if not (cam_cfg_path and anno_path and image_path):
                continue
            
            # === 加载数据 ===
            with open(cam_cfg_path, "r") as f:
                cam_cfgs = json.load(f)
                front_cam_cfg = cam_cfgs[0]
                internal_params = front_cam_cfg["camera_internal"]
                distortion = front_cam_cfg["distortion"]
                K = np.array([
                    internal_params['fx'], 0.0, internal_params['cx'],
                    0.0, internal_params['fy'], internal_params['cy'],
                    0.0, 0.0, 1.0
                ], dtype=np.float64).reshape(3, 3)
                D = np.array([
                    distortion['k1'], distortion['k2'], 
                    distortion['k3'], distortion['k4']
                ], dtype=np.float64)
                ego2cam = np.array(front_cam_cfg["camera_external"], dtype=np.float64).reshape(-1, 4).T
                r_velo2cam = ego2cam[:3, :3]
                t_velo2cam = ego2cam[:3, 3].reshape(3, 1)
                Tr_velo_to_cam = np.hstack((r_velo2cam, t_velo2cam))
            
            # === 解析标注 ===
            annotations = []
            with open(anno_path, "r") as f:
                anno_data = json.load(f)
                for obj in anno_data.get('objects', []):
                    if '2D_bbox' not in obj or 'image0' not in obj['2D_bbox']:
                        continue
                    # lwh = [obj['contour']['size3D']['x'], obj['contour']['size3D']['y'], obj['contour']['size3D']['z']]
                    hwl = [obj['contour']['size3D']['z'], obj['contour']['size3D']['y'], obj['contour']['size3D']['x']]
                    center_ego = [obj['contour']['center3D']['x'], obj['contour']['center3D']['y'], obj['contour']['center3D']['z']]
                    yaw_lidar = obj['contour']['rotation3D']['z']
                    className = obj['className']
                    
                    x, y, z = center_ego #15.58912820440589 -1.9444349600255943 0.8605728043437948
                    h, w, l = hwl #4.5957446808510625 1.937506179389703 1.4496417677522808
                    # z = z - h / 2  #0.8605728043437948-1.4496417677522808/2 = 0.13575192046765439
                    #用几何中心，而不是底面中心
                    bottom_center = [x, y, z] #[15.58912820440589, -1.9444349600255943, 0.13575192046765439]
                    obj_size = [h, w, l] #[4.5957446808510625, 1.937506179389703, 1.4496417677522808]
                    bottom_center_in_cam = r_velo2cam * np.matrix(bottom_center).T + t_velo2cam
                    alpha, yaw_cam, pitch_cam, roll_cam, _ = get_camera_3d_8points(
                        obj_size, yaw_lidar, bottom_center, bottom_center_in_cam, r_velo2cam, t_velo2cam
                    )
                    cam_x, cam_y, cam_z = convert_point(np.array([x, y, z, 1]).T, Tr_velo_to_cam)
                    
                    # 保存 2D bbox for image0
                    bbox_2d = obj['2D_bbox']['image0']
                    
                    # truncation / occlusion
                    truncation = occlusion = None
                    for cv in obj.get('classValues', []):
                        if cv.get('name') == 'truncation':
                            truncation = cv.get('value')[0]
                        elif cv.get('name') == 'occlusion':
                            occlusion = cv.get('value')[0]
                    
                    annotations.append({
                        "class": className,
                        "hwl": hwl,
                        # "lwh": lwh,
                        "center": [cam_x, cam_y, cam_z],  #相机坐标系下的中心点
                        "yaw": yaw_cam ,  # 相机坐标系下的yaw角
                        "pitch": pitch_cam,  # 相机坐标系下的pitch角
                        "roll": roll_cam,    # 相机坐标系下的roll角
                        "bbox_2d": bbox_2d,
                        "alpha": alpha,
                        "truncation": truncation,  
                        "occlusion": occlusion,
                    })
            
            
             #关键：只在有有效标注时才保存所有文件
            if not annotations:
                print(f"Skipping frame {basename}: no valid objects with image0 bbox.")
                continue

            # === 仅当 annotations 非空时，才保存 ===
            
            # === 步骤 1: 保存高精度 label.txt ===
            label_path = save_label_file(annotations, save_dir, basename)
            
            # === 步骤 2: 从 label.txt 读取并可视化 ===
            vis_dir = os.path.join(save_dir, "vis_front_from_label")
            os.makedirs(vis_dir, exist_ok=True)
            vis_path = os.path.join(vis_dir, f"{basename}_vis.jpg")
            
            # === 步骤 3:真值可视化
            img_vis = vis_from_label(label_path, image_path, K, D, ego2cam)
            if img_vis is not None:
                cv2.imwrite(vis_path, img_vis)
                print(f"Saved (from label): {vis_path}")
            
            #=== 步骤 4: 生成 calib.txt（使用 P0）
            calib_path = calib_texts(save_dir, basename, K, Tr_velo_to_cam,D)

            # === 步骤 5:复制 image0
            image_dst_path = image_save(image_path, save_dir, basename)  
            
                
            