import json
import cv2
import numpy as np
import os
from natsort import natsorted
import math
import shutil
from scipy.spatial.transform import Rotation as R
from multiprocessing import Pool


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


mapping = {}
for cat in category_info:
    for sub in cat["description"].split(","):
        mapping[sub.strip()] = cat["name"]


def map_class(chinese_class: str) -> str:
    return mapping.get(chinese_class, "unknown")


CAMERA_ORDER = {
    "front": {"image_key": "image0", "camera_index": 0},
    "right": {"image_key": "image1", "camera_index": 1},
    "left": {"image_key": "image2", "camera_index": 2},
    "back": {"image_key": "image3", "camera_index": 3},
}


def list_stems(directory, suffix=None):
    stems = set()
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        if suffix is not None and not name.lower().endswith(suffix.lower()):
            continue
        stems.add(os.path.splitext(name)[0])
    return stems


def find_file_by_stem(directory, stem):
    for name in os.listdir(directory):
        if os.path.splitext(name)[0] == stem:
            return os.path.join(directory, name)
    return None


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
            ann["class"] = map_class(ann["class"])
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
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    z_corners = [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2]
    corners = np.vstack([x_corners, y_corners, z_corners])
    R_x = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    R_y = np.array([[np.cos(pitch), 0, -np.sin(pitch)], [0, 1, 0], [np.sin(pitch), 0, np.cos(pitch)]])
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    R_ = R_x @ R_y @ R_z
    corners_R = R_ @ corners
    corners_3d = corners_R.T + np.array(center).reshape(-1, 3)
    return corners_3d


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
            img = draw_bbox_simple(img, corners_2d)
    return img


def project_points_fisheye(points_3d, K, dist, lidar2cam):
    cam_points = np.array(points_3d)
    if np.all(cam_points[:, 2] <= 0):
        return np.empty((0, 2))
    num_front = np.sum(cam_points[:, 2] > 0)
    if num_front <= 2:
        return np.empty((0, 2))

    K = np.array(K, dtype=np.float64)
    D = np.array(dist, dtype=np.float64)
    img_points = np.zeros((cam_points.shape[0], 2), dtype=np.float64)
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
    back_mask = ~front_mask
    if np.any(back_mask):
        X = cam_points[back_mask, 0]
        Y = cam_points[back_mask, 1]
        Z = cam_points[back_mask, 2]
        r = np.sqrt(X ** 2 + Y ** 2)
        theta = np.arctan2(r, Z + 1e-8)
        phi = np.arctan2(Y, X)
        k1, k2, k3, k4 = D.flatten()
        theta_d = theta * (1 + k1 * theta ** 2 + k2 * theta ** 4 + k3 * theta ** 6 + k4 * theta ** 8)

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        u = cx + fx * theta_d * np.cos(phi)
        v = cy + fy * theta_d * np.sin(phi)
        img_points[back_mask] = np.stack([u, v], axis=1)

        if img_points.shape[0] != 8 or np.any(np.isnan(img_points)):
            return np.empty((0, 2))
    return img_points


def draw_bbox_simple(img, corners_2d, color=(0, 255, 0), thickness=2):
    if len(corners_2d) == 0:
        return img
    corners_2d = corners_2d.astype(int)
    n = len(corners_2d)
    if n >= 8:
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
        for i, j in edges:
            cv2.line(img, tuple(corners_2d[i]), tuple(corners_2d[j]), color, thickness)
    return img


def normalize_angle(angle):
    alpha_tan = np.tan(angle)
    alpha_arctan = np.arctan(alpha_tan)
    if np.cos(angle) < 0:
        alpha_arctan = alpha_arctan + math.pi
    return alpha_arctan


def get_camera_3d_8points(obj_size, yaw_lidar, center_lidar, center_in_cam, r_velo2cam, t_velo2cam):
    liadr_r = np.matrix([[math.cos(yaw_lidar), -math.sin(yaw_lidar), 0], [math.sin(yaw_lidar), math.cos(yaw_lidar), 0], [0, 0, 1]])
    l, w, h = obj_size
    corners_3d_lidar = np.matrix([[l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2], [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], [0, 0, 0, 0, h, h, h, h]])
    corners_3d_lidar = liadr_r * corners_3d_lidar + np.matrix(center_lidar).T
    corners_3d_cam = r_velo2cam * corners_3d_lidar + t_velo2cam
    x0, z0 = corners_3d_cam[0, 0], corners_3d_cam[2, 0]
    x3, z3 = corners_3d_cam[0, 3], corners_3d_cam[2, 3]
    dx, dz = x0 - x3, z0 - z3
    yaw_cam = math.atan2(dx, dz)
    alpha = yaw_cam - math.atan2(center_in_cam[0], center_in_cam[2])
    if alpha > math.pi:
        alpha = alpha - 2.0 * math.pi
    if alpha <= (-1 * math.pi):
        alpha = alpha + 2.0 * math.pi
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


def calib_texts(save_dir, basename, K, Tr_velo_to_cam, D):
    calib_dir = os.path.join(save_dir, "calibs")
    os.makedirs(calib_dir, exist_ok=True)
    calib_path = os.path.join(calib_dir, f"{basename}.txt")
    P0 = K
    Tr = Tr_velo_to_cam
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
    dst_path = os.path.join(image_dir, f"{basename}.jpg")
    shutil.copy(image_path, dst_path)
    return dst_path


def convert_point(point, matrix):
    return matrix @ point


def get_class_value(obj, key_name, camera_index):
    for cv in obj.get('classValues', []):
        if cv.get('name') == key_name:
            values = cv.get('value', [])
            if camera_index < len(values):
                return values[camera_index]
            if values:
                return values[0]
    return 0


def load_camera_params(cam_cfg_json_path, image_path):
    with open(cam_cfg_json_path, "r") as f:
        cam_cfgs = json.load(f)

    frame_id = os.path.splitext(os.path.basename(cam_cfg_json_path))[0]
    cam_params = {}
    for camera_name, camera_meta in CAMERA_ORDER.items():
        camera_index = camera_meta["camera_index"]
        if camera_index >= len(cam_cfgs):
            continue

        cam_cfg = cam_cfgs[camera_index]
        internal_params = cam_cfg["camera_internal"]
        distortion = cam_cfg["distortion"]
        K = np.array([internal_params['fx'], 0.0, internal_params['cx'], 0.0, internal_params['fy'], internal_params['cy'], 0.0, 0.0, 1.0], dtype=np.float64).reshape(3, 3)
        D = np.array([distortion['k1'], distortion['k2'], distortion['k3'], distortion['k4']], dtype=np.float64)

        ego2cam_raw = np.array(cam_cfg["camera_external"], dtype=np.float64).reshape(4, 4)
        row_major = str(cam_cfg.get("rowMajor", "true")).lower() == "true"
        ego2cam = ego2cam_raw if row_major else ego2cam_raw.T
        r_velo2cam = ego2cam[:3, :3]
        t_velo2cam = ego2cam[:3, 3].reshape(3, 1)
        Tr_velo_to_cam = np.hstack((r_velo2cam, t_velo2cam))

        cam_params[camera_name] = {
            "frame_id": frame_id,
            "image_path": image_path[camera_name],
            "image_key": camera_meta["image_key"],
            "camera_index": camera_index,
            "K": K,
            "D": D,
            "ego2cam": ego2cam,
            "r_velo2cam": r_velo2cam,
            "t_velo2cam": t_velo2cam,
            "Tr_velo_to_cam": Tr_velo_to_cam,
        }
    return cam_params


def load_annotation(anno_json_path):
    with open(anno_json_path, "r",encoding='utf-8') as f:
        anno_data = json.load(f)
    return anno_data.get('objects', [])


def project_all_gt_to_cameras(cam_params, annotations, save_dir):
    for camera_name, params in cam_params.items():
        frame_id = params["frame_id"]
        image_key = params["image_key"]
        camera_index = params["camera_index"]
        K = params["K"]
        D = params["D"]
        ego2cam = params["ego2cam"]
        r_velo2cam = params["r_velo2cam"]
        t_velo2cam = params["t_velo2cam"]
        Tr_velo_to_cam = params["Tr_velo_to_cam"]
        image_path = params["image_path"]

        annotations_one_camera = []
        for obj in annotations:
            if obj.get("type", "3D_BOX") != "3D_BOX":
                continue
            if '2D_bbox' not in obj or image_key not in obj['2D_bbox']:
                continue

            hwl = [obj['contour']['size3D']['z'], obj['contour']['size3D']['y'], obj['contour']['size3D']['x']]
            center_ego = [obj['contour']['center3D']['x'], obj['contour']['center3D']['y'], obj['contour']['center3D']['z']]
            yaw_lidar = obj['contour']['rotation3D']['z']
            className = obj['className']

            x, y, z = center_ego
            h, w, l = hwl
            bottom_center = [x, y, z]
            obj_size = [h, w, l]
            bottom_center_in_cam = r_velo2cam * np.matrix(bottom_center).T + t_velo2cam
            alpha, yaw_cam, pitch_cam, roll_cam, _ = get_camera_3d_8points(obj_size, yaw_lidar, bottom_center, bottom_center_in_cam, r_velo2cam, t_velo2cam)
            cam_x, cam_y, cam_z = convert_point(np.array([x, y, z, 1]).T, Tr_velo_to_cam)

            truncation = float(get_class_value(obj, 'truncation', camera_index))
            occlusion = int(float(get_class_value(obj, 'occlusion', camera_index)))

            annotations_one_camera.append({
                "class": className,
                "hwl": hwl,
                "center": [cam_x, cam_y, cam_z],
                "yaw": yaw_cam,
                "pitch": pitch_cam,
                "roll": roll_cam,
                "bbox_2d": obj['2D_bbox'][image_key],
                "alpha": alpha,
                "truncation": truncation,
                "occlusion": occlusion,
            })

        camera_save_dir = os.path.join(save_dir, camera_name)
        label_path = save_label_file(annotations_one_camera, camera_save_dir, frame_id)

        vis_dir = os.path.join(camera_save_dir, "vis_from_label")
        os.makedirs(vis_dir, exist_ok=True)
        vis_path = os.path.join(vis_dir, f"{frame_id}_vis.jpg")

        img_vis = vis_from_label(label_path, image_path, K, D, ego2cam)
        if img_vis is not None:
            cv2.imwrite(vis_path, img_vis)
            print(f"Saved (from label): {vis_path}")

        calib_texts(camera_save_dir, frame_id, K, Tr_velo_to_cam, D)
        image_save(image_path, camera_save_dir, frame_id)


def process_one_frame(cam_cfg_json_path, anno_json_path, image_path, save_dir):
    cam_params = load_camera_params(cam_cfg_json_path, image_path)
    annotations = load_annotation(anno_json_path)
    project_all_gt_to_cameras(cam_params, annotations, save_dir)


if __name__ == "__main__":
    dataset_dir = "C:/Users/yingxie/Desktop/mono3d-main/fisheye_data_aug"
    scene_list = natsorted(os.listdir(dataset_dir))
    for scene_index in scene_list:
        scene_dir = os.path.join(dataset_dir, scene_index)
        save_dir = os.path.join(
            "C:/Users/yingxie/Desktop/mono3d-main/data_camera/demo_data/trainval_gt_vis",
            scene_dir.split(os.sep)[-1],
        )
        os.makedirs(save_dir, exist_ok=True)
        camera_config_dir = os.path.join(scene_dir, "camera_config")
        image_dir = {
            "front_dir": os.path.join(scene_dir, "image0"),
            "right_dir": os.path.join(scene_dir, "image1"),
            "left_dir": os.path.join(scene_dir, "image2"),
            "back_dir": os.path.join(scene_dir, "image3"),
        }
        annotation_dir = os.path.join(scene_dir, "result")

        if not all(os.path.exists(path) for path in [camera_config_dir, annotation_dir, image_dir["front_dir"], image_dir["right_dir"], image_dir["left_dir"], image_dir["back_dir"]]):
            continue

        cam_cfg_ids = list_stems(camera_config_dir, ".json")
        anno_ids = list_stems(annotation_dir, ".json")
        front_ids = list_stems(image_dir["front_dir"])
        right_ids = list_stems(image_dir["right_dir"])
        left_ids = list_stems(image_dir["left_dir"])
        back_ids = list_stems(image_dir["back_dir"])

        common_frame_ids = natsorted(
            list(cam_cfg_ids & anno_ids & front_ids & right_ids & left_ids & back_ids)
        )
        frame_count = len(common_frame_ids)
        print(f"Scene {scene_index}: found {frame_count} frames")

        args = []
        for frame_id in common_frame_ids:
            cam_cfg_json_path = os.path.join(camera_config_dir, f"{frame_id}.json")
            anno_json_path = os.path.join(annotation_dir, f"{frame_id}.json")
            front_image_path = find_file_by_stem(image_dir["front_dir"], frame_id)
            right_image_path = find_file_by_stem(image_dir["right_dir"], frame_id)
            left_image_path = find_file_by_stem(image_dir["left_dir"], frame_id)
            back_image_path = find_file_by_stem(image_dir["back_dir"], frame_id)
            if not all([front_image_path, right_image_path, left_image_path, back_image_path]):
                continue
            image_path = {
                "front": front_image_path,
                "right": right_image_path,
                "left": left_image_path,
                "back": back_image_path,
            }
            args.append((cam_cfg_json_path, anno_json_path, image_path, save_dir))

        try:
            with Pool(8) as p:
                p.starmap(process_one_frame, args)
        except (PermissionError, OSError) as exc:
            print(f"Pool failed for scene {scene_index}, fallback to serial: {exc}")
            for arg in args:
                process_one_frame(*arg)

    print("All scenes processed.")
