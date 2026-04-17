import os
import cv2
import numpy as np
from multiprocessing import Pool

cv2.setNumThreads(0)


def parse_result_lidar_line(line):
    data = line.strip().split()
    if len(data) < 16:
        return None
    cls = data[0]
    h, w, l = map(float, data[8:11])
    x, y, z = map(float, data[11:14])
    yaw = float(data[14])
    score = float(data[15])
    return cls, [h, w, l], [x, y, z], yaw, score


def get_3d_bbox_corners_lidar(center, size, yaw):
    h, w, l = size
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    z_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
    corners = np.vstack([x_corners, y_corners, z_corners])

    c, s = np.cos(yaw), np.sin(yaw)
    rot_z = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ], dtype=np.float32)
    corners = rot_z @ corners
    return (corners.T + np.array(center, dtype=np.float32)).astype(np.float32)


def project_points_fisheye(points_3d, k_fisheye, dist, lidar2cam):
    points_3d_h = np.hstack([points_3d, np.ones((points_3d.shape[0], 1), dtype=np.float64)])
    cam_points = (lidar2cam @ points_3d_h.T).T[:, :3].astype(np.float64)

    # ===== 极坐标鱼眼投影 =====
    K = np.array(k_fisheye, dtype=np.float64)
    D = np.array(dist, dtype=np.float64).flatten()

    X = cam_points[..., 0]
    Y = cam_points[..., 1]
    Z = cam_points[..., 2]

    eps = 1e-6
    r_xy = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(r_xy, Z + eps)
    phi = np.arctan2(Y, X)

    k1, k2, k3, k4 = D
    theta2 = theta * theta
    theta_d = theta * (
        1 + k1 * theta2
          + k2 * theta2**2
          + k3 * theta2**3
          + k4 * theta2**4
    )

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = cx + fx * theta_d * np.cos(phi)
    v = cy + fy * theta_d * np.sin(phi)

    img_points = np.stack([u, v], axis=-1)   # (B, 8, 2)

    return img_points


def draw_bbox_simple(img, corners_2d, color=(0, 255, 0), thickness=2):
    if len(corners_2d) == 0:
        return img
    corners_2d = corners_2d.astype(int)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    for i, j in edges:
        cv2.line(img, tuple(corners_2d[i]), tuple(corners_2d[j]), color, thickness)
    return img


def read_fisheye_calib(calib_path):
    calib = {}
    with open(calib_path, 'r') as f:
        for line in f:
            if ':' not in line:
                continue
            key, value = line.strip().split(':', 1)
            calib[key] = np.array([float(x) for x in value.strip().split()], dtype=np.float64)

    k_fisheye = calib['P0'].reshape(3, 3)
    dist = calib['dist']
    lidar2cam = calib['Tr_velo_to_cam'].reshape(3, 4)
    return k_fisheye, dist, lidar2cam


def process_single(name, result_dir, image_dir_fisheye, calib_dir, save_dir, score_thresh):
    if not name.endswith('.txt'):
        return

    basename = os.path.splitext(name)[0]
    img_path = os.path.join(image_dir_fisheye, basename + '.jpg')
    if not os.path.exists(img_path):
        img_path = os.path.join(image_dir_fisheye, basename + '.png')
    if not os.path.exists(img_path):
        return

    calib_path = os.path.join(calib_dir, basename + '.txt')
    result_path = os.path.join(result_dir, name)
    if not os.path.exists(calib_path) or not os.path.exists(result_path):
        return

    img = cv2.imread(img_path)
    k_fisheye, dist, lidar2cam = read_fisheye_calib(calib_path)

    with open(result_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parsed = parse_result_lidar_line(line)
        if parsed is None:
            continue

        cls, size, center, yaw, score = parsed
        if score < score_thresh:
            continue

        corners_3d = get_3d_bbox_corners_lidar(center, size, yaw)
        corners_2d = project_points_fisheye(corners_3d, k_fisheye, dist, lidar2cam)
        if len(corners_2d) == 0:
            continue

        img = draw_bbox_simple(img, corners_2d)
        text_pt = tuple(corners_2d[0].astype(int))
        cv2.putText(
            img, f'{cls}:{score:.2f}', text_pt,
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

    save_path = os.path.join(save_dir, basename + '.jpg')
    cv2.imwrite(save_path, img)
    print('Saved:', save_path)


def visualize(result_dir, image_dir_fisheye, calib_dir, save_dir, score_thresh=0.1):
    os.makedirs(save_dir, exist_ok=True)
    file_list = [f for f in os.listdir(result_dir) if f.endswith('.txt')]
    args = [
        (name, result_dir, image_dir_fisheye, calib_dir, save_dir, score_thresh)
        for name in file_list
    ]
    with Pool(40) as p:
        p.starmap(process_single, args)


if __name__ == '__main__':
    result_dir = '/media/whut-user/350b5acb-7342-4f97-b1a7-57e9622bff8b/mono3d/git-apa-perception-mono3dod-t2/mono3DOD_Parking/mono3DOD_Parking_code/exp/ddd/default/results_lidar'
    image_dir_fisheye = '/media/whut-user/350b5acb-7342-4f97-b1a7-57e9622bff8b/mono3d/data_camera/demo_data/trainval/images'
    calib_dir = '/media/whut-user/350b5acb-7342-4f97-b1a7-57e9622bff8b/mono3d/data_camera_cyl/demo_data/trainval/calibs'
    save_dir = '/media/whut-user/350b5acb-7342-4f97-b1a7-57e9622bff8b/mono3d/vis_pred_3d/vis_pred_3d_lidar_fisheye0413_epoch200'

    visualize(result_dir, image_dir_fisheye, calib_dir, save_dir, score_thresh=0.1)