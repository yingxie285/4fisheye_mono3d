import cv2
import numpy as np
import os
from multiprocessing import Pool
import tqdm

# ====================== 工具函数 ======================
def wrap(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi


# ====================== fisheye → cylindrical ======================
def get_mapping(K, D, calib, hfov=np.deg2rad(190), vfov=np.deg2rad(140)):
    R = calib['extrinsic']['R'].copy()
    rdf_to_flu = np.array([[0,0,1],[-1,0,0],[0,-1,0]], dtype=np.float64)
    R = R @ rdf_to_flu

    azimuth = np.arccos(R[2,2] / np.sqrt(R[0,2]**2 + R[2,2]**2))
    if R[0,2] < 0:
        azimuth = 2*np.pi - azimuth
    tilt = -np.arccos(np.sqrt(R[0,2]**2 + R[2,2]**2))

    Ry = np.array([[np.cos(azimuth),0,np.sin(azimuth)],
                   [0,1,0],
                   [-np.sin(azimuth),0,np.cos(azimuth)]]).T
    R_final = R @ Ry

    f = calib['intrinsic']['f']
    h = int(2 * f * np.tan(vfov / 2))
    w = int(f * hfov)

    K_cyl = np.array([[f,0,w/2],
                      [0,f,f*np.tan(vfov/2 + tilt)],
                      [0,0,1]], dtype=np.float64)

    K_cyl_inv = np.linalg.inv(K_cyl)
    xv, yv = np.meshgrid(range(w), range(h), indexing='xy')
    p = np.stack([xv, yv, np.ones_like(xv)], axis=-1).astype(np.float64)[..., np.newaxis]

    r = (K_cyl_inv @ p)[...,0]
    r /= r[:, :, [2]]

    r_cart = np.zeros_like(r)
    r_cart[:,:,2] = np.cos(r[:,:,0])
    r_cart[:,:,0] = np.sin(r[:,:,0])
    r_cart[:,:,1] = r[:,:,1]

    r_cam = (R_final @ r_cart[..., np.newaxis])[...,0]
    rays = r_cam.reshape(-1,1,3).astype(np.float64)

    img_points, _ = cv2.fisheye.projectPoints(
        rays, np.zeros((3,1)), np.zeros((3,1)),
        K.astype(np.float64), D.astype(np.float64)
    )
    img_points = img_points.reshape(h,w,2)

    return img_points[...,0].astype(np.float32), img_points[...,1].astype(np.float32), R_final, K_cyl


def fisheye_to_cylindrical(K, D, image, calib):
    mapx, mapy, R_final, K_cyl = get_mapping(K, D, calib)
    cyl = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)

    # ===== 裁剪 + pad =====
    target_h = 960
    target_w = 1408

    cyl = cyl[:target_h, :]
    h, w = cyl.shape[:2]

    if w < target_w:
        cyl = cv2.copyMakeBorder(
            cyl, 0, 0, 0, target_w - w,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )

    return cyl, R_final, K_cyl


def project_point_to_cylindrical(P_cam, R_final, K_cyl):
    norm = np.linalg.norm(P_cam)
    if norm < 1e-6:
        return None

    dir_cam = P_cam / norm
    dir_cyl = R_final.T @ dir_cam

    x, y, z = dir_cyl
    theta = np.arctan2(x, z)
    rho = np.sqrt(x*x + z*z)
    y_norm = y / rho if rho > 1e-6 else 0.0

    f, cx, cy = K_cyl[0,0], K_cyl[0,2], K_cyl[1,2]
    return f*theta + cx, f*y_norm + cy


# ====================== 3D可视化 ======================
def draw_3d_boxes_on_cylindrical(cyl_img, label_path, R_final, K_cyl):

    f = K_cyl[0, 0]
    u0 = K_cyl[0, 2]
    v0 = K_cyl[1, 2]

    img = cyl_img.copy()
    obj_id = 0   #  新增编号

    with open(label_path, 'r') as fp:
        for line in fp:
            parts = line.strip().split()
            if len(parts) < 8:
                continue

            u = float(parts[1])
            v = float(parts[2])
            H = float(parts[3])
            W = float(parts[4])
            L = float(parts[5])
            yaw_local = float(parts[6])
            rho = float(parts[7])

            phi = (u - u0) / f
            global_yaw = wrap(yaw_local + phi)

            tan_psi = (v - v0) / f
            dir_cyl = np.array([np.sin(phi), tan_psi, np.cos(phi)], dtype=np.float32)
            dir_cyl /= np.linalg.norm(dir_cyl)

            dir_cam = R_final @ dir_cyl
            center_cam = (dir_cam * rho).reshape(3, 1)
            center_upright = R_final.T @ center_cam

            corners_local = np.array([
                [ W/2,  H/2,  L/2], [ W/2,  H/2, -L/2],
                [ W/2, -H/2,  L/2], [ W/2, -H/2, -L/2],
                [-W/2,  H/2,  L/2], [-W/2,  H/2, -L/2],
                [-W/2, -H/2,  L/2], [-W/2, -H/2, -L/2]
            ], dtype=np.float32).T

            c, s = np.cos(global_yaw), np.sin(global_yaw)
            R_yaw = np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=np.float32)

            corners = R_yaw @ corners_local + center_upright

            pts = []
            for i in range(8):
                X, Y, Z = corners[:, i]
                r = np.sqrt(X**2 + Z**2 + 1e-8)
                uc = f * np.arctan2(X, Z) + u0
                vc = f * np.tan(np.arctan(Y / r)) + v0
                pts.append((int(round(uc)), int(round(vc))))

            edges = [(0,1),(2,3),(4,5),(6,7),(1,3),(3,7),(7,5),(5,1),(0,2),(2,6),(6,4),(4,0)]
            for i,j in edges:
                cv2.line(img, pts[i], pts[j], (0,255,0), 2)

            # ===== 在bbox旁边写编号 =====
            # 用中心点或者第一个角点都行
            text_pos = (int(u) + 5, int(v) - 5)

            cv2.putText(
                img,
                f"{obj_id+1}",
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            cv2.circle(img, (int(u), int(v)), 5, (0,0,255), -1)

            obj_id += 1   # 编号递增

    return img


def save_kitti_gts_cyl(save_path, lines, R_final, K_cyl):

    f = K_cyl[0, 0]
    u0 = K_cyl[0, 2]
    v0 = K_cyl[1, 2]

    with open(save_path, 'w') as f_out:

        for line in lines:
            parts = line.split()
            class_name = parts[0]

            h, w, l = map(float, parts[8:11])
            x, y, z = map(float, parts[11:14])

            P_cam = np.array([x, y, z], dtype=np.float32)

            # ===== 投影到柱面 =====
            uv = project_point_to_cylindrical(P_cam, R_final, K_cyl)
            if uv is None:
                continue

            u, v = uv

            # ===== 方向 =====
            yaw = float(parts[3])
            phi = (u - u0) / f
            ry = wrap(yaw + phi)
            alpha = wrap(ry - phi)

            # ===== 构造3D框 =====
            center_upright = R_final.T @ P_cam.reshape(3,1)

            corners_local = np.array([
                [ w/2,  h/2,  l/2], [ w/2,  h/2, -l/2],
                [ w/2, -h/2,  l/2], [ w/2, -h/2, -l/2],
                [-w/2,  h/2,  l/2], [-w/2,  h/2, -l/2],
                [-w/2, -h/2,  l/2], [-w/2, -h/2, -l/2]
            ], dtype=np.float32).T

            c, s = np.cos(ry), np.sin(ry)
            R_yaw = np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=np.float32)

            corners = R_yaw @ corners_local + center_upright

            pts = []
            for i in range(8):
                X, Y, Z = corners[:, i]
                r = np.sqrt(X**2 + Z**2 + 1e-8)
                uc = f * np.arctan2(X, Z) + u0
                vc = f * np.tan(np.arctan(Y / r)) + v0
                pts.append((uc, vc))

            pts = np.array(pts)
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)

            # ===== 写KITTI =====
            f_out.write(
                f"{class_name} 0 0 {alpha:.6f} "
                f"{x1:.3f} {y1:.3f} {x2:.3f} {y2:.3f} "
                f"{h:.3f} {w:.3f} {l:.3f} "
                f"{x:.3f} {y:.3f} {z:.3f} "
                f"{ry:.6f}\n"
            )

def draw_kitti_on_cylindrical(cyl_img, kitti_path, R_final, K_cyl):

    img = cyl_img.copy()

    f = K_cyl[0, 0]
    u0 = K_cyl[0, 2]
    v0 = K_cyl[1, 2]
    obj_id = 0

    with open(kitti_path, 'r') as fp:
        for line in fp:
            parts = line.strip().split()

            if len(parts) < 15:
                continue

            h, w, l = map(float, parts[8:11])
            x, y, z = map(float, parts[11:14])
            ry = float(parts[14])

            P_cam = np.array([x, y, z], dtype=np.float32)
            center_upright = R_final.T @ P_cam.reshape(3,1)

            corners_local = np.array([
                [ w/2,  h/2,  l/2], [ w/2,  h/2, -l/2],
                [ w/2, -h/2,  l/2], [ w/2, -h/2, -l/2],
                [-w/2,  h/2,  l/2], [-w/2,  h/2, -l/2],
                [-w/2, -h/2,  l/2], [-w/2, -h/2, -l/2]
            ], dtype=np.float32).T

            c, s = np.cos(ry), np.sin(ry)
            R_yaw = np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=np.float32)

            corners = R_yaw @ corners_local + center_upright

            pts = []
            for i in range(8):
                X, Y, Z = corners[:, i]
                r = np.sqrt(X**2 + Z**2 + 1e-8)
                uc = f * np.arctan2(X, Z) + u0
                vc = f * np.tan(np.arctan(Y / r)) + v0
                pts.append((int(uc), int(vc)))

            edges = [(0,1),(2,3),(4,5),(6,7),(1,3),(3,7),(7,5),(5,1),(0,2),(2,6),(6,4),(4,0)]
            for i,j in edges:
                cv2.line(img, pts[i], pts[j], (255,0,0), 2)

            # 取8个点的中心
            cx = int(np.mean([p[0] for p in pts]))
            cy = int(np.mean([p[1] for p in pts]))

            cv2.putText(
                img,
                f"{obj_id+1}",
                (cx + 5, cy - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

            obj_id += 1

    return img

# ====================== 单帧 ======================
def process_one(file_id):
    try:
        img_path   = os.path.join(img_dir,   f"{file_id}.jpg")
        calib_txt  = os.path.join(calib_dir, f"{file_id}.txt")
        label_path = os.path.join(label_dir, f"{file_id}.txt")

        if not (os.path.exists(img_path) and os.path.exists(calib_txt) and os.path.exists(label_path)):
            return

        save_img   = os.path.join(base_out, "images", f"{file_id}.jpg")
        save_vis   = os.path.join(base_out, "vis",    f"{file_id}.jpg")
        save_label = os.path.join(base_out, "labels", f"{file_id}.txt")
        save_calib = os.path.join(base_out, "calibs", f"{file_id}.txt")
        save_kitti = os.path.join(base_out, "gts_cyl", f"{file_id}.txt")

        for p in [save_img, save_vis, save_label, save_calib, save_kitti]:
            os.makedirs(os.path.dirname(p), exist_ok=True)

        # ===== 读取calib =====
        K = D = Tr = None
        with open(calib_txt) as f:
            for line in f:
                if line.startswith('P0:'):
                    K = np.array(line.split()[1:10], float).reshape(3,3)
                elif line.startswith('Tr_velo_to_cam:'):
                    vals = np.array(line.split()[1:], float)
                    Tr = np.vstack([vals.reshape(3,4), [0,0,0,1]])
                elif line.startswith('dist:'):
                    D = np.array(line.split()[1:], float)

        calib = {"intrinsic":{"f":K[0,0]}, "extrinsic":{"R":Tr[:3,:3]}}

        img = cv2.imread(img_path)

        # ===== 转柱面 =====
        cyl, R_final, K_cyl = fisheye_to_cylindrical(K, D, img, calib)

        # =====  保存calib（恢复你的逻辑）=====
        with open(calib_txt) as f_in, open(save_calib, 'w') as f_out:
            f_out.writelines(f_in.readlines())
            f_out.write("R_final: " + " ".join(f"{v:.12e}" for v in R_final.flatten()) + "\n")
            f_out.write("K_cyl: " + " ".join(f"{v:.12e}" for v in K_cyl.flatten()) + "\n")

        # ===== label =====
        with open(label_path) as f:
            lines = [l.strip() for l in f if l.strip()]

        with open(save_label, 'w') as f_out:
            for line in lines:
                parts = line.split()
                class_name = parts[0]

                h, w, l = map(float, parts[8:11])
                x, y, z = map(float, parts[11:14])

                uv = project_point_to_cylindrical(np.array([x,y,z]), R_final, K_cyl)
                if uv is None:
                    continue

                u,v = uv
                yaw = float(parts[3])
                rho = np.linalg.norm(R_final.T @ np.array([x,y,z]))

                f_out.write(f"{class_name} {u:.3f} {v:.3f} {h:.3f} {w:.3f} {l:.3f} {yaw:.3f} {rho:.3f}\n")

        # ===== 保存 KITTI GT =====
        save_kitti_gts_cyl(save_kitti, lines, R_final, K_cyl)


        # ===== 可视化 =====
        vis = draw_3d_boxes_on_cylindrical(cyl, save_label, R_final, K_cyl)

        vis_kitti = draw_kitti_on_cylindrical(cyl, save_kitti, R_final, K_cyl)
        cv2.imwrite(save_vis.replace(".jpg", "_kitti.jpg"), vis_kitti)
        
        cv2.imwrite(save_img, cyl)
        cv2.imwrite(save_vis, vis)

    except Exception as e:
        print("error:", file_id, e)


# ====================== 主程序 ======================
if __name__ == "__main__":

    train_ids_path = "/media/whut-user/350b5acb-7342-4f97-b1a7-57e9622bff8b/mono3d/脚本/柱面相关的/trainval.txt"

    img_dir   = "/media/whut-user/350b5acb-7342-4f97-b1a7-57e9622bff8b/mono3d/data_camera/demo_data/trainval/images"
    calib_dir = "/media/whut-user/350b5acb-7342-4f97-b1a7-57e9622bff8b/mono3d/data_camera/demo_data/trainval/calibs"
    label_dir = "/media/whut-user/350b5acb-7342-4f97-b1a7-57e9622bff8b/mono3d/data_camera/demo_data/trainval/labels"
    base_out  = "/media/whut-user/350b5acb-7342-4f97-b1a7-57e9622bff8b/mono3d/data_camera_cyl/demo_data/trainval"

    with open(train_ids_path) as f:
        ids = [l.strip() for l in f if l.strip()]

    print("总数:", len(ids))

    with Pool(40) as p:
        list(tqdm.tqdm(p.imap(process_one, ids), total=len(ids)))

    print(" 完成")