import cv2
import numpy as np
import os
from multiprocessing import Pool
import tqdm
from natsort import natsorted


CAMERA_NAMES = ["front", "right", "left", "back"]
DEFAULT_HFOV = np.deg2rad(190)
DEFAULT_VFOV = np.deg2rad(143)
TARGET_H = 960
TARGET_W = 1408
RDF_TO_FLU = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)


# ====================== utils ======================
def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


# ====================== fisheye -> cylindrical ======================
def build_view_info(f, R_raw, hfov=DEFAULT_HFOV, vfov=DEFAULT_VFOV, target_h=TARGET_H, target_w=TARGET_W):
    R = R_raw.copy() @ RDF_TO_FLU

    forward_norm = np.sqrt(R[0, 2] ** 2 + R[2, 2] ** 2)
    forward_norm = max(forward_norm, 1e-12)
    azimuth = np.arccos(np.clip(R[2, 2] / forward_norm, -1.0, 1.0))
    if R[0, 2] < 0:
        azimuth = 2 * np.pi - azimuth
    tilt = -np.arccos(np.clip(forward_norm, -1.0, 1.0))

    Ry = np.array(
        [
            [np.cos(azimuth), 0, np.sin(azimuth)],
            [0, 1, 0],
            [-np.sin(azimuth), 0, np.cos(azimuth)],
        ]
    ).T
    R_final = R @ Ry

    h = max(int(round(2 * f * np.tan(vfov / 2))), 1)
    w = max(int(round(f * hfov)), 1)

    K_cyl = np.array([[f, 0, w / 2], [0, f, f * np.tan(vfov / 2 + tilt)], [0, 0, 1]], dtype=np.float64)
    crop_top = int(np.clip(np.round(K_cyl[1, 2] - target_h / 2), 0, max(h - target_h, 0)))
    suggested_hfov = target_w / f
    suggested_vfov = 2 * (np.arctan((target_h / 2) / f) - tilt)
    suggested_vfov = np.clip(suggested_vfov, np.deg2rad(1), np.deg2rad(179))

    return {
        "azimuth": azimuth,
        "tilt": tilt,
        "raw_height": h,
        "raw_width": w,
        "R_final": R_final,
        "K_cyl": K_cyl,
        "crop_top": crop_top,
        "suggested_hfov": suggested_hfov,
        "suggested_vfov": suggested_vfov,
    }


def get_mapping(K, D, calib, hfov=DEFAULT_HFOV, vfov=DEFAULT_VFOV):
    f = calib["intrinsic"]["f"]
    view_info = build_view_info(f, calib["extrinsic"]["R"], hfov=hfov, vfov=vfov)
    R_final = view_info["R_final"]
    K_cyl = view_info["K_cyl"]
    h = view_info["raw_height"]
    w = view_info["raw_width"]

    K_cyl_inv = np.linalg.inv(K_cyl)
    xv, yv = np.meshgrid(range(w), range(h), indexing="xy")
    p = np.stack([xv, yv, np.ones_like(xv)], axis=-1).astype(np.float64)[..., np.newaxis]

    r = (K_cyl_inv @ p)[..., 0]
    r /= r[:, :, [2]]

    r_cart = np.zeros_like(r)
    r_cart[:, :, 2] = np.cos(r[:, :, 0])
    r_cart[:, :, 0] = np.sin(r[:, :, 0])
    r_cart[:, :, 1] = r[:, :, 1]

    r_cam = (R_final @ r_cart[..., np.newaxis])[..., 0]
    rays = r_cam.reshape(-1, 1, 3).astype(np.float64)

    img_points, _ = cv2.fisheye.projectPoints(
        rays,
        np.zeros((3, 1)),
        np.zeros((3, 1)),
        K.astype(np.float64),
        D.astype(np.float64),
    )
    img_points = img_points.reshape(h, w, 2)

    return img_points[..., 0].astype(np.float32), img_points[..., 1].astype(np.float32), R_final, K_cyl, view_info


def crop_or_pad_cylindrical(cyl, K_cyl, target_h=TARGET_H, target_w=TARGET_W):
    h, w = cyl.shape[:2]
    crop_top = 0
    crop_left = 0
    pad_top = 0
    pad_bottom = 0
    pad_left = 0
    pad_right = 0

    if h >= target_h:
        crop_top = int(np.clip(np.round(K_cyl[1, 2] - target_h / 2), 0, max(h - target_h, 0)))
        cyl = cyl[crop_top : crop_top + target_h, :]
        K_cyl[1, 2] -= crop_top
    else:
        pad_top = int(np.clip(np.round(target_h / 2 - K_cyl[1, 2]), 0, target_h - h))
        pad_bottom = target_h - h - pad_top
        cyl = cv2.copyMakeBorder(
            cyl,
            pad_top,
            pad_bottom,
            0,
            0,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        K_cyl[1, 2] += pad_top

    h, w = cyl.shape[:2]
    if w >= target_w:
        crop_left = int(np.clip(np.round(K_cyl[0, 2] - target_w / 2), 0, max(w - target_w, 0)))
        cyl = cyl[:, crop_left : crop_left + target_w]
        K_cyl[0, 2] -= crop_left
    else:
        pad_left = int(np.clip(np.round(target_w / 2 - K_cyl[0, 2]), 0, target_w - w))
        pad_right = target_w - w - pad_left
        cyl = cv2.copyMakeBorder(
            cyl,
            0,
            0,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        K_cyl[0, 2] += pad_left

    crop_info = {
        "crop_top": crop_top,
        "crop_left": crop_left,
        "pad_top": pad_top,
        "pad_bottom": pad_bottom,
        "pad_left": pad_left,
        "pad_right": pad_right,
    }
    return cyl, K_cyl, crop_info


def fisheye_to_cylindrical(K, D, image, calib, hfov=DEFAULT_HFOV, vfov=DEFAULT_VFOV, target_h=TARGET_H, target_w=TARGET_W):
    mapx, mapy, R_final, K_cyl, view_info = get_mapping(K, D, calib, hfov=hfov, vfov=vfov)
    cyl = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
    K_cyl = K_cyl.copy()
    cyl, K_cyl, crop_info = crop_or_pad_cylindrical(cyl, K_cyl, target_h=target_h, target_w=target_w)
    K_cyl = K_cyl.copy()
    view_info = dict(view_info)
    view_info.update(crop_info)
    view_info["crop_bottom"] = view_info["crop_top"] + target_h - view_info["pad_top"] - view_info["pad_bottom"]
    view_info["crop_right"] = view_info["crop_left"] + target_w - view_info["pad_left"] - view_info["pad_right"]

    return cyl, R_final, K_cyl, view_info


def project_point_to_cylindrical(P_cam, R_final, K_cyl):
    norm = np.linalg.norm(P_cam)
    if norm < 1e-6:
        return None

    dir_cam = P_cam / norm
    dir_cyl = R_final.T @ dir_cam

    x, y, z = dir_cyl
    theta = np.arctan2(x, z)
    rho = np.sqrt(x * x + z * z)
    y_norm = y / rho if rho > 1e-6 else 0.0

    f, cx, cy = K_cyl[0, 0], K_cyl[0, 2], K_cyl[1, 2]
    return f * theta + cx, f * y_norm + cy


# ====================== visualization ======================
def draw_3d_boxes_on_cylindrical(cyl_img, label_path, R_final, K_cyl):
    f = K_cyl[0, 0]
    u0 = K_cyl[0, 2]
    v0 = K_cyl[1, 2]

    img = cyl_img.copy()

    with open(label_path, "r") as fp:
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

            corners_local = np.array(
                [
                    [W / 2, H / 2, L / 2],
                    [W / 2, H / 2, -L / 2],
                    [W / 2, -H / 2, L / 2],
                    [W / 2, -H / 2, -L / 2],
                    [-W / 2, H / 2, L / 2],
                    [-W / 2, H / 2, -L / 2],
                    [-W / 2, -H / 2, L / 2],
                    [-W / 2, -H / 2, -L / 2],
                ],
                dtype=np.float32,
            ).T

            c, s = np.cos(global_yaw), np.sin(global_yaw)
            R_yaw = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)

            corners = R_yaw @ corners_local + center_upright

            pts = []
            for i in range(8):
                X, Y, Z = corners[:, i]
                r = np.sqrt(X ** 2 + Z ** 2 + 1e-8)
                uc = f * np.arctan2(X, Z) + u0
                vc = f * np.tan(np.arctan(Y / r)) + v0
                pts.append((int(round(uc)), int(round(vc))))

            edges = [(0, 1), (2, 3), (4, 5), (6, 7), (1, 3), (3, 7), (7, 5), (5, 1), (0, 2), (2, 6), (6, 4), (4, 0)]
            for i, j in edges:
                cv2.line(img, pts[i], pts[j], (0, 255, 0), 2)

            cv2.circle(img, (int(u), int(v)), 5, (0, 0, 255), -1)

    return img


def read_fisheye_calib(calib_txt):
    K = D = Tr = None
    with open(calib_txt) as f:
        for line in f:
            if line.startswith("P0:"):
                K = np.array(line.split()[1:10], float).reshape(3, 3)
            elif line.startswith("Tr_velo_to_cam:"):
                vals = np.array(line.split()[1:], float)
                Tr = np.vstack([vals.reshape(3, 4), [0, 0, 0, 1]])
            elif line.startswith("dist:"):
                D = np.array(line.split()[1:], float)
    return K, D, Tr


def print_camera_view_report(input_root, hfov=DEFAULT_HFOV, vfov=DEFAULT_VFOV, target_h=TARGET_H, target_w=TARGET_W):
    print("[camera_view_report] begin")
    scene_list = natsorted(os.listdir(input_root))
    for scene_name in scene_list:
        scene_dir = os.path.join(input_root, scene_name)
        if not os.path.isdir(scene_dir):
            continue

        for camera_name in CAMERA_NAMES:
            calib_dir = os.path.join(scene_dir, camera_name, "calibs")
            if not os.path.isdir(calib_dir):
                continue

            calib_files = [name for name in os.listdir(calib_dir) if name.lower().endswith(".txt")]
            if not calib_files:
                continue

            calib_path = os.path.join(calib_dir, natsorted(calib_files)[0])
            K, D, Tr = read_fisheye_calib(calib_path)
            if K is None or Tr is None:
                continue

            view_info = build_view_info(K[0, 0], Tr[:3, :3], hfov=hfov, vfov=vfov, target_h=target_h, target_w=target_w)
            print(
                "[camera_view] "
                f"scene={scene_name} camera={camera_name} "
                f"azimuth_deg={np.rad2deg(view_info['azimuth']):.3f} "
                f"tilt_deg={np.rad2deg(view_info['tilt']):.3f} "
                f"hfov_deg={np.rad2deg(hfov):.3f} "
                f"vfov_deg={np.rad2deg(vfov):.3f} "
                f"raw_size={view_info['raw_width']}x{view_info['raw_height']} "
                f"K_cyl_cy={view_info['K_cyl'][1, 2]:.3f} "
                f"crop_top={view_info['crop_top']} "
                f"hfov_deg_for_target_w={np.rad2deg(view_info['suggested_hfov']):.3f} "
                f"vfov_deg_for_target_h={np.rad2deg(view_info['suggested_vfov']):.3f}"
            )
    print("[camera_view_report] end")


def convert_one(job):
    scene_name, camera_name, file_id, img_path, calib_txt, label_path, out_camera_dir = job
    try:
        if not (os.path.exists(img_path) and os.path.exists(calib_txt) and os.path.exists(label_path)):
            return

        save_img = os.path.join(out_camera_dir, "images", f"{file_id}.jpg")
        save_vis = os.path.join(out_camera_dir, "vis", f"{file_id}.jpg")
        save_label = os.path.join(out_camera_dir, "labels", f"{file_id}.txt")
        save_calib = os.path.join(out_camera_dir, "calibs", f"{file_id}.txt")

        for path in [save_img, save_vis, save_label, save_calib]:
            os.makedirs(os.path.dirname(path), exist_ok=True)

        K, D, Tr = read_fisheye_calib(calib_txt)
        if K is None or D is None or Tr is None:
            print("skip invalid calib:", scene_name, camera_name, file_id)
            return

        calib = {"intrinsic": {"f": K[0, 0]}, "extrinsic": {"R": Tr[:3, :3]}}
        img = cv2.imread(img_path)
        if img is None:
            print("skip invalid image:", scene_name, camera_name, file_id)
            return

        cyl, R_final, K_cyl, view_info = fisheye_to_cylindrical(K, D, img, calib)

        with open(calib_txt) as f_in, open(save_calib, "w") as f_out:
            f_out.writelines(f_in.readlines())
            f_out.write("R_final: " + " ".join(f"{v:.12e}" for v in R_final.flatten()) + "\n")
            f_out.write("K_cyl: " + " ".join(f"{v:.12e}" for v in K_cyl.flatten()) + "\n")
            f_out.write(f"cyl_azimuth_deg: {np.rad2deg(view_info['azimuth']):.12e}\n")
            f_out.write(f"cyl_tilt_deg: {np.rad2deg(view_info['tilt']):.12e}\n")
            f_out.write(f"cyl_crop_top: {int(view_info['crop_top'])}\n")
            f_out.write(f"cyl_crop_left: {int(view_info['crop_left'])}\n")

        with open(label_path) as f:
            lines = [line.strip() for line in f if line.strip()]

        with open(save_label, "w") as f_out:
            for line in lines:
                parts = line.split()
                class_name = parts[0]

                h, w, l = map(float, parts[8:11])
                x, y, z = map(float, parts[11:14])

                uv = project_point_to_cylindrical(np.array([x, y, z]), R_final, K_cyl)
                if uv is None:
                    continue

                u, v = uv
                yaw = float(parts[3])
                rho = np.linalg.norm(R_final.T @ np.array([x, y, z]))

                f_out.write(f"{class_name} {u:.3f} {v:.3f} {h:.3f} {w:.3f} {l:.3f} {yaw:.3f} {rho:.3f}\n")

        vis = draw_3d_boxes_on_cylindrical(cyl, save_label, R_final, K_cyl)
        cv2.imwrite(save_img, cyl)
        cv2.imwrite(save_vis, vis)

    except Exception as e:
        print("error:", scene_name, camera_name, file_id, e)


def collect_jobs(input_root, output_root):
    jobs = []
    scene_list = natsorted(os.listdir(input_root))
    for scene_name in scene_list:
        scene_dir = os.path.join(input_root, scene_name)
        if not os.path.isdir(scene_dir):
            continue

        for camera_name in CAMERA_NAMES:
            camera_dir = os.path.join(scene_dir, camera_name)
            img_dir = os.path.join(camera_dir, "images")
            calib_dir = os.path.join(camera_dir, "calibs")
            label_dir = os.path.join(camera_dir, "labels")
            if not all(os.path.isdir(path) for path in [img_dir, calib_dir, label_dir]):
                continue

            img_ids = {os.path.splitext(name)[0] for name in os.listdir(img_dir)}
            calib_ids = {os.path.splitext(name)[0] for name in os.listdir(calib_dir)}
            label_ids = {os.path.splitext(name)[0] for name in os.listdir(label_dir)}
            common_ids = natsorted(list(img_ids & calib_ids & label_ids))

            out_camera_dir = os.path.join(output_root, scene_name, camera_name)
            for file_id in common_ids:
                jobs.append(
                    (
                        scene_name,
                        camera_name,
                        file_id,
                        os.path.join(img_dir, f"{file_id}.jpg"),
                        os.path.join(calib_dir, f"{file_id}.txt"),
                        os.path.join(label_dir, f"{file_id}.txt"),
                        out_camera_dir,
                    )
                )

    return jobs


# ====================== main ======================
if __name__ == "__main__":
    input_root = "C:/Users/yingxie/Desktop/mono3d-main/data_camera/demo_data/trainval_gt_vis"
    output_root = "C:/Users/yingxie/Desktop/mono3d-main/data_camera_cyl/demo_data/trainval_gt_vis"

    print_camera_view_report(input_root)
    jobs = collect_jobs(input_root, output_root)
    print("total:", len(jobs))

    try:
        with Pool(8) as p:
            list(tqdm.tqdm(p.imap(convert_one, jobs), total=len(jobs)))
    except (PermissionError, OSError) as exc:
        print("pool failed, fallback to serial:", exc)
        for job in tqdm.tqdm(jobs):
            convert_one(job)

    print("done")
