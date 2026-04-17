from __future__ import absolute_import, division, print_function

import argparse
import json
import math
import os
import shutil
from multiprocessing import Pool

import cv2
import numpy as np
import tqdm


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_INPUT_ROOT = os.path.join(REPO_ROOT, "data_camera_cyl", "demo_data", "trainval_gt_vis")
DEFAULT_OUTPUT_ROOT = os.path.join(REPO_ROOT, "data_camera_cyl", "demo_data", "trainval")
DEFAULT_SCENE_SPLIT_PATH = os.path.join(REPO_ROOT, "脚本", "train_val_split.json")
CAMERA_NAMES = ("front", "right", "left", "back")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _bbox_to_coco_bbox(bbox):
    return [bbox[0], bbox[1], max(0.0, bbox[2] - bbox[0]), max(0.0, bbox[3] - bbox[1])]


def sample_sort_key(sample):
    return sample["scene_name"], sample["camera_name"], sample["frame_id"]


def make_unique_stem(scene_name, camera_name, frame_id):
    return "{}__{}__{}".format(scene_name, camera_name, frame_id)


def find_image_path(image_dir, frame_id):
    for ext in IMAGE_EXTENSIONS:
        candidate = os.path.join(image_dir, frame_id + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def parse_calib_file(calib_path):
    calib = {}
    with open(calib_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or ":" not in line:
                continue

            key, value = line.split(":", 1)
            values = np.fromstring(value.strip(), sep=" ", dtype=np.float32)

            if key == "P0":
                if values.size == 9:
                    calib["P0"] = np.concatenate(
                        [values.reshape(3, 3), np.zeros((3, 1), dtype=np.float32)],
                        axis=1,
                    )
                elif values.size == 12:
                    calib["P0"] = values.reshape(3, 4)
            elif key == "K_cyl" and values.size == 9:
                calib["K_cyl"] = values.reshape(3, 3)
            elif key == "R_final" and values.size == 9:
                calib["R_final"] = values.reshape(3, 3)
            elif key == "Tr_velo_to_cam" and values.size == 12:
                calib["Tr_velo_to_cam"] = values.reshape(3, 4)

    if "P0" not in calib and "K_cyl" in calib:
        calib["P0"] = np.concatenate(
            [calib["K_cyl"], np.zeros((3, 1), dtype=np.float32)],
            axis=1,
        )
    if "K_cyl" not in calib and "P0" in calib:
        calib["K_cyl"] = calib["P0"][:, :3]

    missing = [key for key in ("P0", "K_cyl", "R_final") if key not in calib]
    if missing:
        raise ValueError("Missing {} in {}".format(",".join(missing), calib_path))

    return calib


def load_scene_split(scene_split_path):
    if not os.path.exists(scene_split_path):
        return set(), set()

    with open(scene_split_path, "r", encoding="utf-8") as f:
        scene_split = json.load(f)

    return set(scene_split.get("train", [])), set(scene_split.get("val", []))


def split_by_ratio(samples, val_ratio):
    samples = sorted(samples, key=sample_sort_key)
    if len(samples) <= 1:
        return samples, []

    val_count = int(round(len(samples) * val_ratio))
    val_count = min(max(val_count, 1), len(samples) - 1)
    split_index = len(samples) - val_count
    return samples[:split_index], samples[split_index:]


def split_samples(samples, scene_split_path, val_ratio):
    train_scenes, val_scenes = load_scene_split(scene_split_path)
    train_samples = []
    val_samples = []
    unknown_samples = []
    has_scene_split = bool(train_scenes or val_scenes)

    overlap_scenes = train_scenes & val_scenes
    if overlap_scenes:
        raise ValueError(
            "Scene split file contains scenes in both train and val: {}".format(
                ",".join(sorted(overlap_scenes))
            )
        )

    for sample in samples:
        scene_name = sample["scene_name"]
        if scene_name in train_scenes:
            train_samples.append(sample)
        elif scene_name in val_scenes:
            val_samples.append(sample)
        else:
            unknown_samples.append(sample)

    if has_scene_split:
        if unknown_samples:
            unknown_scene_names = sorted({sample["scene_name"] for sample in unknown_samples})
            raise ValueError(
                "Scene split file does not cover all scenes. Missing: {}".format(
                    ",".join(unknown_scene_names)
                )
            )
        return (
            sorted(train_samples, key=sample_sort_key),
            sorted(val_samples, key=sample_sort_key),
            "scene_split",
        )

    fallback_train, fallback_val = split_by_ratio(samples, val_ratio)
    return fallback_train, fallback_val, "ratio_fallback"


def safe_link_or_copy(src_path, dst_path, use_hardlink):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    if os.path.exists(dst_path):
        os.remove(dst_path)

    if use_hardlink:
        try:
            os.link(src_path, dst_path)
            return
        except OSError:
            pass

    shutil.copy2(src_path, dst_path)


def collect_samples(input_root):
    samples = []

    if not os.path.isdir(input_root):
        raise FileNotFoundError("Input root does not exist: {}".format(input_root))

    for scene_name in sorted(os.listdir(input_root)):
        scene_dir = os.path.join(input_root, scene_name)
        if not os.path.isdir(scene_dir):
            continue

        for camera_name in CAMERA_NAMES:
            camera_dir = os.path.join(scene_dir, camera_name)
            image_dir = os.path.join(camera_dir, "images")
            label_dir = os.path.join(camera_dir, "labels")
            calib_dir = os.path.join(camera_dir, "calibs")

            if not all(os.path.isdir(path) for path in (image_dir, label_dir, calib_dir)):
                continue

            image_ids = {
                os.path.splitext(name)[0]
                for name in os.listdir(image_dir)
                if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
            }
            label_ids = {
                os.path.splitext(name)[0]
                for name in os.listdir(label_dir)
                if name.lower().endswith(".txt")
            }
            calib_ids = {
                os.path.splitext(name)[0]
                for name in os.listdir(calib_dir)
                if name.lower().endswith(".txt")
            }

            for frame_id in sorted(image_ids & label_ids & calib_ids):
                image_path = find_image_path(image_dir, frame_id)
                if image_path is None:
                    continue

                image_ext = os.path.splitext(image_path)[1].lower()
                samples.append(
                    {
                        "scene_name": scene_name,
                        "camera_name": camera_name,
                        "frame_id": frame_id,
                        "stem": make_unique_stem(scene_name, camera_name, frame_id),
                        "image_path": image_path,
                        "image_ext": image_ext,
                        "label_path": os.path.join(label_dir, frame_id + ".txt"),
                        "calib_path": os.path.join(calib_dir, frame_id + ".txt"),
                    }
                )

    return sorted(samples, key=sample_sort_key)


def build_keypoints(points, width, height):
    keypoints = []
    for x, y in points:
        visible = 2 if 0 <= x < width and 0 <= y < height else 1
        keypoints.extend([float(x), float(y), visible])
    return keypoints


def process_one(args):
    sample, image_id, output_root, save_vis, use_hardlink = args

    image = cv2.imread(sample["image_path"])
    if image is None:
        raise ValueError("Failed to read image: {}".format(sample["image_path"]))

    height, width = image.shape[:2]
    calib = parse_calib_file(sample["calib_path"])
    p_cyl = calib["P0"].astype(np.float32)
    k_cyl = calib["K_cyl"].astype(np.float32)
    r_final = calib["R_final"].astype(np.float32)

    f = float(k_cyl[0, 0])
    u0 = float(k_cyl[0, 2])
    v0 = float(k_cyl[1, 2])

    image_file_name = sample["stem"] + sample["image_ext"]
    output_image_path = os.path.join(output_root, "images", image_file_name)
    output_calib_path = os.path.join(output_root, "calibs", sample["stem"] + ".txt")

    safe_link_or_copy(sample["image_path"], output_image_path, use_hardlink)
    safe_link_or_copy(sample["calib_path"], output_calib_path, use_hardlink)

    image_info = {
        "file_name": image_file_name,
        "id": int(image_id),
        "width": int(width),
        "height": int(height),
        "calib": p_cyl.tolist(),
        "scene_name": sample["scene_name"],
        "camera_name": sample["camera_name"],
        "frame_id": sample["frame_id"],
    }

    annotations = []
    vis_image = image.copy() if save_vis else None

    with open(sample["label_path"], "r", encoding="utf-8") as anns:
        for raw_line in anns:
            parts = raw_line.strip().split()
            if len(parts) < 8:
                continue

            cat = parts[0].lower()
            if cat not in CAT_IDS:
                continue

            u, v = map(float, parts[1:3])
            obj_h, obj_w, obj_l = map(float, parts[3:6])
            yaw_local = float(parts[6])
            rho = float(parts[7])

            phi = (u - u0) / f
            tan_psi = (v - v0) / f

            dir_cyl = np.array([math.sin(phi), tan_psi, math.cos(phi)], dtype=np.float32)
            dir_norm = float(np.linalg.norm(dir_cyl))
            if dir_norm < 1e-6:
                continue
            dir_cyl /= dir_norm

            dir_cam = r_final @ dir_cyl
            center_cam = (dir_cam * rho).reshape(3, 1)
            center_upright = r_final.T @ center_cam

            corners_local = np.array(
                [
                    [obj_w / 2, obj_h / 2, obj_l / 2],
                    [obj_w / 2, obj_h / 2, -obj_l / 2],
                    [obj_w / 2, -obj_h / 2, obj_l / 2],
                    [obj_w / 2, -obj_h / 2, -obj_l / 2],
                    [-obj_w / 2, obj_h / 2, obj_l / 2],
                    [-obj_w / 2, obj_h / 2, -obj_l / 2],
                    [-obj_w / 2, -obj_h / 2, obj_l / 2],
                    [-obj_w / 2, -obj_h / 2, -obj_l / 2],
                ],
                dtype=np.float32,
            ).T

            yaw_global = wrap(yaw_local + phi)
            cos_yaw = math.cos(yaw_global)
            sin_yaw = math.sin(yaw_global)
            r_yaw = np.array(
                [[cos_yaw, 0.0, sin_yaw], [0.0, 1.0, 0.0], [-sin_yaw, 0.0, cos_yaw]],
                dtype=np.float32,
            )
            corners = r_yaw @ corners_local + center_upright

            projected_points = []
            for corner_idx in range(8):
                x_val, y_val, z_val = corners[:, corner_idx]
                radial = math.sqrt(float(x_val * x_val + z_val * z_val) + 1e-8)
                phi_proj = math.atan2(float(x_val), float(z_val))
                psi_proj = math.atan(float(y_val / radial))

                projected_u = f * phi_proj + u0
                projected_v = f * math.tan(psi_proj) + v0
                projected_points.append((projected_u, projected_v))

            xs = [point[0] for point in projected_points]
            ys = [point[1] for point in projected_points]
            x1 = max(min(xs), 0.0)
            y1 = max(min(ys), 0.0)
            x2 = min(max(xs), width - 1.0)
            y2 = min(max(ys), height - 1.0)

            if x2 <= x1 or y2 <= y1:
                continue

            keypoint_points = list(projected_points) + [(u, v)]
            keypoints = build_keypoints(keypoint_points, width, height)
            bbox = [float(x1), float(y1), float(x2), float(y2)]

            ann = {
                "segmentation": [[0, 0, 0, 0, 0, 0]],
                "num_keypoints": 9,
                "area": float((x2 - x1) * (y2 - y1)),
                "iscrowd": 0,
                "keypoints": keypoints,
                "image_id": int(image_id),
                "bbox": _bbox_to_coco_bbox(bbox),
                "uv": [float(u), float(v)],
                "category_id": CAT_IDS[cat],
                "dim": [float(obj_h), float(obj_w), float(obj_l)],
                "alpha": float(yaw_local),
                "depth": float(rho),
                "scene_name": sample["scene_name"],
                "camera_name": sample["camera_name"],
                "frame_id": sample["frame_id"],
            }
            annotations.append(ann)

            if vis_image is not None:
                draw_points = [(int(round(x)), int(round(y))) for x, y in projected_points]
                edges = [
                    (0, 1),
                    (2, 3),
                    (4, 5),
                    (6, 7),
                    (1, 3),
                    (3, 7),
                    (7, 5),
                    (5, 1),
                    (0, 2),
                    (2, 6),
                    (6, 4),
                    (4, 0),
                ]
                for start_idx, end_idx in edges:
                    cv2.line(vis_image, draw_points[start_idx], draw_points[end_idx], (0, 255, 0), 2)
                cv2.rectangle(
                    vis_image,
                    (int(round(x1)), int(round(y1))),
                    (int(round(x2)), int(round(y2))),
                    (0, 0, 255),
                    2,
                )
                cv2.circle(vis_image, (int(round(u)), int(round(v))), 4, (255, 0, 0), -1)

    if vis_image is not None:
        vis_path = os.path.join(output_root, "vis_3d_2d", sample["stem"] + ".jpg")
        os.makedirs(os.path.dirname(vis_path), exist_ok=True)
        cv2.imwrite(vis_path, vis_image)

    return sample["stem"], image_info, annotations


def write_annotations_json(out_path, records):
    images = []
    annotations = []
    ann_id = 1

    for _, image_info, image_annotations in records:
        images.append(image_info)
        for ann in image_annotations:
            ann["id"] = ann_id
            ann_id += 1
            annotations.append(ann)

    payload = {
        "images": images,
        "annotations": annotations,
        "categories": CAT_INFO,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def write_id_txt(out_path, samples):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(sample["stem"] + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert 4-camera cylindrical labels to model-ready ann json.")
    parser.add_argument("--input_root", default=DEFAULT_INPUT_ROOT, help="Input root like data_camera_cyl/demo_data/trainval_gt_vis")
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT, help="Output root like data_camera_cyl/demo_data/trainval")
    parser.add_argument("--scene_split", default=DEFAULT_SCENE_SPLIT_PATH, help="Scene-level train/val split json")
    parser.add_argument("--workers", type=int, default=40, help="Number of worker processes")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Fallback val ratio when scene split is unavailable or empty")
    parser.add_argument("--no_vis", action="store_true", help="Disable vis_3d_2d output")
    parser.add_argument("--no_hardlink", action="store_true", help="Copy files instead of trying hard links first")
    return parser.parse_args()


CATS = [
    "car",
    "truck",
    "bus",
    "construction_vehicle",
    "pedestrian",
    "motor",
    "bicycle",
    "animal",
    "traffic_cone",
    "barrier",
    "stopper",
    "trash_bin",
    "sign",
]
CAT_IDS = {cat: idx + 1 for idx, cat in enumerate(CATS)}
CAT_INFO = [{"name": cat, "id": idx + 1} for idx, cat in enumerate(CATS)]


def main():
    args = parse_args()
    output_root = os.path.abspath(args.output_root)
    input_root = os.path.abspath(args.input_root)
    scene_split_path = os.path.abspath(args.scene_split)
    save_vis = not args.no_vis
    use_hardlink = not args.no_hardlink

    samples = collect_samples(input_root)
    if not samples:
        raise RuntimeError("No valid samples found under {}".format(input_root))

    train_samples, val_samples, split_mode = split_samples(samples, scene_split_path, args.val_ratio)

    tasks = []
    for image_id, sample in enumerate(samples, start=1):
        tasks.append((sample, image_id, output_root, save_vis, use_hardlink))

    os.makedirs(os.path.join(output_root, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_root, "calibs"), exist_ok=True)
    os.makedirs(os.path.join(output_root, "annotations"), exist_ok=True)
    if save_vis:
        os.makedirs(os.path.join(output_root, "vis_3d_2d"), exist_ok=True)

    worker_count = max(1, min(args.workers, len(tasks), os.cpu_count() or 1))
    processed = {}

    if worker_count == 1:
        iterator = map(process_one, tasks)
        for stem, image_info, annotations in tqdm.tqdm(iterator, total=len(tasks)):
            processed[stem] = (stem, image_info, annotations)
    else:
        with Pool(worker_count) as pool:
            iterator = pool.imap_unordered(process_one, tasks)
            for stem, image_info, annotations in tqdm.tqdm(iterator, total=len(tasks)):
                processed[stem] = (stem, image_info, annotations)

    ordered_train_records = [processed[sample["stem"]] for sample in train_samples if sample["stem"] in processed]
    ordered_val_records = [processed[sample["stem"]] for sample in val_samples if sample["stem"] in processed]

    write_annotations_json(os.path.join(output_root, "annotations", "ip42_train_all.json"), ordered_train_records)
    write_annotations_json(os.path.join(output_root, "annotations", "ip42_val_all.json"), ordered_val_records)
    write_id_txt(os.path.join(output_root, "train.txt"), train_samples)
    write_id_txt(os.path.join(output_root, "val.txt"), val_samples)

    print("Input root:", input_root)
    print("Output root:", output_root)
    print("Split mode:", split_mode)
    print("Total samples:", len(samples))
    print("Train samples:", len(train_samples))
    print("Val samples:", len(val_samples))
    print("Workers:", worker_count)


if __name__ == "__main__":
    main()
