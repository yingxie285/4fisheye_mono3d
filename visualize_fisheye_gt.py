#!/usr/bin/env python
"""Visualize 3D GT boxes for a fisheye multi-camera scene."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np


IMAGE_DIR_TO_CAM: Dict[str, str] = {
    "image0": "front",
    "image1": "right",
    "image2": "left",
    "image3": "back",
}
CAM_TO_IMAGE_DIR: Dict[str, str] = {value: key for key, value in IMAGE_DIR_TO_CAM.items()}
CAMERA_INDEX_TO_NAME: Dict[int, str] = {
    0: "front",
    1: "right",
    2: "left",
    3: "back",
}

CANONICAL_CLASS_MAP: Dict[str, str] = {
    "普通轿车": "car",
    "敞篷轿车": "car",
    "SUV": "car",
    "MPV": "car",
    "面包车": "car",
    "皮卡": "car",
    "警车": "car",
    "救护车": "car",
    "房车": "truck",
    "普通小型货车": "truck",
    "箱式小型货车": "truck",
    "普通大型货车": "truck",
    "轿运车": "truck",
    "客车": "bus",
    "校车": "bus",
    "普通公交车": "bus",
    "铰链公交车": "bus",
    "有轨电车": "bus",
    "无轨电车": "bus",
    "消防车": "construction_vehicle",
    "清洁车": "construction_vehicle",
    "工程车": "construction_vehicle",
    "拖车": "construction_vehicle",
    "拖拉机": "construction_vehicle",
    "叉车": "construction_vehicle",
    "油罐车": "construction_vehicle",
    "成人": "pedestrian",
    "儿童": "pedestrian",
    "交警": "pedestrian",
    "环卫工人": "pedestrian",
    "道路施工人员": "pedestrian",
    "两轮摩托": "motor",
    "三轮摩托": "motor",
    "两轮电动车": "bicycle",
    "三轮电动车": "bicycle",
    "两轮自行车": "bicycle",
    "三轮自行车": "bicycle",
    "滑板车": "bicycle",
    "平衡车": "bicycle",
    "婴儿车": "bicycle",
    "轮椅": "bicycle",
    "平板小推车": "bicycle",
    "超市购物车": "bicycle",
    "手推车": "bicycle",
    "其他非机动车": "bicycle",
    "未知非机动车": "bicycle",
    "非机动车组": "bicycle",
    "小型动物": "animal",
    "大型动物": "animal",
    "锥桶": "traffic_cone",
    "防撞桶": "barrier",
    "路桩": "barrier",
    "石墩": "barrier",
    "水马": "barrier",
    "车位停止器": "stopper",
    "车位锁": "stopper",
    "减速带": "stopper",
    "垃圾桶": "trash_bin",
    "灭火器": "trash_bin",
    "箱子": "trash_bin",
    "A字牌": "sign",
    "三角牌": "sign",
    "施工警示牌": "sign",
    "道闸杆": "sign",
    "construcion_sign": "sign",
    "柱子": "barrier",
    "石块": "barrier",
    "树枝/树杈": "barrier",
    "空中漂浮物": "barrier",
    "路坑/水洼": "barrier",
    "其他静态障碍物": "barrier",
    "其他机动车": "car",
    "未知机动车": "car",
    "未知机动车车轮": "car",
    "未知机动车车灯": "car",
}


def canonicalize_class_name(class_name: str) -> str:
    key = str(class_name).strip()
    if key in CANONICAL_CLASS_MAP:
        return CANONICAL_CLASS_MAP[key]
    return key if key.isascii() else "unknown"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(data: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def frame_sort_key(frame_id: str) -> Tuple[int, int | str]:
    if frame_id.isdigit():
        return (0, int(frame_id))
    return (1, frame_id)


def find_single_file(directory: Path, stem: str) -> Optional[Path]:
    matches = [path for path in directory.glob(f"{stem}.*") if path.is_file()]
    if not matches:
        return None
    preferred = [".jpeg", ".jpg", ".png", ".json", ".npy", ".pcd"]
    for suffix in preferred:
        for path in matches:
            if path.suffix.lower() == suffix:
                return path
    return sorted(matches)[0]


def collect_frame_ids(scene_dir: Path) -> List[str]:
    result_dir = scene_dir / "result"
    frame_ids = sorted((path.stem for path in result_dir.glob("*.json")), key=frame_sort_key)
    valid_ids: List[str] = []
    for frame_id in frame_ids:
        if not (scene_dir / "camera_config" / f"{frame_id}.json").exists():
            continue
        image_paths = [find_single_file(scene_dir / image_dir, frame_id) for image_dir in IMAGE_DIR_TO_CAM]
        if any(path is None for path in image_paths):
            continue
        valid_ids.append(frame_id)
    return valid_ids


def is_scene_dir(path: Path) -> bool:
    required_dirs = ("camera_config", "result", *IMAGE_DIR_TO_CAM.keys())
    return path.is_dir() and all((path / folder).is_dir() for folder in required_dirs)


def collect_scene_dirs(scene_root: Path) -> List[Path]:
    if is_scene_dir(scene_root):
        return [scene_root]
    if not scene_root.is_dir():
        raise FileNotFoundError(f"Scene root does not exist: {scene_root}")
    scene_dirs = [path for path in scene_root.iterdir() if is_scene_dir(path)]
    return sorted(scene_dirs, key=lambda path: path.name)


def decode_camera_external(camera_cfg: Mapping[str, Any]) -> np.ndarray:
    matrix = np.asarray(camera_cfg["camera_external"], dtype=np.float64).reshape(4, 4)
    row_major = str(camera_cfg.get("rowMajor", "true")).lower() == "true"
    return matrix if row_major else matrix.T


def load_camera_params(camera_config_path: Path, image_paths: Mapping[str, Path]) -> Dict[str, Dict[str, Any]]:
    cfg_list = load_json(camera_config_path)
    cam_params: Dict[str, Dict[str, Any]] = {}
    for index, cam_cfg in enumerate(cfg_list):
        cam_name = CAMERA_INDEX_TO_NAME[index]
        camera_internal = cam_cfg["camera_internal"]
        k_matrix = np.array(
            [
                [camera_internal["fx"], 0.0, camera_internal["cx"]],
                [0.0, camera_internal["fy"], camera_internal["cy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        distortion = cam_cfg["distortion"]
        cam_params[cam_name] = {
            "K": k_matrix,
            "dist": [
                distortion["k1"],
                distortion["k2"],
                distortion["k3"],
                distortion["k4"],
            ],
            "lidar2cam": decode_camera_external(cam_cfg),
            "img_path": str(image_paths[cam_name]),
        }
    return cam_params


def load_annotation(anno_path: Path) -> List[Dict[str, Any]]:
    anno_list = load_json(anno_path)
    annotations: List[Dict[str, Any]] = []
    for annotation in anno_list.get("objects", []):
        contour = annotation.get("contour", {})
        size3d = contour.get("size3D", {})
        center3d = contour.get("center3D", {})
        rotation3d = contour.get("rotation3D", {})
        raw_class_name = str(annotation.get("className", "unknown"))
        annotations.append(
            {
                "class_name": raw_class_name,
                "display_name": canonicalize_class_name(raw_class_name),
                "lwh": [
                    float(size3d.get("x", 0.0)),
                    float(size3d.get("y", 0.0)),
                    float(size3d.get("z", 0.0)),
                ],
                "center": [
                    float(center3d.get("x", 0.0)),
                    float(center3d.get("y", 0.0)),
                    float(center3d.get("z", 0.0)),
                ],
                "yaw": float(rotation3d.get("z", 0.0)),
                "bbox_2d": annotation.get("2D_bbox", {}),
            }
        )
    return annotations


def get_3d_bbox_corners(center: Sequence[float], size: Sequence[float], yaw: float) -> np.ndarray:
    l, w, h = size
    x_c, y_c, z_c = center
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    z_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
    corners = np.vstack([x_corners, y_corners, z_corners])
    rotation = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    corners = rotation @ corners
    corners = corners + np.array(center).reshape(3, 1)
    return corners.T


def project_points_fisheye(
    points_3d: np.ndarray, K: np.ndarray, dist: Sequence[float], lidar2cam: np.ndarray
) -> np.ndarray:
    if points_3d.size == 0:
        return np.empty((0, 2), dtype=np.float64)

    points_3d_h = np.hstack(
        [points_3d, np.ones((points_3d.shape[0], 1), dtype=np.float64)]
    )
    cam_points = (lidar2cam @ points_3d_h.T).T[:, :3].astype(np.float64)
    
    # ===== 极坐标鱼眼投影：保留所有 8 个角点，不按 z>0 裁剪 =====
    K = np.array(K, dtype=np.float64)
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

    img_points = np.stack([u, v], axis=-1)
    return img_points.squeeze()


def draw_bbox(img: np.ndarray, corners_2d: np.ndarray, color: Tuple[int, int, int], thickness: int) -> np.ndarray:
    if corners_2d is None or len(corners_2d) == 0:
        return img

    corners_2d = corners_2d.astype(int)
    if corners_2d.ndim == 1 and corners_2d.shape[0] == 2:
        corners_2d = corners_2d[None, :]

    num_points = len(corners_2d)
    if num_points == 1:
        cv2.circle(img, tuple(corners_2d[0]), 2, color, thickness)
    elif num_points == 2:
        cv2.line(img, tuple(corners_2d[0]), tuple(corners_2d[1]), color, thickness)
    elif num_points == 4:
        for index in range(4):
            cv2.line(
                img,
                tuple(corners_2d[index]),
                tuple(corners_2d[(index + 1) % 4]),
                color,
                thickness,
            )
    elif num_points >= 8:
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        for i, j in edges:
            cv2.line(img, tuple(corners_2d[i]), tuple(corners_2d[j]), color, thickness)
    else:
        for index in range(num_points - 1):
            cv2.line(
                img,
                tuple(corners_2d[index]),
                tuple(corners_2d[index + 1]),
                color,
                thickness,
            )
    return img


def draw_2d_bbox(img: np.ndarray, box: Sequence[float], color: Tuple[int, int, int], thickness: int) -> np.ndarray:
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return img


def color_from_name(name: str) -> Tuple[int, int, int]:
    base = abs(hash(name))
    return (
        64 + (base % 160),
        64 + ((base // 13) % 160),
        64 + ((base // 29) % 160),
    )


def put_lines(
    img: np.ndarray,
    lines: Sequence[str],
    origin: Tuple[int, int],
    color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    x0, y0 = origin
    for index, line in enumerate(lines):
        y = y0 + index * 28
        cv2.putText(
            img,
            line,
            (x0, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            line,
            (x0, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
    return img


def make_mosaic(images: Mapping[str, np.ndarray]) -> np.ndarray:
    top = np.hstack([images["front"], images["right"]])
    bottom = np.hstack([images["left"], images["back"]])
    return np.vstack([top, bottom])


def build_image_paths(scene_dir: Path, frame_id: str) -> Dict[str, Path]:
    image_paths: Dict[str, Path] = {}
    for image_dir, cam_name in IMAGE_DIR_TO_CAM.items():
        image_path = find_single_file(scene_dir / image_dir, frame_id)
        if image_path is None:
            raise FileNotFoundError(f"Missing {image_dir} image for frame {frame_id}")
        image_paths[cam_name] = image_path
    return image_paths


def build_frame_info_lines(
    frame_id: str, manifest_item: Optional[Mapping[str, Any]], camera_name: str
) -> List[str]:
    lines = [f"{camera_name} | frame {frame_id}"]
    if manifest_item is None:
        lines.append("original frame")
    else:
        lines.append(
            f"aug src {manifest_item['source_frame_id']} | idx {manifest_item.get('augment_index', -1)}"
        )
        lines.append(
            f"flip {manifest_item.get('flip_horizontal', False)} | profile {manifest_item.get('profile', '')}"
        )
    return lines


def project_all_gt_to_cameras(
    cam_params: Mapping[str, Mapping[str, Any]],
    gt_labels: Sequence[Mapping[str, Any]],
    frame_output_dir: Path,
    frame_id: str,
    manifest_item: Optional[Mapping[str, Any]],
    draw_2d_boxes: bool,
) -> None:
    rendered_images: Dict[str, np.ndarray] = {}
    for cam_name, params in cam_params.items():
        img = cv2.imread(params["img_path"])
        if img is None:
            raise RuntimeError(f"Failed to read image: {params['img_path']}")
        image_key = CAM_TO_IMAGE_DIR[cam_name]

        for gt in gt_labels:
            bbox_2d = gt.get("bbox_2d", {})
            if not isinstance(bbox_2d, Mapping) or image_key not in bbox_2d:
                continue

            class_name = str(gt.get("display_name", gt.get("class_name", "unknown")))
            color = color_from_name(class_name)
            corners_3d = get_3d_bbox_corners(gt["center"], gt["lwh"], gt["yaw"])
            corners_2d = project_points_fisheye(
                corners_3d,
                params["K"],
                params["dist"],
                params["lidar2cam"],
            )
            if len(corners_2d) != 0:
                img = draw_bbox(img, corners_2d, color=color, thickness=2)
                if corners_2d.ndim == 1:
                    anchor = corners_2d
                else:
                    anchor = corners_2d[np.argmin(corners_2d[:, 1])]
                img = put_lines(
                    img,
                    [class_name],
                    (int(anchor[0]), max(30, int(anchor[1]) - 8)),
                    color=color,
                )

            if draw_2d_boxes:
                if image_key in bbox_2d and len(bbox_2d[image_key]) == 4:
                    img = draw_2d_bbox(img, bbox_2d[image_key], color=(255, 0, 0), thickness=2)

        img = put_lines(
            img,
            build_frame_info_lines(frame_id, manifest_item, cam_name),
            origin=(25, 35),
            color=(0, 255, 255),
        )

        frame_output_dir.mkdir(parents=True, exist_ok=True)
        cam_output_path = frame_output_dir / f"{cam_name}_vis.jpg"
        if not cv2.imwrite(str(cam_output_path), img):
            raise RuntimeError(f"Failed to write visualization: {cam_output_path}")
        rendered_images[cam_name] = img

    mosaic = make_mosaic(rendered_images)
    mosaic = put_lines(
        mosaic,
        [
            f"frame {frame_id}",
            "green: projected 3D bbox",
            "blue: stored 2D bbox",
        ],
        origin=(25, 35),
        color=(0, 255, 255),
    )
    mosaic_output_path = frame_output_dir / "all_cams.jpg"
    if not cv2.imwrite(str(mosaic_output_path), mosaic):
        raise RuntimeError(f"Failed to write mosaic: {mosaic_output_path}")


def load_manifest_lookup(scene_dir: Path) -> Dict[str, Dict[str, Any]]:
    manifest_path = scene_dir / "augmentation_manifest.json"
    if not manifest_path.exists():
        return {}
    manifest = load_json(manifest_path)
    return {
        str(item["new_frame_id"]): item
        for item in manifest.get("manifest", [])
        if "new_frame_id" in item
    }


def run_single_scene(
    scene_dir: Path,
    output_dir: Path,
    max_frames: Optional[int],
    draw_2d_boxes: bool,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_ids = collect_frame_ids(scene_dir)
    if max_frames is not None:
        frame_ids = frame_ids[: max(0, max_frames)]

    manifest_lookup = load_manifest_lookup(scene_dir)
    summary: List[Dict[str, Any]] = []

    for index, frame_id in enumerate(frame_ids, start=1):
        image_paths = build_image_paths(scene_dir, frame_id)
        cam_params = load_camera_params(
            scene_dir / "camera_config" / f"{frame_id}.json",
            image_paths=image_paths,
        )
        annotations = load_annotation(scene_dir / "result" / f"{frame_id}.json")
        frame_output_dir = output_dir / frame_id
        project_all_gt_to_cameras(
            cam_params=cam_params,
            gt_labels=annotations,
            frame_output_dir=frame_output_dir,
            frame_id=frame_id,
            manifest_item=manifest_lookup.get(frame_id),
            draw_2d_boxes=draw_2d_boxes,
        )
        summary.append(
            {
                "frame_id": frame_id,
                "is_augmented": frame_id in manifest_lookup,
                "object_count": len(annotations),
            }
        )
        if index % 10 == 0 or index == len(frame_ids):
            print(f"Visualized {index}/{len(frame_ids)} frames for scene {scene_dir.name}")

    save_json(
        {
            "scene_name": scene_dir.name,
            "scene_dir": str(scene_dir),
            "output_dir": str(output_dir),
            "frame_count": len(frame_ids),
            "augmented_frame_count": sum(1 for item in summary if item["is_augmented"]),
            "original_frame_count": sum(1 for item in summary if not item["is_augmented"]),
            "draw_2d_boxes": draw_2d_boxes,
            "frames": summary,
        },
        output_dir / "visualization_manifest.json",
    )
    print(f"Saved visualizations to: {output_dir}")
    return {
        "scene_name": scene_dir.name,
        "scene_dir": str(scene_dir),
        "output_dir": str(output_dir),
        "frame_count": len(frame_ids),
        "augmented_frame_count": sum(1 for item in summary if item["is_augmented"]),
        "original_frame_count": sum(1 for item in summary if not item["is_augmented"]),
    }


def run(
    scene_root: Path,
    output_root: Path,
    overwrite_output: bool,
    max_frames: Optional[int],
    draw_2d_boxes: bool,
) -> None:
    if output_root.exists():
        if not overwrite_output:
            raise FileExistsError(
                f"Output directory already exists: {output_root}. "
                "Please remove it or pass --overwrite-output."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    scene_dirs = collect_scene_dirs(scene_root)
    if not scene_dirs:
        raise FileNotFoundError(
            f"No valid scene directories were found under: {scene_root}"
        )

    summaries: List[Dict[str, Any]] = []
    total_frames = 0
    total_augmented_frames = 0

    for index, scene_dir in enumerate(scene_dirs, start=1):
        print(f"[{index}/{len(scene_dirs)}] Visualizing scene: {scene_dir.name}")
        scene_summary = run_single_scene(
            scene_dir=scene_dir,
            output_dir=output_root / scene_dir.name,
            max_frames=max_frames,
            draw_2d_boxes=draw_2d_boxes,
        )
        summaries.append(scene_summary)
        total_frames += int(scene_summary["frame_count"])
        total_augmented_frames += int(scene_summary["augmented_frame_count"])

    save_json(
        {
            "scene_root": str(scene_root),
            "output_root": str(output_root),
            "scene_count": len(summaries),
            "total_frame_count": total_frames,
            "total_augmented_frame_count": total_augmented_frames,
            "draw_2d_boxes": draw_2d_boxes,
            "scenes": summaries,
        },
        output_root / "dataset_visualization_summary.json",
    )
    print(f"Processed scenes: {len(summaries)}")
    print(f"Total frames visualized: {total_frames}")
    print(f"Dataset visualization root: {output_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-dir",
        type=Path,
        default=Path("fisheye_data_aug"),
        help="Input dataset root containing scene directories, or a single scene directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("vis"),
        help="Visualization root directory. Each scene keeps its original scene name under this directory.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Delete and recreate --output-dir if it already exists.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional debug limit.",
    )
    parser.add_argument(
        "--no-draw-2d-boxes",
        action="store_true",
        help="Do not overlay stored 2D boxes in blue.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        scene_root=args.scene_dir,
        output_root=args.output_dir,
        overwrite_output=args.overwrite_output,
        max_frames=args.max_frames,
        draw_2d_boxes=not args.no_draw_2d_boxes,
    )


if __name__ == "__main__":
    main()
