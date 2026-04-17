#!/usr/bin/env python
"""Offline augmentation for the demo fisheye multi-camera 3D detection scene.

This script keeps the dataset layout unchanged and writes augmented samples to a
new scene directory. The geometric mirror follows the LiDAR-box convention used
by MMDetection3D's LiDAR flip logic, but is implemented here as plain offline
file processing instead of framework runtime transforms.

Supported augmentations:
1. LiDAR-consistent horizontal mirror
   - bbox center: y -> -y
   - bbox yaw: z -> wrap_to_pi(-z)
   - 2D bbox: horizontal flip inside each image
   - point cloud: y -> -y
   - camera config: mirrored so projection stays synchronized with flipped image
2. Conservative photometric augmentation for all 4 synchronized cameras
   - brightness / contrast / gamma
   - white balance / saturation
   - light shadow / light fog
   - mild JPEG / blur / noise

Example:
    python offline_fisheye_lidar_aug.py ^
        --scene-dir fisheye_data_demo\\20231214-163109_20231214-163126 ^
        --output-dir fisheye_data_demo\\20231214-163109_20231214-163126_offline_aug
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import cv2
import numpy as np


IMAGE_DIRS: Tuple[str, ...] = ("image0", "image1", "image2", "image3")
CAMERA_INDEX_TO_NAME: Dict[int, str] = {
    0: "front",
    1: "right",
    2: "left",
    3: "back",
}

LIDAR_MIRROR = np.diag([1.0, -1.0, 1.0, 1.0])
CAMERA_X_MIRROR = np.diag([-1.0, 1.0, 1.0, 1.0])

RARE_CLASSES = frozenset(
    {"motor", "animal", "bus", "construction_vehicle", "traffic_cone"}
)
TAIL_CLASSES = frozenset(
    {"pedestrian", "truck", "trash_bin", "sign", "barrier", "stopper"}
)
VERY_LIGHT_PHOTO_CLASSES = frozenset({"bus", "animal", "construction_vehicle"})
LIGHT_PHOTO_CLASSES = frozenset({"pedestrian", "bicycle", "motor", "traffic_cone"})

PHOTO_PROFILES: Dict[str, Dict[str, Any]] = {
    "very_light": {
        "brightness_px": 8.0,
        "contrast_delta": 0.06,
        "gamma_delta": 0.05,
        "gamma_prob": 0.45,
        "wb_delta": 0.04,
        "wb_prob": 0.40,
        "saturation_delta": 0.06,
        "saturation_prob": 0.35,
        "shadow_prob": 0.15,
        "shadow_strength": (0.88, 0.96),
        "fog_prob": 0.10,
        "fog_strength": (0.03, 0.08),
        "blur_prob": 0.12,
        "blur_kernels": (3,),
        "noise_prob": 0.10,
        "noise_sigma": (1.0, 3.0),
        "jpeg_prob": 0.25,
        "jpeg_quality": (90, 98),
    },
    "light": {
        "brightness_px": 12.0,
        "contrast_delta": 0.10,
        "gamma_delta": 0.08,
        "gamma_prob": 0.55,
        "wb_delta": 0.06,
        "wb_prob": 0.45,
        "saturation_delta": 0.08,
        "saturation_prob": 0.40,
        "shadow_prob": 0.20,
        "shadow_strength": (0.84, 0.94),
        "fog_prob": 0.12,
        "fog_strength": (0.04, 0.10),
        "blur_prob": 0.16,
        "blur_kernels": (3,),
        "noise_prob": 0.15,
        "noise_sigma": (1.0, 4.0),
        "jpeg_prob": 0.30,
        "jpeg_quality": (88, 96),
    },
    "medium": {
        "brightness_px": 18.0,
        "contrast_delta": 0.15,
        "gamma_delta": 0.12,
        "gamma_prob": 0.60,
        "wb_delta": 0.08,
        "wb_prob": 0.50,
        "saturation_delta": 0.10,
        "saturation_prob": 0.45,
        "shadow_prob": 0.25,
        "shadow_strength": (0.78, 0.92),
        "fog_prob": 0.15,
        "fog_strength": (0.05, 0.12),
        "blur_prob": 0.20,
        "blur_kernels": (3, 5),
        "noise_prob": 0.18,
        "noise_sigma": (1.0, 5.0),
        "jpeg_prob": 0.35,
        "jpeg_quality": (85, 95),
    },
}

CANONICAL_CLASS_MAP: Dict[str, str] = {
    '普通轿车': 'car','敞篷轿车': 'car','SUV': 'car','MPV': 'car','面包车': 'car',
    '皮卡': 'car','警车': 'car','救护车': 'car',

    '房车': 'truck','普通小型货车': 'truck','箱式小型货车': 'truck',
    '普通大型货车': 'truck','轿运车': 'truck',

    '客车': 'bus','校车': 'bus','普通公交车': 'bus','铰链公交车': 'bus',
    '有轨电车': 'bus','无轨电车': 'bus',

    '消防车': 'construction_vehicle','清洁车': 'construction_vehicle',
    '工程车': 'construction_vehicle','拖车': 'construction_vehicle',
    '拖拉机': 'construction_vehicle','叉车': 'construction_vehicle',
    '油罐车': 'construction_vehicle',

    '成人': 'pedestrian','儿童': 'pedestrian','交警': 'pedestrian',
    '环卫工人': 'pedestrian','道路施工人员': 'pedestrian',

    '两轮摩托': 'motor','三轮摩托': 'motor',

    '两轮电动车': 'bicycle','三轮电动车': 'bicycle','两轮自行车': 'bicycle',
    '三轮自行车': 'bicycle','滑板车': 'bicycle','平衡车': 'bicycle',
    '婴儿车': 'bicycle','轮椅': 'bicycle','平板小推车': 'bicycle',
    '超市购物车': 'bicycle','手推车': 'bicycle',
    '其他非机动车': 'bicycle','未知非机动车': 'bicycle','非机动车组': 'bicycle',

    '小型动物': 'animal','大型动物': 'animal',

    '锥桶': 'traffic_cone',

    '防撞桶': 'barrier','路桩': 'barrier','石墩': 'barrier',
    '水马': 'barrier',

    '车位停止器': 'stopper','车位锁': 'stopper','减速带': 'stopper',

    '垃圾桶': 'trash_bin','灭火器': 'trash_bin','箱子': 'trash_bin',

    'A字牌': 'sign','三角牌': 'sign','施工警示牌': 'sign',
    '道闸杆': 'sign','construcion_sign': 'sign',

    '柱子': 'barrier','石块': 'barrier','树枝/树杈': 'barrier',
    '空中漂浮物': 'barrier','路坑/水洼': 'barrier',
    '其他静态障碍物': 'barrier',

    '其他机动车': 'car','未知机动车': 'car',
    '未知机动车车轮': 'car','未知机动车车灯': 'car',
}


@dataclass(frozen=True)
class FrameDecision:
    augment_times: int
    trigger: str
    profile: str
    canonical_classes: Tuple[str, ...]


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def canonicalize_class(class_name: str) -> str:
    key = str(class_name).strip()
    return CANONICAL_CLASS_MAP.get(key, key if key in CANONICAL_CLASS_MAP else "unknown")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(data: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def stable_rng(seed: int, *parts: str) -> np.random.Generator:
    digest = hashlib.sha1("|".join([str(seed), *parts]).encode("utf-8")).digest()
    child_seed = int.from_bytes(digest[:8], "little", signed=False)
    return np.random.default_rng(child_seed)


def ensure_output_root(
    scene_dir: Path, output_dir: Path, copy_originals: bool, overwrite_output: bool
) -> None:
    if output_dir.exists():
        if not overwrite_output:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                "Please remove it, pass --overwrite-output, or choose another --output-dir."
            )
        shutil.rmtree(output_dir)
    if copy_originals:
        try:
            output_dir.relative_to(scene_dir)
        except ValueError:
            return
        raise ValueError(
            "When --copy-originals is enabled, --output-dir must not be inside --scene-dir."
        )


def find_single_file(directory: Path, stem: str) -> Optional[Path]:
    matches = [p for p in directory.glob(f"{stem}.*") if p.is_file()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        preferred_suffixes = [".jpeg", ".jpg", ".png", ".json", ".npy", ".pcd"]
        for suffix in preferred_suffixes:
            for path in matches:
                if path.suffix.lower() == suffix:
                    return path
        return sorted(matches)[0]
    return None


def collect_frame_ids(scene_dir: Path) -> List[str]:
    result_dir = scene_dir / "result"
    frame_ids = sorted(path.stem for path in result_dir.glob("*.json"))
    valid_ids: List[str] = []
    for frame_id in frame_ids:
        required = [
            scene_dir / "result" / f"{frame_id}.json",
            scene_dir / "camera_config" / f"{frame_id}.json",
            scene_dir / "position" / f"{frame_id}.json",
        ]
        if not all(path.exists() for path in required):
            continue
        image_paths = [find_single_file(scene_dir / image_dir, frame_id) for image_dir in IMAGE_DIRS]
        if any(path is None for path in image_paths):
            continue
        valid_ids.append(frame_id)
    return valid_ids


def decide_frame_augmentation(
    label_data: Mapping[str, Any], seed: int, frame_id: str
) -> Optional[FrameDecision]:
    canonical_classes = sorted(
        {
            canonicalize_class(obj.get("className", ""))
            for obj in label_data.get("objects", [])
            if obj.get("type", "3D_BOX") == "3D_BOX"
        }
    )
    class_set = {name for name in canonical_classes if name != "unknown"}
    if not class_set:
        return None

    plan_rng = stable_rng(seed, frame_id, "plan")
    if class_set & RARE_CLASSES:
        augment_times = int(plan_rng.integers(3, 5))
        trigger = "rare"
    elif class_set & TAIL_CLASSES:
        augment_times = 2
        trigger = "tail"
    else:
        return None

    if class_set & VERY_LIGHT_PHOTO_CLASSES:
        profile = "very_light"
    elif class_set & LIGHT_PHOTO_CLASSES:
        profile = "light"
    else:
        profile = "medium"

    return FrameDecision(
        augment_times=augment_times,
        trigger=trigger,
        profile=profile,
        canonical_classes=tuple(canonical_classes),
    )


def horizontal_flip_bbox(box: Sequence[float], width: int) -> List[float]:
    x1, y1, x2, y2 = [float(value) for value in box]
    new_x1 = float(width - 1 - x2)
    new_x2 = float(width - 1 - x1)
    x_min, x_max = sorted((new_x1, new_x2))
    x_min = min(max(x_min, 0.0), float(width - 1))
    x_max = min(max(x_max, 0.0), float(width - 1))
    return [x_min, y1, x_max, y2]


def augment_label_json(
    label_data: Mapping[str, Any], image_widths: Mapping[str, int], do_flip: bool
) -> Dict[str, Any]:
    augmented = copy.deepcopy(label_data)
    if not do_flip:
        return augmented

    for obj in augmented.get("objects", []):
        contour = obj.get("contour", {})
        center3d = contour.get("center3D")
        if isinstance(center3d, MutableMapping) and "y" in center3d:
            center3d["y"] = -float(center3d["y"])

        rotation3d = contour.get("rotation3D")
        if isinstance(rotation3d, MutableMapping) and "z" in rotation3d:
            rotation3d["z"] = wrap_to_pi(-float(rotation3d["z"]))

        bbox_2d = obj.get("2D_bbox")
        if isinstance(bbox_2d, MutableMapping):
            for image_key, box in list(bbox_2d.items()):
                width = image_widths.get(image_key)
                if width is None or not isinstance(box, Sequence) or len(box) != 4:
                    continue
                bbox_2d[image_key] = horizontal_flip_bbox(box, width)

    return augmented


def decode_camera_external(camera_cfg: Mapping[str, Any]) -> np.ndarray:
    matrix = np.asarray(camera_cfg["camera_external"], dtype=np.float64).reshape(4, 4)
    row_major = str(camera_cfg.get("rowMajor", "true")).lower() == "true"
    return matrix if row_major else matrix.T


def encode_camera_external(matrix: np.ndarray, row_major_flag: Any) -> List[float]:
    row_major = str(row_major_flag).lower() == "true"
    serial = matrix if row_major else matrix.T
    return [float(value) for value in serial.reshape(-1).tolist()]


def augment_camera_config(camera_config: Sequence[Mapping[str, Any]], do_flip: bool) -> List[Dict[str, Any]]:
    augmented = copy.deepcopy(list(camera_config))
    if not do_flip:
        return augmented

    for camera_cfg in augmented:
        width = int(camera_cfg["width"])
        camera_internal = camera_cfg.get("camera_internal", {})
        if "cx" in camera_internal:
            camera_internal["cx"] = float(width - 1 - float(camera_internal["cx"]))

        matrix = decode_camera_external(camera_cfg)
        mirrored_matrix = CAMERA_X_MIRROR @ matrix @ LIDAR_MIRROR
        camera_cfg["camera_external"] = encode_camera_external(
            mirrored_matrix, camera_cfg.get("rowMajor", "false")
        )

    return augmented


def augment_position_json(position_data: Mapping[str, Any], new_frame_id: str) -> Dict[str, Any]:
    augmented = copy.deepcopy(position_data)
    augmented["name"] = new_frame_id
    return augmented


def sample_photometric_plan(rng: np.random.Generator, profile_name: str) -> Dict[str, Any]:
    cfg = PHOTO_PROFILES[profile_name]
    brightness_shift = float(rng.uniform(-cfg["brightness_px"], cfg["brightness_px"]))
    contrast_scale = float(rng.uniform(1.0 - cfg["contrast_delta"], 1.0 + cfg["contrast_delta"]))

    gamma_value: Optional[float] = None
    if rng.random() < cfg["gamma_prob"]:
        gamma_value = float(rng.uniform(1.0 - cfg["gamma_delta"], 1.0 + cfg["gamma_delta"]))

    wb_gains: Optional[List[float]] = None
    if rng.random() < cfg["wb_prob"]:
        wb_gains = [
            float(rng.uniform(1.0 - cfg["wb_delta"], 1.0 + cfg["wb_delta"]))
            for _ in range(3)
        ]

    saturation_scale: Optional[float] = None
    if rng.random() < cfg["saturation_prob"]:
        saturation_scale = float(
            rng.uniform(1.0 - cfg["saturation_delta"], 1.0 + cfg["saturation_delta"])
        )

    shadow_strength: Optional[float] = None
    if rng.random() < cfg["shadow_prob"]:
        shadow_strength = float(rng.uniform(*cfg["shadow_strength"]))

    fog_strength: Optional[float] = None
    if rng.random() < cfg["fog_prob"]:
        fog_strength = float(rng.uniform(*cfg["fog_strength"]))

    blur_kernel: int = 0
    if rng.random() < cfg["blur_prob"]:
        blur_kernel = int(rng.choice(cfg["blur_kernels"]))

    noise_sigma: float = 0.0
    if rng.random() < cfg["noise_prob"]:
        noise_sigma = float(rng.uniform(*cfg["noise_sigma"]))

    jpeg_quality: int = 100
    if rng.random() < cfg["jpeg_prob"]:
        jpeg_quality = int(rng.integers(cfg["jpeg_quality"][0], cfg["jpeg_quality"][1] + 1))

    return {
        "profile": profile_name,
        "brightness_shift": brightness_shift,
        "contrast_scale": contrast_scale,
        "gamma": gamma_value,
        "wb_gains_bgr": wb_gains,
        "saturation_scale": saturation_scale,
        "shadow_strength": shadow_strength,
        "fog_strength": fog_strength,
        "blur_kernel": blur_kernel,
        "noise_sigma": noise_sigma,
        "jpeg_quality": jpeg_quality,
    }


def adjust_brightness_contrast(image: np.ndarray, contrast_scale: float, brightness_shift: float) -> np.ndarray:
    out = image.astype(np.float32) * contrast_scale + brightness_shift
    return np.clip(out, 0, 255).astype(np.uint8)


def adjust_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    gamma = max(gamma, 1e-6)
    inverse_gamma = 1.0 / gamma
    lut = np.array(
        [((index / 255.0) ** inverse_gamma) * 255.0 for index in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(image, lut)


def adjust_white_balance(image: np.ndarray, gains_bgr: Sequence[float]) -> np.ndarray:
    gains = np.asarray(gains_bgr, dtype=np.float32).reshape(1, 1, 3)
    out = image.astype(np.float32) * gains
    return np.clip(out, 0, 255).astype(np.uint8)


def adjust_saturation(image: np.ndarray, scale: float) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= scale
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_shadow(image: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    height, width = image.shape[:2]
    polygon = np.array(
        [
            [0, int(rng.integers(0, max(1, height // 2)))],
            [width - 1, int(rng.integers(0, max(1, height // 2)))],
            [width - 1, int(rng.integers(max(1, height // 2), height))],
            [0, int(rng.integers(max(1, height // 2), height))],
        ],
        dtype=np.int32,
    )
    mask = np.zeros((height, width), dtype=np.float32)
    cv2.fillConvexPoly(mask, polygon, 1.0)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(width, height) / 12.0)
    shadow_gain = 1.0 - mask[..., None] * (1.0 - strength)
    out = image.astype(np.float32) * shadow_gain
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_fog(image: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    height, width = image.shape[:2]
    center_x = float(rng.uniform(0, width))
    center_y = float(rng.uniform(0, height))
    radius = float(rng.uniform(max(width, height) * 0.55, max(width, height) * 1.1))
    yy, xx = np.mgrid[0:height, 0:width]
    dist = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2).astype(np.float32)
    fog_mask = np.exp(-(dist ** 2) / (2.0 * (radius ** 2)))
    fog_mask = (fog_mask / max(float(fog_mask.max()), 1e-6)) * strength
    fog_mask = fog_mask[..., None]
    white = np.full_like(image, 255, dtype=np.float32)
    out = image.astype(np.float32) * (1.0 - fog_mask) + white * fog_mask
    return np.clip(out, 0, 255).astype(np.uint8)


def add_gaussian_noise(image: np.ndarray, rng: np.random.Generator, sigma: float) -> np.ndarray:
    noise = rng.normal(0.0, sigma, size=image.shape).astype(np.float32)
    out = image.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def simulate_jpeg(image: np.ndarray, quality: int) -> np.ndarray:
    quality = int(np.clip(quality, 1, 100))
    success, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        return image
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return image if decoded is None else decoded


def apply_photometric_plan(
    image: np.ndarray, plan: Mapping[str, Any], image_rng: np.random.Generator
) -> np.ndarray:
    out = adjust_brightness_contrast(
        image, float(plan["contrast_scale"]), float(plan["brightness_shift"])
    )
    if plan.get("gamma") is not None:
        out = adjust_gamma(out, float(plan["gamma"]))
    if plan.get("wb_gains_bgr") is not None:
        out = adjust_white_balance(out, plan["wb_gains_bgr"])
    if plan.get("saturation_scale") is not None:
        out = adjust_saturation(out, float(plan["saturation_scale"]))
    if plan.get("shadow_strength") is not None:
        out = apply_shadow(out, image_rng, float(plan["shadow_strength"]))
    if plan.get("fog_strength") is not None:
        out = apply_fog(out, image_rng, float(plan["fog_strength"]))
    blur_kernel = int(plan.get("blur_kernel", 0))
    if blur_kernel > 1:
        out = cv2.GaussianBlur(out, (blur_kernel, blur_kernel), sigmaX=0.0)
    noise_sigma = float(plan.get("noise_sigma", 0.0))
    if noise_sigma > 0:
        out = add_gaussian_noise(out, image_rng, noise_sigma)
    jpeg_quality = int(plan.get("jpeg_quality", 100))
    if jpeg_quality < 100:
        out = simulate_jpeg(out, jpeg_quality)
    return out


def augment_images(
    scene_dir: Path,
    output_dir: Path,
    frame_id: str,
    new_frame_id: str,
    do_flip: bool,
    photometric_plan: Mapping[str, Any],
    seed: int,
) -> Dict[str, int]:
    image_widths: Dict[str, int] = {}
    for image_dir in IMAGE_DIRS:
        src_path = find_single_file(scene_dir / image_dir, frame_id)
        if src_path is None:
            raise FileNotFoundError(f"Missing {image_dir} image for frame {frame_id}")
        image = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {src_path}")

        if do_flip:
            image = cv2.flip(image, 1)

        image_rng = stable_rng(seed, new_frame_id, image_dir)
        image = apply_photometric_plan(image, photometric_plan, image_rng)

        dst_path = output_dir / image_dir / f"{new_frame_id}{src_path.suffix.lower()}"
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(dst_path), image):
            raise RuntimeError(f"Failed to write image: {dst_path}")
        image_widths[image_dir] = int(image.shape[1])

    return image_widths


def flip_numpy_point_cloud(array: np.ndarray) -> np.ndarray:
    flipped = array.copy()
    if flipped.dtype.names:
        if "y" not in flipped.dtype.names:
            raise ValueError("Structured point cloud does not contain a 'y' field.")
        flipped["y"] *= -1
        return flipped
    if flipped.ndim != 2 or flipped.shape[1] < 2:
        raise ValueError(f"Unsupported point cloud shape: {flipped.shape}")
    flipped[:, 1] *= -1
    return flipped


def parse_pcd_dtype(header_text: str) -> Tuple[np.dtype, str, int]:
    meta: Dict[str, List[str]] = {}
    for raw_line in header_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        meta[parts[0].upper()] = parts[1:]

    fields = meta["FIELDS"]
    sizes = [int(v) for v in meta["SIZE"]]
    types = meta["TYPE"]
    counts = [int(v) for v in meta.get("COUNT", ["1"] * len(fields))]
    points = int(meta.get("POINTS", ["0"])[0])
    data_format = meta["DATA"][0].lower()

    type_map = {
        ("F", 4): np.float32,
        ("F", 8): np.float64,
        ("I", 1): np.int8,
        ("I", 2): np.int16,
        ("I", 4): np.int32,
        ("I", 8): np.int64,
        ("U", 1): np.uint8,
        ("U", 2): np.uint16,
        ("U", 4): np.uint32,
        ("U", 8): np.uint64,
    }
    dtype_fields: List[Tuple[Any, ...]] = []
    for field, size, type_code, count in zip(fields, sizes, types, counts):
        base_type = type_map[(type_code, size)]
        if count == 1:
            dtype_fields.append((field, base_type))
        else:
            dtype_fields.append((field, base_type, (count,)))
    return np.dtype(dtype_fields), data_format, points


def flip_pcd_file(src_path: Path, dst_path: Path) -> None:
    with src_path.open("rb") as handle:
        header_lines: List[bytes] = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Invalid PCD file: {src_path}")
            header_lines.append(line)
            if line.strip().lower().startswith(b"data"):
                break
        payload = handle.read()

    header_text = b"".join(header_lines).decode("ascii", errors="ignore")
    dtype, data_format, points = parse_pcd_dtype(header_text)
    if data_format != "binary":
        raise NotImplementedError(
            f"Only binary PCD is supported in this script, got '{data_format}' for {src_path}"
        )

    cloud = np.frombuffer(payload, dtype=dtype, count=points).copy()
    if "y" not in cloud.dtype.names:
        raise ValueError(f"PCD file has no 'y' field: {src_path}")
    cloud["y"] *= -1

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with dst_path.open("wb") as handle:
        for line in header_lines:
            handle.write(line)
        handle.write(cloud.tobytes())


def save_point_cloud_variants(
    scene_dir: Path, output_dir: Path, frame_id: str, new_frame_id: str, do_flip: bool
) -> List[str]:
    saved_formats: List[str] = []
    point_cloud_dir = scene_dir / "point_cloud"
    for src_path in point_cloud_dir.glob(f"{frame_id}.*"):
        suffix = src_path.suffix.lower()
        dst_path = output_dir / "point_cloud" / f"{new_frame_id}{suffix}"
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if not do_flip:
            shutil.copy2(src_path, dst_path)
            saved_formats.append(suffix)
            continue

        if suffix == ".npy":
            array = np.load(src_path, allow_pickle=False)
            np.save(dst_path, flip_numpy_point_cloud(array))
            saved_formats.append(suffix)
        elif suffix == ".pcd":
            flip_pcd_file(src_path, dst_path)
            saved_formats.append(suffix)
        else:
            shutil.copy2(src_path, dst_path)
            saved_formats.append(suffix)
    return saved_formats


def save_augmented_metadata_files(
    scene_dir: Path,
    output_dir: Path,
    frame_id: str,
    new_frame_id: str,
    do_flip: bool,
    image_widths: Mapping[str, int],
) -> None:
    label_data = load_json(scene_dir / "result" / f"{frame_id}.json")
    camera_config = load_json(scene_dir / "camera_config" / f"{frame_id}.json")
    position_data = load_json(scene_dir / "position" / f"{frame_id}.json")

    save_json(
        augment_label_json(label_data, image_widths=image_widths, do_flip=do_flip),
        output_dir / "result" / f"{new_frame_id}.json",
    )
    save_json(
        augment_camera_config(camera_config, do_flip=do_flip),
        output_dir / "camera_config" / f"{new_frame_id}.json",
    )
    save_json(
        augment_position_json(position_data, new_frame_id=new_frame_id),
        output_dir / "position" / f"{new_frame_id}.json",
    )


def create_empty_scene_layout(output_dir: Path) -> None:
    for folder in ("camera_config", "position", "point_cloud", "result", *IMAGE_DIRS):
        (output_dir / folder).mkdir(parents=True, exist_ok=True)


def build_output_dir(
    scene_dir: Path, output_dir: Path, copy_originals: bool, overwrite_output: bool
) -> None:
    ensure_output_root(scene_dir, output_dir, copy_originals, overwrite_output)
    if copy_originals:
        shutil.copytree(scene_dir, output_dir)
    else:
        create_empty_scene_layout(output_dir)


def maybe_limit_frame_ids(frame_ids: List[str], max_frames: Optional[int]) -> List[str]:
    if max_frames is None:
        return frame_ids
    return frame_ids[: max(0, max_frames)]


def allocate_numeric_augmented_ids(
    frame_id: str, count: int, used_ids: set[int]
) -> List[str]:
    if not frame_id.isdigit():
        return [f"{frame_id}_{index + 1}" for index in range(count)]

    next_id = int(frame_id)
    new_ids: List[str] = []
    while len(new_ids) < count:
        next_id += 1
        if next_id in used_ids:
            continue
        used_ids.add(next_id)
        new_ids.append(str(next_id))
    return new_ids


def run(
    scene_dir: Path,
    output_dir: Path,
    seed: int,
    copy_originals: bool,
    max_frames: Optional[int],
    overwrite_output: bool,
) -> None:
    build_output_dir(
        scene_dir,
        output_dir,
        copy_originals=copy_originals,
        overwrite_output=overwrite_output,
    )

    frame_ids = maybe_limit_frame_ids(collect_frame_ids(scene_dir), max_frames=max_frames)
    used_numeric_ids = {int(frame_id) for frame_id in frame_ids if frame_id.isdigit()}
    manifest: List[Dict[str, Any]] = []
    generated = 0
    skipped = 0

    for frame_id in frame_ids:
        label_data = load_json(scene_dir / "result" / f"{frame_id}.json")
        decision = decide_frame_augmentation(label_data, seed=seed, frame_id=frame_id)
        if decision is None:
            skipped += 1
            continue

        new_frame_ids = allocate_numeric_augmented_ids(
            frame_id=frame_id,
            count=decision.augment_times,
            used_ids=used_numeric_ids,
        )
        for aug_idx, new_frame_id in enumerate(new_frame_ids):
            aug_rng = stable_rng(seed, frame_id, str(aug_idx))
            do_flip = bool(aug_rng.random() < 0.5)
            photometric_plan = sample_photometric_plan(aug_rng, decision.profile)
            image_widths = augment_images(
                scene_dir=scene_dir,
                output_dir=output_dir,
                frame_id=frame_id,
                new_frame_id=new_frame_id,
                do_flip=do_flip,
                photometric_plan=photometric_plan,
                seed=seed,
            )
            save_augmented_metadata_files(
                scene_dir=scene_dir,
                output_dir=output_dir,
                frame_id=frame_id,
                new_frame_id=new_frame_id,
                do_flip=do_flip,
                image_widths=image_widths,
            )
            saved_point_cloud_formats = save_point_cloud_variants(
                scene_dir=scene_dir,
                output_dir=output_dir,
                frame_id=frame_id,
                new_frame_id=new_frame_id,
                do_flip=do_flip,
            )
            manifest.append(
                {
                    "new_frame_id": new_frame_id,
                    "source_frame_id": frame_id,
                    "augment_index": aug_idx,
                    "trigger": decision.trigger,
                    "profile": decision.profile,
                    "canonical_classes": list(decision.canonical_classes),
                    "flip_horizontal": do_flip,
                    "point_cloud_formats": saved_point_cloud_formats,
                    "photometric_plan": photometric_plan,
                }
            )
            generated += 1

    save_json(
        {
            "scene_dir": str(scene_dir),
            "output_dir": str(output_dir),
            "seed": seed,
            "copy_originals": copy_originals,
            "input_frame_count": len(frame_ids),
            "generated_augmented_frames": generated,
            "skipped_frames": skipped,
            "manifest": manifest,
            "notes": [
                "position JSON is copied with only the top-level name updated.",
                "camera_config is mirrored only when horizontal flip is applied.",
                "2D_bbox is mirrored within the same image, without swapping camera folders.",
                "Augmented frame ids are new numeric ids allocated after each source frame id.",
            ],
        },
        output_dir / "augmentation_manifest.json",
    )

    print(f"Input frames considered: {len(frame_ids)}")
    print(f"Frames skipped (no rare/tail class): {skipped}")
    print(f"Augmented frames generated: {generated}")
    print(f"Saved to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-dir",
        type=Path,
        default=Path("fisheye_data_demo/20231214-163109_20231214-163126"),
        help="Input demo scene directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("fisheye_data_demo/20231214-163109_20231214-163126_offline_aug"),
        help="Output scene directory for originals + augmented data.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260415,
        help="Global random seed used to make augmentation deterministic.",
    )
    parser.add_argument(
        "--no-copy-originals",
        action="store_true",
        help="Only export augmented samples instead of copying the original scene first.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional debug limit for how many source frames to scan.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Delete and recreate --output-dir if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        scene_dir=args.scene_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        copy_originals=not args.no_copy_originals,
        max_frames=args.max_frames,
        overwrite_output=args.overwrite_output,
    )


if __name__ == "__main__":
    main()
