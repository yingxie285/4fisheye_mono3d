#!/usr/bin/env python
"""Standalone offline augmentation for fisheye camera-only 3D detection data."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple
import os
import cv2
import numpy as np


IMAGE_DIRS: Tuple[str, ...] = ("image0", "image1", "image2", "image3")
METADATA_DIRS: Tuple[str, ...] = ("camera_config", "position", "result")

LIDAR_MIRROR = np.diag([1.0, -1.0, 1.0, 1.0])
CAMERA_X_MIRROR = np.diag([-1.0, 1.0, 1.0, 1.0])

TARGET_TAIL_CLASSES = frozenset({"pedestrian", "bicycle", "motor","stopper"})
RARE_CLASSES = frozenset({"animal", "bus", "construction_vehicle", "traffic_cone"})
TAIL_CLASSES = frozenset({"truck", "trash_bin", "sign", "barrier"})
LONG_TAIL_CLASSES = RARE_CLASSES | TAIL_CLASSES
PARALLEL_WORKERS = 10

DEFAULT_TARGET_TOTAL_FRAMES = 500 #150000
BASE_TARGET_COLOR_ONLY_COPIES = 4
BASE_LONG_TAIL_COLOR_ONLY_COPIES = 6
BASE_TARGET_AND_LONG_TAIL_COLOR_ONLY_COPIES = 8
TARGET_CLASS_COLOR_PRIORITY = 0.75
LONG_TAIL_CLASS_COLOR_PRIORITY = 1.5
PEDESTRIAN_COLOR_PRIORITY_BONUS = 0.75
TARGET_AND_LONG_TAIL_PRIORITY_BONUS = 2.0
VERY_LIGHT_PHOTO_CLASSES = frozenset({"bus", "animal", "construction_vehicle"})
LIGHT_PHOTO_CLASSES = frozenset({"pedestrian", "bicycle", "motor", "traffic_cone"})
COLOR_INTENSITY_CYCLES: Dict[str, Tuple[str, ...]] = {
    "target_tail": ("light", "medium", "strong", "medium"),
    "long_tail": ("medium", "strong", "xstrong", "strong"),
    "target_and_long_tail": ("medium", "strong", "xstrong", "strong", "medium"),
}
PHOTO_INTENSITY_SCALES: Dict[str, Dict[str, float]] = {
    "light": {"magnitude_scale": 0.85, "probability_scale": 0.90},
    "medium": {"magnitude_scale": 1.0, "probability_scale": 1.0},
    "strong": {"magnitude_scale": 1.35, "probability_scale": 1.15},
    "xstrong": {"magnitude_scale": 1.70, "probability_scale": 1.30},
}

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
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "construction_vehicle": "construction_vehicle",
    "pedestrian": "pedestrian",
    "motor": "motor",
    "bicycle": "bicycle",
    "animal": "animal",
    "traffic_cone": "traffic_cone",
    "barrier": "barrier",
    "stopper": "stopper",
    "trash_bin": "trash_bin",
    "sign": "sign",
    "普通轿车": "car",
    "敞篷车": "car",
    "敞篷轿车": "car",
    "SUV": "car",
    "MPV": "car",
    "面包车": "car",
    "皮卡": "car",
    "警车": "car",
    "救护车": "car",
    "其他机动车": "car",
    "未知机动车": "car",
    "未知机动车车轮": "car",
    "未知机动车车灯": "car",
    "房车": "truck",
    "普通小型货车": "truck",
    "箱式小型货车": "truck",
    "普通大型货车": "truck",
    "轻运货车": "truck",
    "轿运车": "truck",
    "客车": "bus",
    "校车": "bus",
    "普通公交车": "bus",
    "铰链公交车": "bus",
    "有轨电车": "bus",
    "无轨电车": "bus",
    "消防车": "construction_vehicle",
    "清扫车": "construction_vehicle",
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
    "锥筒": "traffic_cone",
    "防撞桶": "barrier",
    "路障": "barrier",
    "路桩": "barrier",
    "石墩": "barrier",
    "水马": "barrier",
    "柱子": "barrier",
    "石块": "barrier",
    "树枝/树杈": "barrier",
    "树枝/树桩": "barrier",
    "空中漂浮物": "barrier",
    "路坑/水洼": "barrier",
    "路坑": "barrier",
    "其他静态障碍物": "barrier",
    "车位停止器": "stopper",
    "车位止停器": "stopper",
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
}


@dataclass(frozen=True)
class FrameDecision:
    trigger: str
    profile: str
    canonical_classes: Tuple[str, ...]
    class_counts: Dict[str, int]
    has_target_tail: bool
    has_long_tail: bool
    target_tail_count: int
    long_tail_count: int
    base_color_only_count: int
    color_priority: float


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def canonicalize_class(class_name: str) -> str:
    key = str(class_name).strip()
    return CANONICAL_CLASS_MAP.get(key, key if key.isascii() else "unknown")


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


def _collect_scene_frame_decisions_worker(
    args: Tuple[Path, Optional[int]]
) -> Tuple[Path, List[str], Dict[str, FrameDecision]]:
    scene_dir, max_frames = args
    frame_ids, frame_decisions = collect_scene_frame_decisions(
        scene_dir, max_frames=max_frames
    )
    return scene_dir, frame_ids, frame_decisions


def _run_single_scene_worker(
    args: Tuple[Path, Path, int, bool, Optional[int], bool, Mapping[str, int]]
) -> Dict[str, Any]:
    return run_single_scene(*args)


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
    matches = [path for path in directory.glob(f"{stem}.*") if path.is_file()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        preferred_suffixes = [".jpeg", ".jpg", ".png", ".json"]
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


def collect_object_counts(label_data: Mapping[str, Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for obj in label_data.get("objects", []):
        if obj.get("type", "3D_BOX") != "3D_BOX":
            continue
        class_name = canonicalize_class(obj.get("className", ""))
        if class_name == "unknown":
            continue
        counts[class_name] += 1
    return counts


def compute_base_color_only_count(has_target_tail: bool, has_long_tail: bool) -> int:
    if has_target_tail and has_long_tail:
        return BASE_TARGET_AND_LONG_TAIL_COLOR_ONLY_COPIES
    if has_long_tail:
        return BASE_LONG_TAIL_COLOR_ONLY_COPIES
    if has_target_tail:
        return BASE_TARGET_COLOR_ONLY_COPIES
    return 0


def compute_color_priority(
    class_counts: Mapping[str, int], target_tail_count: int, long_tail_count: int
) -> float:
    has_target_tail = target_tail_count > 0
    has_long_tail = long_tail_count > 0
    priority = float(compute_base_color_only_count(has_target_tail, has_long_tail))
    priority += float(target_tail_count) * TARGET_CLASS_COLOR_PRIORITY
    priority += float(long_tail_count) * LONG_TAIL_CLASS_COLOR_PRIORITY
    if class_counts.get("pedestrian", 0) > 0:
        priority += PEDESTRIAN_COLOR_PRIORITY_BONUS
    if has_target_tail and has_long_tail:
        priority += TARGET_AND_LONG_TAIL_PRIORITY_BONUS
    return priority


def decide_frame_augmentation(label_data: Mapping[str, Any]) -> Optional[FrameDecision]:
    class_counts = collect_object_counts(label_data)
    if not class_counts:
        return None

    canonical_classes = sorted(class_counts)
    class_set = set(canonical_classes)
    target_tail_count = sum(class_counts.get(name, 0) for name in TARGET_TAIL_CLASSES)
    long_tail_count = sum(class_counts.get(name, 0) for name in LONG_TAIL_CLASSES)
    has_target_tail = target_tail_count > 0
    has_long_tail = long_tail_count > 0

    if not has_target_tail and not has_long_tail:
        return None

    if has_target_tail and has_long_tail:
        trigger = "target_and_long_tail"
    elif has_long_tail:
        trigger = "long_tail"
    else:
        trigger = "target_tail"

    if class_set & VERY_LIGHT_PHOTO_CLASSES:
        profile = "very_light"
    elif class_set & LIGHT_PHOTO_CLASSES:
        profile = "light"
    else:
        profile = "medium"

    return FrameDecision(
        trigger=trigger,
        profile=profile,
        canonical_classes=tuple(canonical_classes),
        class_counts=dict(sorted(class_counts.items())),
        has_target_tail=has_target_tail,
        has_long_tail=has_long_tail,
        target_tail_count=target_tail_count,
        long_tail_count=long_tail_count,
        base_color_only_count=compute_base_color_only_count(has_target_tail, has_long_tail),
        color_priority=round(
            compute_color_priority(class_counts, target_tail_count, long_tail_count), 3
        ),
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


def augment_camera_config(
    camera_config: Sequence[Mapping[str, Any]], do_flip: bool
) -> List[Dict[str, Any]]:
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


def make_identity_photometric_plan(profile_name: str) -> Dict[str, Any]:
    return {
        "profile": profile_name,
        "intensity": "identity",
        "brightness_shift": 0.0,
        "contrast_scale": 1.0,
        "gamma": None,
        "wb_gains_bgr": None,
        "saturation_scale": None,
        "shadow_strength": None,
        "fog_strength": None,
        "blur_kernel": 0,
        "noise_sigma": 0.0,
        "jpeg_quality": 100,
    }


def scale_probability(probability: float, scale: float) -> float:
    return min(max(probability * scale, 0.0), 1.0)


def scale_range_from_neutral(
    values: Sequence[float], neutral: float, scale: float
) -> Tuple[float, float]:
    scaled_values = [neutral + (float(value) - neutral) * scale for value in values]
    low, high = sorted(scaled_values)
    return low, high


def sample_photometric_plan(
    rng: np.random.Generator, profile_name: str, intensity_name: str
) -> Dict[str, Any]:
    if intensity_name == "identity":
        return make_identity_photometric_plan(profile_name)

    cfg = PHOTO_PROFILES[profile_name]
    intensity_cfg = PHOTO_INTENSITY_SCALES[intensity_name]
    magnitude_scale = float(intensity_cfg["magnitude_scale"])
    probability_scale = float(intensity_cfg["probability_scale"])

    brightness_shift = float(
        rng.uniform(
            -cfg["brightness_px"] * magnitude_scale,
            cfg["brightness_px"] * magnitude_scale,
        )
    )
    contrast_delta = float(cfg["contrast_delta"]) * magnitude_scale
    contrast_scale = float(rng.uniform(1.0 - contrast_delta, 1.0 + contrast_delta))

    gamma_value: Optional[float] = None
    gamma_delta = float(cfg["gamma_delta"]) * magnitude_scale
    if rng.random() < scale_probability(float(cfg["gamma_prob"]), probability_scale):
        gamma_value = float(rng.uniform(1.0 - gamma_delta, 1.0 + gamma_delta))

    wb_gains: Optional[List[float]] = None
    wb_delta = float(cfg["wb_delta"]) * magnitude_scale
    if rng.random() < scale_probability(float(cfg["wb_prob"]), probability_scale):
        wb_gains = [
            float(rng.uniform(1.0 - wb_delta, 1.0 + wb_delta))
            for _ in range(3)
        ]

    saturation_scale: Optional[float] = None
    saturation_delta = float(cfg["saturation_delta"]) * magnitude_scale
    if rng.random() < scale_probability(float(cfg["saturation_prob"]), probability_scale):
        saturation_scale = float(
            rng.uniform(1.0 - saturation_delta, 1.0 + saturation_delta)
        )

    shadow_strength: Optional[float] = None
    if rng.random() < scale_probability(float(cfg["shadow_prob"]), probability_scale):
        shadow_range = scale_range_from_neutral(cfg["shadow_strength"], neutral=1.0, scale=magnitude_scale)
        shadow_strength = float(rng.uniform(*shadow_range))

    fog_strength: Optional[float] = None
    if rng.random() < scale_probability(float(cfg["fog_prob"]), probability_scale):
        fog_low, fog_high = [float(value) * magnitude_scale for value in cfg["fog_strength"]]
        fog_strength = float(rng.uniform(fog_low, fog_high))

    blur_kernel = 0
    if rng.random() < scale_probability(float(cfg["blur_prob"]), probability_scale):
        blur_kernels = sorted({int(value) for value in cfg["blur_kernels"]})
        if magnitude_scale >= 1.35:
            blur_kernels.append(5)
        if magnitude_scale >= 1.60:
            blur_kernels.append(7)
        blur_kernel = int(rng.choice(sorted(set(blur_kernels))))

    noise_sigma = 0.0
    if rng.random() < scale_probability(float(cfg["noise_prob"]), probability_scale):
        noise_low, noise_high = [float(value) * magnitude_scale for value in cfg["noise_sigma"]]
        noise_sigma = float(rng.uniform(noise_low, noise_high))

    jpeg_quality = 100
    if rng.random() < scale_probability(float(cfg["jpeg_prob"]), probability_scale):
        base_quality = int(rng.integers(cfg["jpeg_quality"][0], cfg["jpeg_quality"][1] + 1))
        jpeg_quality = int(np.clip(round(100.0 - (100.0 - base_quality) * magnitude_scale), 1, 100))

    return {
        "profile": profile_name,
        "intensity": intensity_name,
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


def adjust_brightness_contrast(
    image: np.ndarray, contrast_scale: float, brightness_shift: float
) -> np.ndarray:
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


def choose_color_intensity(decision: FrameDecision, color_only_index: int) -> str:
    cycle = COLOR_INTENSITY_CYCLES[decision.trigger]
    return cycle[color_only_index % len(cycle)]


def load_frame_images(scene_dir: Path, frame_id: str) -> Dict[str, Tuple[np.ndarray, str]]:
    source_images: Dict[str, Tuple[np.ndarray, str]] = {}
    for image_dir in IMAGE_DIRS:
        src_path = find_single_file(scene_dir / image_dir, frame_id)
        if src_path is None:
            raise FileNotFoundError(f"Missing {image_dir} image for frame {frame_id}")
        image = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {src_path}")
        source_images[image_dir] = (image, src_path.suffix.lower())
    return source_images


def load_frame_metadata(
    scene_dir: Path, frame_id: str
) -> Tuple[Mapping[str, Any], Sequence[Mapping[str, Any]], Mapping[str, Any]]:
    return (
        load_json(scene_dir / "result" / f"{frame_id}.json"),
        load_json(scene_dir / "camera_config" / f"{frame_id}.json"),
        load_json(scene_dir / "position" / f"{frame_id}.json"),
    )


def save_augmented_images(
    source_images: Mapping[str, Tuple[np.ndarray, str]],
    output_dir: Path,
    new_frame_id: str,
    do_flip: bool,
    photometric_plan: Mapping[str, Any],
    seed: int,
) -> None:
    for image_dir, (source_image, suffix) in source_images.items():
        image = source_image.copy()
        if do_flip:
            image = cv2.flip(image, 1)

        image_rng = stable_rng(seed, new_frame_id, image_dir)
        image = apply_photometric_plan(image, photometric_plan, image_rng)

        dst_path = output_dir / image_dir / f"{new_frame_id}{suffix}"
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(dst_path), image):
            raise RuntimeError(f"Failed to write image: {dst_path}")


def build_image_widths(source_images: Mapping[str, Tuple[np.ndarray, str]]) -> Dict[str, int]:
    return {
        image_dir: int(image.shape[1])
        for image_dir, (image, _) in source_images.items()
    }


def save_augmented_metadata_files(
    output_dir: Path,
    new_frame_id: str,
    label_data: Mapping[str, Any],
    camera_config: Sequence[Mapping[str, Any]],
    position_data: Mapping[str, Any],
    do_flip: bool,
    image_widths: Mapping[str, int],
) -> None:
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


def create_empty_scene_layout(output_dir: Path) -> None:
    for folder in (*METADATA_DIRS, *IMAGE_DIRS):
        (output_dir / folder).mkdir(parents=True, exist_ok=True)


def copy_scene_without_point_cloud(scene_dir: Path, output_dir: Path) -> None:
    create_empty_scene_layout(output_dir)
    for folder in (*METADATA_DIRS, *IMAGE_DIRS):
        src_dir = scene_dir / folder
        dst_dir = output_dir / folder
        if not src_dir.exists():
            continue
        for src_path in src_dir.iterdir():
            if src_path.is_file():
                shutil.copy2(src_path, dst_dir / src_path.name)


def build_output_dir(
    scene_dir: Path, output_dir: Path, copy_originals: bool, overwrite_output: bool
) -> None:
    ensure_output_root(scene_dir, output_dir, copy_originals, overwrite_output)
    if copy_originals:
        copy_scene_without_point_cloud(scene_dir, output_dir)
    else:
        create_empty_scene_layout(output_dir)


def maybe_limit_frame_ids(frame_ids: List[str], max_frames: Optional[int]) -> List[str]:
    if max_frames is None:
        return frame_ids
    return frame_ids[: max(0, max_frames)]


def collect_scene_frame_decisions(
    scene_dir: Path, max_frames: Optional[int]
) -> Tuple[List[str], Dict[str, FrameDecision]]:
    frame_ids = maybe_limit_frame_ids(collect_frame_ids(scene_dir), max_frames=max_frames)
    frame_decisions: Dict[str, FrameDecision] = {}
    for frame_id in frame_ids:
        label_data = load_json(scene_dir / "result" / f"{frame_id}.json")
        decision = decide_frame_augmentation(label_data)
        if decision is not None:
            frame_decisions[frame_id] = decision
    return frame_ids, frame_decisions


def proportional_allocation(weights: Sequence[float], total_count: int) -> List[int]:
    if not weights:
        return []
    if total_count <= 0:
        return [0 for _ in weights]

    safe_weights = [max(float(weight), 0.0) for weight in weights]
    weight_sum = sum(safe_weights)
    if weight_sum <= 0.0:
        base = total_count // len(safe_weights)
        counts = [base for _ in safe_weights]
        for index in range(total_count - base * len(safe_weights)):
            counts[index] += 1
        return counts

    raw_counts = [total_count * weight / weight_sum for weight in safe_weights]
    counts = [int(math.floor(value)) for value in raw_counts]
    remaining = total_count - sum(counts)
    if remaining <= 0:
        return counts

    order = sorted(
        range(len(raw_counts)),
        key=lambda index: (-(raw_counts[index] - counts[index]), -safe_weights[index], index),
    )
    for index in order[:remaining]:
        counts[index] += 1
    return counts


def plan_scene_color_only_counts(
    scene_decisions: Mapping[str, Dict[str, FrameDecision]],
    target_color_only_frames: int,
) -> Dict[str, Dict[str, int]]:
    frame_keys: List[Tuple[str, str]] = []
    decisions: List[FrameDecision] = []
    for scene_name in sorted(scene_decisions):
        for frame_id in sorted(scene_decisions[scene_name]):
            frame_keys.append((scene_name, frame_id))
            decisions.append(scene_decisions[scene_name][frame_id])

    if not frame_keys:
        return {}

    required_color_only_frames = max(0, int(target_color_only_frames))
    base_color_only_counts = [decision.base_color_only_count for decision in decisions]

    if required_color_only_frames >= sum(base_color_only_counts):
        color_only_counts = list(base_color_only_counts)
        remaining_color_only_frames = required_color_only_frames - sum(color_only_counts)
        extra_color_only_counts = proportional_allocation(
            [decision.color_priority for decision in decisions], remaining_color_only_frames
        )
        color_only_counts = [
            base_count + extra_count
            for base_count, extra_count in zip(color_only_counts, extra_color_only_counts)
        ]
    else:
        color_only_counts = proportional_allocation(
            base_color_only_counts, required_color_only_frames
        )

    scene_color_only_counts: Dict[str, Dict[str, int]] = {}
    for (scene_name, frame_id), color_only_count in zip(frame_keys, color_only_counts):
        scene_color_only_counts.setdefault(scene_name, {})[frame_id] = color_only_count
    return scene_color_only_counts


def is_scene_dir(path: Path) -> bool:
    required_dirs = (*METADATA_DIRS, *IMAGE_DIRS)
    return path.is_dir() and all((path / folder).is_dir() for folder in required_dirs)


def collect_scene_dirs(scene_root: Path) -> List[Path]:
    if is_scene_dir(scene_root):
        return [scene_root]
    if not scene_root.is_dir():
        raise FileNotFoundError(f"Scene root does not exist: {scene_root}")
    scene_dirs = [path for path in scene_root.iterdir() if is_scene_dir(path)]
    return sorted(scene_dirs, key=lambda path: path.name)


def run_single_scene(
    scene_dir: Path,
    output_dir: Path,
    seed: int,
    copy_originals: bool,
    max_frames: Optional[int],
    overwrite_output: bool,
    color_only_counts: Mapping[str, int],
) -> Dict[str, Any]:
    build_output_dir(
        scene_dir=scene_dir,
        output_dir=output_dir,
        copy_originals=copy_originals,
        overwrite_output=overwrite_output,
    )

    frame_ids, frame_decisions = collect_scene_frame_decisions(scene_dir, max_frames=max_frames)
    used_numeric_ids = {int(frame_id) for frame_id in frame_ids if frame_id.isdigit()}
    manifest: List[Dict[str, Any]] = []
    generated = 0
    flip_generated = 0
    color_only_generated = 0
    skipped = 0

    for frame_id in frame_ids:
        decision = frame_decisions.get(frame_id)
        color_only_count = int(color_only_counts.get(frame_id, 0)) if decision is not None else 0
        new_frame_ids = allocate_numeric_augmented_ids(
            frame_id=frame_id,
            count=1 + color_only_count,
            used_ids=used_numeric_ids,
        )

        source_images = load_frame_images(scene_dir, frame_id)
        image_widths = build_image_widths(source_images)
        label_data, camera_config, position_data = load_frame_metadata(scene_dir, frame_id)
        frame_class_counts = dict(sorted(collect_object_counts(label_data).items()))

        if decision is None:
            skipped += 1
            flip_trigger = "all_frame_flip"
            flip_profile_name = "identity"
            flip_canonical_classes = list(sorted(frame_class_counts))
            flip_class_counts = frame_class_counts
            flip_target_tail_count = 0
            flip_long_tail_count = 0
        else:
            flip_trigger = decision.trigger
            flip_profile_name = decision.profile
            flip_canonical_classes = list(decision.canonical_classes)
            flip_class_counts = decision.class_counts
            flip_target_tail_count = decision.target_tail_count
            flip_long_tail_count = decision.long_tail_count

        flip_frame_id = new_frame_ids[0]
        flip_plan = make_identity_photometric_plan(flip_profile_name)
        save_augmented_images(
            source_images=source_images,
            output_dir=output_dir,
            new_frame_id=flip_frame_id,
            do_flip=True,
            photometric_plan=flip_plan,
            seed=seed,
        )
        save_augmented_metadata_files(
            output_dir=output_dir,
            new_frame_id=flip_frame_id,
            label_data=label_data,
            camera_config=camera_config,
            position_data=position_data,
            do_flip=True,
            image_widths=image_widths,
        )
        manifest.append(
            {
                "new_frame_id": flip_frame_id,
                "source_frame_id": frame_id,
                "augment_index": 0,
                "augmentation_type": "flip",
                "trigger": flip_trigger,
                "profile": flip_profile_name,
                "canonical_classes": flip_canonical_classes,
                "class_counts": flip_class_counts,
                "target_tail_count": flip_target_tail_count,
                "long_tail_count": flip_long_tail_count,
                "flip_horizontal": True,
                "photometric_plan": flip_plan,
            }
        )
        generated += 1
        flip_generated += 1

        for color_idx in range(color_only_count):
            new_frame_id = new_frame_ids[color_idx + 1]
            aug_rng = stable_rng(seed, frame_id, "color_only", str(color_idx))
            intensity_name = choose_color_intensity(decision, color_idx)
            photometric_plan = sample_photometric_plan(
                aug_rng, decision.profile, intensity_name
            )

            save_augmented_images(
                source_images=source_images,
                output_dir=output_dir,
                new_frame_id=new_frame_id,
                do_flip=False,
                photometric_plan=photometric_plan,
                seed=seed,
            )
            save_augmented_metadata_files(
                output_dir=output_dir,
                new_frame_id=new_frame_id,
                label_data=label_data,
                camera_config=camera_config,
                position_data=position_data,
                do_flip=False,
                image_widths=image_widths,
            )
            manifest.append(
                {
                    "new_frame_id": new_frame_id,
                    "source_frame_id": frame_id,
                    "augment_index": color_idx + 1,
                    "augmentation_type": "color_only",
                    "trigger": decision.trigger,
                    "profile": decision.profile,
                    "canonical_classes": list(decision.canonical_classes),
                    "class_counts": decision.class_counts,
                    "target_tail_count": decision.target_tail_count,
                    "long_tail_count": decision.long_tail_count,
                    "flip_horizontal": False,
                    "photometric_plan": photometric_plan,
                }
            )
            generated += 1
            color_only_generated += 1

    save_json(
        {
            "scene_dir": str(scene_dir),
            "output_dir": str(output_dir),
            "seed": seed,
            "copy_originals": copy_originals,
            "input_frame_count": len(frame_ids),
            "eligible_frame_count": len(frame_decisions),
            "generated_augmented_frames": generated,
            "generated_flip_frames": flip_generated,
            "generated_color_only_frames": color_only_generated,
            "skipped_frames": skipped,
            "point_cloud_included": False,
            "manifest": manifest,
            "notes": [
                "This camera-only export does not copy or generate point_cloud files.",
                "Every input frame always gets exactly one horizontal flip sample.",
                "Only frames containing pedestrian/bicycle/motor or long-tail classes receive extra color-only samples.",
                "Color-only samples only modify image appearance and keep label/calibration geometry unchanged.",
                "position JSON is copied with only the top-level name updated.",
                "camera_config is mirrored only when horizontal flip is applied.",
                "2D_bbox is mirrored within the same image, without swapping camera folders.",
                "Augmented frame ids are new numeric ids allocated after each source frame id.",
            ],
        },
        output_dir / "augmentation_manifest.json",
    )

    print(f"Input frames considered: {len(frame_ids)}")
    print(f"Eligible frames (target/long-tail): {len(frame_decisions)}")
    print(f"Frames without target/long-tail class (flip-only): {skipped}")
    print(f"Augmented flip frames generated: {flip_generated}")
    print(f"Augmented color-only frames generated: {color_only_generated}")
    print(f"Augmented frames generated: {generated}")
    print("Point clouds copied/generated: 0")
    print(f"Saved to: {output_dir}")

    return {
        "scene_name": scene_dir.name,
        "scene_dir": str(scene_dir),
        "output_dir": str(output_dir),
        "input_frame_count": len(frame_ids),
        "eligible_frame_count": len(frame_decisions),
        "generated_augmented_frames": generated,
        "generated_flip_frames": flip_generated,
        "generated_color_only_frames": color_only_generated,
        "skipped_frames": skipped,
    }


def run(
    scene_root: Path,
    output_root: Path,
    seed: int,
    copy_originals: bool,
    max_frames: Optional[int],
    overwrite_output: bool,
    target_total_frames: int,
) -> None:
    scene_dirs = collect_scene_dirs(scene_root)
    if not scene_dirs:
        raise FileNotFoundError(
            f"No valid scene directories were found under: {scene_root}"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    summary: List[Dict[str, Any]] = []
    total_input_frames = 0
    total_generated_frames = 0
    total_flip_frames = 0
    total_color_only_frames = 0
    scene_frame_decisions: Dict[str, Dict[str, FrameDecision]] = {}
    total_eligible_frames = 0

    scene_scan_tasks = [(scene_dir, max_frames) for scene_dir in scene_dirs]

    with ProcessPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        for scene_dir, frame_ids, frame_decisions in executor.map(
            _collect_scene_frame_decisions_worker, scene_scan_tasks
        ):
            total_input_frames += len(frame_ids)
            total_eligible_frames += len(frame_decisions)
            scene_frame_decisions[scene_dir.name] = frame_decisions

        target_generated_frames = max(
            0,
            int(target_total_frames) - (total_input_frames if copy_originals else 0),
        )
        target_color_only_frames = max(0, target_generated_frames - total_input_frames)
        scene_color_only_counts = plan_scene_color_only_counts(
            scene_frame_decisions, target_color_only_frames=target_color_only_frames
        )

        scene_tasks = []
        for index, scene_dir in enumerate(scene_dirs, start=1):
            scene_output_dir = output_root / scene_dir.name
            print(f"[{index}/{len(scene_dirs)}] Processing scene: {scene_dir.name}")
            scene_tasks.append(
                (
                    scene_dir,
                    scene_output_dir,
                    seed,
                    copy_originals,
                    max_frames,
                    overwrite_output,
                    scene_color_only_counts.get(scene_dir.name, {}),
                )
            )

        for scene_summary in executor.map(_run_single_scene_worker, scene_tasks):
            summary.append(scene_summary)
            total_generated_frames += int(scene_summary["generated_augmented_frames"])
            total_flip_frames += int(scene_summary["generated_flip_frames"])
            total_color_only_frames += int(scene_summary["generated_color_only_frames"])

    save_json(
        {
            "scene_root": str(scene_root),
            "output_root": str(output_root),
            "scene_count": len(summary),
            "total_input_frames": total_input_frames,
            "eligible_frame_count": total_eligible_frames,
            "target_total_frames": int(target_total_frames),
            "total_generated_augmented_frames": total_generated_frames,
            "total_generated_flip_frames": total_flip_frames,
            "total_generated_color_only_frames": total_color_only_frames,
            "target_color_only_frames": target_color_only_frames,
            "total_output_frames": total_generated_frames + (total_input_frames if copy_originals else 0),
            "point_cloud_included": False,
            "scenes": summary,
        },
        output_root / "dataset_augmentation_summary.json",
    )
    print(f"Processed scenes: {len(summary)}")
    print(f"Total input frames: {total_input_frames}")
    print(f"Eligible frames: {total_eligible_frames}")
    print(f"Target total frames: {int(target_total_frames)}")
    print(f"Target color-only frames: {target_color_only_frames}")
    print(f"Total generated flip frames: {total_flip_frames}")
    print(f"Total generated color-only frames: {total_color_only_frames}")
    print(f"Total generated augmented frames: {total_generated_frames}")
    print(
        "Total output frames: "
        f"{total_generated_frames + (total_input_frames if copy_originals else 0)}"
    )
    print(f"Dataset output root: {output_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-dir",
        type=Path,
        default=Path("fisheye_2wdata"),
        help="Input dataset root containing multiple scene directories, or a single scene directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("fisheye_data_aug"),
        help="Output dataset root. Each scene keeps its original scene name under this directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260415,
        help="Global random seed used to make augmentation deterministic.",
    )
    parser.add_argument(
        "--target-total-frames",
        type=int,
        default=DEFAULT_TARGET_TOTAL_FRAMES,
        help=(
            "Target total number of frames after augmentation. "
            "When originals are copied, this includes original frames."
        ),
    )
    parser.add_argument(
        "--no-copy-originals",
        action="store_true",
        help="Only export augmented samples instead of copying original camera/label files first.",
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
    #从 Octopus 环境变量读取路径
    scene_root = Path(os.environ["SOURCE_DATASET_FILE_DIR"]) / "fisheye_2wdata"
    # scene_root = Path(os.environ["TARGET_RESULT_DIR"]) / "fisheye_2wdata"
    output_root = Path(os.environ["TARGET_RESULT_DIR"]) / "fisheye_data_aug"
    output_root.mkdir(parents=True, exist_ok=True)

    print("使用输入路径:", scene_root)
    print("使用输出路径:", output_root)
    run(
        scene_root=scene_root,
        output_root=output_root,
        # scene_root=args.scene_dir,
        # output_root=args.output_dir,
        seed=args.seed,
        copy_originals=not args.no_copy_originals,
        max_frames=args.max_frames,
        overwrite_output=args.overwrite_output,
        target_total_frames=args.target_total_frames,
    )


if __name__ == "__main__":
    main()
