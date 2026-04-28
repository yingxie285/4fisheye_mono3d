from __future__ import annotations

import os
import sys
from pathlib import Path


PIPELINE_SOURCE_ROOT = Path(os.environ["SOURCE_DATASET_FILE_DIR"])
PIPELINE_TARGET_ROOT = Path(os.environ["TARGET_RESULT_DIR"])


def _enter_stage(stage_name: str, stage_filename: str, source_root: Path, target_root: Path):
    previous_source = os.environ.get("SOURCE_DATASET_FILE_DIR")
    previous_target = os.environ.get("TARGET_RESULT_DIR")
    previous_argv = sys.argv[:]

    os.environ["SOURCE_DATASET_FILE_DIR"] = str(source_root)
    os.environ["TARGET_RESULT_DIR"] = str(target_root)
    sys.argv = [stage_filename]

    print("=" * 80)
    print(f"[{stage_name}] start")
    print(f"[{stage_name}] SOURCE_DATASET_FILE_DIR={source_root}")
    print(f"[{stage_name}] TARGET_RESULT_DIR={target_root}")
    print("=" * 80)

    return previous_source, previous_target, previous_argv


def _leave_stage(stage_name: str, previous_source, previous_target, previous_argv):
    sys.argv = previous_argv
    if previous_source is None:
        os.environ.pop("SOURCE_DATASET_FILE_DIR", None)
    else:
        os.environ["SOURCE_DATASET_FILE_DIR"] = previous_source
    if previous_target is None:
        os.environ.pop("TARGET_RESULT_DIR", None)
    else:
        os.environ["TARGET_RESULT_DIR"] = previous_target

    print(f"[{stage_name}] done")

# Stage: offline_fisheye_camera_only_aug.py

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

offaug_IMAGE_DIRS: Tuple[str, ...] = ('image0', 'image1', 'image2', 'image3')

offaug_METADATA_DIRS: Tuple[str, ...] = ('camera_config', 'position', 'result')

offaug_LIDAR_MIRROR = np.diag([1.0, -1.0, 1.0, 1.0])

offaug_CAMERA_X_MIRROR = np.diag([-1.0, 1.0, 1.0, 1.0])

offaug_TARGET_TAIL_CLASSES = frozenset({'pedestrian', 'bicycle', 'motor', 'stopper'})

offaug_RARE_CLASSES = frozenset({'animal', 'bus', 'construction_vehicle', 'traffic_cone'})

offaug_TAIL_CLASSES = frozenset({'truck', 'trash_bin', 'sign', 'barrier'})

offaug_LONG_TAIL_CLASSES = offaug_RARE_CLASSES | offaug_TAIL_CLASSES

offaug_PARALLEL_WORKERS = 10

offaug_DEFAULT_TARGET_TOTAL_FRAMES = 500 #用5个场景是500    #目标是150000帧

offaug_BASE_TARGET_COLOR_ONLY_COPIES = 4

offaug_BASE_LONG_TAIL_COLOR_ONLY_COPIES = 6

offaug_BASE_TARGET_AND_LONG_TAIL_COLOR_ONLY_COPIES = 8

offaug_TARGET_CLASS_COLOR_PRIORITY = 0.75

offaug_LONG_TAIL_CLASS_COLOR_PRIORITY = 1.5

offaug_PEDESTRIAN_COLOR_PRIORITY_BONUS = 0.75

offaug_TARGET_AND_LONG_TAIL_PRIORITY_BONUS = 2.0

offaug_VERY_LIGHT_PHOTO_CLASSES = frozenset({'bus', 'animal', 'construction_vehicle'})

offaug_LIGHT_PHOTO_CLASSES = frozenset({'pedestrian', 'bicycle', 'motor', 'traffic_cone'})

offaug_COLOR_INTENSITY_CYCLES: Dict[str, Tuple[str, ...]] = {'target_tail': ('light', 'medium', 'strong', 'medium'), 'long_tail': ('medium', 'strong', 'xstrong', 'strong'), 'target_and_long_tail': ('medium', 'strong', 'xstrong', 'strong', 'medium')}

offaug_PHOTO_INTENSITY_SCALES: Dict[str, Dict[str, float]] = {'light': {'magnitude_scale': 0.85, 'probability_scale': 0.9}, 'medium': {'magnitude_scale': 1.0, 'probability_scale': 1.0}, 'strong': {'magnitude_scale': 1.35, 'probability_scale': 1.15}, 'xstrong': {'magnitude_scale': 1.7, 'probability_scale': 1.3}}

offaug_PHOTO_PROFILES: Dict[str, Dict[str, Any]] = {'very_light': {'brightness_px': 8.0, 'contrast_delta': 0.06, 'gamma_delta': 0.05, 'gamma_prob': 0.45, 'wb_delta': 0.04, 'wb_prob': 0.4, 'saturation_delta': 0.06, 'saturation_prob': 0.35, 'shadow_prob': 0.15, 'shadow_strength': (0.88, 0.96), 'fog_prob': 0.1, 'fog_strength': (0.03, 0.08), 'blur_prob': 0.12, 'blur_kernels': (3,), 'noise_prob': 0.1, 'noise_sigma': (1.0, 3.0), 'jpeg_prob': 0.25, 'jpeg_quality': (90, 98)}, 'light': {'brightness_px': 12.0, 'contrast_delta': 0.1, 'gamma_delta': 0.08, 'gamma_prob': 0.55, 'wb_delta': 0.06, 'wb_prob': 0.45, 'saturation_delta': 0.08, 'saturation_prob': 0.4, 'shadow_prob': 0.2, 'shadow_strength': (0.84, 0.94), 'fog_prob': 0.12, 'fog_strength': (0.04, 0.1), 'blur_prob': 0.16, 'blur_kernels': (3,), 'noise_prob': 0.15, 'noise_sigma': (1.0, 4.0), 'jpeg_prob': 0.3, 'jpeg_quality': (88, 96)}, 'medium': {'brightness_px': 18.0, 'contrast_delta': 0.15, 'gamma_delta': 0.12, 'gamma_prob': 0.6, 'wb_delta': 0.08, 'wb_prob': 0.5, 'saturation_delta': 0.1, 'saturation_prob': 0.45, 'shadow_prob': 0.25, 'shadow_strength': (0.78, 0.92), 'fog_prob': 0.15, 'fog_strength': (0.05, 0.12), 'blur_prob': 0.2, 'blur_kernels': (3, 5), 'noise_prob': 0.18, 'noise_sigma': (1.0, 5.0), 'jpeg_prob': 0.35, 'jpeg_quality': (85, 95)}}

offaug_CANONICAL_CLASS_MAP: Dict[str, str] = {
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
class offaug_FrameDecision:
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

def offaug_wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi

def offaug_canonicalize_class(class_name: str) -> str:
    key = str(class_name).strip()
    return offaug_CANONICAL_CLASS_MAP.get(key, key if key.isascii() else 'unknown')

def offaug_load_json(path: Path) -> Any:
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)

def offaug_save_json(data: Any, path: Path) -> None:
    with path.open('w', encoding='utf-8') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)

def offaug_stable_rng(seed: int, *parts: str) -> np.random.Generator:
    digest = hashlib.sha1('|'.join([str(seed), *parts]).encode('utf-8')).digest()
    child_seed = int.from_bytes(digest[:8], 'little', signed=False)
    return np.random.default_rng(child_seed)

def offaug__collect_scene_frame_decisions_worker(args: Tuple[Path, Optional[int]]) -> Tuple[Path, List[str], Dict[str, offaug_FrameDecision]]:
    scene_dir, max_frames = args
    frame_ids, frame_decisions = offaug_collect_scene_frame_decisions(scene_dir, max_frames=max_frames)
    return (scene_dir, frame_ids, frame_decisions)

def offaug__run_single_scene_worker(args: Tuple[Path, Path, int, bool, Optional[int], bool, Mapping[str, int]]) -> Dict[str, Any]:
    return offaug_run_single_scene(*args)

def offaug_ensure_output_root(scene_dir: Path, output_dir: Path, copy_originals: bool, overwrite_output: bool) -> None:
    if output_dir.exists():
        if not overwrite_output:
            raise FileExistsError(f'Output directory already exists: {output_dir}. Please remove it, pass --overwrite-output, or choose another --output-dir.')
        shutil.rmtree(output_dir)
    if copy_originals:
        try:
            output_dir.relative_to(scene_dir)
        except ValueError:
            return
        raise ValueError('When --copy-originals is enabled, --output-dir must not be inside --scene-dir.')

def offaug_find_single_file(directory: Path, stem: str) -> Optional[Path]:
    matches = [path for path in directory.glob(f'{stem}.*') if path.is_file()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        preferred_suffixes = ['.jpeg', '.jpg', '.png', '.json']
        for suffix in preferred_suffixes:
            for path in matches:
                if path.suffix.lower() == suffix:
                    return path
        return sorted(matches)[0]
    return None

def offaug_collect_frame_ids(scene_dir: Path) -> List[str]:
    result_dir = scene_dir / 'result'
    frame_ids = sorted((path.stem for path in result_dir.glob('*.json')))
    valid_ids: List[str] = []
    for frame_id in frame_ids:
        required = [scene_dir / 'result' / f'{frame_id}.json', scene_dir / 'camera_config' / f'{frame_id}.json', scene_dir / 'position' / f'{frame_id}.json']
        if not all((path.exists() for path in required)):
            continue
        image_paths = [offaug_find_single_file(scene_dir / image_dir, frame_id) for image_dir in offaug_IMAGE_DIRS]
        if any((path is None for path in image_paths)):
            continue
        valid_ids.append(frame_id)
    return valid_ids

def offaug_collect_object_counts(label_data: Mapping[str, Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for obj in label_data.get('objects', []):
        if obj.get('type', '3D_BOX') != '3D_BOX':
            continue
        class_name = offaug_canonicalize_class(obj.get('className', ''))
        if class_name == 'unknown':
            continue
        counts[class_name] += 1
    return counts

def offaug_compute_base_color_only_count(has_target_tail: bool, has_long_tail: bool) -> int:
    if has_target_tail and has_long_tail:
        return offaug_BASE_TARGET_AND_LONG_TAIL_COLOR_ONLY_COPIES
    if has_long_tail:
        return offaug_BASE_LONG_TAIL_COLOR_ONLY_COPIES
    if has_target_tail:
        return offaug_BASE_TARGET_COLOR_ONLY_COPIES
    return 0

def offaug_compute_color_priority(class_counts: Mapping[str, int], target_tail_count: int, long_tail_count: int) -> float:
    has_target_tail = target_tail_count > 0
    has_long_tail = long_tail_count > 0
    priority = float(offaug_compute_base_color_only_count(has_target_tail, has_long_tail))
    priority += float(target_tail_count) * offaug_TARGET_CLASS_COLOR_PRIORITY
    priority += float(long_tail_count) * offaug_LONG_TAIL_CLASS_COLOR_PRIORITY
    if class_counts.get('pedestrian', 0) > 0:
        priority += offaug_PEDESTRIAN_COLOR_PRIORITY_BONUS
    if has_target_tail and has_long_tail:
        priority += offaug_TARGET_AND_LONG_TAIL_PRIORITY_BONUS
    return priority

def offaug_decide_frame_augmentation(label_data: Mapping[str, Any]) -> Optional[offaug_FrameDecision]:
    class_counts = offaug_collect_object_counts(label_data)
    if not class_counts:
        return None
    canonical_classes = sorted(class_counts)
    class_set = set(canonical_classes)
    target_tail_count = sum((class_counts.get(name, 0) for name in offaug_TARGET_TAIL_CLASSES))
    long_tail_count = sum((class_counts.get(name, 0) for name in offaug_LONG_TAIL_CLASSES))
    has_target_tail = target_tail_count > 0
    has_long_tail = long_tail_count > 0
    if not has_target_tail and (not has_long_tail):
        return None
    if has_target_tail and has_long_tail:
        trigger = 'target_and_long_tail'
    elif has_long_tail:
        trigger = 'long_tail'
    else:
        trigger = 'target_tail'
    if class_set & offaug_VERY_LIGHT_PHOTO_CLASSES:
        profile = 'very_light'
    elif class_set & offaug_LIGHT_PHOTO_CLASSES:
        profile = 'light'
    else:
        profile = 'medium'
    return offaug_FrameDecision(trigger=trigger, profile=profile, canonical_classes=tuple(canonical_classes), class_counts=dict(sorted(class_counts.items())), has_target_tail=has_target_tail, has_long_tail=has_long_tail, target_tail_count=target_tail_count, long_tail_count=long_tail_count, base_color_only_count=offaug_compute_base_color_only_count(has_target_tail, has_long_tail), color_priority=round(offaug_compute_color_priority(class_counts, target_tail_count, long_tail_count), 3))

def offaug_horizontal_flip_bbox(box: Sequence[float], width: int) -> List[float]:
    x1, y1, x2, y2 = [float(value) for value in box]
    new_x1 = float(width - 1 - x2)
    new_x2 = float(width - 1 - x1)
    x_min, x_max = sorted((new_x1, new_x2))
    x_min = min(max(x_min, 0.0), float(width - 1))
    x_max = min(max(x_max, 0.0), float(width - 1))
    return [x_min, y1, x_max, y2]

def offaug_augment_label_json(label_data: Mapping[str, Any], image_widths: Mapping[str, int], do_flip: bool) -> Dict[str, Any]:
    augmented = copy.deepcopy(label_data)
    if not do_flip:
        return augmented
    for obj in augmented.get('objects', []):
        contour = obj.get('contour', {})
        center3d = contour.get('center3D')
        if isinstance(center3d, MutableMapping) and 'y' in center3d:
            center3d['y'] = -float(center3d['y'])
        rotation3d = contour.get('rotation3D')
        if isinstance(rotation3d, MutableMapping) and 'z' in rotation3d:
            rotation3d['z'] = offaug_wrap_to_pi(-float(rotation3d['z']))
        bbox_2d = obj.get('2D_bbox')
        if isinstance(bbox_2d, MutableMapping):
            for image_key, box in list(bbox_2d.items()):
                width = image_widths.get(image_key)
                if width is None or not isinstance(box, Sequence) or len(box) != 4:
                    continue
                bbox_2d[image_key] = offaug_horizontal_flip_bbox(box, width)
    return augmented

def offaug_decode_camera_external(camera_cfg: Mapping[str, Any]) -> np.ndarray:
    matrix = np.asarray(camera_cfg['camera_external'], dtype=np.float64).reshape(4, 4)
    row_major = str(camera_cfg.get('rowMajor', 'true')).lower() == 'true'
    return matrix if row_major else matrix.T

def offaug_encode_camera_external(matrix: np.ndarray, row_major_flag: Any) -> List[float]:
    row_major = str(row_major_flag).lower() == 'true'
    serial = matrix if row_major else matrix.T
    return [float(value) for value in serial.reshape(-1).tolist()]

def offaug_augment_camera_config(camera_config: Sequence[Mapping[str, Any]], do_flip: bool) -> List[Dict[str, Any]]:
    augmented = copy.deepcopy(list(camera_config))
    if not do_flip:
        return augmented
    for camera_cfg in augmented:
        width = int(camera_cfg['width'])
        camera_internal = camera_cfg.get('camera_internal', {})
        if 'cx' in camera_internal:
            camera_internal['cx'] = float(width - 1 - float(camera_internal['cx']))
        matrix = offaug_decode_camera_external(camera_cfg)
        mirrored_matrix = offaug_CAMERA_X_MIRROR @ matrix @ offaug_LIDAR_MIRROR
        camera_cfg['camera_external'] = offaug_encode_camera_external(mirrored_matrix, camera_cfg.get('rowMajor', 'false'))
    return augmented

def offaug_augment_position_json(position_data: Mapping[str, Any], new_frame_id: str) -> Dict[str, Any]:
    augmented = copy.deepcopy(position_data)
    augmented['name'] = new_frame_id
    return augmented

def offaug_make_identity_photometric_plan(profile_name: str) -> Dict[str, Any]:
    return {'profile': profile_name, 'intensity': 'identity', 'brightness_shift': 0.0, 'contrast_scale': 1.0, 'gamma': None, 'wb_gains_bgr': None, 'saturation_scale': None, 'shadow_strength': None, 'fog_strength': None, 'blur_kernel': 0, 'noise_sigma': 0.0, 'jpeg_quality': 100}

def offaug_scale_probability(probability: float, scale: float) -> float:
    return min(max(probability * scale, 0.0), 1.0)

def offaug_scale_range_from_neutral(values: Sequence[float], neutral: float, scale: float) -> Tuple[float, float]:
    scaled_values = [neutral + (float(value) - neutral) * scale for value in values]
    low, high = sorted(scaled_values)
    return (low, high)

def offaug_sample_photometric_plan(rng: np.random.Generator, profile_name: str, intensity_name: str) -> Dict[str, Any]:
    if intensity_name == 'identity':
        return offaug_make_identity_photometric_plan(profile_name)
    cfg = offaug_PHOTO_PROFILES[profile_name]
    intensity_cfg = offaug_PHOTO_INTENSITY_SCALES[intensity_name]
    magnitude_scale = float(intensity_cfg['magnitude_scale'])
    probability_scale = float(intensity_cfg['probability_scale'])
    brightness_shift = float(rng.uniform(-cfg['brightness_px'] * magnitude_scale, cfg['brightness_px'] * magnitude_scale))
    contrast_delta = float(cfg['contrast_delta']) * magnitude_scale
    contrast_scale = float(rng.uniform(1.0 - contrast_delta, 1.0 + contrast_delta))
    gamma_value: Optional[float] = None
    gamma_delta = float(cfg['gamma_delta']) * magnitude_scale
    if rng.random() < offaug_scale_probability(float(cfg['gamma_prob']), probability_scale):
        gamma_value = float(rng.uniform(1.0 - gamma_delta, 1.0 + gamma_delta))
    wb_gains: Optional[List[float]] = None
    wb_delta = float(cfg['wb_delta']) * magnitude_scale
    if rng.random() < offaug_scale_probability(float(cfg['wb_prob']), probability_scale):
        wb_gains = [float(rng.uniform(1.0 - wb_delta, 1.0 + wb_delta)) for _ in range(3)]
    saturation_scale: Optional[float] = None
    saturation_delta = float(cfg['saturation_delta']) * magnitude_scale
    if rng.random() < offaug_scale_probability(float(cfg['saturation_prob']), probability_scale):
        saturation_scale = float(rng.uniform(1.0 - saturation_delta, 1.0 + saturation_delta))
    shadow_strength: Optional[float] = None
    if rng.random() < offaug_scale_probability(float(cfg['shadow_prob']), probability_scale):
        shadow_range = offaug_scale_range_from_neutral(cfg['shadow_strength'], neutral=1.0, scale=magnitude_scale)
        shadow_strength = float(rng.uniform(*shadow_range))
    fog_strength: Optional[float] = None
    if rng.random() < offaug_scale_probability(float(cfg['fog_prob']), probability_scale):
        fog_low, fog_high = [float(value) * magnitude_scale for value in cfg['fog_strength']]
        fog_strength = float(rng.uniform(fog_low, fog_high))
    blur_kernel = 0
    if rng.random() < offaug_scale_probability(float(cfg['blur_prob']), probability_scale):
        blur_kernels = sorted({int(value) for value in cfg['blur_kernels']})
        if magnitude_scale >= 1.35:
            blur_kernels.append(5)
        if magnitude_scale >= 1.6:
            blur_kernels.append(7)
        blur_kernel = int(rng.choice(sorted(set(blur_kernels))))
    noise_sigma = 0.0
    if rng.random() < offaug_scale_probability(float(cfg['noise_prob']), probability_scale):
        noise_low, noise_high = [float(value) * magnitude_scale for value in cfg['noise_sigma']]
        noise_sigma = float(rng.uniform(noise_low, noise_high))
    jpeg_quality = 100
    if rng.random() < offaug_scale_probability(float(cfg['jpeg_prob']), probability_scale):
        base_quality = int(rng.integers(cfg['jpeg_quality'][0], cfg['jpeg_quality'][1] + 1))
        jpeg_quality = int(np.clip(round(100.0 - (100.0 - base_quality) * magnitude_scale), 1, 100))
    return {'profile': profile_name, 'intensity': intensity_name, 'brightness_shift': brightness_shift, 'contrast_scale': contrast_scale, 'gamma': gamma_value, 'wb_gains_bgr': wb_gains, 'saturation_scale': saturation_scale, 'shadow_strength': shadow_strength, 'fog_strength': fog_strength, 'blur_kernel': blur_kernel, 'noise_sigma': noise_sigma, 'jpeg_quality': jpeg_quality}

def offaug_adjust_brightness_contrast(image: np.ndarray, contrast_scale: float, brightness_shift: float) -> np.ndarray:
    out = image.astype(np.float32) * contrast_scale + brightness_shift
    return np.clip(out, 0, 255).astype(np.uint8)

def offaug_adjust_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    gamma = max(gamma, 1e-06)
    inverse_gamma = 1.0 / gamma
    lut = np.array([(index / 255.0) ** inverse_gamma * 255.0 for index in range(256)], dtype=np.uint8)
    return cv2.LUT(image, lut)

def offaug_adjust_white_balance(image: np.ndarray, gains_bgr: Sequence[float]) -> np.ndarray:
    gains = np.asarray(gains_bgr, dtype=np.float32).reshape(1, 1, 3)
    out = image.astype(np.float32) * gains
    return np.clip(out, 0, 255).astype(np.uint8)

def offaug_adjust_saturation(image: np.ndarray, scale: float) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= scale
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def offaug_apply_shadow(image: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    height, width = image.shape[:2]
    polygon = np.array([[0, int(rng.integers(0, max(1, height // 2)))], [width - 1, int(rng.integers(0, max(1, height // 2)))], [width - 1, int(rng.integers(max(1, height // 2), height))], [0, int(rng.integers(max(1, height // 2), height))]], dtype=np.int32)
    mask = np.zeros((height, width), dtype=np.float32)
    cv2.fillConvexPoly(mask, polygon, 1.0)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(width, height) / 12.0)
    shadow_gain = 1.0 - mask[..., None] * (1.0 - strength)
    out = image.astype(np.float32) * shadow_gain
    return np.clip(out, 0, 255).astype(np.uint8)

def offaug_apply_fog(image: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    height, width = image.shape[:2]
    center_x = float(rng.uniform(0, width))
    center_y = float(rng.uniform(0, height))
    radius = float(rng.uniform(max(width, height) * 0.55, max(width, height) * 1.1))
    yy, xx = np.mgrid[0:height, 0:width]
    dist = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2).astype(np.float32)
    fog_mask = np.exp(-dist ** 2 / (2.0 * radius ** 2))
    fog_mask = fog_mask / max(float(fog_mask.max()), 1e-06) * strength
    fog_mask = fog_mask[..., None]
    white = np.full_like(image, 255, dtype=np.float32)
    out = image.astype(np.float32) * (1.0 - fog_mask) + white * fog_mask
    return np.clip(out, 0, 255).astype(np.uint8)

def offaug_add_gaussian_noise(image: np.ndarray, rng: np.random.Generator, sigma: float) -> np.ndarray:
    noise = rng.normal(0.0, sigma, size=image.shape).astype(np.float32)
    out = image.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)

def offaug_simulate_jpeg(image: np.ndarray, quality: int) -> np.ndarray:
    quality = int(np.clip(quality, 1, 100))
    success, encoded = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        return image
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return image if decoded is None else decoded

def offaug_apply_photometric_plan(image: np.ndarray, plan: Mapping[str, Any], image_rng: np.random.Generator) -> np.ndarray:
    out = offaug_adjust_brightness_contrast(image, float(plan['contrast_scale']), float(plan['brightness_shift']))
    if plan.get('gamma') is not None:
        out = offaug_adjust_gamma(out, float(plan['gamma']))
    if plan.get('wb_gains_bgr') is not None:
        out = offaug_adjust_white_balance(out, plan['wb_gains_bgr'])
    if plan.get('saturation_scale') is not None:
        out = offaug_adjust_saturation(out, float(plan['saturation_scale']))
    if plan.get('shadow_strength') is not None:
        out = offaug_apply_shadow(out, image_rng, float(plan['shadow_strength']))
    if plan.get('fog_strength') is not None:
        out = offaug_apply_fog(out, image_rng, float(plan['fog_strength']))
    blur_kernel = int(plan.get('blur_kernel', 0))
    if blur_kernel > 1:
        out = cv2.GaussianBlur(out, (blur_kernel, blur_kernel), sigmaX=0.0)
    noise_sigma = float(plan.get('noise_sigma', 0.0))
    if noise_sigma > 0:
        out = offaug_add_gaussian_noise(out, image_rng, noise_sigma)
    jpeg_quality = int(plan.get('jpeg_quality', 100))
    if jpeg_quality < 100:
        out = offaug_simulate_jpeg(out, jpeg_quality)
    return out

def offaug_choose_color_intensity(decision: offaug_FrameDecision, color_only_index: int) -> str:
    cycle = offaug_COLOR_INTENSITY_CYCLES[decision.trigger]
    return cycle[color_only_index % len(cycle)]

def offaug_load_frame_images(scene_dir: Path, frame_id: str) -> Dict[str, Tuple[np.ndarray, str]]:
    source_images: Dict[str, Tuple[np.ndarray, str]] = {}
    for image_dir in offaug_IMAGE_DIRS:
        src_path = offaug_find_single_file(scene_dir / image_dir, frame_id)
        if src_path is None:
            raise FileNotFoundError(f'Missing {image_dir} image for frame {frame_id}')
        image = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f'Failed to read image: {src_path}')
        source_images[image_dir] = (image, src_path.suffix.lower())
    return source_images

def offaug_load_frame_metadata(scene_dir: Path, frame_id: str) -> Tuple[Mapping[str, Any], Sequence[Mapping[str, Any]], Mapping[str, Any]]:
    return (offaug_load_json(scene_dir / 'result' / f'{frame_id}.json'), offaug_load_json(scene_dir / 'camera_config' / f'{frame_id}.json'), offaug_load_json(scene_dir / 'position' / f'{frame_id}.json'))

def offaug_save_augmented_images(source_images: Mapping[str, Tuple[np.ndarray, str]], output_dir: Path, new_frame_id: str, do_flip: bool, photometric_plan: Mapping[str, Any], seed: int) -> None:
    for image_dir, (source_image, suffix) in source_images.items():
        image = source_image.copy()
        if do_flip:
            image = cv2.flip(image, 1)
        image_rng = offaug_stable_rng(seed, new_frame_id, image_dir)
        image = offaug_apply_photometric_plan(image, photometric_plan, image_rng)
        dst_path = output_dir / image_dir / f'{new_frame_id}{suffix}'
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(dst_path), image):
            raise RuntimeError(f'Failed to write image: {dst_path}')

def offaug_build_image_widths(source_images: Mapping[str, Tuple[np.ndarray, str]]) -> Dict[str, int]:
    return {image_dir: int(image.shape[1]) for image_dir, (image, _) in source_images.items()}

def offaug_save_augmented_metadata_files(output_dir: Path, new_frame_id: str, label_data: Mapping[str, Any], camera_config: Sequence[Mapping[str, Any]], position_data: Mapping[str, Any], do_flip: bool, image_widths: Mapping[str, int]) -> None:
    offaug_save_json(offaug_augment_label_json(label_data, image_widths=image_widths, do_flip=do_flip), output_dir / 'result' / f'{new_frame_id}.json')
    offaug_save_json(offaug_augment_camera_config(camera_config, do_flip=do_flip), output_dir / 'camera_config' / f'{new_frame_id}.json')
    offaug_save_json(offaug_augment_position_json(position_data, new_frame_id=new_frame_id), output_dir / 'position' / f'{new_frame_id}.json')

def offaug_allocate_numeric_augmented_ids(frame_id: str, count: int, used_ids: set[int]) -> List[str]:
    if not frame_id.isdigit():
        return [f'{frame_id}_{index + 1}' for index in range(count)]
    next_id = int(frame_id)
    new_ids: List[str] = []
    while len(new_ids) < count:
        next_id += 1
        if next_id in used_ids:
            continue
        used_ids.add(next_id)
        new_ids.append(str(next_id))
    return new_ids

def offaug_create_empty_scene_layout(output_dir: Path) -> None:
    for folder in (*offaug_METADATA_DIRS, *offaug_IMAGE_DIRS):
        (output_dir / folder).mkdir(parents=True, exist_ok=True)

def offaug_copy_scene_without_point_cloud(scene_dir: Path, output_dir: Path) -> None:
    offaug_create_empty_scene_layout(output_dir)
    for folder in (*offaug_METADATA_DIRS, *offaug_IMAGE_DIRS):
        src_dir = scene_dir / folder
        dst_dir = output_dir / folder
        if not src_dir.exists():
            continue
        for src_path in src_dir.iterdir():
            if src_path.is_file():
                shutil.copy2(src_path, dst_dir / src_path.name)

def offaug_build_output_dir(scene_dir: Path, output_dir: Path, copy_originals: bool, overwrite_output: bool) -> None:
    offaug_ensure_output_root(scene_dir, output_dir, copy_originals, overwrite_output)
    if copy_originals:
        offaug_copy_scene_without_point_cloud(scene_dir, output_dir)
    else:
        offaug_create_empty_scene_layout(output_dir)

def offaug_maybe_limit_frame_ids(frame_ids: List[str], max_frames: Optional[int]) -> List[str]:
    if max_frames is None:
        return frame_ids
    return frame_ids[:max(0, max_frames)]

def offaug_collect_scene_frame_decisions(scene_dir: Path, max_frames: Optional[int]) -> Tuple[List[str], Dict[str, offaug_FrameDecision]]:
    frame_ids = offaug_maybe_limit_frame_ids(offaug_collect_frame_ids(scene_dir), max_frames=max_frames)
    frame_decisions: Dict[str, offaug_FrameDecision] = {}
    for frame_id in frame_ids:
        label_data = offaug_load_json(scene_dir / 'result' / f'{frame_id}.json')
        decision = offaug_decide_frame_augmentation(label_data)
        if decision is not None:
            frame_decisions[frame_id] = decision
    return (frame_ids, frame_decisions)

def offaug_proportional_allocation(weights: Sequence[float], total_count: int) -> List[int]:
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
    order = sorted(range(len(raw_counts)), key=lambda index: (-(raw_counts[index] - counts[index]), -safe_weights[index], index))
    for index in order[:remaining]:
        counts[index] += 1
    return counts

def offaug_plan_scene_color_only_counts(scene_decisions: Mapping[str, Dict[str, offaug_FrameDecision]], target_color_only_frames: int) -> Dict[str, Dict[str, int]]:
    frame_keys: List[Tuple[str, str]] = []
    decisions: List[offaug_FrameDecision] = []
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
        extra_color_only_counts = offaug_proportional_allocation([decision.color_priority for decision in decisions], remaining_color_only_frames)
        color_only_counts = [base_count + extra_count for base_count, extra_count in zip(color_only_counts, extra_color_only_counts)]
    else:
        color_only_counts = offaug_proportional_allocation(base_color_only_counts, required_color_only_frames)
    scene_color_only_counts: Dict[str, Dict[str, int]] = {}
    for (scene_name, frame_id), color_only_count in zip(frame_keys, color_only_counts):
        scene_color_only_counts.setdefault(scene_name, {})[frame_id] = color_only_count
    return scene_color_only_counts

def offaug_is_scene_dir(path: Path) -> bool:
    required_dirs = (*offaug_METADATA_DIRS, *offaug_IMAGE_DIRS)
    return path.is_dir() and all(((path / folder).is_dir() for folder in required_dirs))

def offaug_collect_scene_dirs(scene_root: Path) -> List[Path]:
    if offaug_is_scene_dir(scene_root):
        return [scene_root]
    if not scene_root.is_dir():
        raise FileNotFoundError(f'Scene root does not exist: {scene_root}')
    scene_dirs = [path for path in scene_root.iterdir() if offaug_is_scene_dir(path)]
    return sorted(scene_dirs, key=lambda path: path.name)

def offaug_run_single_scene(scene_dir: Path, output_dir: Path, seed: int, copy_originals: bool, max_frames: Optional[int], overwrite_output: bool, color_only_counts: Mapping[str, int]) -> Dict[str, Any]:
    offaug_build_output_dir(scene_dir=scene_dir, output_dir=output_dir, copy_originals=copy_originals, overwrite_output=overwrite_output)
    frame_ids, frame_decisions = offaug_collect_scene_frame_decisions(scene_dir, max_frames=max_frames)
    used_numeric_ids = {int(frame_id) for frame_id in frame_ids if frame_id.isdigit()}
    manifest: List[Dict[str, Any]] = []
    generated = 0
    flip_generated = 0
    color_only_generated = 0
    skipped = 0
    for frame_id in frame_ids:
        decision = frame_decisions.get(frame_id)
        color_only_count = int(color_only_counts.get(frame_id, 0)) if decision is not None else 0
        new_frame_ids = offaug_allocate_numeric_augmented_ids(frame_id=frame_id, count=1 + color_only_count, used_ids=used_numeric_ids)
        source_images = offaug_load_frame_images(scene_dir, frame_id)
        image_widths = offaug_build_image_widths(source_images)
        label_data, camera_config, position_data = offaug_load_frame_metadata(scene_dir, frame_id)
        frame_class_counts = dict(sorted(offaug_collect_object_counts(label_data).items()))
        if decision is None:
            skipped += 1
            flip_trigger = 'all_frame_flip'
            flip_profile_name = 'identity'
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
        flip_plan = offaug_make_identity_photometric_plan(flip_profile_name)
        offaug_save_augmented_images(source_images=source_images, output_dir=output_dir, new_frame_id=flip_frame_id, do_flip=True, photometric_plan=flip_plan, seed=seed)
        offaug_save_augmented_metadata_files(output_dir=output_dir, new_frame_id=flip_frame_id, label_data=label_data, camera_config=camera_config, position_data=position_data, do_flip=True, image_widths=image_widths)
        manifest.append({'new_frame_id': flip_frame_id, 'source_frame_id': frame_id, 'augment_index': 0, 'augmentation_type': 'flip', 'trigger': flip_trigger, 'profile': flip_profile_name, 'canonical_classes': flip_canonical_classes, 'class_counts': flip_class_counts, 'target_tail_count': flip_target_tail_count, 'long_tail_count': flip_long_tail_count, 'flip_horizontal': True, 'photometric_plan': flip_plan})
        generated += 1
        flip_generated += 1
        for color_idx in range(color_only_count):
            new_frame_id = new_frame_ids[color_idx + 1]
            aug_rng = offaug_stable_rng(seed, frame_id, 'color_only', str(color_idx))
            intensity_name = offaug_choose_color_intensity(decision, color_idx)
            photometric_plan = offaug_sample_photometric_plan(aug_rng, decision.profile, intensity_name)
            offaug_save_augmented_images(source_images=source_images, output_dir=output_dir, new_frame_id=new_frame_id, do_flip=False, photometric_plan=photometric_plan, seed=seed)
            offaug_save_augmented_metadata_files(output_dir=output_dir, new_frame_id=new_frame_id, label_data=label_data, camera_config=camera_config, position_data=position_data, do_flip=False, image_widths=image_widths)
            manifest.append({'new_frame_id': new_frame_id, 'source_frame_id': frame_id, 'augment_index': color_idx + 1, 'augmentation_type': 'color_only', 'trigger': decision.trigger, 'profile': decision.profile, 'canonical_classes': list(decision.canonical_classes), 'class_counts': decision.class_counts, 'target_tail_count': decision.target_tail_count, 'long_tail_count': decision.long_tail_count, 'flip_horizontal': False, 'photometric_plan': photometric_plan})
            generated += 1
            color_only_generated += 1
    offaug_save_json({'scene_dir': str(scene_dir), 'output_dir': str(output_dir), 'seed': seed, 'copy_originals': copy_originals, 'input_frame_count': len(frame_ids), 'eligible_frame_count': len(frame_decisions), 'generated_augmented_frames': generated, 'generated_flip_frames': flip_generated, 'generated_color_only_frames': color_only_generated, 'skipped_frames': skipped, 'point_cloud_included': False, 'manifest': manifest, 'notes': ['This camera-only export does not copy or generate point_cloud files.', 'Every input frame always gets exactly one horizontal flip sample.', 'Only frames containing pedestrian/bicycle/motor or long-tail classes receive extra color-only samples.', 'Color-only samples only modify image appearance and keep label/calibration geometry unchanged.', 'position JSON is copied with only the top-level name updated.', 'camera_config is mirrored only when horizontal flip is applied.', '2D_bbox is mirrored within the same image, without swapping camera folders.', 'Augmented frame ids are new numeric ids allocated after each source frame id.']}, output_dir / 'augmentation_manifest.json')
    print(f'Input frames considered: {len(frame_ids)}')
    print(f'Eligible frames (target/long-tail): {len(frame_decisions)}')
    print(f'Frames without target/long-tail class (flip-only): {skipped}')
    print(f'Augmented flip frames generated: {flip_generated}')
    print(f'Augmented color-only frames generated: {color_only_generated}')
    print(f'Augmented frames generated: {generated}')
    print('Point clouds copied/generated: 0')
    print(f'Saved to: {output_dir}')
    return {'scene_name': scene_dir.name, 'scene_dir': str(scene_dir), 'output_dir': str(output_dir), 'input_frame_count': len(frame_ids), 'eligible_frame_count': len(frame_decisions), 'generated_augmented_frames': generated, 'generated_flip_frames': flip_generated, 'generated_color_only_frames': color_only_generated, 'skipped_frames': skipped}

def offaug_run(scene_root: Path, output_root: Path, seed: int, copy_originals: bool, max_frames: Optional[int], overwrite_output: bool, target_total_frames: int) -> None:
    scene_dirs = offaug_collect_scene_dirs(scene_root)
    if not scene_dirs:
        raise FileNotFoundError(f'No valid scene directories were found under: {scene_root}')
    output_root.mkdir(parents=True, exist_ok=True)
    summary: List[Dict[str, Any]] = []
    total_input_frames = 0
    total_generated_frames = 0
    total_flip_frames = 0
    total_color_only_frames = 0
    scene_frame_decisions: Dict[str, Dict[str, offaug_FrameDecision]] = {}
    total_eligible_frames = 0
    scene_scan_tasks = [(scene_dir, max_frames) for scene_dir in scene_dirs]
    with ProcessPoolExecutor(max_workers=offaug_PARALLEL_WORKERS) as executor:
        for scene_dir, frame_ids, frame_decisions in executor.map(offaug__collect_scene_frame_decisions_worker, scene_scan_tasks):
            total_input_frames += len(frame_ids)
            total_eligible_frames += len(frame_decisions)
            scene_frame_decisions[scene_dir.name] = frame_decisions
        target_generated_frames = max(0, int(target_total_frames) - (total_input_frames if copy_originals else 0))
        target_color_only_frames = max(0, target_generated_frames - total_input_frames)
        scene_color_only_counts = offaug_plan_scene_color_only_counts(scene_frame_decisions, target_color_only_frames=target_color_only_frames)
        scene_tasks = []
        for index, scene_dir in enumerate(scene_dirs, start=1):
            scene_output_dir = output_root / scene_dir.name
            print(f'[{index}/{len(scene_dirs)}] Processing scene: {scene_dir.name}')
            scene_tasks.append((scene_dir, scene_output_dir, seed, copy_originals, max_frames, overwrite_output, scene_color_only_counts.get(scene_dir.name, {})))
        for scene_summary in executor.map(offaug__run_single_scene_worker, scene_tasks):
            summary.append(scene_summary)
            total_generated_frames += int(scene_summary['generated_augmented_frames'])
            total_flip_frames += int(scene_summary['generated_flip_frames'])
            total_color_only_frames += int(scene_summary['generated_color_only_frames'])
    offaug_save_json({'scene_root': str(scene_root), 'output_root': str(output_root), 'scene_count': len(summary), 'total_input_frames': total_input_frames, 'eligible_frame_count': total_eligible_frames, 'target_total_frames': int(target_total_frames), 'total_generated_augmented_frames': total_generated_frames, 'total_generated_flip_frames': total_flip_frames, 'total_generated_color_only_frames': total_color_only_frames, 'target_color_only_frames': target_color_only_frames, 'total_output_frames': total_generated_frames + (total_input_frames if copy_originals else 0), 'point_cloud_included': False, 'scenes': summary}, output_root / 'dataset_augmentation_summary.json')
    print(f'Processed scenes: {len(summary)}')
    print(f'Total input frames: {total_input_frames}')
    print(f'Eligible frames: {total_eligible_frames}')
    print(f'Target total frames: {int(target_total_frames)}')
    print(f'Target color-only frames: {target_color_only_frames}')
    print(f'Total generated flip frames: {total_flip_frames}')
    print(f'Total generated color-only frames: {total_color_only_frames}')
    print(f'Total generated augmented frames: {total_generated_frames}')
    print(f'Total output frames: {total_generated_frames + (total_input_frames if copy_originals else 0)}')
    print(f'Dataset output root: {output_root}')

def offaug_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-dir', type=Path, default=Path('fisheye_2wdata'), help='Input dataset root containing multiple scene directories, or a single scene directory.')
    parser.add_argument('--output-dir', type=Path, default=Path('fisheye_data_aug'), help='Output dataset root. Each scene keeps its original scene name under this directory.')
    parser.add_argument('--seed', type=int, default=20260415, help='Global random seed used to make augmentation deterministic.')
    parser.add_argument('--target-total-frames', type=int, default=offaug_DEFAULT_TARGET_TOTAL_FRAMES, help='Target total number of frames after augmentation. When originals are copied, this includes original frames.')
    parser.add_argument('--no-copy-originals', action='store_true', help='Only export augmented samples instead of copying original camera/label files first.')
    parser.add_argument('--max-frames', type=int, default=None, help='Optional debug limit for how many source frames to scan.')
    parser.add_argument('--overwrite-output', action='store_true', help='Delete and recreate --output-dir if it already exists.')
    return parser.parse_args()

def offaug_main() -> None:
    args = offaug_parse_args()
    scene_root = Path(os.environ['SOURCE_DATASET_FILE_DIR']) / 'fisheye_2wdata'
    output_root = Path(os.environ['TARGET_RESULT_DIR']) / 'fisheye_data_aug'
    output_root.mkdir(parents=True, exist_ok=True)
    print('使用输入路径:', scene_root)
    print('使用输出路径:', output_root)
    offaug_run(scene_root=scene_root, output_root=output_root, seed=args.seed, copy_originals=not args.no_copy_originals, max_frames=args.max_frames, overwrite_output=args.overwrite_output, target_total_frames=args.target_total_frames)

def offaug_stage_entry() -> None:
    offaug_main()


def run_stage_offline_fisheye_camera_only_aug(source_root: Path, target_root: Path) -> None:
    previous_source, previous_target, previous_argv = _enter_stage(
        stage_name="offline_fisheye_camera_only_aug",
        stage_filename="offline_fisheye_camera_only_aug.py",
        source_root=source_root,
        target_root=target_root,
    )
    try:
        offaug_stage_entry()
    finally:
        _leave_stage("offline_fisheye_camera_only_aug", previous_source, previous_target, previous_argv)

# Stage: 40CPU_gt_4fisheye_camera_cord.py
import json

import cv2

import numpy as np

import os

from natsort import natsorted

import math

import shutil

from scipy.spatial.transform import Rotation as R

from multiprocessing import Pool

from pathlib import Path

gtcam_category_info = [{'name': 'car', 'description': '普通轿车, 敞篷轿车, SUV, MPV,面包车,皮卡,警车,救护车,房车,其他机动车,未知机动车,未知机动车车轮,未知机动车车灯'}, {'name': 'truck', 'description': '房车,普通小型货车,箱式小型货车,普通大型货车,轿运车'}, {'name': 'bus', 'description': '客车,校车, 普通公交车, 铰链公交车, 有轨电车,无轨电车'}, {'name': 'construction_vehicle', 'description': '消防车,清洁车,工程车,拖车,拖拉机,叉车,油罐车'}, {'name': 'pedestrian', 'description': '成人,儿童,交警,环卫工人,道路施工人员'}, {'name': 'motor', 'description': '两轮摩托,三轮摩托'}, {'name': 'bicycle', 'description': '两轮电动车,三轮电动车,两轮自行车,三轮自行车,滑板车,平衡车,婴儿车,轮椅,平板小推车,超市购物车,手推车,其他非机动车,未知非机动车,非机动车组'}, {'name': 'animal', 'description': '小型动物,大型动物'}, {'name': 'traffic_cone', 'description': '锥桶'}, {'name': 'barrier', 'description': '防撞桶,路桩,石墩,水马,柱子,石块,树枝/树杈,空中漂浮物,路坑/水洼,其他静态障碍物'}, {'name': 'stopper', 'description': '车位停止器,车位锁,减速带'}, {'name': 'trash_bin', 'description': '垃圾桶,灭火器,箱子'}, {'name': 'sign', 'description': 'A字牌,三角牌,施工警示牌,道闸杆'}]

gtcam_mapping = {}

for cat in gtcam_category_info:
    for sub in cat['description'].split(','):
        gtcam_mapping[sub.strip()] = cat['name']

def gtcam_map_class(chinese_class: str) -> str:
    return gtcam_mapping.get(chinese_class, 'unknown')

gtcam_CAMERA_ORDER = {'front': {'image_key': 'image0', 'camera_index': 0}, 'right': {'image_key': 'image1', 'camera_index': 1}, 'left': {'image_key': 'image2', 'camera_index': 2}, 'back': {'image_key': 'image3', 'camera_index': 3}}

def gtcam_list_stems(directory, suffix=None):
    stems = set()
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        if suffix is not None and (not name.lower().endswith(suffix.lower())):
            continue
        stems.add(os.path.splitext(name)[0])
    return stems

def gtcam_find_file_by_stem(directory, stem):
    for name in os.listdir(directory):
        if os.path.splitext(name)[0] == stem:
            return os.path.join(directory, name)
    return None

def gtcam_parse_label_line(line):
    items = line.strip().split()
    cls = items[0]
    truncation = float(items[1])
    occlusion = int(float(items[2]))
    alpha = float(items[3])
    bbox = list(map(float, items[4:8]))
    h, w, l = map(float, items[8:11])
    x, y, z = map(float, items[11:14])
    yaw, pitch, roll = map(float, items[14:17])
    return (cls, truncation, occlusion, alpha, bbox, h, w, l, x, y, z, yaw, pitch, roll)

def gtcam_save_label_file(annotations, save_dir, frame_id):
    label_dir = os.path.join(save_dir, 'labels')
    os.makedirs(label_dir, exist_ok=True)
    label_path = os.path.join(label_dir, f'{frame_id}.txt')
    with open(label_path, 'w') as f:
        for ann in annotations:
            ann['class'] = gtcam_map_class(ann['class'])
            line = (
                f"{ann['class']} {ann['truncation']} {ann['occlusion']} {ann['alpha']:.12f} "
                f"{ann['bbox_2d'][0]:.12f} {ann['bbox_2d'][1]:.12f} {ann['bbox_2d'][2]:.12f} {ann['bbox_2d'][3]:.12f} "
                f"{ann['hwl'][0]:.12f} {ann['hwl'][1]:.12f} {ann['hwl'][2]:.12f} "
                f"{ann['center'][0]:.12f} {ann['center'][1]:.12f} {ann['center'][2]:.12f} "
                f"{ann['yaw']:.12f} {ann['pitch']:.12f} {ann['roll']:.12f}\n"
            )
            f.write(line)
    return label_path

def gtcam_get_3d_bbox_corners(center, size, yaw, pitch, roll):
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

def gtcam_vis_from_label(label_path, image_path, K, D, ego2cam):
    img = cv2.imread(image_path)
    if img is None:
        return None
    with open(label_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if not line.strip():
            continue
        cls, truncation, occlusion, alpha, bbox, h, w, l, x, y, z, yaw, pitch, roll = gtcam_parse_label_line(line)
        corners_3d = gtcam_get_3d_bbox_corners([x, y, z], [h, w, l], yaw, pitch, roll)
        corners_2d = gtcam_project_points_fisheye(corners_3d, K, D, ego2cam)
        if len(corners_2d) > 0:
            img = gtcam_draw_bbox_simple(img, corners_2d)
    return img

def gtcam_project_points_fisheye(points_3d, K, dist, lidar2cam):
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
        img_front, _ = cv2.fisheye.projectPoints(cam_points[front_mask].reshape(-1, 1, 3), rvec=np.zeros((3, 1), dtype=np.float64), tvec=np.zeros((3, 1), dtype=np.float64), K=K, D=D)
        img_points[front_mask] = img_front.reshape(-1, 2)
    back_mask = ~front_mask
    if np.any(back_mask):
        X = cam_points[back_mask, 0]
        Y = cam_points[back_mask, 1]
        Z = cam_points[back_mask, 2]
        r = np.sqrt(X ** 2 + Y ** 2)
        theta = np.arctan2(r, Z + 1e-08)
        phi = np.arctan2(Y, X)
        k1, k2, k3, k4 = D.flatten()
        theta_d = theta * (1 + k1 * theta ** 2 + k2 * theta ** 4 + k3 * theta ** 6 + k4 * theta ** 8)
        fx, fy = (K[0, 0], K[1, 1])
        cx, cy = (K[0, 2], K[1, 2])
        u = cx + fx * theta_d * np.cos(phi)
        v = cy + fy * theta_d * np.sin(phi)
        img_points[back_mask] = np.stack([u, v], axis=1)
        if img_points.shape[0] != 8 or np.any(np.isnan(img_points)):
            return np.empty((0, 2))
    return img_points

def gtcam_draw_bbox_simple(img, corners_2d, color=(0, 255, 0), thickness=2):
    if len(corners_2d) == 0:
        return img
    corners_2d = corners_2d.astype(int)
    n = len(corners_2d)
    if n >= 8:
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
        for i, j in edges:
            cv2.line(img, tuple(corners_2d[i]), tuple(corners_2d[j]), color, thickness)
    return img

def gtcam_normalize_angle(angle):
    alpha_tan = np.tan(angle)
    alpha_arctan = np.arctan(alpha_tan)
    if np.cos(angle) < 0:
        alpha_arctan = alpha_arctan + math.pi
    return alpha_arctan

def gtcam_get_camera_3d_8points(obj_size, yaw_lidar, center_lidar, center_in_cam, r_velo2cam, t_velo2cam):
    liadr_r = np.array([[math.cos(yaw_lidar), -math.sin(yaw_lidar), 0], [math.sin(yaw_lidar), math.cos(yaw_lidar), 0], [0, 0, 1]], dtype=np.float64)
    l, w, h = obj_size
    corners_3d_lidar = np.array([[l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2], [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], [0, 0, 0, 0, h, h, h, h]], dtype=np.float64)
    center_lidar = np.asarray(center_lidar, dtype=np.float64).reshape(3, 1)
    center_in_cam = np.asarray(center_in_cam, dtype=np.float64).reshape(-1)
    corners_3d_lidar = liadr_r @ corners_3d_lidar + center_lidar
    corners_3d_cam = r_velo2cam @ corners_3d_lidar + t_velo2cam
    x0, z0 = (corners_3d_cam[0, 0], corners_3d_cam[2, 0])
    x3, z3 = (corners_3d_cam[0, 3], corners_3d_cam[2, 3])
    dx, dz = (x0 - x3, z0 - z3)
    yaw_cam = math.atan2(dx, dz)
    alpha = yaw_cam - math.atan2(center_in_cam[0], center_in_cam[2])
    if alpha > math.pi:
        alpha = alpha - 2.0 * math.pi
    if alpha <= -1 * math.pi:
        alpha = alpha + 2.0 * math.pi
    rt_matrix1 = np.eye(4)
    rt_matrix1[:3, 3] = center_lidar.reshape(-1)
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
    return (alpha, yaw_cam, pitch_cam, roll_cam, corners_3d_cam)

def gtcam_calib_texts(save_dir, basename, K, Tr_velo_to_cam, D):
    calib_dir = os.path.join(save_dir, 'calibs')
    os.makedirs(calib_dir, exist_ok=True)
    calib_path = os.path.join(calib_dir, f'{basename}.txt')
    P0 = K
    Tr = Tr_velo_to_cam
    T = np.eye(3, 4).reshape(-1)
    dist = D
    with open(calib_path, 'w') as f:
        f.write("P0: " + " ".join(map(lambda x: f"{x:.12e}", P0.flatten())) + "\n")
        f.write("P1: " + " ".join(map(str, np.zeros(12))) + "\n")
        f.write("P2: " + " ".join(map(str, np.zeros(12))) + "\n")
        f.write("P3: " + " ".join(map(str, np.zeros(12))) + "\n")
        f.write("R0_rect: " + " ".join(map(str, np.eye(3).flatten())) + "\n")
        f.write("Tr_velo_to_cam: " + " ".join(map(lambda x: f"{x:.12e}", Tr.flatten())) + "\n")
        f.write("Tr_imu_to_velo: " + " ".join(f"{v:.12e}" for v in T) + "\n")
        f.write("dist: " + " ".join(map(lambda x: f"{x:.12e}", dist.flatten())) + "\n")
    return calib_path

def gtcam_image_save(image_path, save_dir, basename):
    image_dir = os.path.join(save_dir, 'images')
    os.makedirs(image_dir, exist_ok=True)
    dst_path = os.path.join(image_dir, f'{basename}.jpg')
    shutil.copy(image_path, dst_path)
    return dst_path

def gtcam_convert_point(point, matrix):
    return matrix @ point

def gtcam_get_class_value(obj, key_name, camera_index):
    for cv in obj.get('classValues', []):
        if cv.get('name') == key_name:
            values = cv.get('value', [])
            if camera_index < len(values):
                return values[camera_index]
            if values:
                return values[0]
    return 0

def gtcam_load_camera_params(cam_cfg_json_path, image_path):
    with open(cam_cfg_json_path, 'r') as f:
        cam_cfgs = json.load(f)
    frame_id = os.path.splitext(os.path.basename(cam_cfg_json_path))[0]
    cam_params = {}
    for camera_name, camera_meta in gtcam_CAMERA_ORDER.items():
        camera_index = camera_meta['camera_index']
        if camera_index >= len(cam_cfgs):
            continue
        cam_cfg = cam_cfgs[camera_index]
        internal_params = cam_cfg['camera_internal']
        distortion = cam_cfg['distortion']
        K = np.array([internal_params['fx'], 0.0, internal_params['cx'], 0.0, internal_params['fy'], internal_params['cy'], 0.0, 0.0, 1.0], dtype=np.float64).reshape(3, 3)
        D = np.array([distortion['k1'], distortion['k2'], distortion['k3'], distortion['k4']], dtype=np.float64)
        ego2cam_raw = np.array(cam_cfg['camera_external'], dtype=np.float64).reshape(4, 4)
        row_major = str(cam_cfg.get('rowMajor', 'true')).lower() == 'true'
        ego2cam = ego2cam_raw if row_major else ego2cam_raw.T
        r_velo2cam = ego2cam[:3, :3]
        t_velo2cam = ego2cam[:3, 3].reshape(3, 1)
        Tr_velo_to_cam = np.hstack((r_velo2cam, t_velo2cam))
        cam_params[camera_name] = {'frame_id': frame_id, 'image_path': image_path[camera_name], 'image_key': camera_meta['image_key'], 'camera_index': camera_index, 'K': K, 'D': D, 'ego2cam': ego2cam, 'r_velo2cam': r_velo2cam, 't_velo2cam': t_velo2cam, 'Tr_velo_to_cam': Tr_velo_to_cam}
    return cam_params

def gtcam_load_annotation(anno_json_path):
    with open(anno_json_path, 'r', encoding='utf-8') as f:
        anno_data = json.load(f)
    return anno_data.get('objects', [])

def gtcam_project_all_gt_to_cameras(cam_params, annotations, save_dir):
    for camera_name, params in cam_params.items():
        frame_id = params['frame_id']
        image_key = params['image_key']
        camera_index = params['camera_index']
        K = params['K']
        D = params['D']
        ego2cam = params['ego2cam']
        r_velo2cam = params['r_velo2cam']
        t_velo2cam = params['t_velo2cam']
        Tr_velo_to_cam = params['Tr_velo_to_cam']
        image_path = params['image_path']
        annotations_one_camera = []
        for obj in annotations:
            if obj.get('type', '3D_BOX') != '3D_BOX':
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
            bottom_center_in_cam = r_velo2cam @ np.array(bottom_center, dtype=np.float64).reshape(3, 1) + t_velo2cam
            alpha, yaw_cam, pitch_cam, roll_cam, _ = gtcam_get_camera_3d_8points(obj_size, yaw_lidar, bottom_center, bottom_center_in_cam, r_velo2cam, t_velo2cam)
            cam_x, cam_y, cam_z = gtcam_convert_point(np.array([x, y, z, 1]).T, Tr_velo_to_cam)
            truncation = float(gtcam_get_class_value(obj, 'truncation', camera_index))
            occlusion = int(float(gtcam_get_class_value(obj, 'occlusion', camera_index)))
            annotations_one_camera.append({'class': className, 'hwl': hwl, 'center': [cam_x, cam_y, cam_z], 'yaw': yaw_cam, 'pitch': pitch_cam, 'roll': roll_cam, 'bbox_2d': obj['2D_bbox'][image_key], 'alpha': alpha, 'truncation': truncation, 'occlusion': occlusion})
        camera_save_dir = os.path.join(save_dir, camera_name)
        label_path = gtcam_save_label_file(annotations_one_camera, camera_save_dir, frame_id)
        vis_dir = os.path.join(camera_save_dir, 'vis_from_label')
        os.makedirs(vis_dir, exist_ok=True)
        vis_path = os.path.join(vis_dir, f'{frame_id}_vis.jpg')
        img_vis = gtcam_vis_from_label(label_path, image_path, K, D, ego2cam)
        if img_vis is not None:
            cv2.imwrite(vis_path, img_vis)
            print(f'Saved (from label): {vis_path}')
        gtcam_calib_texts(camera_save_dir, frame_id, K, Tr_velo_to_cam, D)
        gtcam_image_save(image_path, camera_save_dir, frame_id)

def gtcam_process_one_frame(cam_cfg_json_path, anno_json_path, image_path, save_dir):
    cam_params = gtcam_load_camera_params(cam_cfg_json_path, image_path)
    annotations = gtcam_load_annotation(anno_json_path)
    gtcam_project_all_gt_to_cameras(cam_params, annotations, save_dir)

def gtcam_stage_entry() -> None:
    dataset_dir = Path(os.environ['SOURCE_DATASET_FILE_DIR']) / 'fisheye_data_aug'

    output_root = Path(os.environ['TARGET_RESULT_DIR'])

    print('经过数据增强的路径:', dataset_dir)

    scene_list = natsorted(os.listdir(dataset_dir))

    for scene_index in scene_list:
        scene_dir = os.path.join(dataset_dir, scene_index)
        save_dir = output_root / 'data_camera' / 'demo_data' / 'trainval_gt_vis' / scene_dir.split(os.sep)[-1]
        os.makedirs(save_dir, exist_ok=True)
        camera_config_dir = os.path.join(scene_dir, 'camera_config')
        image_dir = {'front_dir': os.path.join(scene_dir, 'image0'), 'right_dir': os.path.join(scene_dir, 'image1'), 'left_dir': os.path.join(scene_dir, 'image2'), 'back_dir': os.path.join(scene_dir, 'image3')}
        annotation_dir = os.path.join(scene_dir, 'result')
        if not all((os.path.exists(path) for path in [camera_config_dir, annotation_dir, image_dir['front_dir'], image_dir['right_dir'], image_dir['left_dir'], image_dir['back_dir']])):
            continue
        cam_cfg_ids = gtcam_list_stems(camera_config_dir, '.json')
        anno_ids = gtcam_list_stems(annotation_dir, '.json')
        front_ids = gtcam_list_stems(image_dir['front_dir'])
        right_ids = gtcam_list_stems(image_dir['right_dir'])
        left_ids = gtcam_list_stems(image_dir['left_dir'])
        back_ids = gtcam_list_stems(image_dir['back_dir'])
        common_frame_ids = natsorted(list(cam_cfg_ids & anno_ids & front_ids & right_ids & left_ids & back_ids))
        frame_count = len(common_frame_ids)
        print(f'Scene {scene_index}: found {frame_count} frames')
        args = []
        for frame_id in common_frame_ids:
            cam_cfg_json_path = os.path.join(camera_config_dir, f'{frame_id}.json')
            anno_json_path = os.path.join(annotation_dir, f'{frame_id}.json')
            front_image_path = gtcam_find_file_by_stem(image_dir['front_dir'], frame_id)
            right_image_path = gtcam_find_file_by_stem(image_dir['right_dir'], frame_id)
            left_image_path = gtcam_find_file_by_stem(image_dir['left_dir'], frame_id)
            back_image_path = gtcam_find_file_by_stem(image_dir['back_dir'], frame_id)
            if not all([front_image_path, right_image_path, left_image_path, back_image_path]):
                continue
            image_path = {'front': front_image_path, 'right': right_image_path, 'left': left_image_path, 'back': back_image_path}
            args.append((cam_cfg_json_path, anno_json_path, image_path, save_dir))
        try:
            with Pool(10) as p:
                p.starmap(gtcam_process_one_frame, args)
        except (PermissionError, OSError) as exc:
            print(f'Pool failed for scene {scene_index}, fallback to serial: {exc}')
            for arg in args:
                gtcam_process_one_frame(*arg)

    print('All scenes processed.')


def run_stage_40cpu_gt_4fisheye_camera_cord(source_root: Path, target_root: Path) -> None:
    previous_source, previous_target, previous_argv = _enter_stage(
        stage_name="40CPU_gt_4fisheye_camera_cord",
        stage_filename="40CPU_gt_4fisheye_camera_cord.py",
        source_root=source_root,
        target_root=target_root,
    )
    try:
        gtcam_stage_entry()
    finally:
        _leave_stage("40CPU_gt_4fisheye_camera_cord", previous_source, previous_target, previous_argv)

# Stage: 40CPU_fisheye2cyl.py
import cv2

import numpy as np

import os

from multiprocessing import Pool

import tqdm

from natsort import natsorted

from pathlib import Path

cylconv_CAMERA_NAMES = ['front', 'right', 'left', 'back']

cylconv_DEFAULT_HFOV = np.deg2rad(190)

cylconv_DEFAULT_VFOV = np.deg2rad(143)

cylconv_TARGET_H = 960

cylconv_TARGET_W = 1408

cylconv_RDF_TO_FLU = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)

def cylconv_wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def cylconv_build_view_info(f, R_raw, hfov=cylconv_DEFAULT_HFOV, vfov=cylconv_DEFAULT_VFOV, target_h=cylconv_TARGET_H, target_w=cylconv_TARGET_W):
    R = R_raw.copy() @ cylconv_RDF_TO_FLU
    forward_norm = np.sqrt(R[0, 2] ** 2 + R[2, 2] ** 2)
    forward_norm = max(forward_norm, 1e-12)
    azimuth = np.arccos(np.clip(R[2, 2] / forward_norm, -1.0, 1.0))
    if R[0, 2] < 0:
        azimuth = 2 * np.pi - azimuth
    tilt = -np.arccos(np.clip(forward_norm, -1.0, 1.0))
    Ry = np.array([[np.cos(azimuth), 0, np.sin(azimuth)], [0, 1, 0], [-np.sin(azimuth), 0, np.cos(azimuth)]]).T
    R_final = R @ Ry
    h = max(int(round(2 * f * np.tan(vfov / 2))), 1)
    w = max(int(round(f * hfov)), 1)
    K_cyl = np.array([[f, 0, w / 2], [0, f, f * np.tan(vfov / 2 + tilt)], [0, 0, 1]], dtype=np.float64)
    crop_top = int(np.clip(np.round(K_cyl[1, 2] - target_h / 2), 0, max(h - target_h, 0)))
    suggested_hfov = target_w / f
    suggested_vfov = 2 * (np.arctan(target_h / 2 / f) - tilt)
    suggested_vfov = np.clip(suggested_vfov, np.deg2rad(1), np.deg2rad(179))
    return {'azimuth': azimuth, 'tilt': tilt, 'raw_height': h, 'raw_width': w, 'R_final': R_final, 'K_cyl': K_cyl, 'crop_top': crop_top, 'suggested_hfov': suggested_hfov, 'suggested_vfov': suggested_vfov}

def cylconv_get_mapping(K, D, calib, hfov=cylconv_DEFAULT_HFOV, vfov=cylconv_DEFAULT_VFOV):
    f = calib['intrinsic']['f']
    view_info = cylconv_build_view_info(f, calib['extrinsic']['R'], hfov=hfov, vfov=vfov)
    R_final = view_info['R_final']
    K_cyl = view_info['K_cyl']
    h = view_info['raw_height']
    w = view_info['raw_width']
    K_cyl_inv = np.linalg.inv(K_cyl)
    xv, yv = np.meshgrid(range(w), range(h), indexing='xy')
    p = np.stack([xv, yv, np.ones_like(xv)], axis=-1).astype(np.float64)[..., np.newaxis]
    r = (K_cyl_inv @ p)[..., 0]
    r /= r[:, :, [2]]
    r_cart = np.zeros_like(r)
    r_cart[:, :, 2] = np.cos(r[:, :, 0])
    r_cart[:, :, 0] = np.sin(r[:, :, 0])
    r_cart[:, :, 1] = r[:, :, 1]
    r_cam = (R_final @ r_cart[..., np.newaxis])[..., 0]
    rays = r_cam.reshape(-1, 1, 3).astype(np.float64)
    img_points, _ = cv2.fisheye.projectPoints(rays, np.zeros((3, 1)), np.zeros((3, 1)), K.astype(np.float64), D.astype(np.float64))
    img_points = img_points.reshape(h, w, 2)
    return (img_points[..., 0].astype(np.float32), img_points[..., 1].astype(np.float32), R_final, K_cyl, view_info)

def cylconv_crop_or_pad_cylindrical(cyl, K_cyl, target_h=cylconv_TARGET_H, target_w=cylconv_TARGET_W):
    h, w = cyl.shape[:2]
    crop_top = 0
    crop_left = 0
    pad_top = 0
    pad_bottom = 0
    pad_left = 0
    pad_right = 0
    if h >= target_h:
        crop_top = int(np.clip(np.round(K_cyl[1, 2] - target_h / 2), 0, max(h - target_h, 0)))
        cyl = cyl[crop_top:crop_top + target_h, :]
        K_cyl[1, 2] -= crop_top
    else:
        pad_top = int(np.clip(np.round(target_h / 2 - K_cyl[1, 2]), 0, target_h - h))
        pad_bottom = target_h - h - pad_top
        cyl = cv2.copyMakeBorder(cyl, pad_top, pad_bottom, 0, 0, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
        K_cyl[1, 2] += pad_top
    h, w = cyl.shape[:2]
    if w >= target_w:
        crop_left = int(np.clip(np.round(K_cyl[0, 2] - target_w / 2), 0, max(w - target_w, 0)))
        cyl = cyl[:, crop_left:crop_left + target_w]
        K_cyl[0, 2] -= crop_left
    else:
        pad_left = int(np.clip(np.round(target_w / 2 - K_cyl[0, 2]), 0, target_w - w))
        pad_right = target_w - w - pad_left
        cyl = cv2.copyMakeBorder(cyl, 0, 0, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
        K_cyl[0, 2] += pad_left
    crop_info = {'crop_top': crop_top, 'crop_left': crop_left, 'pad_top': pad_top, 'pad_bottom': pad_bottom, 'pad_left': pad_left, 'pad_right': pad_right}
    return (cyl, K_cyl, crop_info)

def cylconv_fisheye_to_cylindrical(K, D, image, calib, hfov=cylconv_DEFAULT_HFOV, vfov=cylconv_DEFAULT_VFOV, target_h=cylconv_TARGET_H, target_w=cylconv_TARGET_W):
    mapx, mapy, R_final, K_cyl, view_info = cylconv_get_mapping(K, D, calib, hfov=hfov, vfov=vfov)
    cyl = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
    K_cyl = K_cyl.copy()
    cyl, K_cyl, crop_info = cylconv_crop_or_pad_cylindrical(cyl, K_cyl, target_h=target_h, target_w=target_w)
    K_cyl = K_cyl.copy()
    view_info = dict(view_info)
    view_info.update(crop_info)
    view_info['crop_bottom'] = view_info['crop_top'] + target_h - view_info['pad_top'] - view_info['pad_bottom']
    view_info['crop_right'] = view_info['crop_left'] + target_w - view_info['pad_left'] - view_info['pad_right']
    return (cyl, R_final, K_cyl, view_info)

def cylconv_project_point_to_cylindrical(P_cam, R_final, K_cyl):
    norm = np.linalg.norm(P_cam)
    if norm < 1e-06:
        return None
    dir_cam = P_cam / norm
    dir_cyl = R_final.T @ dir_cam
    x, y, z = dir_cyl
    theta = np.arctan2(x, z)
    rho = np.sqrt(x * x + z * z)
    y_norm = y / rho if rho > 1e-06 else 0.0
    f, cx, cy = (K_cyl[0, 0], K_cyl[0, 2], K_cyl[1, 2])
    return (f * theta + cx, f * y_norm + cy)

def cylconv_draw_3d_boxes_on_cylindrical(cyl_img, label_path, R_final, K_cyl):
    f = K_cyl[0, 0]
    u0 = K_cyl[0, 2]
    v0 = K_cyl[1, 2]
    img = cyl_img.copy()
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
            global_yaw = cylconv_wrap(yaw_local + phi)
            tan_psi = (v - v0) / f
            dir_cyl = np.array([np.sin(phi), tan_psi, np.cos(phi)], dtype=np.float32)
            dir_cyl /= np.linalg.norm(dir_cyl)
            dir_cam = R_final @ dir_cyl
            center_cam = (dir_cam * rho).reshape(3, 1)
            center_upright = R_final.T @ center_cam
            corners_local = np.array([[W / 2, H / 2, L / 2], [W / 2, H / 2, -L / 2], [W / 2, -H / 2, L / 2], [W / 2, -H / 2, -L / 2], [-W / 2, H / 2, L / 2], [-W / 2, H / 2, -L / 2], [-W / 2, -H / 2, L / 2], [-W / 2, -H / 2, -L / 2]], dtype=np.float32).T
            c, s = (np.cos(global_yaw), np.sin(global_yaw))
            R_yaw = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
            corners = R_yaw @ corners_local + center_upright
            pts = []
            for i in range(8):
                X, Y, Z = corners[:, i]
                r = np.sqrt(X ** 2 + Z ** 2 + 1e-08)
                uc = f * np.arctan2(X, Z) + u0
                vc = f * np.tan(np.arctan(Y / r)) + v0
                pts.append((int(round(uc)), int(round(vc))))
            edges = [(0, 1), (2, 3), (4, 5), (6, 7), (1, 3), (3, 7), (7, 5), (5, 1), (0, 2), (2, 6), (6, 4), (4, 0)]
            for i, j in edges:
                cv2.line(img, pts[i], pts[j], (0, 255, 0), 2)
            cv2.circle(img, (int(u), int(v)), 5, (0, 0, 255), -1)
    return img

def cylconv_read_fisheye_calib(calib_txt):
    K = D = Tr = None
    with open(calib_txt) as f:
        for line in f:
            if line.startswith('P0:'):
                K = np.array(line.split()[1:10], float).reshape(3, 3)
            elif line.startswith('Tr_velo_to_cam:'):
                vals = np.array(line.split()[1:], float)
                Tr = np.vstack([vals.reshape(3, 4), [0, 0, 0, 1]])
            elif line.startswith('dist:'):
                D = np.array(line.split()[1:], float)
    return (K, D, Tr)

def cylconv_print_camera_view_report(input_root, hfov=cylconv_DEFAULT_HFOV, vfov=cylconv_DEFAULT_VFOV, target_h=cylconv_TARGET_H, target_w=cylconv_TARGET_W):
    print('[camera_view_report] begin')
    scene_list = natsorted(os.listdir(input_root))
    for scene_name in scene_list:
        scene_dir = os.path.join(input_root, scene_name)
        if not os.path.isdir(scene_dir):
            continue
        for camera_name in cylconv_CAMERA_NAMES:
            calib_dir = os.path.join(scene_dir, camera_name, 'calibs')
            if not os.path.isdir(calib_dir):
                continue
            calib_files = [name for name in os.listdir(calib_dir) if name.lower().endswith('.txt')]
            if not calib_files:
                continue
            calib_path = os.path.join(calib_dir, natsorted(calib_files)[0])
            K, D, Tr = cylconv_read_fisheye_calib(calib_path)
            if K is None or Tr is None:
                continue
            view_info = cylconv_build_view_info(K[0, 0], Tr[:3, :3], hfov=hfov, vfov=vfov, target_h=target_h, target_w=target_w)
            print(
                f"[camera_view] scene={scene_name} camera={camera_name} "
                f"azimuth_deg={np.rad2deg(view_info['azimuth']):.3f} "
                f"tilt_deg={np.rad2deg(view_info['tilt']):.3f} "
                f"hfov_deg={np.rad2deg(hfov):.3f} vfov_deg={np.rad2deg(vfov):.3f} "
                f"raw_size={view_info['raw_width']}x{view_info['raw_height']} "
                f"K_cyl_cy={view_info['K_cyl'][1, 2]:.3f} crop_top={view_info['crop_top']} "
                f"hfov_deg_for_target_w={np.rad2deg(view_info['suggested_hfov']):.3f} "
                f"vfov_deg_for_target_h={np.rad2deg(view_info['suggested_vfov']):.3f}"
            )
    print('[camera_view_report] end')

def cylconv_convert_one(job):
    scene_name, camera_name, file_id, img_path, calib_txt, label_path, out_camera_dir = job
    try:
        if not (os.path.exists(img_path) and os.path.exists(calib_txt) and os.path.exists(label_path)):
            return
        save_img = os.path.join(out_camera_dir, 'images', f'{file_id}.jpg')
        save_vis = os.path.join(out_camera_dir, 'vis', f'{file_id}.jpg')
        save_label = os.path.join(out_camera_dir, 'labels', f'{file_id}.txt')
        save_calib = os.path.join(out_camera_dir, 'calibs', f'{file_id}.txt')
        for path in [save_img, save_vis, save_label, save_calib]:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        K, D, Tr = cylconv_read_fisheye_calib(calib_txt)
        if K is None or D is None or Tr is None:
            print('skip invalid calib:', scene_name, camera_name, file_id)
            return
        calib = {'intrinsic': {'f': K[0, 0]}, 'extrinsic': {'R': Tr[:3, :3]}}
        img = cv2.imread(img_path)
        if img is None:
            print('skip invalid image:', scene_name, camera_name, file_id)
            return
        cyl, R_final, K_cyl, view_info = cylconv_fisheye_to_cylindrical(K, D, img, calib)
        with open(calib_txt) as f_in, open(save_calib, 'w') as f_out:
            f_out.writelines(f_in.readlines())
            f_out.write('R_final: ' + ' '.join((f'{v:.12e}' for v in R_final.flatten())) + '\n')
            f_out.write('K_cyl: ' + ' '.join((f'{v:.12e}' for v in K_cyl.flatten())) + '\n')
            f_out.write(f"cyl_azimuth_deg: {np.rad2deg(view_info['azimuth']):.12e}\n")
            f_out.write(f"cyl_tilt_deg: {np.rad2deg(view_info['tilt']):.12e}\n")
            f_out.write(f"cyl_crop_top: {int(view_info['crop_top'])}\n")
            f_out.write(f"cyl_crop_left: {int(view_info['crop_left'])}\n")
        with open(label_path) as f:
            lines = [line.strip() for line in f if line.strip()]
        with open(save_label, 'w') as f_out:
            for line in lines:
                parts = line.split()
                class_name = parts[0]
                h, w, l = map(float, parts[8:11])
                x, y, z = map(float, parts[11:14])
                uv = cylconv_project_point_to_cylindrical(np.array([x, y, z]), R_final, K_cyl)
                if uv is None:
                    continue
                u, v = uv
                yaw = float(parts[3])
                rho = np.linalg.norm(R_final.T @ np.array([x, y, z]))
                f_out.write(f'{class_name} {u:.3f} {v:.3f} {h:.3f} {w:.3f} {l:.3f} {yaw:.3f} {rho:.3f}\n')
        vis = cylconv_draw_3d_boxes_on_cylindrical(cyl, save_label, R_final, K_cyl)
        cv2.imwrite(save_img, cyl)
        cv2.imwrite(save_vis, vis)
    except Exception as e:
        print('error:', scene_name, camera_name, file_id, e)

def cylconv_collect_jobs(input_root, output_root):
    jobs = []
    scene_list = natsorted(os.listdir(input_root))
    for scene_name in scene_list:
        scene_dir = os.path.join(input_root, scene_name)
        if not os.path.isdir(scene_dir):
            continue
        for camera_name in cylconv_CAMERA_NAMES:
            camera_dir = os.path.join(scene_dir, camera_name)
            img_dir = os.path.join(camera_dir, 'images')
            calib_dir = os.path.join(camera_dir, 'calibs')
            label_dir = os.path.join(camera_dir, 'labels')
            if not all((os.path.isdir(path) for path in [img_dir, calib_dir, label_dir])):
                continue
            img_ids = {os.path.splitext(name)[0] for name in os.listdir(img_dir)}
            calib_ids = {os.path.splitext(name)[0] for name in os.listdir(calib_dir)}
            label_ids = {os.path.splitext(name)[0] for name in os.listdir(label_dir)}
            common_ids = natsorted(list(img_ids & calib_ids & label_ids))
            out_camera_dir = os.path.join(output_root, scene_name, camera_name)
            for file_id in common_ids:
                jobs.append((scene_name, camera_name, file_id, os.path.join(img_dir, f'{file_id}.jpg'), os.path.join(calib_dir, f'{file_id}.txt'), os.path.join(label_dir, f'{file_id}.txt'), out_camera_dir))
    return jobs

def cylconv_stage_entry() -> None:
    dataset_dir = Path(os.environ['SOURCE_DATASET_FILE_DIR'])

    input_root = dataset_dir / 'data_camera' / 'demo_data' / 'trainval_gt_vis'

    out_dir = Path(os.environ['TARGET_RESULT_DIR'])

    output_root = out_dir / 'data_camera_cyl' / 'demo_data' / 'trainval_gt_vis'

    cylconv_print_camera_view_report(input_root)

    jobs = cylconv_collect_jobs(input_root, output_root)

    print('total:', len(jobs))

    try:
        with Pool(10) as p:
            list(tqdm.tqdm(p.imap(cylconv_convert_one, jobs), total=len(jobs)))
    except (PermissionError, OSError) as exc:
        print('pool failed, fallback to serial:', exc)
        for job in tqdm.tqdm(jobs):
            cylconv_convert_one(job)

    print('done')


def run_stage_40cpu_fisheye2cyl(source_root: Path, target_root: Path) -> None:
    previous_source, previous_target, previous_argv = _enter_stage(
        stage_name="40CPU_fisheye2cyl",
        stage_filename="40CPU_fisheye2cyl.py",
        source_root=source_root,
        target_root=target_root,
    )
    try:
        cylconv_stage_entry()
    finally:
        _leave_stage("40CPU_fisheye2cyl", previous_source, previous_target, previous_argv)

# Stage: 40CPU_cyl_kitti2ann.py
import argparse

import json

import math

import os

import shutil

from multiprocessing import Pool

import cv2

import numpy as np

import tqdm

from pathlib import Path

cylann_script_root = Path(os.environ['SOURCE_DATASET_FILE_DIR']) / 'scripts'

cylann_train_val_split_path = cylann_script_root / 'train_val_split.json'

cylann_dataset_dir = Path(os.environ['SOURCE_DATASET_FILE_DIR'])

cylann_DEFAULT_INPUT_ROOT = cylann_dataset_dir / 'data_camera_cyl' / 'demo_data' / 'trainval_gt_vis'

cylann_out_dataset_dir = Path(os.environ['TARGET_RESULT_DIR'])

cylann_DEFAULT_OUTPUT_ROOT = cylann_out_dataset_dir / 'data_camera_cyl' / 'demo_data' / 'trainval'

cylann_DEFAULT_SCENE_SPLIT_PATH = os.path.join(cylann_train_val_split_path)

cylann_CAMERA_NAMES = ('front', 'right', 'left', 'back')

cylann_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')

def cylann_wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def cylann__bbox_to_coco_bbox(bbox):
    return [bbox[0], bbox[1], max(0.0, bbox[2] - bbox[0]), max(0.0, bbox[3] - bbox[1])]

def cylann_sample_sort_key(sample):
    return (sample['scene_name'], sample['camera_name'], sample['frame_id'])

def cylann_make_unique_stem(scene_name, camera_name, frame_id):
    return '{}__{}__{}'.format(scene_name, camera_name, frame_id)

def cylann_find_image_path(image_dir, frame_id):
    for ext in cylann_IMAGE_EXTENSIONS:
        candidate = os.path.join(image_dir, frame_id + ext)
        if os.path.exists(candidate):
            return candidate
    return None

def cylann_parse_calib_file(calib_path):
    calib = {}
    with open(calib_path, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or ':' not in line:
                continue
            key, value = line.split(':', 1)
            values = np.fromstring(value.strip(), sep=' ', dtype=np.float32)
            if key == 'P0':
                if values.size == 9:
                    calib['P0'] = np.concatenate([values.reshape(3, 3), np.zeros((3, 1), dtype=np.float32)], axis=1)
                elif values.size == 12:
                    calib['P0'] = values.reshape(3, 4)
            elif key == 'K_cyl' and values.size == 9:
                calib['K_cyl'] = values.reshape(3, 3)
            elif key == 'R_final' and values.size == 9:
                calib['R_final'] = values.reshape(3, 3)
            elif key == 'Tr_velo_to_cam' and values.size == 12:
                calib['Tr_velo_to_cam'] = values.reshape(3, 4)
    if 'P0' not in calib and 'K_cyl' in calib:
        calib['P0'] = np.concatenate([calib['K_cyl'], np.zeros((3, 1), dtype=np.float32)], axis=1)
    if 'K_cyl' not in calib and 'P0' in calib:
        calib['K_cyl'] = calib['P0'][:, :3]
    missing = [key for key in ('P0', 'K_cyl', 'R_final') if key not in calib]
    if missing:
        raise ValueError('Missing {} in {}'.format(','.join(missing), calib_path))
    return calib

def cylann_load_scene_split(scene_split_path):
    if not os.path.exists(scene_split_path):
        return (set(), set())
    with open(scene_split_path, 'r', encoding='utf-8') as f:
        scene_split = json.load(f)
    return (set(scene_split.get('train', [])), set(scene_split.get('val', [])))

def cylann_split_by_ratio(samples, val_ratio):
    samples = sorted(samples, key=cylann_sample_sort_key)
    if len(samples) <= 1:
        return (samples, [])
    val_count = int(round(len(samples) * val_ratio))
    val_count = min(max(val_count, 1), len(samples) - 1)
    split_index = len(samples) - val_count
    return (samples[:split_index], samples[split_index:])

def cylann_split_samples(samples, scene_split_path, val_ratio):
    train_scenes, val_scenes = cylann_load_scene_split(scene_split_path)
    train_samples = []
    val_samples = []
    unknown_samples = []
    has_scene_split = bool(train_scenes or val_scenes)
    overlap_scenes = train_scenes & val_scenes
    if overlap_scenes:
        raise ValueError('Scene split file contains scenes in both train and val: {}'.format(','.join(sorted(overlap_scenes))))
    for sample in samples:
        scene_name = sample['scene_name']
        if scene_name in train_scenes:
            train_samples.append(sample)
        elif scene_name in val_scenes:
            val_samples.append(sample)
        else:
            unknown_samples.append(sample)
    if has_scene_split:
        if unknown_samples:
            unknown_scene_names = sorted({sample['scene_name'] for sample in unknown_samples})
            raise ValueError('Scene split file does not cover all scenes. Missing: {}'.format(','.join(unknown_scene_names)))
        return (sorted(train_samples, key=cylann_sample_sort_key), sorted(val_samples, key=cylann_sample_sort_key), 'scene_split')
    fallback_train, fallback_val = cylann_split_by_ratio(samples, val_ratio)
    return (fallback_train, fallback_val, 'ratio_fallback')

def cylann_safe_link_or_copy(src_path, dst_path, use_hardlink):
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

def cylann_collect_samples(input_root):
    samples = []
    if not os.path.isdir(input_root):
        raise FileNotFoundError('Input root does not exist: {}'.format(input_root))
    for scene_name in sorted(os.listdir(input_root)):
        scene_dir = os.path.join(input_root, scene_name)
        if not os.path.isdir(scene_dir):
            continue
        for camera_name in cylann_CAMERA_NAMES:
            camera_dir = os.path.join(scene_dir, camera_name)
            image_dir = os.path.join(camera_dir, 'images')
            label_dir = os.path.join(camera_dir, 'labels')
            calib_dir = os.path.join(camera_dir, 'calibs')
            if not all((os.path.isdir(path) for path in (image_dir, label_dir, calib_dir))):
                continue
            image_ids = {os.path.splitext(name)[0] for name in os.listdir(image_dir) if os.path.splitext(name)[1].lower() in cylann_IMAGE_EXTENSIONS}
            label_ids = {os.path.splitext(name)[0] for name in os.listdir(label_dir) if name.lower().endswith('.txt')}
            calib_ids = {os.path.splitext(name)[0] for name in os.listdir(calib_dir) if name.lower().endswith('.txt')}
            for frame_id in sorted(image_ids & label_ids & calib_ids):
                image_path = cylann_find_image_path(image_dir, frame_id)
                if image_path is None:
                    continue
                image_ext = os.path.splitext(image_path)[1].lower()
                samples.append({'scene_name': scene_name, 'camera_name': camera_name, 'frame_id': frame_id, 'stem': cylann_make_unique_stem(scene_name, camera_name, frame_id), 'image_path': image_path, 'image_ext': image_ext, 'label_path': os.path.join(label_dir, frame_id + '.txt'), 'calib_path': os.path.join(calib_dir, frame_id + '.txt')})
    return sorted(samples, key=cylann_sample_sort_key)

def cylann_build_keypoints(points, width, height):
    keypoints = []
    for x, y in points:
        visible = 2 if 0 <= x < width and 0 <= y < height else 1
        keypoints.extend([float(x), float(y), visible])
    return keypoints

def cylann_process_one(args):
    sample, image_id, output_root, save_vis, use_hardlink = args
    image = cv2.imread(sample['image_path'])
    if image is None:
        raise ValueError('Failed to read image: {}'.format(sample['image_path']))
    height, width = image.shape[:2]
    calib = cylann_parse_calib_file(sample['calib_path'])
    p_cyl = calib['P0'].astype(np.float32)
    k_cyl = calib['K_cyl'].astype(np.float32)
    r_final = calib['R_final'].astype(np.float32)
    f = float(k_cyl[0, 0])
    u0 = float(k_cyl[0, 2])
    v0 = float(k_cyl[1, 2])
    image_file_name = sample['stem'] + sample['image_ext']
    output_image_path = os.path.join(output_root, 'images', image_file_name)
    output_calib_path = os.path.join(output_root, 'calibs', sample['stem'] + '.txt')
    cylann_safe_link_or_copy(sample['image_path'], output_image_path, use_hardlink)
    cylann_safe_link_or_copy(sample['calib_path'], output_calib_path, use_hardlink)
    image_info = {'file_name': image_file_name, 'id': int(image_id), 'width': int(width), 'height': int(height), 'calib': p_cyl.tolist(), 'scene_name': sample['scene_name'], 'camera_name': sample['camera_name'], 'frame_id': sample['frame_id']}
    annotations = []
    vis_image = image.copy() if save_vis else None
    with open(sample['label_path'], 'r', encoding='utf-8') as anns:
        for raw_line in anns:
            parts = raw_line.strip().split()
            if len(parts) < 8:
                continue
            cat = parts[0].lower()
            if cat not in cylann_CAT_IDS:
                continue
            u, v = map(float, parts[1:3])
            obj_h, obj_w, obj_l = map(float, parts[3:6])
            yaw_local = float(parts[6])
            rho = float(parts[7])
            phi = (u - u0) / f
            tan_psi = (v - v0) / f
            dir_cyl = np.array([math.sin(phi), tan_psi, math.cos(phi)], dtype=np.float32)
            dir_norm = float(np.linalg.norm(dir_cyl))
            if dir_norm < 1e-06:
                continue
            dir_cyl /= dir_norm
            dir_cam = r_final @ dir_cyl
            center_cam = (dir_cam * rho).reshape(3, 1)
            center_upright = r_final.T @ center_cam
            corners_local = np.array([[obj_w / 2, obj_h / 2, obj_l / 2], [obj_w / 2, obj_h / 2, -obj_l / 2], [obj_w / 2, -obj_h / 2, obj_l / 2], [obj_w / 2, -obj_h / 2, -obj_l / 2], [-obj_w / 2, obj_h / 2, obj_l / 2], [-obj_w / 2, obj_h / 2, -obj_l / 2], [-obj_w / 2, -obj_h / 2, obj_l / 2], [-obj_w / 2, -obj_h / 2, -obj_l / 2]], dtype=np.float32).T
            yaw_global = cylann_wrap(yaw_local + phi)
            cos_yaw = math.cos(yaw_global)
            sin_yaw = math.sin(yaw_global)
            r_yaw = np.array([[cos_yaw, 0.0, sin_yaw], [0.0, 1.0, 0.0], [-sin_yaw, 0.0, cos_yaw]], dtype=np.float32)
            corners = r_yaw @ corners_local + center_upright
            projected_points = []
            for corner_idx in range(8):
                x_val, y_val, z_val = corners[:, corner_idx]
                radial = math.sqrt(float(x_val * x_val + z_val * z_val) + 1e-08)
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
            keypoints = cylann_build_keypoints(keypoint_points, width, height)
            bbox = [float(x1), float(y1), float(x2), float(y2)]
            ann = {'segmentation': [[0, 0, 0, 0, 0, 0]], 'num_keypoints': 9, 'area': float((x2 - x1) * (y2 - y1)), 'iscrowd': 0, 'keypoints': keypoints, 'image_id': int(image_id), 'bbox': cylann__bbox_to_coco_bbox(bbox), 'uv': [float(u), float(v)], 'category_id': cylann_CAT_IDS[cat], 'dim': [float(obj_h), float(obj_w), float(obj_l)], 'alpha': float(yaw_local), 'depth': float(rho), 'scene_name': sample['scene_name'], 'camera_name': sample['camera_name'], 'frame_id': sample['frame_id']}
            annotations.append(ann)
            if vis_image is not None:
                draw_points = [(int(round(x)), int(round(y))) for x, y in projected_points]
                edges = [(0, 1), (2, 3), (4, 5), (6, 7), (1, 3), (3, 7), (7, 5), (5, 1), (0, 2), (2, 6), (6, 4), (4, 0)]
                for start_idx, end_idx in edges:
                    cv2.line(vis_image, draw_points[start_idx], draw_points[end_idx], (0, 255, 0), 2)
                cv2.rectangle(vis_image, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), (0, 0, 255), 2)
                cv2.circle(vis_image, (int(round(u)), int(round(v))), 4, (255, 0, 0), -1)
    if vis_image is not None:
        vis_path = os.path.join(output_root, 'vis_3d_2d', sample['stem'] + '.jpg')
        os.makedirs(os.path.dirname(vis_path), exist_ok=True)
        cv2.imwrite(vis_path, vis_image)
    return (sample['stem'], image_info, annotations)

def cylann_write_annotations_json(out_path, records):
    images = []
    annotations = []
    ann_id = 1
    for _, image_info, image_annotations in records:
        images.append(image_info)
        for ann in image_annotations:
            ann['id'] = ann_id
            ann_id += 1
            annotations.append(ann)
    payload = {'images': images, 'annotations': annotations, 'categories': cylann_CAT_INFO}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False)

def cylann_write_id_txt(out_path, samples):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(sample['stem'] + '\n')

def cylann_parse_args():
    parser = argparse.ArgumentParser(description='Convert 4-camera cylindrical labels to model-ready ann json.')
    parser.add_argument('--input_root', default=cylann_DEFAULT_INPUT_ROOT, help='Input root like data_camera_cyl/demo_data/trainval_gt_vis')
    parser.add_argument('--output_root', default=cylann_DEFAULT_OUTPUT_ROOT, help='Output root like data_camera_cyl/demo_data/trainval')
    parser.add_argument('--scene_split', default=cylann_DEFAULT_SCENE_SPLIT_PATH, help='Scene-level train/val split json')
    parser.add_argument('--workers', type=int, default=10, help='Number of worker processes')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Fallback val ratio when scene split is unavailable or empty')
    parser.add_argument('--no_vis', action='store_true', help='Disable vis_3d_2d output')
    parser.add_argument('--no_hardlink', action='store_true', help='Copy files instead of trying hard links first')
    return parser.parse_args()

cylann_CATS = ['car', 'truck', 'bus', 'construction_vehicle', 'pedestrian', 'motor', 'bicycle', 'animal', 'traffic_cone', 'barrier', 'stopper', 'trash_bin', 'sign']

cylann_CAT_IDS = {cat: idx + 1 for idx, cat in enumerate(cylann_CATS)}

cylann_CAT_INFO = [{'name': cat, 'id': idx + 1} for idx, cat in enumerate(cylann_CATS)]

def cylann_main():
    args = cylann_parse_args()
    output_root = os.path.abspath(args.output_root)
    input_root = os.path.abspath(args.input_root)
    scene_split_path = os.path.abspath(args.scene_split)
    save_vis = not args.no_vis
    use_hardlink = not args.no_hardlink
    samples = cylann_collect_samples(input_root)
    if not samples:
        raise RuntimeError('No valid samples found under {}'.format(input_root))
    train_samples, val_samples, split_mode = cylann_split_samples(samples, scene_split_path, args.val_ratio)
    tasks = []
    for image_id, sample in enumerate(samples, start=1):
        tasks.append((sample, image_id, output_root, save_vis, use_hardlink))
    os.makedirs(os.path.join(output_root, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_root, 'calibs'), exist_ok=True)
    os.makedirs(os.path.join(output_root, 'annotations'), exist_ok=True)
    if save_vis:
        os.makedirs(os.path.join(output_root, 'vis_3d_2d'), exist_ok=True)
    worker_count = max(1, min(args.workers, len(tasks), os.cpu_count() or 1))
    processed = {}
    if worker_count == 1:
        iterator = map(cylann_process_one, tasks)
        for stem, image_info, annotations in tqdm.tqdm(iterator, total=len(tasks)):
            processed[stem] = (stem, image_info, annotations)
    else:
        with Pool(worker_count) as pool:
            iterator = pool.imap_unordered(cylann_process_one, tasks)
            for stem, image_info, annotations in tqdm.tqdm(iterator, total=len(tasks)):
                processed[stem] = (stem, image_info, annotations)
    ordered_train_records = [processed[sample['stem']] for sample in train_samples if sample['stem'] in processed]
    ordered_val_records = [processed[sample['stem']] for sample in val_samples if sample['stem'] in processed]
    cylann_write_annotations_json(os.path.join(output_root, 'annotations', 'ip42_train_all.json'), ordered_train_records)
    cylann_write_annotations_json(os.path.join(output_root, 'annotations', 'ip42_val_all.json'), ordered_val_records)
    cylann_write_id_txt(os.path.join(output_root, 'train.txt'), train_samples)
    cylann_write_id_txt(os.path.join(output_root, 'val.txt'), val_samples)
    print('Input root:', input_root)
    print('Output root:', output_root)
    print('Split mode:', split_mode)
    print('Total samples:', len(samples))
    print('Train samples:', len(train_samples))
    print('Val samples:', len(val_samples))
    print('Workers:', worker_count)

def cylann_stage_entry() -> None:
    global cylann_script_root
    global cylann_train_val_split_path
    global cylann_dataset_dir
    global cylann_DEFAULT_INPUT_ROOT
    global cylann_out_dataset_dir
    global cylann_DEFAULT_OUTPUT_ROOT
    global cylann_DEFAULT_SCENE_SPLIT_PATH
    cylann_script_root = Path(os.environ["SOURCE_DATASET_FILE_DIR"]) / "scripts"
    cylann_train_val_split_path = cylann_script_root / "train_val_split.json"
    cylann_dataset_dir = Path(os.environ["SOURCE_DATASET_FILE_DIR"])
    cylann_DEFAULT_INPUT_ROOT = cylann_dataset_dir / "data_camera_cyl" / "demo_data" / "trainval_gt_vis"
    cylann_out_dataset_dir = Path(os.environ["TARGET_RESULT_DIR"])
    cylann_DEFAULT_OUTPUT_ROOT = cylann_out_dataset_dir / "data_camera_cyl" / "demo_data" / "trainval"
    cylann_DEFAULT_SCENE_SPLIT_PATH = os.path.join(cylann_train_val_split_path)
    cylann_main()


def run_stage_40cpu_cyl_kitti2ann(source_root: Path, target_root: Path) -> None:
    previous_source, previous_target, previous_argv = _enter_stage(
        stage_name="40CPU_cyl_kitti2ann",
        stage_filename="40CPU_cyl_kitti2ann.py",
        source_root=source_root,
        target_root=target_root,
    )
    try:
        cylann_stage_entry()
    finally:
        _leave_stage("40CPU_cyl_kitti2ann", previous_source, previous_target, previous_argv)

# Stage: split_train_test.py

def split_stage_entry() -> None:
    import json
    import os
    import random
    from pathlib import Path

    root = Path(os.environ["SOURCE_DATASET_FILE_DIR"]) / "fisheye_data_aug"

    all_folders = [
        name for name in os.listdir(root)
        if os.path.isdir(os.path.join(root, name))
    ]

    random.shuffle(all_folders)

    n_total = len(all_folders)
    n_train = int(n_total * (4 / 5))

    train_list = all_folders[:n_train]
    val_list = all_folders[n_train:]

    data = {
        "train": train_list,
        "val": val_list,
    }

    script_root = Path(os.environ["TARGET_RESULT_DIR"]) / "scripts"
    script_root.mkdir(parents=True, exist_ok=True)
    train_val_split_path = script_root / "train_val_split.json"
    with open(train_val_split_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print("Total scenes:", n_total)
    print("Train scenes:", len(train_list))
    print("Val scenes:", len(val_list))
    print("Saved train_val_split.json")


def run_stage_split_train_test(source_root: Path, target_root: Path) -> None:
    previous_source, previous_target, previous_argv = _enter_stage(
        stage_name="split_train_test",
        stage_filename="split_train_test.py",
        source_root=source_root,
        target_root=target_root,
    )
    try:
        split_stage_entry()
    finally:
        _leave_stage("split_train_test", previous_source, previous_target, previous_argv)



def main() -> None:
    if not PIPELINE_SOURCE_ROOT.exists():
        raise FileNotFoundError(f"SOURCE_DATASET_FILE_DIR does not exist: {PIPELINE_SOURCE_ROOT}")
    if not (PIPELINE_SOURCE_ROOT / "fisheye_2wdata").exists():
        raise FileNotFoundError(
            f"Input fisheye dataset does not exist: {PIPELINE_SOURCE_ROOT / 'fisheye_2wdata'}"
        )

    PIPELINE_TARGET_ROOT.mkdir(parents=True, exist_ok=True)

    run_stage_offline_fisheye_camera_only_aug(PIPELINE_SOURCE_ROOT, PIPELINE_TARGET_ROOT)
    run_stage_split_train_test(PIPELINE_TARGET_ROOT, PIPELINE_TARGET_ROOT)
    run_stage_40cpu_gt_4fisheye_camera_cord(PIPELINE_TARGET_ROOT, PIPELINE_TARGET_ROOT)
    run_stage_40cpu_fisheye2cyl(PIPELINE_TARGET_ROOT, PIPELINE_TARGET_ROOT)
    run_stage_40cpu_cyl_kitti2ann(PIPELINE_TARGET_ROOT, PIPELINE_TARGET_ROOT)

    print("=" * 80)
    print("Pipeline finished.")
    print(f"Input root: {PIPELINE_SOURCE_ROOT / 'fisheye_2wdata'}")
    print(f"Final output root: {PIPELINE_TARGET_ROOT / 'data_camera_cyl' / 'demo_data' / 'trainval'}")
    print("=" * 80)


if __name__ == "__main__":
    main()

