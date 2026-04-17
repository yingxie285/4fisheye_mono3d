import os
import numpy as np
from collections import defaultdict

# ================== 路径配置 ==================
gt_dir = "/media/whut-user/350b5acb-7342-4f97-b1a7-57e9622bff8b/mono3d/gt_lidar_cord/label_front"
pred_dir = "/media/whut-user/350b5acb-7342-4f97-b1a7-57e9622bff8b/mono3d/git-apa-perception-mono3dod-t2/mono3DOD_Parking/mono3DOD_Parking_code/exp/ddd/default/results_lidar_epoch200"
val_txt = "/media/whut-user/350b5acb-7342-4f97-b1a7-57e9622bff8b/mono3d/data_camera_cyl/demo_data/trainval/val.txt"

output_path = "/media/whut-user/350b5acb-7342-4f97-b1a7-57e9622bff8b/mono3d/recall_3712.txt"

classes = ['car', 'truck', 'bus', 'construction_vehicle', 'pedestrian',
        'motor', 'bicycle', 'animal', 'traffic_cone', 'barrier',
        'stopper', 'trash_bin', 'sign']

# ================== 工具函数 ==================
def read_kitti_txt(path, is_pred=False):
    objs = []
    if not os.path.exists(path):
        return objs

    with open(path, 'r') as f:
        for line in f:
            data = line.strip().split()
            if len(data) < 15:
                continue

            obj = {}
            obj["type"] = data[0]
            obj["h"], obj["w"], obj["l"] = map(float, data[8:11])
            obj["x"], obj["y"], obj["z"] = map(float, data[11:14])
            obj["ry"] = float(data[14])

            if is_pred:
                score = float(data[15]) if len(data) > 15 else 0.0
                obj["score"] = score

            objs.append(obj)
    return objs


def angle_diff(a, b):
    return np.arctan2(np.sin(a - b), np.cos(a - b))


# ================== 读取帧ID ==================
with open(val_txt, 'r') as f:
    frame_ids = [line.strip() for line in f.readlines()]

print(f"Total frames to evaluate: {len(frame_ids)}")

# ================== 距离分段 ==================
range_thresholds = {
    (0, 5):   [0.1, 0.3, 0.5],
    (5, 10):  [0.3, 0.5, 0.7],
    (10, 20): [0.5, 0.7, 1.0, 1.5],
    (20, 30): [1.0, 1.5, 2.0],
    (30, 40): [1.5, 2.0, 2.5],
}

# ================== 全局 GT 统计 ==================
global_total_gt = 0
global_class_count = defaultdict(int)
all_gt_distances = []

all_data = []

for fid in frame_ids:
    gt_path = os.path.join(gt_dir, f"{fid}.txt")
    pred_path = os.path.join(pred_dir, f"{fid}.txt")

    gt_objs = read_kitti_txt(gt_path, False)
    pred_objs = read_kitti_txt(pred_path, True)

    all_data.append((fid, gt_objs, pred_objs))

    for g in gt_objs:
        if g["type"] not in classes:
            continue
        global_total_gt += 1
        global_class_count[g["type"]] += 1
        dist = np.linalg.norm([g["x"], g["y"]])
        all_gt_distances.append(dist)

# ================== 写文件 ==================
os.makedirs(os.path.dirname(output_path), exist_ok=True)
log_file = open(output_path, "w")

def log_print(*args):
    print(*args)
    print(*args, file=log_file)

# ================== 打印基础信息 ==================
log_print("=" * 80)
log_print(f"Total evaluated frames: {len(frame_ids)}")
log_print("=" * 80)

log_print("\n" + "=" * 80)
log_print("Global GT Distribution (All Classes)")
log_print("=" * 80)

sum_check = 0.0

for c in classes:
    cnt = global_class_count[c]
    ratio = cnt / global_total_gt if global_total_gt > 0 else 0.0
    sum_check += ratio
    log_print(f"{c:<20} {cnt} ({ratio:.2%})")

log_print("-" * 80)
log_print(f"Total GT: {global_total_gt}")
log_print(f"Total Ratio Check: {sum_check:.2%}")
log_print("=" * 80)

if len(all_gt_distances) > 0:
    log_print(f"Min dist: {np.min(all_gt_distances):.2f}")
    log_print(f"Max dist: {np.max(all_gt_distances):.2f}")

# ================== 主评估 ==================
for current_range, thresholds in range_thresholds.items():
    for dist_threshold in thresholds:

        log_print("\n" + "=" * 140)
        log_print(f"Range {current_range}, Threshold={dist_threshold}")
        log_print("=" * 140)

        log_print("{:<20} {:<18} {:<10} {:<10} {:<10} {:<10} {:<8} {:<15} {:<8} {:<8} {:<8} {:<20}".format(
            "Class","GT(%)","Matched","Recall","Pred","Precision",
            "mATE","mAOE(deg/rad)","mALE","mAWE","mAHE","角度相似度"
        ))

        gt_count = defaultdict(int)
        matched_count = defaultdict(int)
        pred_count = defaultdict(int)

        ate_cls = defaultdict(list)
        aoe_cls = defaultdict(list)
        ale_cls = defaultdict(list)
        awe_cls = defaultdict(list)
        ahe_cls = defaultdict(list)

        total_gt = 0
        total_matched = 0
        total_pred = 0

        for fid, gt_objs, pred_objs in all_data:

            # 筛选预测
            valid_preds = []
            for i, p in enumerate(pred_objs):
                if p["type"] not in classes:
                    continue
                if p.get("score", 0.0) <= 0.1:   #过滤噪声框
                    continue
                dist = np.linalg.norm([p["x"], p["y"]])
                if current_range[0] <= dist < current_range[1]:
                    valid_preds.append((i, p))
                    pred_count[p["type"]] += 1
                    total_pred += 1

            # 按score排序
            valid_preds = sorted(valid_preds, key=lambda x: x[1].get("score", 0), reverse=True)
            used = set()

            for g in gt_objs:
                if g["type"] not in classes:
                    continue

                dist = np.linalg.norm([g["x"], g["y"]])
                if not (current_range[0] <= dist < current_range[1]):
                    continue

                gt_count[g["type"]] += 1
                total_gt += 1

                matched = False

                for idx, p in valid_preds:
                    if idx in used:
                        continue
                    if p["type"] != g["type"]:
                        continue

                    d = np.linalg.norm([p["x"]-g["x"], p["y"]-g["y"]])
                    if d < dist_threshold:
                        used.add(idx)
                        matched = True

                        ate_cls[g["type"]].append(d)
                        aoe_cls[g["type"]].append(abs(angle_diff(p["ry"], g["ry"])))
                        ale_cls[g["type"]].append(abs(p["l"]-g["l"]))
                        awe_cls[g["type"]].append(abs(p["w"]-g["w"]))
                        ahe_cls[g["type"]].append(abs(p["h"]-g["h"]))
                        break

                if matched:
                    matched_count[g["type"]] += 1
                    total_matched += 1

        # ================== 输出 ==================
        for c in sorted(classes, key=lambda x: gt_count[x], reverse=True):
            gt = gt_count[c]
            m = matched_count[c]
            p = pred_count[c]

            recall = m/gt if gt>0 else 0
            precision = m/p if p>0 else 0

            aoe_rad = np.mean(aoe_cls[c]) if aoe_cls[c] else 0.0
            aoe_deg = np.degrees(aoe_rad)
            aoe_str = f"{aoe_deg:.1f}({aoe_rad:.4f})"  
            cos_sim = np.cos(aoe_rad) if aoe_cls[c] else 0.0 

            log_print("{:<20} {:<18} {:<10} {:<10.2%} {:<10} {:<10.2%} {:<8.2f} {:<15} {:<8.2f} {:<8.2f} {:<8.2f} {:<20.4f}".format(
                c,
                f"{gt} ({gt/global_total_gt:.2%})" if global_total_gt>0 else "0",
                m,
                recall,
                p,
                precision,
                np.mean(ate_cls[c]) if ate_cls[c] else 0,
                aoe_str,   
                np.mean(ale_cls[c]) if ale_cls[c] else 0,
                np.mean(awe_cls[c]) if awe_cls[c] else 0,
                np.mean(ahe_cls[c]) if ahe_cls[c] else 0,
                cos_sim
            ))

        log_print("=" * 140)
        log_print(f"Overall Recall: {total_matched/total_gt:.2%}" if total_gt>0 else "0")
        log_print(f"Overall Precision: {total_matched/total_pred:.2%}" if total_pred>0 else "0")
        log_print("=" * 140)

log_file.close()
print(f"\nSaved to: {output_path}")