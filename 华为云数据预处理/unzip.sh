#!/bin/bash

set -e

echo "开始处理..."

# =========================
# 1. 路径定义
# =========================
SRC_DIR="$SOURCE_DATASET_FILE_DIR"

# 最终输出目录（要的是 fisheye_2wdata）
OUT_DIR="$TARGET_RESULT_DIR/fisheye_2wdata"

TEMP_DIR="$TARGET_RESULT_DIR/temp_unzip"

echo "源目录(只读): $SRC_DIR"
echo "目标目录: $OUT_DIR"
echo "临时目录: $TEMP_DIR"

# =========================
# 2. 创建目录（只在可写区）
# =========================
mkdir -p "$OUT_DIR"
mkdir -p "$TEMP_DIR"

# =========================
# 3. 进入源目录（只读）
# =========================
cd "$SRC_DIR"

# =========================
# 4. 解压
# =========================
echo "开始解压 zip..."
for f in *.zip; do
    echo "解压 $f"
    unzip -q "$f" -d "$TEMP_DIR"
done

# =========================
# 5. 打印目录层级（检查结构）
# =========================
echo "解压后的目录层级："

find "$TEMP_DIR" -type d | while read dir; do
    rel_path="${dir#$TEMP_DIR}"
    depth=$(echo "$rel_path" | awk -F'/' '{print NF-1}')
    printf "第%d层: %s\n" "$depth" "$dir"
done


# =========================
# 5. 查看结构（可选）
# =========================
echo "解压结构："
find "$TEMP_DIR" -type d

# =========================
# 6. 移动第3层目录
# =========================
echo "开始移动..."

find "$TEMP_DIR" -mindepth 3 -maxdepth 3 -type d | while read dir; do
    echo "移动: $dir"
    mv "$dir" "$OUT_DIR/"
done

# =========================
# 7. 清理
# =========================
rm -rf "$TEMP_DIR"

echo "完成！"