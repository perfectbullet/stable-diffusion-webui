#!/bin/bash
# 修复 HuggingFace 缓存和目录权限问题

set -e

echo "=========================================="
echo "修复 Stable Diffusion WebUI 启动问题"
echo "=========================================="
echo ""

BASE_DIR="/data/metahuman_work/stable-diffusion-webui"

# 1. 创建 HuggingFace 缓存目录
echo "✓ 创建 HuggingFace 缓存目录..."
mkdir -p "$BASE_DIR/huggingface-cache"
chmod 777 "$BASE_DIR/huggingface-cache"

# 1.5. 清理可能存在的空 venv 目录（避免冲突）
if [ -d "$BASE_DIR/venv" ] && [ ! -f "$BASE_DIR/venv/bin/python" ]; then
    echo "✓ 清理空的 venv 目录..."
    rm -rf "$BASE_DIR/venv"
fi

# 2. 创建必需的模型子目录
echo "✓ 创建必需的模型子目录..."
mkdir -p "$BASE_DIR/models/Stable-diffusion"
mkdir -p "$BASE_DIR/models/Lora"
mkdir -p "$BASE_DIR/models/hypernetworks"
mkdir -p "$BASE_DIR/models/VAE"

# 3. 设置权限
echo "✓ 设置目录权限..."
chmod -R 755 "$BASE_DIR/models" 2>/dev/null || true
chmod -R 755 "$BASE_DIR/outputs" 2>/dev/null || true

# 4. 停止现有容器
echo "✓ 停止现有容器..."
cd "$BASE_DIR"
docker compose down

# 5. 拉取最新镜像（可选）
echo "✓ 拉取最新镜像..."
docker compose pull

# 6. 启动容器
echo "✓ 启动容器..."
docker compose up -d

echo ""
echo "=========================================="
echo "修复完成！"
echo "=========================================="
echo ""
echo "查看日志："
echo "  docker compose logs -f"
echo ""
echo "如果仍然失败，请检查："
echo "  1. 容器是否能访问 https://hf-mirror.com"
echo "  2. 模型文件是否存在于 $BASE_DIR/models/Stable-diffusion/"
echo ""
