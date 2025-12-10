#!/bin/bash
# Stable Diffusion 模型下载脚本

set -e

MODEL_DIR="/data/metahuman_work/stable-diffusion-webui/models/Stable-diffusion"
MIRROR="https://hf-mirror.com"

echo "=========================================="
echo "Stable Diffusion 模型下载工具"
echo "=========================================="
echo ""

# 创建目录
mkdir -p "$MODEL_DIR"
cd "$MODEL_DIR"

echo "模型保存目录: $MODEL_DIR"
echo ""

# 菜单
echo "请选择要下载的模型："
echo "1. SD 1.5 (4GB, 显存需求低, 推荐新手)"
echo "2. SDXL Base (6.9GB, 高质量, 显存需求高)"
echo "3. 全部下载"
echo "0. 退出"
echo ""

read -p "请输入选项 [0-3]: " choice

case $choice in
    1)
        echo ""
        echo "下载 SD 1.5 模型..."
        wget -c "$MIRROR/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors"
        echo "✓ SD 1.5 下载完成！"
        ;;
    2)
        echo ""
        echo "下载 SDXL Base 模型..."
        wget -c "$MIRROR/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors"
        echo "✓ SDXL Base 下载完成！"
        ;;
    3)
        echo ""
        echo "下载所有模型（约 11GB）..."
        wget -c "$MIRROR/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors"
        wget -c "$MIRROR/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors"
        echo "✓ 所有模型下载完成！"
        ;;
    0)
        echo "退出"
        exit 0
        ;;
    *)
        echo "无效选项"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "下载完成！"
echo "=========================================="
echo ""
echo "已下载的模型："
ls -lh "$MODEL_DIR"/*.safetensors 2>/dev/null || echo "无模型文件"
echo ""
echo "重启容器以加载模型："
echo "  cd /data/metahuman_work/stable-diffusion-webui"
echo "  docker compose restart"
echo ""
