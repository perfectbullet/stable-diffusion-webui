# HuggingFace Tokenizer 问题修复指南

## 问题说明

错误信息：
```
OSError: Can't load tokenizer for 'openai/clip-vit-large-patch14'
```

**原因**：SDXL 模型需要从 HuggingFace Hub 下载 CLIP tokenizer，但容器内无法访问或下载失败。

## 已实施的修复

### 1. 更新了 `docker-compose.yml`

添加了以下配置：

```yaml
environment:
  - HF_HOME=/app/.cache/huggingface
  - TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers
  - HF_ENDPOINT=https://hf-mirror.com  # 使用国内镜像加速

volumes:
  - /data/metahuman_work/stable-diffusion-webui/huggingface-cache:/app/.cache/huggingface
```

### 2. 创建了修复脚本 `fix-huggingface-cache.sh`

自动创建目录、设置权限并重启容器。

## 立即执行步骤

### 在 Linux 服务器上执行：

```bash
# 1. 进入工作目录
cd /data/metahuman_work/stable-diffusion-webui

# 2. 给脚本执行权限
chmod +x fix-huggingface-cache.sh

# 3. 运行修复脚本
./fix-huggingface-cache.sh

# 4. 查看启动日志
docker compose logs -f
```

## 预期结果

成功后日志应该显示：

```
Loading weights [31e35c80fc] from /app/stable-diffusion-webui/models/Stable-diffusion/sd_xl_base_1.0.safetensors
Creating model from config: /app/stable-diffusion-webui/repositories/generative-models/configs/inference/sd_xl_base.yaml
Model loaded in XXs
```

## 如果仍然失败

### 方案 A：测试网络连接

```bash
# 进入容器测试
docker exec -it stable-diffusion-webui-1 bash

# 测试 HuggingFace 镜像站
curl -I https://hf-mirror.com

# 测试 tokenizer 下载
python -c "
from transformers import CLIPTokenizer
tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
print('✓ Tokenizer loaded successfully')
"
```

### 方案 B：手动下载 Tokenizer（离线方案）

在有网络的机器上：

```bash
pip install transformers
python << EOF
from transformers import CLIPTokenizer
tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
tokenizer.save_pretrained('./clip-vit-large-patch14')
print('Downloaded to ./clip-vit-large-patch14')
EOF
```

然后上传到服务器：

```bash
# 创建目录
mkdir -p /data/metahuman_work/stable-diffusion-webui/huggingface-cache/hub

# 上传文件到该目录
# 重启容器
docker compose restart
```

### 方案 C：使用 SD 1.5 模型（临时方案）

如果 SDXL 一直有问题，可以先用 SD 1.5：

```bash
cd /data/metahuman_work/stable-diffusion-webui/models/Stable-diffusion

# 下载 SD 1.5（约 4GB）
wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors

# 或使用国内镜像
wget https://hf-mirror.com/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors

# 删除或移走 SDXL 模型（让它优先加载 SD 1.5）
mv sd_xl_base_1.0.safetensors sd_xl_base_1.0.safetensors.bak

# 重启容器
docker compose restart
```

## 验证清单

- [ ] huggingface-cache 目录已创建且有写权限
- [ ] docker-compose.yml 已更新
- [ ] 容器已重启
- [ ] 日志中没有 OSError
- [ ] WebUI 可以通过 http://服务器IP:50010 访问

## 快速命令参考

```bash
# 查看实时日志
docker compose logs -f

# 重启容器
docker compose restart

# 完全重建
docker compose down && docker compose up -d

# 进入容器调试
docker exec -it stable-diffusion-webui-1 bash

# 查看 HuggingFace 缓存
docker exec -it stable-diffusion-webui-1 ls -la /app/.cache/huggingface/

# 检查模型文件
ls -lh /data/metahuman_work/stable-diffusion-webui/models/Stable-diffusion/
```
