# Stable Diffusion 文生图服务部署指南

## 📋 目录
1. [GitHub 代码仓库对比](#github-代码仓库对比)
2. [推荐方案](#推荐方案)
3. [其他可选方案](#其他可选方案)
4. [HuggingFace 模型库说明](#huggingface-模型库说明)
5.  [代码与模型的关系](#代码与模型的关系)
6. [部署流程概述](#部署流程概述)

---

## GitHub 代码仓库对比

### 1.  Stability-AI/stablediffusion
- **GitHub**: https://github.com/Stability-AI/stablediffusion
- ⭐ **Stars**: 42,089
- 🍴 **Forks**: 5,358
- 📝 **描述**: High-Resolution Image Synthesis with Latent Diffusion Models
- 📄 **许可**: MIT License
- 🎯 **定位**: 官方研究实现
- **特点**: 
  - Stability AI 官方的研究级代码
  - 更偏向底层和研究用途
  - 适合二次开发和学术研究
  - 需要较多编程知识

### 2.  AUTOMATIC1111/stable-diffusion-webui ⭐推荐
- **GitHub**: https://github.com/AUTOMATIC1111/stable-diffusion-webui
- ⭐ **Stars**: 158,765（最流行！）
- 🍴 **Forks**: 29,474
- 📝 **描述**: Stable Diffusion web UI
- 📄 **许可**: AGPL-3.0
- 🏷️ **标签**: ai-art, image-generation, text2image, img2img, web, gradio
- 🎯 **定位**: 生产级 Web UI
- **特点**:
  - 提供完整的 Web 图形界面
  - 功能丰富，开箱即用
  - 庞大的社区和插件生态
  - 支持多种扩展（ControlNet, Lora 等）
  - 适合生产环境部署

---

## 推荐方案

### 🎯 首选：AUTOMATIC1111/stable-diffusion-webui

**推荐理由：**

✅ **开箱即用**
- 提供完整的 Web 界面，无需额外开发
- 安装脚本自动化程度高

✅ **功能全面**
- 文生图 (txt2img)
- 图生图 (img2img)
- 图像放大 (upscaling)
- 局部重绘 (inpainting)
- ControlNet 支持（精确控制生成）
- Lora 模型支持（风格定制）
- 提示词权重调整
- 批量生成

✅ **社区活跃**
- 158,765 stars，GitHub 上最受欢迎的 SD 项目
- 大量中文教程和文档
- 丰富的第三方插件
- 问题解答资源丰富

✅ **易于部署**
- Windows/Linux/Mac 一键安装脚本
- Docker 部署支持
- 云服务器友好

✅ **适合场景**
- 个人使用
- 小团队部署
- 原型开发
- 生产环境

---

## 其他可选方案

### 1. ComfyUI
- **特点**: 基于节点的工作流 UI
- **优势**: 高度灵活，可视化工作流，适合复杂流程
- **劣势**: 学习曲线较陡
- **适合**: 需要定制化工作流的高级用户

### 2.  InvokeAI
- **特点**: 另一个成熟的 Web UI 方案
- **优势**: 界面现代化，功能完善
- **劣势**: 社区规模小于 AUTOMATIC1111
- **适合**: 追求更好 UI/UX 的用户

### 3. Fooocus
- **特点**: 简化版 UI，类似 Midjourney 体验
- **优势**: 极简操作，自动优化参数
- **劣势**: 可控性较低
- **适合**: 新手或不需要复杂调参的用户

### 4. SD.Next (vladmandic/automatic)
- **特点**: AUTOMATIC1111 的现代化分支
- **优势**: 更新更快，支持更多新功能
- **劣势**: 可能不够稳定
- **适合**: 追求最新功能的用户

**选择建议：**
- 快速上手、功能全面 → **AUTOMATIC1111**
- 研究或二次开发 → **Stability-AI**
- 工作流定制化 → **ComfyUI**
- 极简体验 → **Fooocus**

---

## HuggingFace 模型库说明

### 什么是 HuggingFace 上的模型仓库？

**示例**: https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5

这是 **AI 模型权重文件的存储库**，与 GitHub 代码仓库完全不同。

### 两者的本质区别

| 对比项 | GitHub 仓库 | HuggingFace 仓库 |
|--------|-------------|------------------|
| **存储内容** | 代码、程序、Web UI | 训练好的模型权重文件 |
| **文件类型** | `.py`, `.js`, `.sh` 等 | `.ckpt`, `.safetensors`, `.bin` 等 |
| **文件大小** | 通常几 MB 到几百 MB | 通常 4-7 GB |
| **作用** | 提供生成图片的"引擎"和"界面" | 提供生成图片的"大脑"和"知识" |
| **类比** | 播放器软件 | 音乐文件 |

### HuggingFace 仓库包含什么

典型的 SD 模型仓库包含：

```
stable-diffusion-v1-5/
├── v1-5-pruned. ckpt          # 主模型文件（约 4GB）
├── v1-5-pruned-emaonly.safetensors  # 安全格式（推荐）
├── vae/                       # VAE 变分自编码器
├── text_encoder/              # 文本编码器
├── tokenizer/                 # 分词器
├── unet/                      # UNet 扩散模型
└── model_index.json           # 配置文件
```

### 常见的模型版本

#### 官方基础模型
- **SD v1.4** - 最早的公开版本
- **SD v1. 5** - 最经典，兼容性最好（推荐新手）
- **SD v2.1** - 更新但兼容性稍差
- **SDXL 1.0** - 最新，质量最高，需要 8GB+ 显存

#### 社区微调模型（更多选择）
- **Realistic Vision** - 真实感照片风格
- **DreamShaper** - 梦幻风格
- **Anything V5** - 动漫风格
- **ChilloutMix** - 亚洲面孔优化
- 还有成千上万的其他模型...

可以在 https://civitai.com/ 找到更多社区模型。

---

## 代码与模型的关系

### 工作原理图解

```
┌─────────────────────────────┐
│    GitHub 代码仓库           │
│  (AUTOMATIC1111/SD-WebUI)   │  ← 你要部署的软件程序
│                             │
│  • Web 界面                 │
│  • 图像生成逻辑             │
│  • 参数控制                 │
│  • 插件系统                 │
└──────────────┬──────────────┘
               │
               │ 需要加载 ↓
               │
┌──────────────▼──────────────┐
│   HuggingFace 模型库         │
│  (stable-diffusion-v1-5)    │  ← AI 模型权重文件
│                             │
│  • 模型权重 (4-7GB)         │
│  • VAE, Text Encoder        │
│  • 配置文件                 │
└─────────────────────────────┘
```

### 类比理解

**就像音乐播放器和音乐文件的关系：**

- **代码（GitHub）** = 播放器软件（如 VLC、iTunes）
  - 提供播放功能
  - 提供用户界面
  - 控制播放参数

- **模型（HuggingFace）** = 音乐文件（MP3、FLAC）
  - 实际的内容数据
  - 可以更换不同的"音乐"
  - 可以有不同风格和质量

### 为什么需要两者？

- ❌ **只有代码**：程序能运行，但没有模型不能生成图片
- ❌ **只有模型**：有模型文件，但没有程序来运行它
- ✅ **两者都有**：完整的文生图服务

---

## 部署流程概述

### 第一步：安装程序代码（从 GitHub）

```bash
# 克隆 AUTOMATIC1111 仓库
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
cd stable-diffusion-webui
```

### 第二步：下载模型文件（从 HuggingFace 或其他来源）

**方式 1: 手动下载**
```bash
# 1. 访问 HuggingFace
# https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5

# 2. 下载 v1-5-pruned-emaonly.safetensors

# 3. 放到 models/Stable-diffusion/ 目录
mv v1-5-pruned-emaonly.safetensors models/Stable-diffusion/
```

**方式 2: 使用 WebUI 自动下载**
- 首次运行时 WebUI 会提示下载
- 或在设置中配置下载地址

**方式 3: 使用 huggingface-cli**
```bash
pip install huggingface_hub
huggingface-cli download stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --local-dir models/Stable-diffusion/
```

### 第三步：运行服务

**Windows:**
```bash
webui-user.bat
```

**Linux/Mac:**
```bash
./webui.sh
```

### 第四步：访问 Web 界面

打开浏览器访问：
```
http://localhost:7860
```

### 完整目录结构示例

```
stable-diffusion-webui/
├── webui.sh                   # 启动脚本
├── webui-user.bat             # Windows 启动脚本
├── models/
│   ├── Stable-diffusion/      # 主模型存放处
│   │   ├── v1-5-pruned-emaonly.safetensors
│   │   └── sd_xl_base_1.0.safetensors
│   ├── VAE/                   # VAE 模型
│   ├── Lora/                  # Lora 模型
│   └── ControlNet/            # ControlNet 模型
├── extensions/                # 插件目录
└── outputs/                   # 生成的图片
```

---

## 系统要求

### 最低配置
- **GPU**: NVIDIA GTX 1060 6GB（或同等性能）
- **内存**: 8GB RAM
- **存储**: 20GB 可用空间
- **系统**: Windows 10+, Ubuntu 18.04+, macOS

### 推荐配置
- **GPU**: NVIDIA RTX 3060 12GB 或更高
- **内存**: 16GB RAM
- **存储**: 50GB+ SSD
- **系统**: Ubuntu 22.04 或 Windows 11

### 显存需求对比

| 模型版本 | 最低显存 | 推荐显存 | 生成速度 |
|---------|---------|---------|---------|
| SD v1.5 | 4GB | 6GB+ | 快 |
| SD v2.1 | 6GB | 8GB+ | 中等 |
| SDXL 1.0 | 8GB | 12GB+ | 慢 |

---

## 快速决策树

```
你的需求是什么？
│
├─ 我想快速开始，有 Web 界面
│  └─> 选择 AUTOMATIC1111/stable-diffusion-webui
│
├─ 我要做研究或深度定制开发
│  └─> 选择 Stability-AI/stablediffusion
│
├─ 我需要复杂的工作流控制
│  └─> 考虑 ComfyUI
│
└─ 我是新手，要最简单的
   └─> 考虑 Fooocus
```

---

## 常见问题 FAQ

### Q1: 我可以在没有 GPU 的情况下运行吗？
A: 可以，但速度会非常慢。CPU 生成一张图可能需要几分钟到几十分钟。

### Q2: 模型文件可以从其他地方下载吗？
A: 可以，除了 HuggingFace，还可以从：
- Civitai.com（最大的社区模型库）
- LiblibAI（国内模型库）
- 其他用户分享的网盘

### Q3: 我需要懂编程吗？
A: 使用 AUTOMATIC1111 不需要编程知识，跟着教程操作即可。

### Q4: 商业使用有限制吗？
A: 
- AUTOMATIC1111 使用 AGPL-3.0 许可
- 模型许可各不相同，使用前请查看具体模型的许可证
- SD v1.5 允许商业使用

### Q5: 可以同时使用多个模型吗？
A: 可以，你可以下载多个模型放在 `models/Stable-diffusion/` 目录，在 WebUI 中切换。

---

## 总结

### 核心要点

1. **GitHub 仓库 = 软件程序**（你要部署的代码）
2. **HuggingFace 仓库 = 模型文件**（软件需要加载的 AI 大脑）
3. **两者缺一不可**才能运行完整的文生图服务

### 推荐配置

- **软件**: AUTOMATIC1111/stable-diffusion-webui
- **模型**: Stable Diffusion v1. 5（新手）或 SDXL 1.0（高质量）
- **硬件**: 至少 6GB 显存的 NVIDIA 显卡

### 下一步行动

1. ✅ 确认硬件配置满足要求
2. ✅ 从 GitHub 克隆 AUTOMATIC1111 代码
3. ✅ 从 HuggingFace 下载 SD v1.5 模型
4. ✅ 运行安装脚本
5. ✅ 启动服务并开始创作

---

## 参考链接

- **AUTOMATIC1111 GitHub**: https://github.com/AUTOMATIC1111/stable-diffusion-webui
- **Stability-AI GitHub**: https://github.com/Stability-AI/stablediffusion
- **SD v1.5 HuggingFace**: https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
- **Civitai 模型库**: https://civitai.com/
- **AUTOMATIC1111 Wiki**: https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki

---

**文档版本**: 1.0  
**最后更新**: 2025-12-05  
**作者**: GitHub Copilot