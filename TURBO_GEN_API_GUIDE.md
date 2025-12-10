# Turbo-Gen 异步API使用指南

## 概述

Turbo-Gen 提供了一组异步的 txt2img 接口，允许您提交图片生成任务后立即返回，而不需要等待图片生成完成。任务会在后台执行，生成的图片会自动保存到本地。

**任务数据使用 MongoDB 持久化存储**，支持服务重启后继续查询任务状态。

## MongoDB 配置

### 1. 安装 MongoDB

如果还没有安装 MongoDB，请先安装：

**Docker 方式（推荐）：**
```bash
docker run -d \
  --name mongodb \
  -p 27017:27017 \
  -e MONGO_INITDB_ROOT_USERNAME=admin \
  -e MONGO_INITDB_ROOT_PASSWORD=yourpassword \
  -v mongodb_data:/data/db \
  mongo:latest
```

**或使用 docker-compose：**
```yaml
services:
  mongodb:
    image: mongo:latest
    container_name: mongodb
    ports:
      - "27017:27017"
    environment:
      MONGO_INITDB_ROOT_USERNAME: admin
      MONGO_INITDB_ROOT_PASSWORD: yourpassword
    volumes:
      - mongodb_data:/data/db

volumes:
  mongodb_data:
```

### 2. 配置环境变量

设置以下环境变量来连接 MongoDB（如果不设置，使用默认值）：

```bash
# MongoDB 连接URI（默认: mongodb://localhost:27017/）
export MONGODB_URI="mongodb://admin:yourpassword@localhost:27017/"

# 数据库名称（默认: stable_diffusion）
export MONGODB_DATABASE="stable_diffusion"

# 集合名称（默认: turbo_gen_tasks）
export MONGODB_COLLECTION="turbo_gen_tasks"
```

**Windows PowerShell:**
```powershell
$env:MONGODB_URI="mongodb://admin:yourpassword@localhost:27017/"
$env:MONGODB_DATABASE="stable_diffusion"
$env:MONGODB_COLLECTION="turbo_gen_tasks"
```

**Docker Compose 配置示例：**
```yaml
services:
  sd-webui:
    image: universonic/stable-diffusion-webui:full
    environment:
      - MONGODB_URI=mongodb://admin:yourpassword@mongodb:27017/
      - MONGODB_DATABASE=stable_diffusion
      - MONGODB_COLLECTION=turbo_gen_tasks
    depends_on:
      - mongodb
```

### 3. 自动回退机制

如果 MongoDB 连接失败或 pymongo 未安装，系统会**自动回退到内存模式**，仍然可以正常使用，但任务数据不会持久化。

启动时会显示以下信息：
- ✓ MongoDB连接成功 - 使用持久化存储
- ⚠ MongoDB连接失败 - 使用内存模式

### 4. 数据库索引

系统会自动创建以下索引以优化查询性能：
- `task_id`: 唯一索引，用于快速查找任务
- `status`: 普通索引，用于按状态筛选
- `created_at`: 降序索引，用于按时间排序

## 接口列表

### 1. 提交异步生成任务

**接口路径**: `POST /sdapi/v1/turbo-gen/txt2img-async`

**描述**: 提交txt2img任务到后台队列，立即返回任务ID

**请求参数**:

```json
{
  "prompt": "a beautiful landscape",
  "negative_prompt": "bad quality",
  "steps": 20,
  "width": 512,
  "height": 512,
  "cfg_scale": 7.0,
  "sampler_name": "Euler",
  "seed": -1,
  "batch_size": 4,
  "n_iter": 1,
  "override_settings": {},
  "override_settings_restore_afterwards": true
}
```

**参数说明**:

- `prompt` (string): 图片生成的提示词
- `negative_prompt` (string): 负面提示词
- `steps` (integer): 采样步数，默认20
- `width` (integer): 图片宽度，默认512
- `height` (integer): 图片高度，默认512
- `cfg_scale` (float): CFG缩放系数，默认7.0
- `sampler_name` (string, optional): 采样器名称
- `seed` (integer): 随机种子，-1表示随机，默认-1
- `batch_size` (integer): 批次大小，**固定为4**（会自动设置为4）
- `n_iter` (integer): 迭代次数，默认1
- `override_settings` (object, optional): 临时覆盖的设置
- `override_settings_restore_afterwards` (boolean): 是否在处理后恢复设置，默认true

**响应示例**:

```json
{
  "task_id": "e7b3c8d1-9f4a-4b2e-8c7d-1a2b3c4d5e6f",
  "message": "任务已创建，任务ID: e7b3c8d1-9f4a-4b2e-8c7d-1a2b3c4d5e6f"
}
```

### 2. 查询任务状态

**接口路径**: `GET /sdapi/v1/turbo-gen/task/{task_id}`

**描述**: 根据任务ID查询任务执行状态、进度和结果

**路径参数**:

- `task_id` (string): 任务的唯一标识符

**响应示例**:

```json
{
  "task_id": "e7b3c8d1-9f4a-4b2e-8c7d-1a2b3c4d5e6f",
  "status": "completed",
  "created_at": "2023-12-10T10:30:00.123456",
  "started_at": "2023-12-10T10:30:01.234567",
  "completed_at": "2023-12-10T10:30:45.678901",
  "execution_time": 44.444444,
  "image_urls": [
    "/sdapi/v1/turbo-gen/download/txt2img-images/2023-12-10/00001.png",
    "/sdapi/v1/turbo-gen/download/txt2img-images/2023-12-10/00002.png",
    "/sdapi/v1/turbo-gen/download/txt2img-images/2023-12-10/00003.png",
    "/sdapi/v1/turbo-gen/download/txt2img-images/2023-12-10/00004.png"
  ],
  "error_message": null,
  "progress": 100.0,
  "parameters": {
    "prompt": "a beautiful landscape",
    "steps": 20,
    ...
  }
}
```

**状态说明**:

- `pending`: 等待中 - 任务已创建，等待执行
- `running`: 运行中 - 任务正在执行
- `completed`: 已完成 - 任务成功完成
- `failed`: 失败 - 任务执行失败

**字段说明**:

- `task_id`: 任务唯一标识符
- `status`: 任务状态
- `created_at`: 任务创建时间（ISO 8601格式）
- `started_at`: 任务开始执行时间
- `completed_at`: 任务完成时间
- `execution_time`: 任务执行时长（秒）
- `image_urls`: 生成的图片下载链接列表
- `error_message`: 如果失败，包含错误信息
- `progress`: 任务进度百分比（0-100）
- `parameters`: 生成时使用的参数

### 3. 下载生成的图片

**接口路径**: `GET /sdapi/v1/turbo-gen/download/{filename}`

**描述**: 下载指定的生成图片文件

**路径参数**:

- `filename` (string): 图片文件路径（相对于outputs目录），例如 `txt2img-images/2023-12-10/00001.png`

**响应**: 图片文件（Content-Type: image/png）

## 使用示例

### Python 示例

```python
import requests
import time

# 服务器地址
BASE_URL = "http://localhost:50010"

# 1. 提交异步任务
response = requests.post(
    f"{BASE_URL}/sdapi/v1/turbo-gen/txt2img-async",
    json={
        "prompt": "a beautiful sunset over mountains",
        "negative_prompt": "blurry, bad quality",
        "steps": 30,
        "width": 768,
        "height": 768,
        "cfg_scale": 7.5,
        "sampler_name": "DPM++ 2M Karras",
        "seed": -1
    }
)

task_data = response.json()
task_id = task_data["task_id"]
print(f"任务已提交: {task_id}")

# 2. 轮询任务状态
while True:
    response = requests.get(f"{BASE_URL}/sdapi/v1/turbo-gen/task/{task_id}")
    task_status = response.json()
    
    status = task_status["status"]
    progress = task_status.get("progress", 0)
    
    print(f"状态: {status}, 进度: {progress}%")
    
    if status == "completed":
        print(f"任务完成! 执行时间: {task_status['execution_time']:.2f}秒")
        print(f"生成了 {len(task_status['image_urls'])} 张图片")
        
        # 3. 下载所有图片
        for i, image_url in enumerate(task_status["image_urls"], 1):
            full_url = f"{BASE_URL}{image_url}"
            img_response = requests.get(full_url)
            
            if img_response.status_code == 200:
                filename = f"generated_image_{i}.png"
                with open(filename, "wb") as f:
                    f.write(img_response.content)
                print(f"已保存: {filename}")
        
        break
    
    elif status == "failed":
        print(f"任务失败: {task_status.get('error_message')}")
        break
    
    # 等待5秒后再次查询
    time.sleep(5)
```

### cURL 示例

```bash
# 1. 提交任务
TASK_ID=$(curl -X POST "http://localhost:50010/sdapi/v1/turbo-gen/txt2img-async" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a cat sitting on a table",
    "steps": 20,
    "width": 512,
    "height": 512
  }' | jq -r '.task_id')

echo "Task ID: $TASK_ID"

# 2. 查询任务状态
curl "http://localhost:50010/sdapi/v1/turbo-gen/task/$TASK_ID"

# 3. 下载图片（假设图片路径为 txt2img-images/2023-12-10/00001.png）
curl "http://localhost:50010/sdapi/v1/turbo-gen/download/txt2img-images/2023-12-10/00001.png" \
  -o image.png
```

### JavaScript (Node.js) 示例

```javascript
const axios = require('axios');
const fs = require('fs');

const BASE_URL = 'http://localhost:50010';

async function generateImages() {
  // 1. 提交任务
  const submitResponse = await axios.post(
    `${BASE_URL}/sdapi/v1/turbo-gen/txt2img-async`,
    {
      prompt: 'a futuristic city at night',
      negative_prompt: 'blurry',
      steps: 25,
      width: 768,
      height: 512,
      cfg_scale: 7.0
    }
  );

  const taskId = submitResponse.data.task_id;
  console.log(`任务已提交: ${taskId}`);

  // 2. 轮询任务状态
  while (true) {
    const statusResponse = await axios.get(
      `${BASE_URL}/sdapi/v1/turbo-gen/task/${taskId}`
    );

    const taskStatus = statusResponse.data;
    console.log(`状态: ${taskStatus.status}, 进度: ${taskStatus.progress}%`);

    if (taskStatus.status === 'completed') {
      console.log(`任务完成! 执行时间: ${taskStatus.execution_time.toFixed(2)}秒`);

      // 3. 下载所有图片
      for (let i = 0; i < taskStatus.image_urls.length; i++) {
        const imageUrl = taskStatus.image_urls[i];
        const fullUrl = `${BASE_URL}${imageUrl}`;

        const imageResponse = await axios.get(fullUrl, {
          responseType: 'arraybuffer'
        });

        const filename = `generated_image_${i + 1}.png`;
        fs.writeFileSync(filename, imageResponse.data);
        console.log(`已保存: ${filename}`);
      }

      break;
    } else if (taskStatus.status === 'failed') {
      console.error(`任务失败: ${taskStatus.error_message}`);
      break;
    }

    // 等待5秒后再次查询
    await new Promise(resolve => setTimeout(resolve, 5000));
  }
}

generateImages().catch(console.error);
```

## API文档

启动服务后，可以访问自动生成的交互式API文档：

- **Swagger UI**: `http://localhost:50010/docs`
- **ReDoc**: `http://localhost:50010/redoc`

在文档中可以找到 `turbo-gen` 标签下的所有接口。

## 注意事项

1. **批次大小固定为4**: 无论请求中 `batch_size` 设置为多少，系统都会强制设置为4，每次任务会生成4张图片。

2. **图片保存位置**: 生成的图片会保存在 `outputs/txt2img-images/` 目录下，按日期自动分目录保存。

3. **任务持久化存储**: 任务信息存储在 MongoDB 中，服务重启后仍可查询历史任务。如果 MongoDB 不可用，会自动回退到内存模式。

4. **并发限制**: 任务会排队执行，同时只有一个任务在处理（使用 `queue_lock` 控制）。

5. **MongoDB 连接**: 确保 MongoDB 服务正常运行，否则系统会使用内存模式（任务数据不持久化）。

6. **资源清理**: 建议定期清理旧任务数据和图片文件：
   ```javascript
   // MongoDB Shell 示例：删除30天前的已完成任务
   db.turbo_gen_tasks.deleteMany({
     status: "completed",
     created_at: { $lt: new Date(Date.now() - 30*24*60*60*1000).toISOString() }
   })
   ```

7. **错误处理**: 如果任务失败，`error_message` 字段会包含详细的错误信息。

8. **安全性**: 图片下载接口包含路径验证，确保只能访问 `outputs` 目录下的文件。

## 常见问题

**Q: 为什么任务状态一直是 pending？**

A: 可能有其他任务正在执行。任务会按顺序排队处理。

**Q: 如何取消正在运行的任务？**

A: 当前版本不支持取消任务。可以使用现有的 `/sdapi/v1/interrupt` 接口中断当前正在执行的任务。

**Q: 图片URL的有效期是多久？**

A: 图片URL永久有效，直到对应的文件被删除。

**Q: 可以同时提交多个任务吗？**

A: 可以，但任务会按提交顺序排队执行，不会并行处理。

**Q: 如何获取所有任务的列表？**

A: 可以直接查询 MongoDB：
```javascript
// 获取最近10个任务
db.turbo_gen_tasks.find().sort({created_at: -1}).limit(10)

// 获取所有运行中的任务
db.turbo_gen_tasks.find({status: "running"})

// 获取所有失败的任务
db.turbo_gen_tasks.find({status: "failed"})
```

**Q: MongoDB 连接失败怎么办？**

A: 系统会自动回退到内存模式，功能正常可用，但任务数据在服务重启后会丢失。检查：
- MongoDB 服务是否运行
- 连接字符串是否正确
- 防火墙设置
- pymongo 是否已安装

**Q: 如何查看 MongoDB 中的任务数据？**

A: 使用 MongoDB Compass（GUI工具）或命令行：
```bash
# 连接到MongoDB
mongosh "mongodb://admin:yourpassword@localhost:27017/"

# 切换到数据库
use stable_diffusion

# 查看任务
db.turbo_gen_tasks.find().pretty()

# 统计任务状态
db.turbo_gen_tasks.aggregate([
  { $group: { _id: "$status", count: { $sum: 1 } } }
])
```

**Q: 如何提高性能？**

A: 
1. 确保 MongoDB 索引已创建（系统会自动创建）
2. 定期清理旧任务数据
3. 考虑使用 MongoDB 副本集提高可用性
4. 对于高并发场景，可以部署多个 worker 实例处理任务队列

## MongoDB 数据结构

任务文档结构：
```json
{
  "task_id": "uuid",
  "status": "pending|running|completed|failed",
  "created_at": "ISO8601 datetime",
  "started_at": "ISO8601 datetime or null",
  "completed_at": "ISO8601 datetime or null", 
  "execution_time": "float or null",
  "image_urls": ["url1", "url2", "url3", "url4"],
  "error_message": "string or null",
  "progress": "float 0-100",
  "parameters": {
    "prompt": "...",
    "steps": 20,
    ...
  }
}
```

## 生产环境建议

1. **MongoDB 部署**：使用副本集或 MongoDB Atlas 确保高可用性
2. **备份策略**：定期备份 MongoDB 数据和图片文件
3. **监控告警**：监控 MongoDB 连接状态、任务队列长度、失败率
4. **日志记录**：启用详细日志记录便于问题排查
5. **资源限制**：设置 MongoDB 连接池大小、任务并发数限制
