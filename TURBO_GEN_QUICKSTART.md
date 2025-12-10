# Turbo-Gen 快速启动指南

## 一键启动（推荐）

使用 Docker Compose 同时启动 Stable Diffusion WebUI、MongoDB 和 Mongo Express 管理界面：

```bash
# 启动所有服务
docker compose -f docker-compose-with-mongodb.yml up -d

# 查看日志
docker compose -f docker-compose-with-mongodb.yml logs -f

# 停止服务
docker compose -f docker-compose-with-mongodb.yml down
```

启动后可以访问：
- **SD WebUI**: http://localhost:50010
- **API 文档**: http://localhost:50010/docs
- **Mongo Express**: http://localhost:8081 (用户名/密码: admin/admin)

## 分步启动

### 1. 启动 MongoDB

```bash
docker run -d \
  --name sd-mongodb \
  -p 27017:27017 \
  -e MONGO_INITDB_ROOT_USERNAME=admin \
  -e MONGO_INITDB_ROOT_PASSWORD=yourStrongPassword123 \
  -v mongodb_data:/data/db \
  mongo:7.0
```

### 2. 配置环境变量

**Linux/Mac:**
```bash
export MONGODB_URI="mongodb://admin:yourStrongPassword123@localhost:27017/"
export MONGODB_DATABASE="stable_diffusion"
export MONGODB_COLLECTION="turbo_gen_tasks"
```

**Windows PowerShell:**
```powershell
$env:MONGODB_URI="mongodb://admin:yourStrongPassword123@localhost:27017/"
$env:MONGODB_DATABASE="stable_diffusion"
$env:MONGODB_COLLECTION="turbo_gen_tasks"
```

### 3. 启动 SD WebUI

```bash
# 确保已安装 pymongo
pip install pymongo>=4.6.0

# 启动服务
python webui.py --api --listen
```

## 验证安装

### 1. 检查 MongoDB 连接

启动 SD WebUI 时应该看到：
```
✓ MongoDB连接成功: mongodb://localhost:27017/
✓ MongoDB集合初始化完成: stable_diffusion.turbo_gen_tasks
```

如果看到警告，检查 MongoDB 是否正在运行：
```bash
docker ps | grep mongodb
```

### 2. 测试 API

提交一个测试任务：
```bash
curl -X POST "http://localhost:50010/sdapi/v1/turbo-gen/txt2img-async" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a beautiful sunset",
    "steps": 20
  }'
```

应该返回：
```json
{
  "task_id": "uuid-here",
  "message": "任务已创建，任务ID: uuid-here"
}
```

### 3. 查询任务状态

```bash
curl "http://localhost:50010/sdapi/v1/turbo-gen/task/{task_id}"
```

### 4. 检查 MongoDB 数据

使用 Mongo Express（浏览器访问 http://localhost:8081）或命令行：
```bash
mongosh "mongodb://admin:yourStrongPassword123@localhost:27017/"

use stable_diffusion
db.turbo_gen_tasks.find().pretty()
```

## 故障排除

### MongoDB 连接失败

**症状**：启动时显示 "⚠ MongoDB连接失败，将使用内存模式"

**解决方案**：
1. 确认 MongoDB 正在运行：
   ```bash
   docker ps | grep mongodb
   ```

2. 检查连接字符串是否正确：
   ```bash
   echo $MONGODB_URI
   ```

3. 测试 MongoDB 连接：
   ```bash
   mongosh "$MONGODB_URI"
   ```

4. 检查防火墙设置（确保 27017 端口开放）

### pymongo 未安装

**症状**：启动时显示 "⚠ pymongo未安装，使用内存模式存储任务"

**解决方案**：
```bash
pip install pymongo>=4.6.0
```

### 权限错误

**症状**：MongoDB 连接时出现认证错误

**解决方案**：
1. 检查用户名密码是否正确
2. 确保连接字符串格式正确：
   ```
   mongodb://用户名:密码@主机:端口/
   ```

### 索引创建失败

**症状**：日志中显示索引创建错误

**解决方案**：
手动创建索引：
```javascript
db.turbo_gen_tasks.createIndex({task_id: 1}, {unique: true})
db.turbo_gen_tasks.createIndex({status: 1})
db.turbo_gen_tasks.createIndex({created_at: -1})
```

## 性能优化

### 1. MongoDB 配置优化

编辑 `/etc/mongod.conf`：
```yaml
storage:
  wiredTiger:
    engineConfig:
      cacheSizeGB: 2  # 根据可用内存调整

net:
  maxIncomingConnections: 1000

operationProfiling:
  mode: slowOp
  slowOpThresholdMs: 100
```

### 2. 定期清理旧数据

创建 cron 任务清理 30 天前的已完成任务：
```javascript
// cleanup_script.js
db.turbo_gen_tasks.deleteMany({
  status: "completed",
  created_at: { 
    $lt: new Date(Date.now() - 30*24*60*60*1000).toISOString() 
  }
})
```

执行：
```bash
mongosh "$MONGODB_URI" cleanup_script.js
```

### 3. 监控任务队列

```javascript
// 查看各状态任务数量
db.turbo_gen_tasks.aggregate([
  { $group: { _id: "$status", count: { $sum: 1 } } }
])

// 查看平均执行时间
db.turbo_gen_tasks.aggregate([
  { $match: { status: "completed" } },
  { $group: { _id: null, avgTime: { $avg: "$execution_time" } } }
])
```

## 下一步

1. 阅读完整 API 文档：`TURBO_GEN_API_GUIDE.md`
2. 下载模型文件到 `models/Stable-diffusion/` 目录
3. 访问 API 文档页面：http://localhost:50010/docs
4. 开始使用 Turbo-Gen 异步接口！

## 生产环境部署

对于生产环境，建议：

1. **使用 MongoDB 副本集**确保高可用性
2. **配置备份策略**定期备份数据和图片
3. **启用 SSL/TLS** 加密连接
4. **设置访问控制**限制网络访问
5. **监控系统**监控 MongoDB 性能和任务队列
6. **负载均衡**使用 Nginx 或 HAProxy 分发请求

示例 MongoDB 副本集配置请参考 MongoDB 官方文档：
https://www.mongodb.com/docs/manual/replication/
