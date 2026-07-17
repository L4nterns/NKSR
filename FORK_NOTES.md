# NKSR Fork Notes

这个 fork 保留了运行预训练推理所需的最小改动，并补充了面向自定义 `.pcd` / `.ply` 数据的命令行推理入口。

## 保留的改动

1. 依赖兼容性修复

- `requirements.txt`
  - 将 `torch-scatter` 从默认依赖安装中移除，避免在较低 `glibc` 环境下误装不兼容的预编译 wheel
- `pyproject.toml` / `uv.lock`
  - 使用 `uv` 管理 Python 运行时依赖，并锁定 PyTorch CUDA 12.4 wheel 源
  - 保留 `torch==2.6.0+cu124`、`torchvision==0.21.0+cu124` 为默认 PyTorch 组合，匹配 NVIDIA Driver 550.x / CUDA 12.4 服务器
  - 锁定可复现的普通 Python 依赖；`torch-scatter` 和 `./package` 的 CUDA 扩展仍在目标机器上显式编译安装
  - Docker 构建时禁用缓存并源码安装 `torch-scatter`，随后在最终环境导入校验，避免 PyTorch/CUDA 版本调整后沿用旧 ABI 构建产物

2. 原仓库推理可用性修复

- `package/nksr/__init__.py`
  - 修正 `get_estimate_normal_preprocess_fn()` 对 `estimate_normals()` 的调用
- `package/nksr/svh.py`
  - 将 `pycg.vis` 改为延迟导入，避免普通推理路径被 `open3d` 链式硬依赖阻塞

3. 新增最小推理入口

- `scripts/infer_pcd.py`
  - 单文件推理入口
- `scripts/batch_infer_pcd.py`
  - 批量推理入口
- `scripts/http_infer_pcd.py`
  - FastAPI HTTP 建模服务入口，接口为 `POST /v1/reconstruct`
  - 同步返回 zip；zip 内包含输出 `.ply` 和 `.stats.json`
  - 请求处理使用临时目录保存中间文件，响应结束后自动删除，不持久化推理文件
  - 推理 device 和建模参数只从服务环境变量读取，HTTP 请求不能切换 GPU、调资源参数或传本地路径
  - 服务进程内对推理请求串行执行，避免同一 GPU 上多个 NKSR job 同时竞争显存；需要多 GPU 并发时建议按 GPU 启多个服务进程，由上游做负载均衡
- `tools/pcd_inference.py`
  - 复用的核心推理逻辑
  - 分块建模按点数和可选空间尺寸控制显存峰值，默认 `NKSR_MAX_CHUNK_POINTS=0` 不分块，`NKSR_MAX_CHUNK_INPUT_EXTENT` 关闭
  - 分块模式下输入点云先保留在 CPU，按 chunk 进入 GPU；每个 chunk 完成后释放中间张量并清理 CUDA cache
  - 默认 `NKSR_TEXTURE_MODE=auto`，有颜色字段时保留顶点颜色；需要极限省显存时可显式设置 `NKSR_TEXTURE_MODE=off`
  - 默认 `NKSR_FUSED_MODE=true`，保持原 NKSR fused kernel 求解路径；需要排查求解路径差异时可显式关闭
  - 默认 `NKSR_RECONSTRUCT_ATTEMPTS=5`，仅当 mesh 提取为空时用同一参数重新构建隐式场；用于处理 CUDA 稀疏求解未收敛时的偶发空 mesh
  - 输出 chunk 规划和 chunk 点数/包围盒日志，便于定位异常输入和性能瓶颈

## 功能边界

当前推理入口支持：

- 单文件推理
- 批量目录推理
- HTTP 单文件建模服务
- `--device`，指定推理设备，例如 `cuda:0` 或 `cpu`；默认值为 `cuda:0`
- `--config`，选择内置预训练配置：`ks`、`snet`、`snet-wonormal`；默认值见 `.env.example`
- `--detail-level`，控制重建细节等级；默认值见 `.env.example`
- `--voxel-size`，显式覆盖最细体素尺寸；设置后会忽略 `--detail-level`
- `--max-chunk-extent`，限制单个 chunk 的最大空间尺寸；默认关闭
- `--max-chunk-input-extent`，限制单个 chunk 的输入空间尺寸；默认关闭
- `--max-chunk-points`，限制单个 chunk 的最大点数；默认值为 `0`，即不按点数分块
- `--chunk-overlap-ratio`，控制自适应分块的重叠比例；默认值为 `0.05`
- `--chunk-tmp-device`，控制已完成 chunk 的临时存储设备；默认值为 `cpu`
- `--texture-mode auto|off`，控制颜色纹理导出；默认 `auto` 保留颜色，`off` 跳过顶点颜色以降低资源占用
- `--obb-lock-axis auto|none|x|y|z|pca`，按导出坐标系控制网格方向；`auto` 锁定 `--output-up-axis` 指定的竖轴，`none` 不做 OBB 旋转归一化，`x|y|z` 表示锁定指定轴并在另外两个轴所在平面内做 2D OBB 对齐，`pca` 按 PCA 主轴做 3D OBB 对齐；默认值见 `.env.example`
- `--output-up-axis y|z`，控制导出网格的上轴；`y` 会将 z-up 结果转换为 y-up，适配 three.js 常见场景；默认值见 `.env.example`
- `--simplify-method auto|quadric|clustering|none`，控制减面方式；默认值为 `auto`，优先使用基于 `open3d` 的 `quadric` 减面
- `--face-ratio`，按比例减面；只有当 `--simplify-method` 不是 `none` 且 `--face-ratio < 1.0` 时才会真正执行减面；默认值为 `0.1`
- `--fused-mode` / `--no-fused-mode`，控制 kernel 求解路径；默认 `--fused-mode`，保持原 NKSR 默认行为
- `--skip-existing`，仅用于批量命令；当目标输出文件已存在时直接跳过，避免重复处理；默认值见 `.env.example`
- 输出 `*.stats.json`，用于记录包围盒尺寸、体积和导出网格统计信息

`*.stats.json` 包含：

- `input_path` / `output_path`，输入点云和导出网格路径
- `bbox_type` / `obb_lock_axis`，包围盒类型和锁定轴设置
- `output_up_axis` / `axis_transform`，导出坐标系及其变换类型
- `normalization_rotation_degrees`，导出 mesh 局部归一化时的 OBB 旋转角度；该角度描述本次 mesh 导出变换，不表示业务系统中的物体朝向
- `x_length` / `y_length` / `z_length`，导出坐标系下包围盒三轴长度
- `bbox_volume`，包围盒体积，不是物体真实体积
- `bbox_min` / `bbox_max` / `bbox_center`，导出坐标系下包围盒范围与中心
- `normalization_center` / `normalization_bbox_center` / `normalization_rotation_matrix`，mesh 局部归一化的中心、旋转后 bbox 中心和旋转矩阵
- `transform_convention` / `input_to_export_matrix` / `export_to_input_matrix`，4x4 齐次变换矩阵及其约定；矩阵用于在输入点云坐标和导出 mesh 坐标之间转换
- `axis_transform_matrix`，导出上轴变换矩阵
- `face_ratio`，请求的减面比例
- `requested_simplify_method` / `applied_simplify_method`，请求的减面方式和最终实际应用的减面方式
- `raw_vertices` / `raw_faces`，减面前网格规模
- `export_vertices` / `export_faces`，导出网格规模

## GPU 使用

多卡机器上的显卡切换建议：

- 如果直接传 `--device cuda:1`、`cuda:2`、`cuda:3`，某些 NKSR CUDA 扩展路径可能不稳定
- 更稳的做法是先用 `CUDA_VISIBLE_DEVICES` 选择物理卡，再在脚本里统一使用 `--device cuda:0`
- 例如使用物理 `GPU 1`：`CUDA_VISIBLE_DEVICES=1 python scripts/infer_pcd.py ... --device cuda:0`
- 如果分两行写，记得使用 `export CUDA_VISIBLE_DEVICES=1`，否则后续 `python` 进程可能看不到这个变量

## 安装建议

建议使用 `uv` 管理 Python 虚拟环境和可锁定依赖；CUDA/NVCC 仍由系统环境提供。`environment.yml` 可作为服务器 CUDA 编译依赖参考，不再作为推荐的 Python 包管理入口。

基础安装示例：

```bash
uv sync --locked
export CUDA_HOME=/usr/local/cuda  # 请与本机 CUDA 安装路径保持一致
export TORCH_CUDA_ARCH_LIST="8.6"  # RTX 3090 / Ampere; 请按实际 GPU 架构调整
uv pip install --no-cache --no-binary torch-scatter --no-build-isolation torch-scatter
uv pip install --no-build-isolation ./package
uv run python -c "import nksr, torch, torch_scatter; print(torch.__version__, torch_scatter.__version__, nksr.__version__)"
```

如果本机较老、`glibc` 版本偏低，建议始终让 `torch-scatter` 在本机源码编译，不要直接安装预编译 wheel。
如果源码包下载有问题，可改用：`uv pip install --no-build-isolation git+https://github.com/rusty1s/pytorch_scatter.git`

## 预训练模型

当前命令行入口支持 3 个内置预训练配置，可通过 `--config` 选择：

- `ks`：通用的 `kitchen-sink` 模型，依赖法向输入，默认重建尺度更偏通用场景，适合作为自定义数据的首选基线
- `snet`：更偏单物体重建，依赖法向输入，默认体素尺度更细，通常更适合干净的物体级点云
- `snet-wonormal`：`snet` 的无法向版本，适合输入没有可靠法向时使用

推理前需要把 `NKSR_CONFIG` 对应的权重文件放入 `NKSR_CHECKPOINTS_DIR`：

| `NKSR_CONFIG` | 文件名 | 下载地址 |
|---|---|---|
| `ks` | `ks.pth` | https://huggingface.co/heiwang1997/nksr-checkpoints/resolve/main/checkpoints/ks.pth |
| `snet` | `snet-n3k-wnormal.pth` | https://huggingface.co/heiwang1997/nksr-checkpoints/resolve/main/checkpoints/snet-n3k-wnormal.pth |
| `snet-wonormal` | `snet-n3k-wonormal.pth` | https://huggingface.co/heiwang1997/nksr-checkpoints/resolve/main/checkpoints/snet-n3k-wonormal.pth |

选择建议：

- 场景扫描、尺度不稳定或先跑通流程时，优先使用 `ks`
- 单物体、法向质量较好、希望保留更多几何细节时，优先使用 `snet`
- 没有法向或法向质量较差时，使用 `snet-wonormal`

## 推理命令

`scripts/infer_pcd.py`、`scripts/batch_infer_pcd.py` 和 `scripts/http_infer_pcd.py` 都会自动加载 `.env`。加载顺序是当前工作目录 `.env`，再尝试仓库根目录 `.env`；已存在的系统环境变量不会被 `.env` 覆盖。
仓库提供 `.env.example`，可以复制为 `.env` 后按部署环境调整。

配置优先级：

- 单文件/批量 CLI：命令行参数优先，其次环境变量/`.env`，最后使用脚本内默认值
- HTTP 服务：请求只上传文件，推理策略全部来自环境变量/`.env`

环境变量、可选值和参数含义统一写在 `.env.example` 注释里，避免 HTTP 参数、CLI 参数和部署环境混在文档正文里。

单文件：

```bash
uv run python scripts/infer_pcd.py ~/pcds/example.pcd ~/plys/example.ply \
  --device cuda:0 \
  --config snet-wonormal \
  --detail-level 1 \
  --max-chunk-points 0 \
  --chunk-overlap-ratio 0.05 \
  --obb-lock-axis auto \
  --output-up-axis y \
  --simplify-method none \
  --face-ratio 1.0
```

批量：

```bash
uv run python scripts/batch_infer_pcd.py ~/pcds ~/plys \
  --device cuda:0 \
  --config snet-wonormal \
  --detail-level 1 \
  --max-chunk-points 0 \
  --chunk-overlap-ratio 0.05 \
  --obb-lock-axis auto \
  --output-up-axis y \
  --simplify-method none \
  --face-ratio 1.0 \
  --skip-existing
```

## HTTP 推理接口

推荐使用 Docker Compose 启动 HTTP 服务：

```bash
docker compose up -d
```

默认 `docker compose up -d` 只启动 `nksr-http`。调试 shell 不随默认服务启动，需要显式使用 `tools` profile：

```bash
docker compose --profile tools run --rm shell
```

查看日志：

```bash
docker compose logs -f nksr-http
```

停止服务：

```bash
docker compose down
```

本地开发或临时调试也可以直接运行：

```bash
uv run --group http python scripts/http_infer_pcd.py --host 0.0.0.0 --port 8012
```

后台启动：

```bash
mkdir -p logs run
nohup uv run --group http python scripts/http_infer_pcd.py --host 0.0.0.0 --port 8012 \
  > logs/nksr-http.log 2>&1 &
echo $! > run/nksr-http.pid
```

停止后台服务：

```bash
kill "$(cat run/nksr-http.pid)"
```

HTTP 服务环境变量见 `.env.example`。

HTTP endpoint：

```text
POST /v1/reconstruct
GET  /health
```

HTTP 请求表单字段：

```text
object_pcd  必填，上传单个 .pcd 或 .ply 点云
request_id  可选，用于响应头、临时目录和返回 zip 文件名的安全前缀，不会持久化
```

设置 `HTTP_AUTH_TOKEN` 后，`POST /v1/reconstruct` 请求必须带 `Authorization: Bearer <token>`；不设置时不启用鉴权。

HTTP 请求不能覆盖建模策略、GPU 或本地路径相关配置；这些配置全部由服务启动环境控制。需要调整时，改 `.env` 或进程环境后重启服务。

HTTP 建模请求：

```bash
curl -X POST http://127.0.0.1:8012/v1/reconstruct \
  -H "Authorization: Bearer $HTTP_AUTH_TOKEN" \
  -F "object_pcd=@~/pcds/example.pcd" \
  -o result.zip
```

HTTP 成功返回 `application/zip`，响应头包含 `X-Request-ID`。zip 文件名为 `<request_id>_nksr.zip`；未传 `request_id` 时使用服务生成的随机 ID。zip 内包含输出 `.ply`，`write_stats=true` 时还包含同名 `.stats.json`。

HTTP 错误返回 JSON：

```json
{
  "ok": false,
  "error": {
    "code": "bad_request",
    "message": "object_pcd 只支持 .pcd 或 .ply 文件。"
  },
  "request_id": "example"
}
```

多 GPU HTTP 部署建议按“一张物理 GPU 一个服务进程”拆分，进程内统一使用 `DEVICE=cuda:0`，通过 `CUDA_VISIBLE_DEVICES` 选择物理卡。例如：

```bash
CUDA_VISIBLE_DEVICES=0 DEVICE=cuda:0 HTTP_PORT=8012 uv run --group http python scripts/http_infer_pcd.py
CUDA_VISIBLE_DEVICES=1 DEVICE=cuda:0 HTTP_PORT=8013 uv run --group http python scripts/http_infer_pcd.py
```
