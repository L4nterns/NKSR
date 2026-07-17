#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import logging
import math
import os
import secrets
import shutil
import sys
import tempfile
import threading
import uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.pcd_inference import (
    NKSR_CONFIG_CHOICES,
    ReconstructionFailedError,
    checkpoint_file_name,
    configured_checkpoint_path,
    env_bool,
    env_float,
    env_int,
    env_optional_float,
    env_str,
    infer_one,
    load_env_file,
)


load_env_file()


LOGGER = logging.getLogger(__name__)
INFER_LOCK = threading.Lock()


class CheckpointUnavailableError(RuntimeError):
    pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = configured_config_name()
    LOGGER.info("checking NKSR checkpoint: config=%s", config)
    ensure_checkpoint_available(config)
    LOGGER.info("NKSR checkpoint ready: config=%s", config)
    yield


app = FastAPI(title="NKSR HTTP 建模服务", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    request.state.request_id = uuid.uuid4().hex
    if request.url.path == "/v1/reconstruct":
        token = os.environ.get("HTTP_AUTH_TOKEN", "").strip()
        authorization = request.headers.get("authorization")
        if token and not valid_bearer_token(authorization, token):
            return error_response(401, "unauthorized", "缺少或无效的 Authorization Bearer token。", request.state.request_id)

    response = await call_next(request)
    request_id = getattr(request.state, "request_id", None)
    if request_id and "x-request-id" not in response.headers:
        response.headers["X-Request-ID"] = header_request_id(request_id)
    return response


def error_response(status_code: int, code: str, message: str, request_id: str | None = None) -> JSONResponse:
    payload = {"ok": False, "error": {"code": code, "message": message}}
    if request_id:
        payload["request_id"] = request_id
    headers = {"X-Request-ID": header_request_id(request_id)} if request_id else {}
    if status_code == 401:
        headers["WWW-Authenticate"] = "Bearer"
    return JSONResponse(payload, status_code=status_code, headers=headers)


def http_error_code(status_code: int) -> str:
    if status_code == 401:
        return "unauthorized"
    if status_code == 403:
        return "forbidden"
    if status_code == 404:
        return "not_found"
    if status_code == 405:
        return "method_not_allowed"
    if status_code == 422:
        return "validation_error"
    return "bad_request" if status_code < 500 else "server_error"


def validation_error_message(exc: RequestValidationError) -> str:
    messages = []
    for error in exc.errors():
        loc = ".".join(str(item) for item in error.get("loc", []) if item != "body")
        msg = error.get("msg", "参数无效")
        messages.append(f"{loc}: {msg}" if loc else str(msg))
    return "；".join(messages) or "请求参数校验失败。"


@app.exception_handler(HTTPException)
def handle_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    message = str(exc.detail)
    code = http_error_code(exc.status_code)
    return error_response(exc.status_code, code, message, request_id)


@app.exception_handler(RequestValidationError)
def handle_validation_exception(request: Request, exc: RequestValidationError) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    return error_response(422, "validation_error", validation_error_message(exc), request_id)


@app.exception_handler(Exception)
def handle_unexpected_exception(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    if isinstance(exc, CheckpointUnavailableError):
        LOGGER.error("NKSR checkpoint 不可用 request_id=%s: %s", request_id, exc)
        return error_response(503, "checkpoint_unavailable", str(exc), request_id)
    if is_cuda_oom(exc):
        device = env_str("DEVICE", "cuda:0")
        LOGGER.error(
            "NKSR HTTP 请求处理失败：CUDA 显存不足 request_id=%s",
            request_id,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        cleanup_cuda_memory(device)
        return error_response(503, "cuda_oom", "当前 GPU 显存不足，请检查 GPU 占用或资源配置。", request_id)
    LOGGER.error("NKSR HTTP 请求处理失败", exc_info=(type(exc), exc, exc.__traceback__))
    return error_response(500, "internal_error", "服务内部错误，请查看服务日志。", request_id)


def safe_token(value: str, fallback: str, max_len: int = 80) -> str:
    token = Path(value).name
    token = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in token).strip("._-")
    return (token[:max_len] or fallback)


def header_request_id(value: str, max_len: int = 80) -> str:
    token = Path(value).name
    token = "".join(
        ch if ch.isascii() and (ch.isalnum() or ch in "._-") else "_"
        for ch in token
    ).strip("._-")
    return (token[:max_len] or uuid.uuid4().hex)


def safe_stem(name: str, fallback: str) -> str:
    return safe_token(Path(name).stem, fallback)


def cleanup_dir(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def cleanup_cuda_memory(device_name: str) -> None:
    gc.collect()
    try:
        import torch
    except Exception:
        return

    if not torch.cuda.is_available():
        return

    try:
        device = torch.device(device_name)
    except Exception:
        device = None

    try:
        if device is not None and device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        else:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        LOGGER.warning("CUDA 显存清理失败", exc_info=True)


def iter_exception_chain(exc: BaseException):
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def is_cuda_oom(exc: Exception) -> bool:
    for current in iter_exception_chain(exc):
        if current.__class__.__name__ == "OutOfMemoryError" and current.__class__.__module__.startswith("torch"):
            return True
        if "CUDA out of memory" in str(current):
            return True
    return False


def valid_bearer_token(authorization: str | None, token: str) -> bool:
    if authorization is None:
        return False
    scheme, separator, credential = authorization.strip().partition(" ")
    return bool(separator) and scheme.lower() == "bearer" and secrets.compare_digest(credential.strip(), token)


def validate_choice(name: str, value: str, choices: set[str], status_code: int = 400) -> None:
    if value not in choices:
        allowed = ", ".join(sorted(choices))
        raise HTTPException(status_code=status_code, detail=f"{name} 必须是以下值之一：{allowed}。")


def configured_config_name() -> str:
    config = env_str("NKSR_CONFIG", "snet-wonormal")
    if config not in NKSR_CONFIG_CHOICES:
        allowed = ", ".join(NKSR_CONFIG_CHOICES)
        raise CheckpointUnavailableError(f"NKSR_CONFIG 必须是以下值之一：{allowed}。")
    return config


def ensure_checkpoint_available(config: str) -> None:
    checkpoint_path = configured_checkpoint_path(config)
    if checkpoint_path is None:
        raise CheckpointUnavailableError(
            "NKSR_CHECKPOINTS_DIR 未配置，请配置本地 checkpoint 目录。"
        )
    if not checkpoint_path.is_file():
        raise CheckpointUnavailableError(
            f"NKSR checkpoint 文件不存在：{checkpoint_path}。"
            f"当前 NKSR_CONFIG={config} 需要 {checkpoint_file_name(config)}。"
        )


def validate_finite(name: str, value: float, status_code: int = 400) -> None:
    if not math.isfinite(value):
        raise HTTPException(status_code=status_code, detail=f"{name} 必须是有限数值。")


def validate_request(
    upload: UploadFile,
) -> None:
    suffix = Path(upload.filename or "object.pcd").suffix.lower()
    if suffix and suffix not in {".pcd", ".ply"}:
        raise HTTPException(status_code=400, detail="object_pcd 只支持 .pcd 或 .ply 文件。")


def build_infer_args(
    input_path: Path,
    output_path: Path,
    *,
    device: str,
    config: str,
    input_mode: str,
    xyz_cols: str,
    normal_cols: str,
    sensor_cols: str,
    color_cols: str,
    obb_lock_axis: str,
    write_stats: bool,
    simplify_method: str,
    face_ratio: float,
    output_up_axis: str,
    detail_level: float,
    voxel_size: float | None,
    max_chunk_extent: float | None,
    max_chunk_points: int,
    chunk_overlap_ratio: float,
    chunk_tmp_device: str,
    max_input_extent: float,
    max_chunk_input_extent: float | None,
    outlier_quantile: float,
    min_filtered_points: int,
    mise_iter: int,
    grid_upsample: int,
    extract_on_cpu: bool,
    texture_mode: str,
    estimate_knn: int,
    estimate_orient: str,
    drop_threshold_deg: float,
    approx_kernel_grad: bool,
    solver_max_iter: int,
    solver_tol: float,
    nystrom_min_depth: int,
    reconstruct_attempts: int,
    fused_mode: bool,
) -> argparse.Namespace:
    return argparse.Namespace(
        input=input_path,
        output=output_path,
        device=device,
        config=config,
        input_mode=input_mode,
        xyz_cols=xyz_cols,
        normal_cols=normal_cols,
        sensor_cols=sensor_cols,
        color_cols=color_cols,
        obb_lock_axis=obb_lock_axis,
        write_stats=write_stats,
        simplify_method=simplify_method,
        face_ratio=face_ratio,
        output_up_axis=output_up_axis,
        detail_level=detail_level,
        voxel_size=voxel_size,
        max_chunk_extent=max_chunk_extent,
        max_chunk_points=max_chunk_points,
        chunk_overlap_ratio=chunk_overlap_ratio,
        chunk_tmp_device=chunk_tmp_device,
        max_input_extent=max_input_extent,
        max_chunk_input_extent=max_chunk_input_extent,
        outlier_quantile=outlier_quantile,
        min_filtered_points=min_filtered_points,
        mise_iter=mise_iter,
        grid_upsample=grid_upsample,
        extract_on_cpu=extract_on_cpu,
        texture_mode=texture_mode,
        estimate_knn=estimate_knn,
        estimate_orient=estimate_orient,
        drop_threshold_deg=drop_threshold_deg,
        approx_kernel_grad=approx_kernel_grad,
        solver_max_iter=solver_max_iter,
        solver_tol=solver_tol,
        nystrom_min_depth=nystrom_min_depth,
        reconstruct_attempts=reconstruct_attempts,
        fused_mode=fused_mode,
    )


def configured_infer_args(input_path: Path, output_path: Path) -> argparse.Namespace:
    args = build_infer_args(
        input_path,
        output_path,
        device=env_str("DEVICE", "cuda:0"),
        config=configured_config_name(),
        input_mode=env_str("NKSR_INPUT_MODE", "auto"),
        xyz_cols=env_str("NKSR_XYZ_COLS", "x,y,z"),
        normal_cols=env_str("NKSR_NORMAL_COLS", "normal_x,normal_y,normal_z"),
        sensor_cols=env_str("NKSR_SENSOR_COLS", "sensor_x,sensor_y,sensor_z"),
        color_cols=env_str("NKSR_COLOR_COLS", "red,green,blue"),
        obb_lock_axis=env_str("NKSR_OBB_LOCK_AXIS", "auto"),
        write_stats=env_bool("NKSR_WRITE_STATS", True),
        simplify_method=env_str("NKSR_SIMPLIFY_METHOD", "auto"),
        face_ratio=env_float("NKSR_FACE_RATIO", 0.1),
        output_up_axis=env_str("NKSR_OUTPUT_UP_AXIS", "y"),
        detail_level=env_float("NKSR_DETAIL_LEVEL", 1.0),
        voxel_size=env_optional_float("NKSR_VOXEL_SIZE"),
        max_chunk_extent=env_optional_float("NKSR_MAX_CHUNK_EXTENT"),
        max_chunk_points=env_int("NKSR_MAX_CHUNK_POINTS", 0),
        chunk_overlap_ratio=env_float("NKSR_CHUNK_OVERLAP_RATIO", 0.05),
        chunk_tmp_device=env_str("NKSR_CHUNK_TMP_DEVICE", "cpu"),
        max_input_extent=env_float("NKSR_MAX_INPUT_EXTENT", 200.0),
        max_chunk_input_extent=env_optional_float("NKSR_MAX_CHUNK_INPUT_EXTENT"),
        outlier_quantile=env_float("NKSR_OUTLIER_QUANTILE", 0.0),
        min_filtered_points=env_int("NKSR_MIN_FILTERED_POINTS", 1000),
        mise_iter=env_int("NKSR_MISE_ITER", 1),
        grid_upsample=env_int("NKSR_GRID_UPSAMPLE", 1),
        extract_on_cpu=env_bool("NKSR_EXTRACT_ON_CPU", False),
        texture_mode=env_str("NKSR_TEXTURE_MODE", "auto"),
        estimate_knn=env_int("NKSR_ESTIMATE_KNN", 64),
        estimate_orient=env_str("NKSR_ESTIMATE_ORIENT", "centroid"),
        drop_threshold_deg=env_float("NKSR_DROP_THRESHOLD_DEG", 85.0),
        approx_kernel_grad=env_bool("NKSR_APPROX_KERNEL_GRAD", False),
        solver_max_iter=env_int("NKSR_SOLVER_MAX_ITER", 2000),
        solver_tol=env_float("NKSR_SOLVER_TOL", 1.0e-5),
        nystrom_min_depth=env_int("NKSR_NYSTROM_MIN_DEPTH", 100),
        reconstruct_attempts=env_int("NKSR_RECONSTRUCT_ATTEMPTS", 5),
        fused_mode=env_bool("NKSR_FUSED_MODE", True),
    )
    validate_configured_args(args)
    return args


def validate_configured_args(args: argparse.Namespace) -> None:
    validate_choice("NKSR_CONFIG", args.config, set(NKSR_CONFIG_CHOICES), status_code=500)
    validate_choice("NKSR_INPUT_MODE", args.input_mode, {"auto", "normal", "sensor", "estimate"}, status_code=500)
    validate_choice("NKSR_OBB_LOCK_AXIS", args.obb_lock_axis, {"auto", "none", "x", "y", "z", "pca"}, status_code=500)
    validate_choice("NKSR_SIMPLIFY_METHOD", args.simplify_method, {"auto", "quadric", "clustering", "none"}, status_code=500)
    validate_choice("NKSR_OUTPUT_UP_AXIS", args.output_up_axis, {"z", "y"}, status_code=500)
    validate_choice("NKSR_TEXTURE_MODE", args.texture_mode, {"auto", "off"}, status_code=500)
    validate_choice("NKSR_ESTIMATE_ORIENT", args.estimate_orient, {"none", "centroid"}, status_code=500)

    for name, value in (
        ("NKSR_FACE_RATIO", args.face_ratio),
        ("NKSR_DETAIL_LEVEL", args.detail_level),
        ("NKSR_CHUNK_OVERLAP_RATIO", args.chunk_overlap_ratio),
        ("NKSR_MAX_INPUT_EXTENT", args.max_input_extent),
        ("NKSR_OUTLIER_QUANTILE", args.outlier_quantile),
        ("NKSR_DROP_THRESHOLD_DEG", args.drop_threshold_deg),
        ("NKSR_SOLVER_TOL", args.solver_tol),
    ):
        validate_finite(name, value, status_code=500)
    for name, value in (
        ("NKSR_VOXEL_SIZE", args.voxel_size),
        ("NKSR_MAX_CHUNK_EXTENT", args.max_chunk_extent),
        ("NKSR_MAX_CHUNK_INPUT_EXTENT", args.max_chunk_input_extent),
    ):
        if value is not None:
            validate_finite(name, value, status_code=500)

    if not 0.0 < args.face_ratio <= 1.0:
        raise HTTPException(status_code=500, detail="NKSR_FACE_RATIO 必须在 (0, 1] 范围内。")
    if not 0.0 <= args.detail_level <= 1.0:
        raise HTTPException(status_code=500, detail="NKSR_DETAIL_LEVEL 必须在 [0, 1] 范围内。")
    if not 0.0 <= args.chunk_overlap_ratio <= 0.49:
        raise HTTPException(status_code=500, detail="NKSR_CHUNK_OVERLAP_RATIO 必须在 [0, 0.49] 范围内。")
    if not 0.0 <= args.outlier_quantile < 0.5:
        raise HTTPException(status_code=500, detail="NKSR_OUTLIER_QUANTILE 必须在 [0, 0.5) 范围内。")
    if args.voxel_size is not None and args.voxel_size <= 0.0:
        raise HTTPException(status_code=500, detail="NKSR_VOXEL_SIZE 必须大于 0。")
    chunk_limited = (
        (args.max_chunk_extent is not None and args.max_chunk_extent > 0.0)
        or (args.max_chunk_input_extent is not None and args.max_chunk_input_extent > 0.0)
    )
    if args.voxel_size is not None and chunk_limited:
        raise HTTPException(status_code=500, detail="NKSR_VOXEL_SIZE 不能和 NKSR_MAX_CHUNK_EXTENT/NKSR_MAX_CHUNK_INPUT_EXTENT 同时使用。")

    for name, value in (
        ("NKSR_MAX_CHUNK_POINTS", args.max_chunk_points),
        ("NKSR_MIN_FILTERED_POINTS", args.min_filtered_points),
        ("NKSR_MISE_ITER", args.mise_iter),
        ("NKSR_GRID_UPSAMPLE", args.grid_upsample),
        ("NKSR_ESTIMATE_KNN", args.estimate_knn),
        ("NKSR_SOLVER_MAX_ITER", args.solver_max_iter),
        ("NKSR_NYSTROM_MIN_DEPTH", args.nystrom_min_depth),
    ):
        if value < 0:
            raise HTTPException(status_code=500, detail=f"{name} 不能为负数。")


def write_result_zip(zip_path: Path, output_path: Path, stats_path: Path | None) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(output_path, output_path.name)
        if stats_path is not None and stats_path.exists():
            archive.write(stats_path, stats_path.name)


@app.get("/health")
def health() -> dict:
    return {"ok": True, "service": "nksr-http"}


@app.post("/v1/reconstruct", response_model=None)
def reconstruct(
    request: Request,
    background_tasks: BackgroundTasks,
    object_pcd: UploadFile = File(...),
    request_id: str | None = Form(None),
) -> Response:
    job_id = safe_token(request_id or getattr(request.state, "request_id", None) or uuid.uuid4().hex, "object")
    request.state.request_id = job_id

    validate_request(object_pcd)

    work_dir = Path(tempfile.mkdtemp(prefix=f"nksr_{job_id}_"))
    cleanup_device = env_str("DEVICE", "cuda:0")
    oom_response: JSONResponse | None = None
    reconstruction_failed_response: JSONResponse | None = None
    try:
        input_dir = work_dir / "input"
        output_dir = work_dir / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        suffix = Path(object_pcd.filename or "object.pcd").suffix.lower()
        if suffix not in {".pcd", ".ply"}:
            suffix = ".pcd"
        input_path = input_dir / f"{safe_stem(object_pcd.filename or 'object', 'object')}{suffix}"
        output_path = output_dir / f"{input_path.stem}.ply"

        with input_path.open("wb") as fh:
            shutil.copyfileobj(object_pcd.file, fh)

        args = configured_infer_args(input_path, output_path)
        cleanup_device = args.device
        with INFER_LOCK:
            try:
                result = infer_one(args)
            finally:
                cleanup_cuda_memory(args.device)
        zip_path = work_dir / f"{job_id}_nksr.zip"
        write_result_zip(zip_path, result.output_path, result.stats_path)
    except HTTPException:
        cleanup_dir(work_dir)
        raise
    except ReconstructionFailedError as exc:
        cleanup_dir(work_dir)
        LOGGER.warning("NKSR 重建未生成有效 mesh request_id=%s: %s", job_id, exc)
        reconstruction_failed_response = error_response(422, "reconstruction_failed", str(exc), job_id)
    except Exception as exc:
        cleanup_dir(work_dir)
        if is_cuda_oom(exc):
            LOGGER.error(
                "NKSR HTTP 请求处理失败：CUDA 显存不足 request_id=%s",
                job_id,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            oom_response = error_response(503, "cuda_oom", "当前 GPU 显存不足，请检查 GPU 占用或资源配置。", job_id)
        else:
            raise

    if oom_response is not None:
        cleanup_cuda_memory(cleanup_device)
        return oom_response
    if reconstruction_failed_response is not None:
        cleanup_cuda_memory(cleanup_device)
        return reconstruction_failed_response

    background_tasks.add_task(cleanup_dir, work_dir)
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=zip_path.name,
        headers={"X-Request-ID": header_request_id(job_id)},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="启动 NKSR HTTP 建模服务。")
    parser.add_argument("--host", default=env_str("HTTP_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=env_int("HTTP_PORT", 8012))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import uvicorn

    uvicorn.run(
        "scripts.http_infer_pcd:app",
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
