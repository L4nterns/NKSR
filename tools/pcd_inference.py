from __future__ import annotations

import argparse
import json
import math
import os
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np


ROOT_DIR = Path(__file__).resolve().parent.parent
NKSR_CONFIG_CHOICES = ("ks", "snet", "snet-wonormal")
NKSR_CHECKPOINT_FILES = {
    "ks": "ks.pth",
    "snet": "snet-n3k-wnormal.pth",
    "snet-wonormal": "snet-n3k-wonormal.pth",
}


def load_env_file(path: Optional[Path] = None) -> None:
    """加载 .env 到环境变量；已有环境变量优先，不覆盖。"""
    candidates = [Path.cwd() / ".env", ROOT_DIR / ".env"] if path is None else [Path(path)]
    seen = set()
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate in seen or not candidate.exists():
            continue
        seen.add(candidate)
        for raw_line in candidate.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key or key in os.environ:
                continue
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            os.environ[key] = value


def env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value in (None, "") else value


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value in (None, "") else int(value)


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value in (None, "") else float(value)


def env_optional_float(name: str, default: Optional[float] = None) -> Optional[float]:
    value = os.environ.get(name)
    return default if value in (None, "") else float(value)


def checkpoint_file_name(config: str) -> str:
    return NKSR_CHECKPOINT_FILES[config]


def configured_checkpoint_path(config: str) -> Optional[Path]:
    checkpoint_dir = os.environ.get("NKSR_CHECKPOINTS_DIR", "").strip()
    if not checkpoint_dir:
        return None
    return Path(checkpoint_dir).expanduser() / checkpoint_file_name(config)


@dataclass
class PlannedChunk:
    xyz: np.ndarray
    normal: Optional[np.ndarray]
    sensor: Optional[np.ndarray]
    color: Optional[np.ndarray]
    center: np.ndarray


@dataclass
class InferenceResult:
    output_path: Path
    stats_path: Optional[Path]
    stats: dict


class ReconstructionFailedError(RuntimeError):
    pass


def plan_adaptive_chunks(
    xyz_np: np.ndarray,
    normal_np: Optional[np.ndarray],
    sensor_np: Optional[np.ndarray],
    color_np: Optional[np.ndarray],
    *,
    max_chunk_extent: Optional[float],
    max_chunk_points: Optional[int],
    overlap_ratio: float,
) -> list[PlannedChunk]:
    """Recursively split a point cloud into local chunks that satisfy extent and point-count constraints."""
    max_chunk_extent = None if max_chunk_extent is None or max_chunk_extent <= 0.0 else float(max_chunk_extent)
    max_chunk_points = None if max_chunk_points is None or max_chunk_points <= 0 else int(max_chunk_points)
    if max_chunk_extent is None and max_chunk_points is None:
        return [PlannedChunk(
            xyz=xyz_np.astype(np.float32, copy=False),
            normal=None if normal_np is None else normal_np.astype(np.float32, copy=False),
            sensor=None if sensor_np is None else sensor_np.astype(np.float32, copy=False),
            color=None if color_np is None else color_np.astype(np.float32, copy=False),
            center=np.zeros(3, dtype=np.float32),
        )]

    overlap_ratio = min(max(float(overlap_ratio), 0.0), 0.49)
    pending = [(xyz_np, normal_np, sensor_np, color_np)]
    planned_chunks: list[PlannedChunk] = []

    while pending:
        node_xyz, node_normal, node_sensor, node_color = pending.pop()
        if node_xyz.shape[0] == 0:
            continue

        node_min = np.min(node_xyz, axis=0)
        node_max = np.max(node_xyz, axis=0)
        node_lengths = node_max - node_min
        extent_ok = max_chunk_extent is None or bool(np.all(node_lengths <= max_chunk_extent + 1.0e-8))
        points_ok = max_chunk_points is None or node_xyz.shape[0] <= max_chunk_points
        if extent_ok and points_ok:
            center = 0.5 * (node_min + node_max)
            planned_chunks.append(PlannedChunk(
                xyz=(node_xyz - center[None, :]).astype(np.float32, copy=False),
                normal=None if node_normal is None else node_normal.astype(np.float32, copy=False),
                sensor=None if node_sensor is None else (node_sensor - center[None, :]).astype(np.float32, copy=False),
                color=None if node_color is None else node_color.astype(np.float32, copy=False),
                center=center.astype(np.float32),
            ))
            continue

        if max_chunk_extent is not None:
            violating_axes = np.where(node_lengths > max_chunk_extent + 1.0e-8)[0]
            split_axis = int(violating_axes[np.argmax(node_lengths[violating_axes])]) if violating_axes.size > 0 else int(np.argmax(node_lengths))
        else:
            split_axis = int(np.argmax(node_lengths))

        split_value = float(0.5 * (node_min[split_axis] + node_max[split_axis]))
        overlap = node_lengths[split_axis] * overlap_ratio * 0.5
        coord = node_xyz[:, split_axis]
        left_mask = coord <= split_value + overlap
        right_mask = coord >= split_value - overlap

        if np.all(left_mask) or np.all(right_mask):
            order = np.argsort(coord, kind="mergesort")
            mid = order.shape[0] // 2
            if mid == 0 or mid == order.shape[0]:
                center = 0.5 * (node_min + node_max)
                planned_chunks.append(PlannedChunk(
                    xyz=(node_xyz - center[None, :]).astype(np.float32, copy=False),
                    normal=None if node_normal is None else node_normal.astype(np.float32, copy=False),
                    sensor=None if node_sensor is None else (node_sensor - center[None, :]).astype(np.float32, copy=False),
                    color=None if node_color is None else node_color.astype(np.float32, copy=False),
                    center=center.astype(np.float32),
                ))
                continue
            left_indices = order[:mid]
            right_indices = order[mid:]
        else:
            left_indices = np.where(left_mask)[0]
            right_indices = np.where(right_mask)[0]

        pending.append((
            node_xyz[right_indices],
            None if node_normal is None else node_normal[right_indices],
            None if node_sensor is None else node_sensor[right_indices],
            None if node_color is None else node_color[right_indices],
        ))
        pending.append((
            node_xyz[left_indices],
            None if node_normal is None else node_normal[left_indices],
            None if node_sensor is None else node_sensor[left_indices],
            None if node_color is None else node_color[left_indices],
        ))

    planned_chunks.sort(key=lambda chunk: (chunk.center[0], chunk.center[1], chunk.center[2]))
    return planned_chunks


def add_infer_arguments(parser: argparse.ArgumentParser, include_paths: bool = True) -> argparse.ArgumentParser:
    if include_paths:
        parser.add_argument("input", type=Path, help="Input point cloud path, e.g. .pcd/.ply")
        parser.add_argument("output", type=Path, help="Output mesh path, only .ply is supported")
    parser.add_argument("--device", default=env_str("DEVICE", "cuda:0"), help="Torch device, e.g. cuda:0 or cpu")
    parser.add_argument(
        "--config",
        default=env_str("NKSR_CONFIG", "snet-wonormal"),
        choices=NKSR_CONFIG_CHOICES,
        help="Pretrained NKSR config",
    )
    parser.add_argument(
        "--input-mode",
        default=env_str("NKSR_INPUT_MODE", "auto"),
        choices=["auto", "normal", "sensor", "estimate"],
        help="How to obtain orientation information for reconstruction",
    )
    parser.add_argument("--xyz-cols", default=env_str("NKSR_XYZ_COLS", "x,y,z"), help="Comma-separated xyz column names")
    parser.add_argument(
        "--normal-cols",
        default=env_str("NKSR_NORMAL_COLS", "normal_x,normal_y,normal_z"),
        help="Comma-separated normal column names",
    )
    parser.add_argument(
        "--sensor-cols",
        default=env_str("NKSR_SENSOR_COLS", "sensor_x,sensor_y,sensor_z"),
        help="Comma-separated sensor column names",
    )
    parser.add_argument(
        "--color-cols",
        default=env_str("NKSR_COLOR_COLS", "red,green,blue"),
        help="Comma-separated RGB column names",
    )
    parser.add_argument(
        "--obb-lock-axis",
        default=env_str("NKSR_OBB_LOCK_AXIS", "auto"),
        choices=["auto", "none", "x", "y", "z", "pca"],
        help="OBB alignment mode in output coordinates; auto locks output up axis, none disables OBB alignment, pca runs unlocked OBB alignment",
    )
    parser.add_argument(
        "--write-stats",
        action=argparse.BooleanOptionalAction,
        default=env_bool("NKSR_WRITE_STATS", True),
        help="Write mesh bbox stats to a sidecar JSON file",
    )
    parser.add_argument(
        "--simplify-method",
        default=env_str("NKSR_SIMPLIFY_METHOD", "auto"),
        choices=["auto", "quadric", "clustering", "none"],
        help="Mesh simplification method; only applied when --face-ratio < 1.0",
    )
    parser.add_argument(
        "--face-ratio",
        type=float,
        default=env_float("NKSR_FACE_RATIO", 0.1),
        help="Relative face count after simplification, in (0, 1]; ignored when --simplify-method none",
    )
    parser.add_argument(
        "--output-up-axis",
        default=env_str("NKSR_OUTPUT_UP_AXIS", "y"),
        choices=["z", "y"],
        help="Export mesh in z-up or y-up coordinates",
    )
    parser.add_argument(
        "--detail-level",
        type=float,
        default=env_float("NKSR_DETAIL_LEVEL", 1.0),
        help="Reconstruction detail level in [0, 1], ignored when --voxel-size is given",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=env_optional_float("NKSR_VOXEL_SIZE"),
        help="Override the finest voxel size; do not combine with adaptive chunking",
    )
    parser.add_argument(
        "--max-chunk-extent",
        type=float,
        default=env_optional_float("NKSR_MAX_CHUNK_EXTENT"),
        help="Maximum spatial extent per chunk; <=0 disables extent constraint",
    )
    parser.add_argument(
        "--max-chunk-points",
        type=int,
        default=env_int("NKSR_MAX_CHUNK_POINTS", 0),
        help="Maximum point count per chunk; lower values reduce OOM risk but increase chunk count; <=0 disables point-count constraint",
    )
    parser.add_argument(
        "--chunk-overlap-ratio",
        type=float,
        default=env_float("NKSR_CHUNK_OVERLAP_RATIO", 0.05),
        help="Overlap ratio used when adaptively splitting chunks; clamped to [0, 0.49]",
    )
    parser.add_argument(
        "--chunk-tmp-device",
        default=env_str("NKSR_CHUNK_TMP_DEVICE", "cpu"),
        help="Temporary device used to store finished chunks",
    )
    parser.add_argument(
        "--max-input-extent",
        type=float,
        default=env_float("NKSR_MAX_INPUT_EXTENT", 200.0),
        help="Fail before reconstruction when the input bbox exceeds this extent; <=0 disables the guard",
    )
    parser.add_argument(
        "--max-chunk-input-extent",
        type=float,
        default=env_optional_float("NKSR_MAX_CHUNK_INPUT_EXTENT"),
        help="Also split chunks by this spatial extent; default disabled, <=0 disables this extra guard",
    )
    parser.add_argument(
        "--outlier-quantile",
        type=float,
        default=env_float("NKSR_OUTLIER_QUANTILE", 0.0),
        help="Trim symmetric coordinate outliers before reconstruction; 0 disables trimming",
    )
    parser.add_argument(
        "--min-filtered-points",
        type=int,
        default=env_int("NKSR_MIN_FILTERED_POINTS", 1000),
        help="Minimum points required after finite/outlier filtering",
    )
    parser.add_argument("--mise-iter", type=int, default=env_int("NKSR_MISE_ITER", 1), help="Dual mesh MISE iterations")
    parser.add_argument("--grid-upsample", type=int, default=env_int("NKSR_GRID_UPSAMPLE", 1), help="Mesh extraction grid upsample")
    parser.add_argument(
        "--extract-on-cpu",
        action=argparse.BooleanOptionalAction,
        default=env_bool("NKSR_EXTRACT_ON_CPU", False),
        help="Move the field and network to CPU before mesh extraction",
    )
    parser.add_argument(
        "--texture-mode",
        default=env_str("NKSR_TEXTURE_MODE", "auto"),
        choices=["auto", "off"],
        help="Texture export mode; auto preserves input color when available, off skips vertex colors",
    )
    parser.add_argument("--estimate-knn", type=int, default=env_int("NKSR_ESTIMATE_KNN", 64), help="KNN used when estimating normals")
    parser.add_argument(
        "--estimate-orient",
        default=env_str("NKSR_ESTIMATE_ORIENT", "centroid"),
        choices=["none", "centroid"],
        help="Normal orientation rule used by --input-mode estimate",
    )
    parser.add_argument(
        "--drop-threshold-deg",
        type=float,
        default=env_float("NKSR_DROP_THRESHOLD_DEG", 85.0),
        help="Only used in sensor mode when estimating normals from sensor positions",
    )
    parser.add_argument(
        "--approx-kernel-grad",
        action=argparse.BooleanOptionalAction,
        default=env_bool("NKSR_APPROX_KERNEL_GRAD", False),
        help="Minor efficiency optimization for large scenes",
    )
    parser.add_argument("--solver-max-iter", type=int, default=env_int("NKSR_SOLVER_MAX_ITER", 2000))
    parser.add_argument("--solver-tol", type=float, default=env_float("NKSR_SOLVER_TOL", 1.0e-5))
    parser.add_argument("--nystrom-min-depth", type=int, default=env_int("NKSR_NYSTROM_MIN_DEPTH", 100))
    parser.add_argument(
        "--reconstruct-attempts",
        type=int,
        default=env_int("NKSR_RECONSTRUCT_ATTEMPTS", 5),
        help="Maximum same-parameter reconstruction attempts when mesh extraction is empty",
    )
    parser.add_argument(
        "--fused-mode",
        action=argparse.BooleanOptionalAction,
        default=env_bool("NKSR_FUSED_MODE", True),
        help="Use fused kernel solve",
    )
    return parser


def build_infer_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run NKSR reconstruction on a point cloud file.")
    return add_infer_arguments(parser, include_paths=True)


def build_batch_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch run NKSR reconstruction on a directory.")
    parser.add_argument("input_dir", type=Path, help="Input directory containing .pcd/.ply files")
    parser.add_argument("output_dir", type=Path, help="Output directory")
    parser.add_argument(
        "--extensions",
        default=env_str("NKSR_BATCH_EXTENSIONS", ".pcd,.ply"),
        help="Comma-separated input extensions to scan recursively",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=env_bool("NKSR_SKIP_EXISTING", True),
        help="Skip inference when the target mesh already exists",
    )
    return add_infer_arguments(parser, include_paths=False)


def resolve_chunk_constraints(args: argparse.Namespace) -> tuple[Optional[float], Optional[int], float]:
    """Normalize adaptive chunking CLI arguments into internal constraints."""
    max_chunk_extent = args.max_chunk_extent
    if max_chunk_extent is not None and max_chunk_extent <= 0.0:
        max_chunk_extent = None

    max_chunk_input_extent = getattr(args, "max_chunk_input_extent", None)
    if max_chunk_input_extent is not None and max_chunk_input_extent <= 0.0:
        max_chunk_input_extent = None
    if max_chunk_extent is None:
        max_chunk_extent = max_chunk_input_extent
    elif max_chunk_input_extent is not None:
        max_chunk_extent = min(max_chunk_extent, max_chunk_input_extent)

    max_chunk_points = args.max_chunk_points
    if max_chunk_points is not None and max_chunk_points <= 0:
        max_chunk_points = None

    overlap_ratio = min(max(float(args.chunk_overlap_ratio), 0.0), 0.49)
    return max_chunk_extent, max_chunk_points, overlap_ratio


def split_columns(spec: str) -> Sequence[str]:
    return [part.strip() for part in spec.split(",") if part.strip()]


def read_columns(points, columns: Iterable[str]) -> Optional[np.ndarray]:
    columns = list(columns)
    if not columns:
        return None
    if any(col not in points.columns for col in columns):
        return None
    return points.loc[:, columns].to_numpy(dtype=np.float32, copy=True)


def autodetect_columns(points, preferred: Sequence[str], candidates: Sequence[Sequence[str]]) -> Optional[np.ndarray]:
    values = read_columns(points, preferred)
    if values is not None:
        return values
    for candidate in candidates:
        values = read_columns(points, candidate)
        if values is not None:
            return values
    return None


def infer_packed_color_storage_type(values: np.ndarray, storage_type: Optional[str]) -> Optional[str]:
    """Infer packed rgb/rgba storage kind from explicit metadata first, then fall back to numpy dtype."""
    if storage_type in {"F", "U", "I"}:
        return storage_type

    values = np.asarray(values)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if np.issubdtype(values.dtype, np.floating):
        return "F"
    if np.issubdtype(values.dtype, np.integer):
        return "U" if np.issubdtype(values.dtype, np.unsignedinteger) else "I"
    return None


def unpack_packed_rgb(values: np.ndarray, storage_type: Optional[str] = None) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]

    resolved_type = infer_packed_color_storage_type(values, storage_type)
    if resolved_type == "F":
        packed = values.astype(np.float32, copy=False).view(np.uint32)
    else:
        packed = values.astype(np.uint32, copy=False)

    return np.stack(
        [
            (packed >> 16) & 255,
            (packed >> 8) & 255,
            packed & 255,
        ],
        axis=1,
    ).astype(np.float32)


def normalize_colors(colors: np.ndarray) -> np.ndarray:
    colors = np.asarray(colors, dtype=np.float32)
    if colors.ndim != 2 or colors.shape[1] != 3:
        raise ValueError(f"颜色数组形状不合法: {colors.shape}")
    max_value = float(np.nanmax(colors))
    if max_value > 1.0:
        if max_value <= 255.0:
            colors = colors / 255.0
        elif max_value <= 65535.0:
            colors = colors / 65535.0
        else:
            colors = colors / max_value
    return np.clip(colors, 0.0, 1.0)


def autodetect_colors(points, preferred: Sequence[str], packed_color_type: Optional[str] = None) -> Optional[np.ndarray]:
    colors = read_columns(points, preferred)
    if colors is not None:
        return normalize_colors(colors)

    for candidate in (
        ["red", "green", "blue"],
        ["r", "g", "b"],
        ["diffuse_red", "diffuse_green", "diffuse_blue"],
    ):
        colors = read_columns(points, candidate)
        if colors is not None:
            return normalize_colors(colors)

    for packed_name in ("rgb", "rgba"):
        if packed_name in points.columns:
            packed_values = points.loc[:, packed_name].to_numpy(copy=True)
            return normalize_colors(
                unpack_packed_rgb(
                    packed_values,
                    storage_type=packed_color_type,
                )
            )

    return None


def load_ply_with_plyfile(input_path: Path):
    from plyfile import PlyData

    ply = PlyData.read(str(input_path))
    vertex = ply["vertex"].data
    columns = list(vertex.dtype.names or [])
    packed_color_type = None
    for packed_name in ("rgb", "rgba"):
        if packed_name in columns:
            field_dtype = np.asarray(vertex[packed_name]).dtype
            packed_color_type = infer_packed_color_storage_type(np.empty((0,), dtype=field_dtype), None)
            break

    class SimplePoints:
        def __init__(self, data, names, packed_color_type=None):
            self._data = data
            self.columns = list(names)
            self.packed_color_type = packed_color_type

        @property
        def loc(self):
            return self

        def __getitem__(self, key):
            if isinstance(key, tuple):
                _, cols = key
                return SimpleSelection(self._data, cols)
            raise KeyError(key)

    class SimpleSelection:
        def __init__(self, data, cols):
            self._data = data
            if isinstance(cols, str):
                self._cols = [cols]
            else:
                self._cols = list(cols)

        def to_numpy(self, dtype=np.float32, copy=True):
            arr = np.column_stack([self._data[col] for col in self._cols])
            return np.array(arr, dtype=dtype, copy=copy)

    return SimplePoints(vertex, columns, packed_color_type=packed_color_type)


def parse_pcd_header(raw: bytes) -> tuple[dict, int]:
    header = {}
    offset = 0
    data_offset = -1

    while offset < len(raw):
        next_offset = raw.find(b"\n", offset)
        if next_offset < 0:
            line_bytes = raw[offset:]
            offset = len(raw)
        else:
            line_bytes = raw[offset:next_offset]
            offset = next_offset + 1

        line = line_bytes.decode("latin1", errors="ignore").strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        key = parts[0].upper()
        values = parts[1:]
        header[key] = values

        if key == "DATA":
            data_offset = offset
            break

    if data_offset < 0:
        raise ValueError("PCD 文件缺少 DATA 头")

    return header, data_offset


def pcd_dtype(header: dict) -> np.dtype:
    fields = header["FIELDS"]
    sizes = list(map(int, header["SIZE"]))
    types = header["TYPE"]
    counts = list(map(int, header.get("COUNT", ["1"] * len(fields))))

    dtype_fields = []
    for name, size, typ, count in zip(fields, sizes, types, counts):
        if typ == "F":
            base = {4: np.float32, 8: np.float64}[size]
        elif typ == "U":
            base = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[size]
        elif typ == "I":
            base = {1: np.int8, 2: np.int16, 4: np.int32, 8: np.int64}[size]
        else:
            raise ValueError(f"不支持的 PCD TYPE: {typ}")

        if count == 1:
            dtype_fields.append((name, base))
        else:
            dtype_fields.append((name, base, (count,)))

    return np.dtype(dtype_fields)


def lzf_decompress(compressed: bytes, expected_size: int) -> bytes:
    out = bytearray(expected_size)
    in_pos = 0
    out_pos = 0
    in_len = len(compressed)

    while in_pos < in_len:
        ctrl = compressed[in_pos]
        in_pos += 1

        if ctrl < 32:
            length = ctrl + 1
            if in_pos + length > in_len:
                raise ValueError("LZF literal 数据越界")
            if out_pos + length > expected_size:
                raise ValueError("LZF 解压数据超过头信息长度")
            out[out_pos:out_pos + length] = compressed[in_pos:in_pos + length]
            in_pos += length
            out_pos += length
            continue

        length = ctrl >> 5
        ref_offset = (ctrl & 0x1f) << 8
        if length == 7:
            if in_pos >= in_len:
                raise ValueError("LZF 扩展长度数据越界")
            length += compressed[in_pos]
            in_pos += 1
        if in_pos >= in_len:
            raise ValueError("LZF back-reference 数据越界")
        ref_offset += compressed[in_pos]
        in_pos += 1
        length += 2

        ref_pos = out_pos - ref_offset - 1
        if ref_pos < 0:
            raise ValueError("LZF back-reference 指向无效位置")
        if out_pos + length > expected_size:
            raise ValueError("LZF 解压数据超过头信息长度")

        for _ in range(length):
            out[out_pos] = out[ref_pos]
            out_pos += 1
            ref_pos += 1

    if out_pos != expected_size:
        raise ValueError(f"LZF 解压长度与头信息不一致: {out_pos} != {expected_size}")
    return bytes(out)


def decompress_pcd_binary_compressed(compressed: bytes, expected_size: int) -> bytes:
    errors = []
    for name, decompress_fn in (
        ("lzf", lambda data: lzf_decompress(data, expected_size)),
        ("zlib", zlib.decompress),
    ):
        try:
            decompressed = decompress_fn(compressed)
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            continue
        if len(decompressed) != expected_size:
            errors.append(f"{name}: 解压长度 {len(decompressed)} != {expected_size}")
            continue
        return decompressed

    raise ValueError("binary_compressed PCD 解压失败；尝试 " + "; ".join(errors))


def decode_pcd_binary_compressed(payload: bytes, header: dict) -> np.ndarray:
    if len(payload) < 8:
        raise ValueError("binary_compressed PCD 数据长度非法")

    compressed_size, uncompressed_size = struct.unpack("<II", payload[:8])
    compressed = payload[8:8 + compressed_size]
    if len(compressed) != compressed_size:
        raise ValueError("binary_compressed PCD 压缩数据长度与头信息不一致")
    decompressed = decompress_pcd_binary_compressed(compressed, uncompressed_size)

    fields = header["FIELDS"]
    sizes = list(map(int, header["SIZE"]))
    types = header["TYPE"]
    counts = list(map(int, header.get("COUNT", ["1"] * len(fields))))
    points = int(header.get("POINTS", [header.get("WIDTH", ["0"])[0]])[0])
    dtype = pcd_dtype(header)

    out = np.empty(points, dtype=dtype)
    cursor = 0
    for name, size, typ, count in zip(fields, sizes, types, counts):
        if typ == "F":
            base = {4: np.float32, 8: np.float64}[size]
        elif typ == "U":
            base = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[size]
        elif typ == "I":
            base = {1: np.int8, 2: np.int16, 4: np.int32, 8: np.int64}[size]
        else:
            raise ValueError(f"不支持的 PCD TYPE: {typ}")

        item_count = points * count
        byte_count = item_count * size
        field_bytes = decompressed[cursor:cursor + byte_count]
        field_array = np.frombuffer(field_bytes, dtype=base, count=item_count)

        if count == 1:
            out[name] = field_array
        else:
            out[name] = field_array.reshape(points, count)

        cursor += byte_count

    return out


def load_pcd_points(input_path: Path):
    raw = input_path.read_bytes()
    header, data_offset = parse_pcd_header(raw)

    fields = header["FIELDS"]
    points = int(header.get("POINTS", [header.get("WIDTH", ["0"])[0]])[0])
    data_type = header["DATA"][0].lower()
    packed_color_type = None
    for name, typ in zip(fields, header["TYPE"]):
        if name in {"rgb", "rgba"}:
            packed_color_type = typ
            break

    class SimplePoints:
        def __init__(self, data, names, packed_color_type=None):
            self._data = data
            self.columns = list(names)
            self.packed_color_type = packed_color_type

        @property
        def loc(self):
            return self

        def __getitem__(self, key):
            if isinstance(key, tuple):
                _, cols = key
                return SimpleSelection(self._data, cols)
            raise KeyError(key)

    class SimpleSelection:
        def __init__(self, data, cols):
            self._data = data
            if isinstance(cols, str):
                self._cols = [cols]
            else:
                self._cols = list(cols)

        def to_numpy(self, dtype=np.float32, copy=True):
            arr = np.column_stack([self._data[col] for col in self._cols])
            return np.array(arr, dtype=dtype, copy=copy)

    if data_type == "ascii":
        text = raw[data_offset:].decode("latin1", errors="ignore")
        lines = text.splitlines()
        rows = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(stripped.split())

        if not rows:
            raise ValueError("PCD ascii 数据为空")

        sizes = list(map(int, header["SIZE"]))
        types = header["TYPE"]
        counts = list(map(int, header.get("COUNT", ["1"] * len(fields))))

        data = {}
        cursor = 0
        matrix = np.asarray(rows, dtype=object)
        for name, size, typ, count in zip(fields, sizes, types, counts):
            field_values = matrix[:, cursor:cursor + count]
            cursor += count

            if typ == "F":
                base = np.float32 if size == 4 else np.float64
            elif typ == "U":
                base = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[size]
            elif typ == "I":
                base = {1: np.int8, 2: np.int16, 4: np.int32, 8: np.int64}[size]
            else:
                raise ValueError(f"不支持的 PCD TYPE: {typ}")

            converted = np.asarray(field_values, dtype=base)
            if count == 1:
                data[name] = converted[:, 0]
            else:
                data[name] = converted

        return SimplePoints(data, fields, packed_color_type=packed_color_type)

    payload = raw[data_offset:]
    if data_type == "binary":
        dtype = pcd_dtype(header)
        array = np.frombuffer(payload, dtype=dtype, count=points)
    elif data_type == "binary_compressed":
        array = decode_pcd_binary_compressed(payload, header)
    else:
        raise ValueError(f"当前仅支持 ascii/binary/binary_compressed PCD，收到: {data_type}")

    data = {name: array[name] for name in fields}
    return SimplePoints(data, fields, packed_color_type=packed_color_type)


def load_point_cloud(
    input_path: Path,
    xyz_cols: Sequence[str],
    normal_cols: Sequence[str],
    sensor_cols: Sequence[str],
    color_cols: Sequence[str],
):
    suffix = input_path.suffix.lower()
    if suffix == ".pcd":
        points = load_pcd_points(input_path)
    elif suffix == ".ply":
        points = load_ply_with_plyfile(input_path)
    else:
        from pyntcloud import PyntCloud

        cloud = PyntCloud.from_file(str(input_path))
        points = cloud.points

    xyz = autodetect_columns(points, xyz_cols, [["x", "y", "z"]])
    if xyz is None:
        raise ValueError(f"点云中未找到 xyz 列，当前列为: {list(points.columns)}")

    normals = autodetect_columns(
        points,
        normal_cols,
        [["normal_x", "normal_y", "normal_z"], ["nx", "ny", "nz"]],
    )
    sensors = autodetect_columns(
        points,
        sensor_cols,
        [["sensor_x", "sensor_y", "sensor_z"], ["vp_x", "vp_y", "vp_z"]],
    )
    colors = autodetect_colors(points, color_cols, getattr(points, "packed_color_type", None))
    return xyz, normals, sensors, colors, list(points.columns)


def compute_point_cloud_bbox_stats(xyz: np.ndarray) -> dict:
    mins = np.min(xyz, axis=0)
    maxs = np.max(xyz, axis=0)
    lengths = maxs - mins
    return {
        "point_bbox_min": mins.astype(float).tolist(),
        "point_bbox_max": maxs.astype(float).tolist(),
        "point_bbox_lengths": lengths.astype(float).tolist(),
        "point_bbox_max_extent": float(np.max(lengths)),
    }


def filter_point_cloud_data(
    xyz: np.ndarray,
    normal: Optional[np.ndarray],
    sensor: Optional[np.ndarray],
    color: Optional[np.ndarray],
    *,
    outlier_quantile: float,
    min_filtered_points: int,
) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], dict]:
    original_points = int(xyz.shape[0])
    finite_mask = np.all(np.isfinite(xyz), axis=1)
    filter_stats = {
        "original_points": original_points,
        "finite_points": int(np.count_nonzero(finite_mask)),
        "outlier_quantile": float(outlier_quantile),
    }
    if not np.all(finite_mask):
        raise ValueError(f"点云包含 {original_points - filter_stats['finite_points']} 个非有限 xyz 坐标")

    trimmed_mask = np.ones(xyz.shape[0], dtype=bool)
    q = float(outlier_quantile)
    if q > 0.0 and xyz.shape[0] > 0:
        q = min(q, 0.49)
        lo = np.quantile(xyz, q, axis=0)
        hi = np.quantile(xyz, 1.0 - q, axis=0)
        trimmed_mask = np.all((xyz >= lo[None, :]) & (xyz <= hi[None, :]), axis=1)
        filter_stats["outlier_bounds_min"] = lo.astype(float).tolist()
        filter_stats["outlier_bounds_max"] = hi.astype(float).tolist()

    kept_points = int(np.count_nonzero(trimmed_mask))
    filter_stats["filtered_points"] = kept_points
    filter_stats["dropped_points"] = original_points - kept_points
    if kept_points < int(min_filtered_points):
        raise ValueError(f"过滤后点数 {kept_points} 小于 --min-filtered-points={min_filtered_points}")

    xyz = xyz[trimmed_mask]
    normal = None if normal is None else normal[trimmed_mask]
    sensor = None if sensor is None else sensor[trimmed_mask]
    color = None if color is None else color[trimmed_mask]
    filter_stats.update(compute_point_cloud_bbox_stats(xyz))

    return xyz, normal, sensor, color, filter_stats


def validate_input_extent(xyz: np.ndarray, max_input_extent: Optional[float]) -> dict:
    stats = compute_point_cloud_bbox_stats(xyz)
    if max_input_extent is not None and max_input_extent > 0.0:
        max_extent = stats["point_bbox_max_extent"]
        if max_extent > float(max_input_extent):
            lengths = ", ".join(f"{value:.3f}" for value in stats["point_bbox_lengths"])
            raise ValueError(
                f"输入点云 bbox 过大，最长边 {max_extent:.3f} 超过 --max-input-extent={max_input_extent}. "
                f"三轴长度: {lengths}. "
            )
    return stats


def estimate_normals_from_xyz(xyz: torch.Tensor, knn: int, orient_mode: str) -> torch.Tensor:
    import torch
    from nksr import ext

    if xyz.size(0) < knn:
        raise ValueError(f"点数 {xyz.size(0)} 小于法向估计所需 knn={knn}")

    knn_dist, knn_indices = ext.pcproc.nearest_neighbours(xyz, knn)
    normal = ext.pcproc.estimate_normals_knn(xyz, knn_dist, knn_indices)
    normal = normal / (torch.linalg.norm(normal, dim=-1, keepdim=True) + 1e-6)

    if orient_mode == "centroid":
        center = torch.mean(xyz, dim=0, keepdim=True)
        outward = xyz - center
        flip_mask = torch.sum(normal * outward, dim=-1) < 0.0
        normal[flip_mask] = -normal[flip_mask]

    return normal


def tensor_like_to_numpy_float32(value) -> Optional[np.ndarray]:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float32)


def chunk_orientation_arrays(
    mode: str,
    normal_np: Optional[np.ndarray],
    sensor_np: Optional[np.ndarray],
    estimated_normal,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if mode == "none":
        return None, None
    if mode == "estimate":
        return tensor_like_to_numpy_float32(estimated_normal), None
    return normal_np, sensor_np


def rotation_matrix_2d(angle_rad: float) -> np.ndarray:
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    return np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)


def convex_hull_2d(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] <= 1:
        return points

    points = np.unique(points, axis=0)
    if points.shape[0] <= 1:
        return points

    order = np.lexsort((points[:, 1], points[:, 0]))
    pts = points[order]

    def cross(o, a, b) -> float:
        oa = a - o
        ob = b - o
        return float(oa[0] * ob[1] - oa[1] * ob[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    return np.asarray(hull, dtype=np.float32)


def min_area_rect_2d(points: np.ndarray) -> dict:
    hull = convex_hull_2d(points)
    if hull.shape[0] <= 1:
        return {"angle_rad": 0.0}

    best = None
    for idx in range(hull.shape[0]):
        p0 = hull[idx]
        p1 = hull[(idx + 1) % hull.shape[0]]
        edge = p1 - p0
        edge_norm = np.linalg.norm(edge)
        if edge_norm <= 1.0e-12:
            continue

        edge_angle = math.atan2(float(edge[1]), float(edge[0]))
        align_rot = rotation_matrix_2d(-edge_angle)
        rotated = hull @ align_rot.T
        mins = np.min(rotated, axis=0)
        maxs = np.max(rotated, axis=0)
        lengths = maxs - mins
        area = float(lengths[0] * lengths[1])

        candidate = {
            "angle_rad": -edge_angle,
            "area": area,
        }
        if best is None or candidate["area"] < best["area"]:
            best = candidate

    if best is None:
        best = {"angle_rad": 0.0}
    return best


def get_locked_axis_indices(lock_axis: str) -> tuple[int, tuple[int, int]]:
    if lock_axis == "x":
        return 0, (1, 2)
    if lock_axis == "y":
        return 1, (0, 2)
    if lock_axis == "z":
        return 2, (0, 1)
    raise ValueError(f"未知锁定轴: {lock_axis}")


def rotation_matrix_locked_axis(lock_axis: str, angle_rad: float) -> np.ndarray:
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    rot = np.eye(3, dtype=np.float32)
    if lock_axis == "x":
        rot[1:, 1:] = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    elif lock_axis == "y":
        rot[np.ix_([0, 2], [0, 2])] = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    elif lock_axis == "z":
        rot[:2, :2] = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    else:
        raise ValueError(f"未知锁定轴: {lock_axis}")
    return rot


def rotation_matrix_z(angle_rad: float) -> np.ndarray:
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    return np.array(
        [
            [cos_a, -sin_a, 0.0],
            [sin_a, cos_a, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def compute_bbox_stats(vertices: np.ndarray, bbox_type: str, lock_axis: Optional[str], angle_rad: float) -> dict:
    mins = np.min(vertices, axis=0)
    maxs = np.max(vertices, axis=0)
    lengths = maxs - mins
    center = 0.5 * (mins + maxs)
    return {
        "bbox_type": bbox_type,
        "obb_lock_axis": lock_axis,
        "normalization_rotation_degrees": float(np.rad2deg(angle_rad)),
        "bbox_min": mins.astype(float).tolist(),
        "bbox_max": maxs.astype(float).tolist(),
        "bbox_center": center.astype(float).tolist(),
        "x_length": float(lengths[0]),
        "y_length": float(lengths[1]),
        "z_length": float(lengths[2]),
        "bbox_volume": float(lengths[0] * lengths[1] * lengths[2]),
    }


def make_row_vector_affine_matrix(linear: np.ndarray, translation: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(linear, dtype=np.float64)
    matrix[3, :3] = np.asarray(translation, dtype=np.float64)
    return matrix


def invert_row_vector_affine_matrix(matrix: np.ndarray) -> np.ndarray:
    linear = matrix[:3, :3]
    translation = matrix[3, :3]
    inverse_linear = np.linalg.inv(linear)
    inverse_translation = -translation @ inverse_linear
    return make_row_vector_affine_matrix(inverse_linear, inverse_translation)


def build_export_transform_stats(bbox_stats: dict, axis_stats: dict) -> dict:
    normalization_rotation_matrix = np.asarray(bbox_stats["normalization_rotation_matrix"], dtype=np.float64)
    normalization_center = np.asarray(bbox_stats["normalization_center"], dtype=np.float64)
    normalization_bbox_center = np.asarray(bbox_stats["normalization_bbox_center"], dtype=np.float64)
    axis_transform = np.asarray(axis_stats["axis_transform_matrix"], dtype=np.float64)
    normalization_linear = normalization_rotation_matrix.T
    normalization_translation = -normalization_center @ normalization_rotation_matrix.T - normalization_bbox_center
    export_linear = axis_transform.T @ normalization_linear
    export_translation = normalization_translation
    input_to_export = make_row_vector_affine_matrix(export_linear, export_translation)
    export_to_input = invert_row_vector_affine_matrix(input_to_export)
    return {
        "transform_convention": "row-vector homogeneous: [x, y, z, 1] @ matrix",
        "transform_order": "axis_transform_then_obb_normalization",
        "normalization_rotation_matrix": normalization_rotation_matrix.astype(float).tolist(),
        "input_to_export_matrix": input_to_export.astype(float).tolist(),
        "export_to_input_matrix": export_to_input.astype(float).tolist(),
    }


def build_ordered_stats(
    *,
    args: argparse.Namespace,
    bbox_stats: dict,
    axis_stats: dict,
    input_stats: dict,
    raw_vertex_count: int,
    raw_face_count: int,
    export_vertex_count: int,
    export_face_count: int,
    requested_simplify_method: str,
    applied_simplify_method: str,
) -> dict:
    return {
        "input_path": str(args.input),
        "output_path": str(args.output),
        "bbox_type": bbox_stats["bbox_type"],
        "obb_lock_axis": bbox_stats["obb_lock_axis"],
        "output_up_axis": axis_stats["output_up_axis"],
        "axis_transform": axis_stats["axis_transform"],
        "normalization_rotation_degrees": bbox_stats["normalization_rotation_degrees"],
        "x_length": bbox_stats["x_length"],
        "y_length": bbox_stats["y_length"],
        "z_length": bbox_stats["z_length"],
        "bbox_volume": bbox_stats["bbox_volume"],
        "bbox_min": bbox_stats["bbox_min"],
        "bbox_max": bbox_stats["bbox_max"],
        "bbox_center": bbox_stats["bbox_center"],
        "input_filter": input_stats,
        "normalization_center": bbox_stats["normalization_center"],
        "normalization_bbox_center": bbox_stats["normalization_bbox_center"],
        "axis_transform_matrix": axis_stats["axis_transform_matrix"],
        **build_export_transform_stats(bbox_stats, axis_stats),
        "face_ratio": float(args.face_ratio),
        "requested_simplify_method": requested_simplify_method,
        "applied_simplify_method": applied_simplify_method,
        "raw_vertices": raw_vertex_count,
        "raw_faces": raw_face_count,
        "export_vertices": export_vertex_count,
        "export_faces": export_face_count,
    }


def convert_output_up_axis(vertices: np.ndarray, output_up_axis: str) -> tuple[np.ndarray, dict]:
    if output_up_axis == "z":
        return vertices, {
            "output_up_axis": "z",
            "axis_transform": "identity",
            "axis_transform_matrix": np.eye(3, dtype=np.float32).astype(float).tolist(),
        }

    if output_up_axis == "y":
        transform = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float32,
        )
        converted = vertices @ transform.T
        return converted, {
            "output_up_axis": "y",
            "axis_transform": "z-up_to_y-up",
            "axis_transform_matrix": transform.astype(float).tolist(),
        }

    raise ValueError(f"未知输出上轴: {output_up_axis}")


def align_vertices_to_locked_obb(vertices: np.ndarray, lock_axis: str) -> tuple[np.ndarray, dict]:
    center = np.mean(vertices, axis=0)
    centered = vertices - center[None, :]
    _, plane_idx = get_locked_axis_indices(lock_axis)
    obb_2d = min_area_rect_2d(centered[:, plane_idx])
    angle_rad = float(obb_2d["angle_rad"])
    rot = rotation_matrix_locked_axis(lock_axis, angle_rad)
    aligned = centered @ rot.T
    bbox_center = 0.5 * (np.min(aligned, axis=0) + np.max(aligned, axis=0))
    aligned = aligned - bbox_center[None, :]
    stats = compute_bbox_stats(
        aligned,
        bbox_type=f"{lock_axis}-locked-obb",
        lock_axis=lock_axis,
        angle_rad=angle_rad,
    )
    stats["normalization_center"] = center.astype(float).tolist()
    stats["normalization_bbox_center"] = bbox_center.astype(float).tolist()
    stats["normalization_rotation_matrix"] = rot.astype(float).tolist()
    return aligned.astype(np.float32), stats


def align_vertices_to_obb(vertices: np.ndarray) -> tuple[np.ndarray, dict]:
    center = np.mean(vertices, axis=0)
    centered = vertices - center[None, :]
    if centered.shape[0] < 3:
        rot = np.eye(3, dtype=np.float32)
    else:
        covariance = np.cov(centered.astype(np.float64), rowvar=False)
        _, eigenvectors = np.linalg.eigh(covariance)
        rot = eigenvectors[:, ::-1].T.astype(np.float32)
        if np.linalg.det(rot) < 0.0:
            rot[-1, :] *= -1.0
    angle_rad = 0.0
    aligned = centered @ rot.T
    bbox_center = 0.5 * (np.min(aligned, axis=0) + np.max(aligned, axis=0))
    aligned = aligned - bbox_center[None, :]
    stats = compute_bbox_stats(
        aligned,
        bbox_type="obb",
        lock_axis=None,
        angle_rad=angle_rad,
    )
    stats["normalization_center"] = center.astype(float).tolist()
    stats["normalization_bbox_center"] = bbox_center.astype(float).tolist()
    stats["normalization_rotation_matrix"] = rot.astype(float).tolist()
    return aligned.astype(np.float32), stats


def preserve_vertices_orientation(vertices: np.ndarray) -> tuple[np.ndarray, dict]:
    center = np.mean(vertices, axis=0)
    centered = vertices - center[None, :]
    bbox_center = 0.5 * (np.min(centered, axis=0) + np.max(centered, axis=0))
    aligned = centered - bbox_center[None, :]
    stats = compute_bbox_stats(
        aligned,
        bbox_type="input-axis-aabb",
        lock_axis=None,
        angle_rad=0.0,
    )
    stats["normalization_center"] = center.astype(float).tolist()
    stats["normalization_bbox_center"] = bbox_center.astype(float).tolist()
    stats["normalization_rotation_matrix"] = np.eye(3, dtype=np.float32).astype(float).tolist()
    return aligned.astype(np.float32), stats


def resolve_obb_lock_axis(lock_axis: str, output_up_axis: str) -> str:
    if lock_axis == "auto":
        return output_up_axis
    return lock_axis


def simplify_mesh_with_vertex_clustering(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: Optional[np.ndarray],
    face_ratio: float,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if not (0.0 < face_ratio <= 1.0):
        raise ValueError("--face-ratio 必须在 (0, 1] 范围内")
    if face_ratio >= 1.0 or faces.shape[0] == 0:
        return vertices, faces, colors

    target_faces = max(4, int(round(faces.shape[0] * face_ratio)))
    mins = np.min(vertices, axis=0)
    maxs = np.max(vertices, axis=0)
    max_extent = float(np.max(maxs - mins))
    if max_extent <= 1.0e-9:
        return vertices, faces, colors

    def cluster_once(voxel_size: float):
        quantized = np.floor((vertices - mins[None, :]) / voxel_size).astype(np.int64)
        _, inverse = np.unique(quantized, axis=0, return_inverse=True)
        new_vertex_count = int(np.max(inverse)) + 1

        new_vertices = np.zeros((new_vertex_count, 3), dtype=np.float32)
        np.add.at(new_vertices, inverse, vertices)
        counts = np.bincount(inverse).astype(np.float32)
        new_vertices /= counts[:, None]

        new_colors = None
        if colors is not None:
            new_colors = np.zeros((new_vertex_count, 3), dtype=np.float32)
            np.add.at(new_colors, inverse, colors)
            new_colors /= counts[:, None]

        new_faces = inverse[faces]
        non_degenerate = (
            (new_faces[:, 0] != new_faces[:, 1]) &
            (new_faces[:, 0] != new_faces[:, 2]) &
            (new_faces[:, 1] != new_faces[:, 2])
        )
        new_faces = new_faces[non_degenerate]
        if new_faces.shape[0] == 0:
            return new_vertices, new_faces.astype(np.int32), new_colors

        canonical = np.sort(new_faces, axis=1)
        _, unique_idx = np.unique(canonical, axis=0, return_index=True)
        new_faces = new_faces[np.sort(unique_idx)].astype(np.int32)
        return new_vertices, new_faces, new_colors

    best = (vertices, faces, colors)
    best_gap = abs(faces.shape[0] - target_faces)
    low = max_extent * 1.0e-6
    high = max_extent

    for _ in range(14):
        voxel_size = math.sqrt(low * high)
        cand_vertices, cand_faces, cand_colors = cluster_once(voxel_size)
        cand_gap = abs(cand_faces.shape[0] - target_faces)
        if cand_gap < best_gap and cand_faces.shape[0] > 0:
            best = (cand_vertices, cand_faces, cand_colors)
            best_gap = cand_gap

        if cand_faces.shape[0] > target_faces:
            low = voxel_size
        else:
            high = voxel_size

    return best


def simplify_mesh_with_quadric_decimation(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: Optional[np.ndarray],
    face_ratio: float,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if not (0.0 < face_ratio <= 1.0):
        raise ValueError("--face-ratio 必须在 (0, 1] 范围内")
    if face_ratio >= 1.0 or faces.shape[0] == 0:
        return vertices, faces, colors

    import open3d as o3d

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64, copy=False))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32, copy=False))
    if colors is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors.astype(np.float64, copy=False))

    target_faces = max(4, int(round(faces.shape[0] * face_ratio)))
    simplified = mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)
    simplified.remove_degenerate_triangles()
    simplified.remove_duplicated_triangles()
    simplified.remove_unreferenced_vertices()
    simplified.remove_duplicated_vertices()

    out_vertices = np.asarray(simplified.vertices, dtype=np.float32)
    out_faces = np.asarray(simplified.triangles, dtype=np.int32)
    out_colors = None
    if simplified.has_vertex_colors():
        out_colors = np.asarray(simplified.vertex_colors, dtype=np.float32)

    if out_faces.shape[0] == 0:
        return vertices, faces, colors
    return out_vertices, out_faces, out_colors


def simplify_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: Optional[np.ndarray],
    face_ratio: float,
    method: str,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], str]:
    if method == "none" or face_ratio >= 1.0 or faces.shape[0] == 0:
        return vertices, faces, colors, "none"

    if method == "clustering":
        out = simplify_mesh_with_vertex_clustering(vertices, faces, colors, face_ratio)
        return out[0], out[1], out[2], "clustering"

    if method == "quadric":
        out = simplify_mesh_with_quadric_decimation(vertices, faces, colors, face_ratio)
        return out[0], out[1], out[2], "quadric"

    if method == "auto":
        try:
            out = simplify_mesh_with_quadric_decimation(vertices, faces, colors, face_ratio)
            return out[0], out[1], out[2], "quadric"
        except Exception as exc:
            print(f"WARNING: quadric 减面失败，回退到 clustering: {exc}")
            out = simplify_mesh_with_vertex_clustering(vertices, faces, colors, face_ratio)
            return out[0], out[1], out[2], "clustering"

    raise ValueError(f"未知减面方式: {method}")


def save_ply_mesh(
    output_path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: Optional[np.ndarray] = None,
) -> None:
    from plyfile import PlyData, PlyElement

    output_path.parent.mkdir(parents=True, exist_ok=True)

    vertex_dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    if colors is not None:
        vertex_dtype.extend([("red", "u1"), ("green", "u1"), ("blue", "u1")])

    vertex_data = np.empty(vertices.shape[0], dtype=vertex_dtype)
    vertex_data["x"] = vertices[:, 0]
    vertex_data["y"] = vertices[:, 1]
    vertex_data["z"] = vertices[:, 2]
    if colors is not None:
        color_u8 = np.clip(np.rint(colors * 255.0), 0, 255).astype(np.uint8)
        vertex_data["red"] = color_u8[:, 0]
        vertex_data["green"] = color_u8[:, 1]
        vertex_data["blue"] = color_u8[:, 2]

    face_data = np.empty(faces.shape[0], dtype=[("vertex_indices", "i4", (3,))])
    face_data["vertex_indices"] = faces.astype(np.int32, copy=False)

    ply = PlyData(
        [
            PlyElement.describe(vertex_data, "vertex"),
            PlyElement.describe(face_data, "face"),
        ],
        text=False,
    )
    ply.write(str(output_path))


def infer_one(args: argparse.Namespace) -> InferenceResult:
    import torch
    import nksr
    import nksr.fields

    def set_texture_from_numpy(field, texture_xyz_np: np.ndarray, texture_color_np: np.ndarray, texture_device: torch.device) -> None:
        texture_xyz = torch.from_numpy(np.ascontiguousarray(texture_xyz_np, dtype=np.float32))
        texture_color = torch.from_numpy(np.ascontiguousarray(texture_color_np, dtype=np.float32)).to(texture_device)
        field.set_texture_field(nksr.fields.PCNNField(texture_xyz, texture_color))

    if args.output.suffix.lower() != ".ply":
        raise ValueError("当前脚本仅支持输出 .ply 网格")
    max_chunk_extent, max_chunk_points, chunk_overlap_ratio = resolve_chunk_constraints(args)
    if max_chunk_extent is not None and args.voxel_size is not None:
        raise ValueError("--max-chunk-extent/--chunk-size 与 --voxel-size 不能同时使用")
    if not (0.0 < args.face_ratio <= 1.0):
        raise ValueError("--face-ratio 必须在 (0, 1] 范围内")
    chunking_enabled = max_chunk_extent is not None or max_chunk_points is not None

    xyz_np, normal_np, sensor_np, color_np, columns = load_point_cloud(
        args.input,
        split_columns(args.xyz_cols),
        split_columns(args.normal_cols),
        split_columns(args.sensor_cols),
        split_columns(args.color_cols),
    )

    xyz_np, normal_np, sensor_np, color_np, input_stats = filter_point_cloud_data(
        xyz_np,
        normal_np,
        sensor_np,
        color_np,
        outlier_quantile=getattr(args, "outlier_quantile", 0.0),
        min_filtered_points=getattr(args, "min_filtered_points", 1000),
    )
    extent_stats = validate_input_extent(xyz_np, getattr(args, "max_input_extent", 200.0))
    input_stats.update(extent_stats)

    print(f"Loaded {args.input} with {input_stats['original_points']} points")
    if input_stats["dropped_points"] > 0:
        print(f"Filtered points: {input_stats['filtered_points']} kept, {input_stats['dropped_points']} dropped")
    print(
        "Input bbox lengths: "
        + ", ".join(f"{value:.6f}" for value in input_stats["point_bbox_lengths"])
    )
    print(f"Available columns: {columns}")
    print(f"Has color: {color_np is not None}")

    device = torch.device(args.device)
    input_xyz = None
    input_normal = None
    input_sensor = None
    if not chunking_enabled:
        input_xyz = torch.from_numpy(xyz_np).float().to(device)
        input_normal = torch.from_numpy(normal_np).float().to(device) if normal_np is not None else None
        input_sensor = torch.from_numpy(sensor_np).float().to(device) if sensor_np is not None else None

    reconstructor = nksr.Reconstructor(device, config=args.config)
    reconstructor.chunk_tmp_device = torch.device(args.chunk_tmp_device)

    mode = args.input_mode
    if mode == "auto":
        if normal_np is not None:
            mode = "normal"
        elif sensor_np is not None:
            mode = "sensor"
        else:
            mode = "none"

    print(f"Input mode: {mode}")

    if mode in {"sensor", "estimate"} and device.type != "cuda":
        raise ValueError("sensor/estimate 模式当前依赖 CUDA 扩展，请使用 --device cuda:0 之类的设备")

    preprocess_fn = None
    if mode == "none":
        input_normal = None
        input_sensor = None
    elif mode == "normal":
        if normal_np is None:
            raise ValueError("选择了 normal 模式，但输入点云不包含法向列")
        if not chunking_enabled:
            input_normal = torch.from_numpy(normal_np).float().to(device)
    elif mode == "sensor":
        if sensor_np is None:
            raise ValueError("选择了 sensor 模式，但输入点云不包含传感器列")
        if not chunking_enabled:
            input_sensor = torch.from_numpy(sensor_np).float().to(device)
        preprocess_fn = nksr.get_estimate_normal_preprocess_fn(
            args.estimate_knn,
            args.drop_threshold_deg,
        )
    elif mode == "estimate":
        if not chunking_enabled:
            if input_xyz is None:
                input_xyz = torch.from_numpy(xyz_np).float().to(device)
            input_normal = estimate_normals_from_xyz(input_xyz, args.estimate_knn, args.estimate_orient)
        input_sensor = None
    else:
        raise ValueError(f"未知输入模式: {mode}")
    chunk_normal_np = None
    chunk_sensor_np = None
    if chunking_enabled:
        chunk_normal_np, chunk_sensor_np = chunk_orientation_arrays(
            mode,
            normal_np,
            sensor_np,
            input_normal,
        )

    mesh = None
    field = None
    all_fields = []
    all_transforms = []
    all_centers = []
    all_texture_inputs = []
    single_chunk_center = None
    single_chunk_texture_xyz = None
    single_chunk_texture_color = None
    reconstruct_attempts = max(1, int(getattr(args, "reconstruct_attempts", 1)))
    for attempt_index in range(1, reconstruct_attempts + 1):
        field = None
        all_fields = []
        all_transforms = []
        all_centers = []
        all_texture_inputs = []
        single_chunk_center = None
        single_chunk_texture_xyz = None
        single_chunk_texture_color = None
        reconstructor.network.to(device)
        with torch.inference_mode():
            if chunking_enabled:
                planned_chunks = plan_adaptive_chunks(
                    xyz_np,
                    chunk_normal_np,
                    chunk_sensor_np,
                    color_np,
                    max_chunk_extent=max_chunk_extent,
                    max_chunk_points=max_chunk_points,
                    overlap_ratio=chunk_overlap_ratio,
                )
                if not planned_chunks:
                    raise ReconstructionFailedError("重建失败，切分后没有可用 chunk")
                point_counts = [chunk.xyz.shape[0] for chunk in planned_chunks]
                print(
                    "Chunk plan: "
                    f"count={len(planned_chunks)} "
                    f"min_points={min(point_counts)} "
                    f"max_points={max(point_counts)} "
                    f"max_chunk_points={max_chunk_points} "
                    f"max_chunk_extent={max_chunk_extent} "
                    f"overlap={chunk_overlap_ratio}"
                )

                for chunk_index, chunk in enumerate(planned_chunks, start=1):
                    chunk_min = np.min(chunk.xyz, axis=0)
                    chunk_max = np.max(chunk.xyz, axis=0)
                    chunk_lengths = chunk_max - chunk_min
                    print(
                        "Chunk "
                        f"{chunk_index}/{len(planned_chunks)}: "
                        f"points={chunk.xyz.shape[0]} "
                        f"bbox_lengths="
                        + ",".join(f"{value:.3f}" for value in chunk_lengths)
                    )
                    chunk_xyz = torch.from_numpy(chunk.xyz).float().to(device)
                    chunk_normal = torch.from_numpy(chunk.normal).float().to(device) if chunk.normal is not None else None
                    chunk_sensor = torch.from_numpy(chunk.sensor).float().to(device) if chunk.sensor is not None else None
                    if mode == "estimate":
                        chunk_normal = estimate_normals_from_xyz(chunk_xyz, args.estimate_knn, args.estimate_orient)
                        chunk_sensor = None
                    chunk_field = None
                    try:
                        chunk_field = reconstructor.reconstruct(
                            chunk_xyz,
                            normal=chunk_normal,
                            sensor=chunk_sensor,
                            detail_level=args.detail_level,
                            voxel_size=args.voxel_size,
                            chunk_size=-1.0,
                            approx_kernel_grad=args.approx_kernel_grad,
                            solver_max_iter=args.solver_max_iter,
                            solver_tol=args.solver_tol,
                            nystrom_min_depth=args.nystrom_min_depth,
                            fused_mode=args.fused_mode,
                            preprocess_fn=preprocess_fn,
                        )
                    finally:
                        del chunk_xyz, chunk_normal, chunk_sensor
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                    if chunk_field is None:
                        print(f"Chunk {chunk_index}/{len(planned_chunks)}: skipped after preprocess")
                        continue
                    chunk_field.to_(torch.device(args.chunk_tmp_device))
                    all_fields.append(chunk_field)
                    all_transforms.append(nksr.Isometry(t=chunk.center))
                    all_centers.append(chunk.center)
                    if chunk.color is None:
                        all_texture_inputs.append(None)
                    else:
                        all_texture_inputs.append((chunk.xyz, chunk.color))
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    print(f"Chunk {chunk_index}/{len(planned_chunks)}: done")

                if not all_fields:
                    raise ReconstructionFailedError("重建失败，所有 chunk 在预处理后都为空")

                if len(all_fields) == 1:
                    extract_device = torch.device("cpu") if args.extract_on_cpu else device
                    field = all_fields[0]
                    field.to_(extract_device)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    single_chunk_center = all_centers[0]
                    if all_texture_inputs[0] is not None:
                        single_chunk_texture_xyz, single_chunk_texture_color = all_texture_inputs[0]
                    print(f"Reconstruction field ready on {extract_device}")
                else:
                    single_chunk_texture_xyz = None
                    single_chunk_texture_color = None
                    fuse_device = torch.device("cpu") if args.extract_on_cpu else device
                    print(f"Fusing {len(all_fields)} chunks on {fuse_device}")
                    for field_index, chunk_field in enumerate(all_fields, start=1):
                        chunk_field.to_(fuse_device)
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        print(f"Chunk field {field_index}/{len(all_fields)} moved to {fuse_device}")
                    field = nksr.fields.FusedField(all_fields, all_transforms)
            else:
                if input_xyz is None:
                    input_xyz = torch.from_numpy(xyz_np).float().to(device)
                if mode == "normal" and input_normal is None:
                    input_normal = torch.from_numpy(normal_np).float().to(device)
                elif mode == "sensor" and input_sensor is None:
                    input_sensor = torch.from_numpy(sensor_np).float().to(device)
                elif mode == "estimate" and input_normal is None:
                    input_normal = estimate_normals_from_xyz(input_xyz, args.estimate_knn, args.estimate_orient)
                field = reconstructor.reconstruct(
                    input_xyz,
                    normal=input_normal,
                    sensor=input_sensor,
                    detail_level=args.detail_level,
                    voxel_size=args.voxel_size,
                    chunk_size=-1.0,
                    approx_kernel_grad=args.approx_kernel_grad,
                    solver_max_iter=args.solver_max_iter,
                    solver_tol=args.solver_tol,
                    nystrom_min_depth=args.nystrom_min_depth,
                    fused_mode=args.fused_mode,
                    preprocess_fn=preprocess_fn,
                )

            if field is None:
                raise ReconstructionFailedError("重建失败，输入点云在预处理后为空")

            if args.texture_mode == "off":
                print("Texture export disabled by texture mode")
            elif chunking_enabled and single_chunk_texture_color is not None and single_chunk_texture_xyz is not None:
                texture_device = torch.device("cpu") if args.extract_on_cpu else device
                set_texture_from_numpy(field, single_chunk_texture_xyz, single_chunk_texture_color, texture_device)
            elif color_np is not None:
                texture_device = torch.device("cpu") if args.extract_on_cpu else device
                set_texture_from_numpy(field, xyz_np, color_np, texture_device)

            if device.type == "cuda":
                torch.cuda.empty_cache()

            if args.extract_on_cpu:
                field.to_("cpu")
                reconstructor.network.to("cpu")

            print(f"Extract mesh start attempt={attempt_index}/{reconstruct_attempts}")
            mesh = field.extract_dual_mesh(
                mise_iter=args.mise_iter,
                grid_upsample=args.grid_upsample,
            )
            print("Extract mesh done")

        if mesh is not None and mesh.v.shape[0] > 0 and mesh.f.numel() > 0:
            break
        if attempt_index < reconstruct_attempts:
            print(f"NKSR empty mesh on attempt {attempt_index}/{reconstruct_attempts}; retry reconstruction")
            del mesh, field
            mesh = None
            field = None
            all_fields = []
            all_transforms = []
            all_centers = []
            all_texture_inputs = []
            if device.type == "cuda":
                torch.cuda.empty_cache()

    input_xyz = None
    input_normal = None
    input_sensor = None
    field = None
    all_fields = []
    all_transforms = []
    all_centers = []
    all_texture_inputs = []
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if mesh is None:
        raise ReconstructionFailedError("NKSR 未提取到有效 mesh，重建未返回可提取结果")
    vertices = mesh.v.detach().cpu().numpy().astype(np.float32, copy=False)
    if vertices.shape[0] == 0 or mesh.f.numel() == 0:
        raise ReconstructionFailedError(
            f"NKSR 未提取到有效 mesh，已尝试 {reconstruct_attempts} 次；请检查输入点云是否足以形成闭合等值面"
        )
    if single_chunk_center is not None:
        vertices = vertices + single_chunk_center[None, :].astype(np.float32, copy=False)
    faces = mesh.f.detach().cpu().numpy().astype(np.int32, copy=False)
    colors = None
    if mesh.c is not None:
        colors = mesh.c.detach().cpu().numpy().astype(np.float32, copy=False)

    export_space_vertices, axis_stats = convert_output_up_axis(vertices, args.output_up_axis)
    resolved_obb_lock_axis = resolve_obb_lock_axis(args.obb_lock_axis, args.output_up_axis)
    if resolved_obb_lock_axis == "none":
        export_space_vertices, pre_axis_stats = preserve_vertices_orientation(export_space_vertices)
    elif resolved_obb_lock_axis == "pca":
        export_space_vertices, pre_axis_stats = align_vertices_to_obb(export_space_vertices)
    else:
        export_space_vertices, pre_axis_stats = align_vertices_to_locked_obb(export_space_vertices, resolved_obb_lock_axis)

    bbox_stats = compute_bbox_stats(
        export_space_vertices,
        bbox_type=pre_axis_stats["bbox_type"],
        lock_axis=pre_axis_stats["obb_lock_axis"],
        angle_rad=float(np.deg2rad(pre_axis_stats["normalization_rotation_degrees"])),
    )
    for key in ("normalization_center", "normalization_bbox_center", "normalization_rotation_matrix"):
        bbox_stats[key] = pre_axis_stats[key]

    raw_face_count = int(faces.shape[0])
    raw_vertex_count = int(vertices.shape[0])
    requested_simplify_method = args.simplify_method
    export_vertices, export_faces, export_colors, applied_simplify_method = simplify_mesh(
        export_space_vertices,
        faces,
        colors,
        args.face_ratio,
        requested_simplify_method,
    )

    stats = build_ordered_stats(
        args=args,
        bbox_stats=bbox_stats,
        axis_stats=axis_stats,
        input_stats=input_stats,
        raw_vertex_count=raw_vertex_count,
        raw_face_count=raw_face_count,
        export_vertex_count=int(export_vertices.shape[0]),
        export_face_count=int(export_faces.shape[0]),
        requested_simplify_method=requested_simplify_method,
        applied_simplify_method=applied_simplify_method,
    )

    save_ply_mesh(args.output, export_vertices, export_faces, export_colors)

    print(f"Saved mesh to {args.output}")
    print(f"BBox type: {stats['bbox_type']}")
    print(f"Lengths xyz: {stats['x_length']:.6f}, {stats['y_length']:.6f}, {stats['z_length']:.6f}")
    print(f"BBox volume: {stats['bbox_volume']:.6f}")
    print(f"Vertices/Faces: {stats['export_vertices']}, {stats['export_faces']}")
    print(f"Simplify method: {stats['applied_simplify_method']} (requested: {stats['requested_simplify_method']})")

    stats_path = None
    if args.write_stats:
        stats_path = args.output.with_suffix(".stats.json")
        with stats_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"Saved stats to {stats_path}")

    return InferenceResult(output_path=args.output, stats_path=stats_path, stats=stats)


def iter_input_files(input_dir: Path, extensions: Sequence[str]) -> list[Path]:
    normalized_exts = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
    files = []
    for path in sorted(input_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in normalized_exts:
            files.append(path)
    return files


def batch_infer(args: argparse.Namespace) -> list[InferenceResult]:
    input_dir = args.input_dir.resolve()
    output_dir = Path(args.output_dir).resolve()
    if not input_dir.is_dir():
        raise ValueError(f"输入目录不存在: {input_dir}")

    extensions = [part.strip() for part in args.extensions.split(",") if part.strip()]
    files = iter_input_files(input_dir, extensions)
    if not files:
        raise ValueError(f"输入目录下未找到匹配文件: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"文件数量: {len(files)}")

    results = []
    for input_path in files:
        rel = input_path.relative_to(input_dir)
        output_path = (output_dir / rel).with_suffix(".ply")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cur_args = argparse.Namespace(**vars(args))
        cur_args.input = input_path
        cur_args.output = output_path

        print()
        print(f"[NKSR] {input_path}")
        print(f"   -> {output_path}")

        if args.skip_existing and output_path.exists():
            print("   skipped: output exists")
            continue

        results.append(infer_one(cur_args))

    print()
    print("全部处理完成。")
    return results
