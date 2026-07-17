import numpy as np

from tools.pcd_inference import (
    align_vertices_to_obb,
    align_vertices_to_locked_obb,
    build_export_transform_stats,
    build_infer_parser,
    checkpoint_file_name,
    chunk_orientation_arrays,
    configured_checkpoint_path,
    convert_output_up_axis,
)


class TensorLike:

    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float64)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


def test_chunk_orientation_arrays_uses_estimated_normals_for_chunks():
    estimated = TensorLike([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    normal_np, sensor_np = chunk_orientation_arrays(
        "estimate",
        normal_np=None,
        sensor_np=np.ones((2, 3), dtype=np.float32),
        estimated_normal=estimated,
    )

    assert normal_np.dtype == np.float32
    assert np.allclose(normal_np, estimated.values)
    assert sensor_np is None


def test_chunk_orientation_arrays_clears_orientation_for_none_mode():
    normal_np, sensor_np = chunk_orientation_arrays(
        "none",
        normal_np=np.ones((2, 3), dtype=np.float32),
        sensor_np=np.ones((2, 3), dtype=np.float32),
        estimated_normal=None,
    )

    assert normal_np is None
    assert sensor_np is None


def test_align_vertices_to_obb_uses_full_3d_pca_axes():
    base = np.array([
        [-2.0, -1.0, -0.5],
        [-2.0, -1.0, 0.5],
        [-2.0, 1.0, -0.5],
        [-2.0, 1.0, 0.5],
        [2.0, -1.0, -0.5],
        [2.0, -1.0, 0.5],
        [2.0, 1.0, -0.5],
        [2.0, 1.0, 0.5],
    ], dtype=np.float32)
    angle = np.deg2rad(35.0)
    rot_y = np.array([
        [np.cos(angle), 0.0, np.sin(angle)],
        [0.0, 1.0, 0.0],
        [-np.sin(angle), 0.0, np.cos(angle)],
    ], dtype=np.float32)
    vertices = base @ rot_y.T

    aligned, stats = align_vertices_to_obb(vertices)
    lengths = np.max(aligned, axis=0) - np.min(aligned, axis=0)

    assert stats["bbox_type"] == "obb"
    assert stats["obb_lock_axis"] is None
    assert np.isclose(np.prod(lengths), 8.0, atol=1.0e-4)


def test_export_transform_stats_match_locked_obb_and_axis_conversion():
    vertices = np.array([
        [-2.0, -1.0, -0.5],
        [-2.0, -1.0, 0.5],
        [-2.0, 1.0, -0.5],
        [-2.0, 1.0, 0.5],
        [2.0, -1.0, -0.5],
        [2.0, -1.0, 0.5],
        [2.0, 1.0, -0.5],
        [2.0, 1.0, 0.5],
    ], dtype=np.float32)
    angle = np.deg2rad(25.0)
    rot_z = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    translated = vertices @ rot_z.T + np.array([10.0, -3.0, 2.0], dtype=np.float32)

    aligned, bbox_stats = align_vertices_to_locked_obb(translated, "z")
    export_vertices, axis_stats = convert_output_up_axis(aligned, "y")
    transform_stats = build_export_transform_stats(bbox_stats, axis_stats)

    for legacy_key in ("rotation_degrees", "rotation_matrix", "alignment_center", "post_rotation_bbox_center"):
        assert legacy_key not in bbox_stats
        assert legacy_key not in transform_stats

    ones = np.ones((translated.shape[0], 1), dtype=np.float64)
    homogeneous_input = np.concatenate([translated.astype(np.float64), ones], axis=1)
    input_to_export = np.asarray(transform_stats["input_to_export_matrix"], dtype=np.float64)
    export_to_input = np.asarray(transform_stats["export_to_input_matrix"], dtype=np.float64)
    transformed = (homogeneous_input @ input_to_export)[:, :3]
    restored = (np.concatenate([transformed, ones], axis=1) @ export_to_input)[:, :3]

    assert transform_stats["transform_convention"] == "row-vector homogeneous: [x, y, z, 1] @ matrix"
    assert "normalization_rotation_matrix" in transform_stats
    assert np.allclose(transformed, export_vertices, atol=1.0e-5)
    assert np.allclose(restored, translated, atol=1.0e-5)


def test_checkpoint_file_name_matches_config_url():
    assert checkpoint_file_name("ks") == "ks.pth"
    assert checkpoint_file_name("snet") == "snet-n3k-wnormal.pth"
    assert checkpoint_file_name("snet-wonormal") == "snet-n3k-wonormal.pth"


def test_configured_checkpoint_path_uses_local_checkpoint_dir(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "snet-n3k-wonormal.pth"
    checkpoint_path.write_bytes(b"placeholder")
    monkeypatch.setenv("NKSR_CHECKPOINTS_DIR", str(tmp_path))

    assert configured_checkpoint_path("snet-wonormal") == checkpoint_path


def test_default_inference_settings_prefer_delivery_quality(monkeypatch):
    for name in (
        "NKSR_MAX_CHUNK_POINTS",
        "NKSR_MAX_CHUNK_INPUT_EXTENT",
        "NKSR_CHUNK_OVERLAP_RATIO",
        "NKSR_TEXTURE_MODE",
        "NKSR_SIMPLIFY_METHOD",
        "NKSR_FACE_RATIO",
    ):
        monkeypatch.delenv(name, raising=False)

    args = build_infer_parser().parse_args(["input.pcd", "output.ply"])

    assert args.max_chunk_points == 750000
    assert args.max_chunk_input_extent is None
    assert args.chunk_overlap_ratio == 0.05
    assert args.texture_mode == "auto"
    assert args.simplify_method == "none"
    assert args.face_ratio == 1.0
