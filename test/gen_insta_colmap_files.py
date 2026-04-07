#!/usr/bin/env python3

import argparse
import json
import math
import sys
from pathlib import Path


INSTA_CAMERA_CONFIGS = {
    "INSTA_VIDEO": {
        "cameras": [
            {
                "id": 1,
                "name": "camera_0",
                "model": "OPENCV_FISHEYE",
                "width": 2944,
                "height": 2880,
                "params": [
                    935.087283,
                    933.809055,
                    1474.293589,
                    1446.077297,
                    0.028118,
                    -0.016894,
                    0.000993,
                    -0.000267,
                ],
            },
            {
                "id": 2,
                "name": "camera_1",
                "model": "OPENCV_FISHEYE",
                "width": 2944,
                "height": 2880,
                "params": [
                    934.142356,
                    933.064808,
                    1474.955869,
                    1445.242114,
                    0.027524,
                    -0.016272,
                    0.000846,
                    -0.000245,
                ],
            },
        ],
    },
    "INSTA_VIDEO_NORMAL": {
        "cameras": [
            {
                "id": 1,
                "name": "camera_0",
                "model": "OPENCV_FISHEYE",
                "width": 3072,
                "height": 3072,
                "params": [
                    955.912563,
                    955.973425,
                    1566.184363,
                    1552.666712,
                    0.030456,
                    -0.020892,
                    0.00342,
                    -0.000747,
                ],
            },
            {
                "id": 2,
                "name": "camera_1",
                "model": "OPENCV_FISHEYE",
                "width": 3072,
                "height": 3072,
                "params": [
                    951.374377,
                    951.83608,
                    1553.469005,
                    1537.895787,
                    0.029403,
                    -0.018581,
                    0.002153,
                    -0.000493,
                ],
            },
        ],
    },
    "INSTA_VIDEO_PANO": {
        "cameras": [
            {
                "id": 1,
                "name": "pano",
                "model": "SPHERE",
                "width": 3840,
                "height": 1920,
                "params": [1.0, 1920.0, 960.0],
            }
        ],
    },
}

# 文档中给的是 "camera1 to camera0"，即 T_c0_c1。
DUAL_FISHEYE_T_C0_C1 = {
    "qvec": [-0.000949003, 0.00672011, -0.999976, 0.00120144],
    "tvec": [0.00226111, -0.000805, -0.0294827],
}


def normalize_quaternion(qvec):
    norm = math.sqrt(sum(v * v for v in qvec))
    if norm < 1e-12:
        return [1.0, 0.0, 0.0, 0.0]
    return [v / norm for v in qvec]


def quaternion_to_rotation_matrix(qvec):
    w, x, y, z = normalize_quaternion(qvec)
    return [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]


def mat3_transpose(matrix):
    return [[matrix[j][i] for j in range(3)] for i in range(3)]


def mat3_vec3_mul(matrix, vector):
    return [
        sum(matrix[row][col] * vector[col] for col in range(3))
        for row in range(3)
    ]


def invert_pose(qvec, tvec):
    qvec = normalize_quaternion(qvec)
    inv_qvec = [qvec[0], -qvec[1], -qvec[2], -qvec[3]]
    rot = quaternion_to_rotation_matrix(qvec)
    rot_t = mat3_transpose(rot)
    inv_tvec = [-value for value in mat3_vec3_mul(rot_t, tvec)]
    return inv_qvec, inv_tvec


def format_float(value):
    return f"{value:.15g}"


def write_cameras_txt(cameras, output_path):
    lines = [
        "# Camera list with one line of data per camera:",
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
        f"# Number of cameras: {len(cameras)}",
    ]
    for camera in cameras:
        row = [
            str(camera["id"]),
            camera["model"],
            str(camera["width"]),
            str(camera["height"]),
        ]
        row.extend(format_float(param) for param in camera["params"])
        lines.append(" ".join(row))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_dual_fisheye_rig(ref_camera_name):
    camera_0 = {
        "camera_id": 1,
        "image_prefix": "camera_0",
        "rel_qvec": [1.0, 0.0, 0.0, 0.0],
        "rel_tvec": [0.0, 0.0, 0.0],
    }
    camera_1 = {
        "camera_id": 2,
        "image_prefix": "camera_1",
        "rel_qvec": [1.0, 0.0, 0.0, 0.0],
        "rel_tvec": [0.0, 0.0, 0.0],
    }

    if ref_camera_name == "camera_1":
        camera_0["rel_qvec"] = normalize_quaternion(DUAL_FISHEYE_T_C0_C1["qvec"])
        camera_0["rel_tvec"] = DUAL_FISHEYE_T_C0_C1["tvec"][:]
        ref_camera_id = 2
    else:
        inv_qvec, inv_tvec = invert_pose(
            DUAL_FISHEYE_T_C0_C1["qvec"], DUAL_FISHEYE_T_C0_C1["tvec"]
        )
        camera_1["rel_qvec"] = inv_qvec
        camera_1["rel_tvec"] = inv_tvec
        ref_camera_id = 1

    return [
        {
            "ref_camera_id": ref_camera_id,
            "cameras": [camera_0, camera_1],
        }
    ]


def make_single_camera_rig(camera):
    return [
        {
            "ref_camera_id": camera["id"],
            "cameras": [
                {
                    "camera_id": camera["id"],
                    "image_prefix": camera["name"],
                    "rel_qvec": [1.0, 0.0, 0.0, 0.0],
                    "rel_tvec": [0.0, 0.0, 0.0],
                }
            ],
        }
    ]


def round_floats(value):
    if isinstance(value, float):
        return float(format_float(value))
    if isinstance(value, list):
        return [round_floats(item) for item in value]
    if isinstance(value, dict):
        return {key: round_floats(item) for key, item in value.items()}
    return value


def write_rig_json(rig_config, output_path):
    output_path.write_text(
        json.dumps(round_floats(rig_config), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def generate_one(camera_type, ref_camera_name, output_dir):
    config = INSTA_CAMERA_CONFIGS[camera_type]
    output_dir.mkdir(parents=True, exist_ok=True)

    cameras_txt_path = output_dir / "cameras.txt"
    rig_json_path = output_dir / "rig.json"

    write_cameras_txt(config["cameras"], cameras_txt_path)

    if camera_type == "INSTA_VIDEO_PANO":
        rig_config = make_single_camera_rig(config["cameras"][0])
    else:
        rig_config = make_dual_fisheye_rig(ref_camera_name)

    write_rig_json(rig_config, rig_json_path)
    return cameras_txt_path, rig_json_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="根据硬编码的 Insta 参数生成 COLMAP cameras.txt 和 rig.json。"
    )
    parser.add_argument(
        "--camera-type",
        default="all",
        choices=["all", "INSTA_VIDEO", "INSTA_VIDEO_NORMAL", "INSTA_VIDEO_PANO"],
        help="要生成的相机类型，默认一次生成全部。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "generated_insta_colmap",
        help="输出目录；当 --camera-type=all 时会在该目录下按类型分子目录输出。",
    )
    parser.add_argument(
        "--ref-camera",
        default="camera_0",
        choices=["camera_0", "camera_1"],
        help="双鱼眼 rig 使用哪个相机作为参考相机，默认 camera_0。",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.camera_type == "all":
        camera_types = list(INSTA_CAMERA_CONFIGS.keys())
    else:
        camera_types = [args.camera_type]

    for camera_type in camera_types:
        target_dir = args.output_dir / camera_type.lower()
        cameras_txt_path, rig_json_path = generate_one(
            camera_type, args.ref_camera, target_dir
        )
        print(f"[{camera_type}] cameras: {cameras_txt_path}")
        print(f"[{camera_type}] rig: {rig_json_path}")
        if camera_type == "INSTA_VIDEO_PANO":
            print(
                "[INSTA_VIDEO_PANO] note: 当前仓库的 C++ 相机模型列表里没有 SPHERE，"
                "这里按文档要求写出文本格式参数。",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
