#!/usr/bin/env python3
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.pcd_inference import build_infer_parser, infer_one, load_env_file


def main() -> None:
    load_env_file()
    args = build_infer_parser().parse_args()
    infer_one(args)


if __name__ == "__main__":
    main()
