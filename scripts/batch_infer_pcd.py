#!/usr/bin/env python3
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.pcd_inference import batch_infer, build_batch_cli_parser, load_env_file


def main() -> None:
    load_env_file()
    args = build_batch_cli_parser().parse_args()
    batch_infer(args)


if __name__ == "__main__":
    main()
