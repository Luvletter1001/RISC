#!/usr/bin/env python3
"""Run MMEngine validation for FOCUS-TAC under the training val protocol."""

from __future__ import annotations

import argparse
import os
import os.path as osp
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Config file path")
    parser.add_argument("checkpoint", help="Checkpoint file path")
    parser.add_argument("--work-dir", help="Directory for validation logs")
    parser.add_argument(
        "--launcher",
        choices=("none", "pytorch", "slurm", "mpi"),
        default="none")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def main() -> int:
    args = parse_args()
    repo_root = osp.abspath(osp.join(osp.dirname(__file__), "../../.."))
    tools_dir = osp.join(repo_root, "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from openrsd_env import preload_installed_mmengine

    preload_installed_mmengine()

    from mmdet.utils import register_all_modules as register_mmdet_modules
    from mmengine.config import Config
    from mmengine.registry import RUNNERS
    from mmengine.runner import Runner
    from mmrotate.utils import register_all_modules as register_mmrotate_modules

    register_mmdet_modules(init_default_scope=False)
    register_mmrotate_modules(init_default_scope=False)

    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.seed is not None:
        cfg.randomness = dict(seed=args.seed, deterministic=False)
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        cfg.work_dir = osp.join(
            "./work_dirs", osp.splitext(osp.basename(args.config))[0])
    cfg.load_from = args.checkpoint

    runner = RUNNERS.build(cfg) if "runner_type" in cfg else Runner.from_cfg(cfg)
    runner.val()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
