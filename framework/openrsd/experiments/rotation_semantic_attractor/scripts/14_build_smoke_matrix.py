#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from common import PROJECT_ROOT  # noqa: F401 - ensures project root is on sys.path
from experiments.rotation_semantic_attractor.src.utils.io import read_json, write_csv
from experiments.rotation_semantic_attractor.src.utils.status import ExperimentStatus


def _verdict_path(run_dir: Path) -> Path:
    return run_dir / "smoke_verdict.txt"


def _read_verdict(run_dir: Path) -> str:
    path = _verdict_path(run_dir)
    if not path.exists():
        return "FAIL"
    for line in path.read_text().splitlines():
        if line.startswith("verdict="):
            return line.split("=", 1)[1]
    return "FAIL"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--closedset-run-dir", required=True)
    parser.add_argument("--openvocab-run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    closedset_dir = Path(args.closedset_run_dir)
    openvocab_dir = Path(args.openvocab_run_dir)
    output_dir = Path(args.output_dir)
    closedset_verdict = _read_verdict(closedset_dir)
    openvocab_manifest = read_json(openvocab_dir / "manifest.json") if (openvocab_dir / "manifest.json").exists() else {}
    openvocab_status = openvocab_manifest.get("status", ExperimentStatus.NOT_AVAILABLE_ASSET)
    rows = [
        {
            "smoke": "closedset",
            "run_dir": str(closedset_dir),
            "status": ExperimentStatus.DONE_SMOKE if closedset_verdict.startswith("PASS") else ExperimentStatus.FAILED,
            "status_reason": closedset_verdict,
        },
        {
            "smoke": "openvocab",
            "run_dir": str(openvocab_dir),
            "status": openvocab_status,
            "status_reason": openvocab_manifest.get("status_reason", ""),
        },
    ]
    write_csv(output_dir / "smoke_matrix_summary.csv", rows)
    if not closedset_verdict.startswith("PASS"):
        verdict = "FAIL"
    elif (
        "NOT_APPLICABLE" in closedset_verdict
        or "NOT_SELECTED" in closedset_verdict
        or openvocab_status in {ExperimentStatus.NOT_AVAILABLE_ASSET, ExperimentStatus.NOT_SELECTED_IN_THIS_SMOKE, ExperimentStatus.NOT_APPLICABLE}
    ):
        verdict = "PASS_WITH_NOT_APPLICABLE"
    elif "UNSUPPORTED" in closedset_verdict or openvocab_status == ExperimentStatus.UNSUPPORTED_BY_CURRENT_CODE:
        verdict = "PASS_WITH_UNSUPPORTED_HOOKS"
    elif "PARTIAL" in closedset_verdict:
        verdict = "PASS_WITH_PARTIAL_HOOKS"
    else:
        verdict = "PASS"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "smoke_verdict.txt").write_text(
        "\n".join(
            [
                f"verdict={verdict}",
                f"closedset_verdict={closedset_verdict}",
                f"openvocab_status={openvocab_status}",
                f"closedset_run_dir={closedset_dir}",
                f"openvocab_run_dir={openvocab_dir}",
            ]
        )
        + "\n"
    )
    print(f"matrix_verdict={verdict}")
    print(f"matrix_run_dir={output_dir}")


if __name__ == "__main__":
    main()
