import subprocess
from pathlib import Path


RUNNER = Path("M_Tools/experiments/run_hrrsd_bass_gsf_p2_train_20260620.sh")


def _parse_p2_runner_args(*args: str) -> dict[str, str]:
    lines = RUNNER.read_text(encoding="utf-8").splitlines()
    prefix = []
    for line in lines:
        prefix.append(line)
        if line.startswith("EPOCHS="):
            break
    script = "\n".join(prefix + [
        'printf "GPU_ID=%s\\n" "$GPU_ID"',
        'printf "VARIANT=%s\\n" "$VARIANT"',
        'printf "EPOCHS=%s\\n" "$EPOCHS"',
    ])
    proc = subprocess.run(
        ["bash", "-s", "--", *args],
        input=script,
        text=True,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.returncode == 0, proc.stderr
    parsed = {}
    for line in proc.stdout.splitlines():
        key, value = line.split("=", 1)
        parsed[key] = value
    return parsed


def test_p2_runner_preserves_assign_rank_variant_without_usage_suffix():
    parsed = _parse_p2_runner_args("0", "assign_rank", "2")

    assert parsed["GPU_ID"] == "0"
    assert parsed["VARIANT"] == "assign_rank"
    assert parsed["EPOCHS"] == "2"


def test_p2_runner_preserves_posterior_rank_variant_without_usage_suffix():
    parsed = _parse_p2_runner_args("1", "posterior_rank", "2")

    assert parsed["GPU_ID"] == "1"
    assert parsed["VARIANT"] == "posterior_rank"
    assert parsed["EPOCHS"] == "2"
