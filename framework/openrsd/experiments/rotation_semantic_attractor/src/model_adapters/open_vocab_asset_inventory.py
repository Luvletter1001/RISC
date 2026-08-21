from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
from typing import Iterable


KEYWORDS = {
    "openrsd",
    "prompt",
    "text",
    "visual",
    "support",
    "alignment",
    "fusion",
    "dota",
    "dehub",
    "b5k",
    "b8k",
    "step2",
    "step3",
    "epoch",
    "latest",
    "best",
    "grounding",
    "groundingdino",
    "yoloworld",
    "yolo-world",
    "lae",
}

SUPPORTED_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".txt", ".pkl", ".pth", ".pt"}
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", "node_modules"}


def _read_probe(path: Path, limit: int = 65536) -> str:
    if path.suffix.lower() not in {".py", ".yaml", ".yml", ".json", ".txt"}:
        return ""
    try:
        return path.read_text(errors="ignore")[:limit]
    except Exception:
        return ""


def _matched_keywords(path: Path, probe: str = "") -> list[str]:
    text = f"{_path_context(path)} {probe}".lower()
    return sorted(keyword for keyword in KEYWORDS if keyword in text)


def _path_context(path: Path) -> str:
    parts = [part for part in path.parts if part.lower() not in {"openrsd"}]
    return "/".join(parts[-6:])


def _asset_type(path: Path) -> str:
    suffix = path.suffix.lower()
    name = path.name.lower()
    if suffix in {".pth", ".pt"}:
        return "checkpoint"
    if suffix == ".pkl":
        return "support_pkl" if "support" in name else "support_pkl"
    if suffix == ".py":
        return "config" if any(token in name for token in ("config", "a10_", "a12_", "open", "grounding", "yolo", "lae")) else "script"
    if suffix in {".yaml", ".yml"}:
        return "config"
    if suffix in {".json", ".txt"}:
        return "prompt_file" if any(token in name for token in ("prompt", "class", "vocab")) else "class_file"
    return "script"


def _candidate_family(path: Path, probe: str = "") -> str:
    text = f"{_path_context(path)} {probe}".lower()
    if "grounding_dino" in text or "groundingdino" in text:
        return "groundingdino"
    if "yoloworld" in text or "yolo-world" in text:
        return "yoloworld"
    if "lae" in text:
        return "lae"
    if "openrtmdet" in text or "m_ad" in text or "step2" in text or "step3" in text or "mmr_ad" in text:
        return "openrsd"
    return "unknown"


def _blocker(asset_type: str, family: str, path: Path, probe: str = "") -> str:
    if asset_type == "checkpoint":
        try:
            if path.name.startswith("._") or path.stat().st_size < 1024 * 1024:
                return "checkpoint sidecar or too small to be a runnable model checkpoint"
        except OSError:
            return "checkpoint path cannot be stat()ed"
    if asset_type == "config" and family == "openrsd" and "openrtmdet" not in probe.lower():
        return "candidate config path/name matched, but OpenRTMDet marker was not found in file content"
    if asset_type == "support_pkl":
        return ""
    if family in {"groundingdino", "yoloworld", "lae"} and asset_type == "config":
        return "config candidate only; checkpoint/prompt protocol still must be paired"
    return ""


def classify_asset(path: str | Path) -> dict:
    p = Path(path)
    probe = _read_probe(p)
    asset_type = _asset_type(p)
    family = _candidate_family(p, probe)
    exists = p.exists()
    stat = p.stat() if exists else None
    blocker = _blocker(asset_type, family, p, probe)
    return {
        "asset_type": asset_type,
        "path": str(p),
        "exists": exists,
        "size_bytes": int(stat.st_size) if stat else 0,
        "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat() if stat else "",
        "matched_keywords": _matched_keywords(p, probe),
        "candidate_model_family": family,
        "usable": exists and not blocker,
        "blocker": blocker,
    }


def _iter_candidate_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    if root.is_file():
        if root.suffix.lower() in SUPPORTED_SUFFIXES:
            yield root
        return
    for dirpath_str, dirnames, filenames in os.walk(root):
        dirpath = Path(dirpath_str)
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS]
        for filename in filenames:
            path = dirpath / filename
            if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            probe = ""
            if not _matched_keywords(path) and path.suffix.lower() in {".py", ".yaml", ".yml", ".json"}:
                probe = _read_probe(path)
            if _matched_keywords(path, probe):
                yield path


def scan_open_vocab_assets(roots: Iterable[str | Path]) -> list[dict]:
    seen = set()
    assets = []
    for root in roots:
        for path in _iter_candidate_files(Path(root)):
            resolved = str(path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            asset = classify_asset(path)
            if asset["matched_keywords"]:
                assets.append(asset)
    assets.sort(key=lambda item: (item["candidate_model_family"], item["asset_type"], item["path"]))
    return assets


def find_runnable_open_vocab_pairs(assets: list[dict]) -> list[dict]:
    configs = [
        asset for asset in assets
        if asset["asset_type"] == "config"
        and asset["candidate_model_family"] in {"openrsd", "groundingdino", "yoloworld", "lae"}
        and asset["usable"]
    ]
    checkpoints = sorted(
        [asset for asset in assets if asset["asset_type"] == "checkpoint" and asset["usable"]],
        key=lambda asset: (
            "epoch_24_weights_only" not in asset["path"].lower(),
            "epoch_24" not in asset["path"].lower(),
            asset["path"],
        ),
    )
    supports = sorted(
        [
            asset for asset in assets
            if asset["asset_type"] == "support_pkl"
            and asset["usable"]
            and "support" in Path(asset["path"]).name.lower()
        ],
        key=lambda asset: ("step5_3" not in asset["path"].lower(), asset["path"]),
    )
    pairs = []
    for config in configs:
        family = config["candidate_model_family"]
        family_checkpoints = [
            ckpt for ckpt in checkpoints
            if family in ckpt["candidate_model_family"] or family in ckpt["path"].lower() or family == "openrsd"
        ]
        if not family_checkpoints:
            continue
        support = supports[0] if supports else None
        if family == "openrsd" and support is None:
            continue
        checkpoint = family_checkpoints[0]
        pairs.append(
            {
                "candidate_model_family": family,
                "config": config["path"],
                "checkpoint": checkpoint["path"],
                "support_pkl": support["path"] if support else "",
                "usable": True,
                "blocker": "",
            }
        )
    return pairs


def inventory_summary(assets: list[dict], pairs: list[dict]) -> dict:
    exp_src = Path(__file__).resolve().parents[1]
    hook_candidates = [
        exp_src / "model_adapters/open_vocab_hooks.py",
        exp_src / "model_adapters/openrsd_hook_registry.py",
        exp_src / "model_adapters/openrsd_adapter.py",
    ]
    missing = {
        "missing_configs": [],
        "missing_checkpoints": [],
        "missing_prompt_paths": [],
        "missing_support_pkls": [],
        "missing_embedding_hooks": [
            str(path.relative_to(exp_src.parent)) for path in hook_candidates if not path.exists()
        ],
        "missing_model_code": [],
    }
    families = {asset["candidate_model_family"] for asset in assets}
    if "openrsd" not in families:
        missing["missing_configs"].append("OpenRSD/OpenRTMDet config")
    if not any(asset["asset_type"] == "checkpoint" for asset in assets):
        missing["missing_checkpoints"].append("open-vocabulary/OpenRSD checkpoint")
    if not any(asset["asset_type"] in {"prompt_file", "class_file"} for asset in assets):
        missing["missing_prompt_paths"].append("fixed DOTA prompt/class vocabulary protocol")
    if not any(asset["asset_type"] == "support_pkl" for asset in assets):
        missing["missing_support_pkls"].append("OpenRSD support embedding pkl")
    return {
        "num_assets": len(assets),
        "num_runnable_pairs": len(pairs),
        "families": sorted(families),
        "status": "DONE_SMOKE" if pairs else "NOT_AVAILABLE_ASSET",
        "status_reason": (
            "runnable open-vocabulary/OpenRSD config+checkpoint pair candidates found"
            if pairs
            else "no runnable open-vocabulary/OpenRSD config+checkpoint pair found"
        ),
        **missing,
    }


def render_inventory_markdown(summary: dict, assets: list[dict], pairs: list[dict]) -> str:
    lines = [
        "# Open-Vocabulary Asset Inventory",
        "",
        f"- status: `{summary['status']}`",
        f"- status_reason: `{summary['status_reason']}`",
        f"- num_assets: `{summary['num_assets']}`",
        f"- num_runnable_pairs: `{summary['num_runnable_pairs']}`",
        "",
        "## Runnable Pair Candidates",
        "",
    ]
    if pairs:
        lines.extend(["| family | config | checkpoint | support_pkl |", "| --- | --- | --- | --- |"])
        for pair in pairs:
            lines.append(
                f"| {pair['candidate_model_family']} | {pair['config']} | {pair['checkpoint']} | {pair['support_pkl']} |"
            )
    else:
        lines.append("No runnable pair found.")
    lines.extend(["", "## Asset Candidates", "", "| type | family | usable | path | blocker |", "| --- | --- | --- | --- | --- |"])
    for asset in assets[:200]:
        lines.append(
            f"| {asset['asset_type']} | {asset['candidate_model_family']} | {asset['usable']} | {asset['path']} | {asset['blocker']} |"
        )
    if len(assets) > 200:
        lines.append(f"\nShowing 200/{len(assets)} assets.")
    return "\n".join(lines) + "\n"


def render_asset_request_markdown(summary: dict) -> str:
    lines = [
        "# Open-Vocabulary Asset Request",
        "",
        "This file lists the concrete files or hooks needed before OpenRSD/open-vocabulary results can become DONE_FULL.",
        "",
        "## Missing / Required",
        "",
        f"- Config: `{', '.join(summary['missing_configs']) or 'no core OpenRSD config missing from local candidates'}`",
        f"- Checkpoint: `{', '.join(summary['missing_checkpoints']) or 'no core OpenRSD checkpoint missing from local candidates'}`",
        f"- Prompt/class file: `{', '.join(summary['missing_prompt_paths']) or 'fixed prompt protocol still should be committed under experiment configs'}`",
        f"- Support pkl: `{', '.join(summary['missing_support_pkls']) or 'OpenRSD support pkl candidate found'}`",
        f"- Adapter hooks: `{', '.join(summary['missing_embedding_hooks'])}`",
        "",
        "## Recommended Placement",
        "",
        "- Open-vocab registry: `experiments/rotation_semantic_attractor/configs/open_vocab_model_registry.yaml`",
        "- Prompt protocol: `experiments/rotation_semantic_attractor/configs/open_vocab_prompt_protocol.json`",
        "- Adapter: `experiments/rotation_semantic_attractor/src/model_adapters/openrsd_adapter.py`",
        "- Hook registry: `experiments/rotation_semantic_attractor/src/model_adapters/openrsd_hook_registry.py`",
        "",
        "## Next Command",
        "",
        "```bash",
        "rtk python3 experiments/rotation_semantic_attractor/scripts/12_open_vocab_asset_inventory.py",
        "```",
    ]
    return "\n".join(lines) + "\n"
