#!/usr/bin/env python3
"""Download external VLM weights for fres_002 inventory under pretrained/."""
from __future__ import annotations

import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PRE = REPO / "pretrained"
CHUNK = 8 * 1024 * 1024


def _remote_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return int(resp.headers["Content-Length"])


def _wget_resume(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = _remote_size(url)
    have = dest.stat().st_size if dest.exists() else 0
    if have >= total * 0.99:
        print(f"SKIP complete {dest} ({have}/{total} bytes)")
        return
    mode = "ab" if have > 0 else "wb"
    headers = {}
    if have > 0:
        headers["Range"] = f"bytes={have}-"
        print(f"RESUME {dest} from {have}/{total} ({100 * have / total:.1f}%)")
    else:
        print(f"GET {url}\n -> {dest} ({total} bytes)")
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as resp:
        with open(dest, mode) as out:
            while True:
                chunk = resp.read(CHUNK)
                if not chunk:
                    break
                out.write(chunk)
                have = dest.stat().st_size
                if total > 0 and have % (64 * CHUNK) < CHUNK:
                    print(f"  ... {have}/{total} ({100 * have / total:.1f}%)", flush=True)
    final = dest.stat().st_size
    if final < total * 0.99:
        raise RuntimeError(f"SIZE_MISMATCH {dest}: got {final}, expected {total}")
    print(f"OK {dest} ({final} bytes)")


def _hf(repo_id: str, filename: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1024:
        print(f"SKIP exists {dest} ({dest.stat().st_size} bytes)")
        return
    from huggingface_hub import hf_hub_download

    print(f"HF {repo_id}/{filename} -> {dest}")
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(dest.parent),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    got = Path(path)
    if got.resolve() != dest.resolve():
        if dest.exists():
            dest.unlink()
        got.rename(dest)
    print(f"OK {dest.stat().st_size} bytes")


def main() -> int:
    jobs: list[tuple[str, callable]] = []

    expected = [
        PRE / "remoteclip/RemoteCLIP-ViT-B-32.pt",
        PRE / "georsclip/ckpt/RS5M_ViT-B-32.pt",
        PRE / "lae-dino/checkpoints/lae_dino_swint_fintune_dota-9e1e8782.pth",
        PRE / "lae-dino/checkpoints/lae_dino_swint_fintune_dior-e612b298.pth",
        PRE / "open_clip/OpenCLIP-ViT-B-32-laion2b.bin",
    ]
    for p in expected:
        if not p.exists() or p.stat().st_size < 1024:
            print(f"WARN missing/incomplete: {p}")

    jobs.append((
        "CLIP ViT-B/32",
        lambda: _wget_resume(
            "https://openaipublic.azureedge.net/clip/models/"
            "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
            PRE / "clip/ViT-B-32.pt",
        ),
    ))
    jobs.append((
        "DINOv2 ViT-L/14",
        lambda: _wget_resume(
            "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
            PRE / "dinov2/dinov2_vitl14_pretrain.pth",
        ),
    ))
    jobs.append((
        "DINO ViT-B/16",
        lambda: _wget_resume(
            "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth",
            PRE / "dino/dino_vitbase16_pretrain.pth",
        ),
    ))

    failed = []
    for name, fn in jobs:
        try:
            fn()
        except Exception as exc:
            print(f"FAIL {name}: {exc}", file=sys.stderr)
            failed.append(name)

    for sub, note in (
        ("rotclip", "RotCLIP: no public checkpoint found; add weights here if you obtain them."),
        ("roroclip", "RoRoCLIP: no public checkpoint (2025 ElL paper); add weights here if released."),
    ):
        d = PRE / sub
        d.mkdir(parents=True, exist_ok=True)
        readme = d / "README_NO_PUBLIC_WEIGHTS.txt"
        if not readme.exists():
            readme.write_text(note + "\n", encoding="utf-8")

    print("\n=== Summary ===")
    for sub in ("remoteclip", "georsclip", "lae-dino", "clip", "open_clip", "dinov2", "dino", "rotclip", "roroclip"):
        p = PRE / sub
        if not p.exists():
            continue
        files = [f for f in p.rglob("*") if f.is_file() and f.suffix in {".pt", ".pth", ".bin", ".safetensors"}]
        for f in files:
            print(f"  {f.name}: {f.stat().st_size} bytes")
    if failed:
        print("Failed:", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
