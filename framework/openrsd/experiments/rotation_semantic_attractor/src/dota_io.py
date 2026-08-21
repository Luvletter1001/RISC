from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from .geometry import flat_poly_to_points


def read_dota_txt(path: str | Path) -> List[Dict]:
    p = Path(path)
    if not p.exists():
        return []
    records = []
    for line_no, line in enumerate(p.read_text().splitlines(), start=1):
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) < 9:
            records.append({"error": f"line_{line_no}_too_short", "raw": line})
            continue
        coords = [float(v) for v in parts[:8]]
        label = parts[8]
        difficulty = parts[9] if len(parts) > 9 else "0"
        records.append(
            {
                "polygon": flat_poly_to_points(coords),
                "class_name": label,
                "difficulty": difficulty,
                "line_no": line_no,
            }
        )
    return records


def write_dota_txt(path: str | Path, records: Iterable[Dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for rec in records:
        coords = []
        for x, y in rec["polygon"]:
            coords.extend([f"{float(x):.3f}", f"{float(y):.3f}"])
        lines.append(" ".join(coords + [rec["class_name"], str(rec.get("difficulty", "0"))]))
    p.write_text("\n".join(lines) + ("\n" if lines else ""))


def tile_id_from_path(path: str | Path) -> str:
    return Path(path).stem


def records_to_polygons(records: Sequence[Dict]) -> List[List[List[float]]]:
    return [rec["polygon"] for rec in records if "polygon" in rec]

