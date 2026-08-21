#!/usr/bin/env python
"""Build a local Focus-OVD eval dependency bundle.

The original Focus-OVD configs read several small pickle files from
``./data`` at model initialization.  Those files are not part of the model
checkpoint.  This helper creates an explicit, auditable bundle for eval-time
inference without mutating the repository-level ``data`` symlink.
"""

import argparse
import json
import os
import pickle
from pathlib import Path


DEFAULT_DOTA2_CLASSES = [
    "airport",
    "baseball-diamond",
    "basketball-court",
    "bridge",
    "container-crane",
    "ground-track-field",
    "harbor",
    "helicopter",
    "helipad",
    "large-vehicle",
    "plane",
    "roundabout",
    "ship",
    "small-vehicle",
    "soccer-ball-field",
    "storage-tank",
    "swimming-pool",
    "tennis-court",
]


def normalize_class_name(class_name):
    return "_".join(
        word.capitalize()
        for word in str(class_name).replace("-", "_").split("_")
    )


def add_normalized_variants(mapping, class_name):
    normalized = normalize_class_name(class_name)
    variants = {
        class_name,
        str(class_name).lower(),
        str(class_name).upper(),
        str(class_name).replace("_", " "),
        str(class_name).replace("-", " "),
        str(class_name).replace("_", "-"),
        normalized,
        normalized[:1].lower() + normalized[1:],
    }
    for variant in variants:
        mapping[variant] = normalized
    mapping[normalized] = normalized


def build_normalized_mapping(class_names):
    mapping = {}
    for class_name in class_names:
        add_normalized_variants(mapping, class_name)
    return mapping


def load_config_val_classes(config_path):
    if not config_path:
        return DEFAULT_DOTA2_CLASSES
    from mmengine.config import Config

    cfg = Config.fromfile(str(config_path))
    model_cfg = cfg.get("model", {})
    classes = model_cfg.get("val_support_classes")
    if classes:
        return list(classes)
    dataset_cfg = cfg.get("test_dataloader", {}).get("dataset", {})
    metainfo = dataset_cfg.get("metainfo", {})
    return list(metainfo.get("classes", DEFAULT_DOTA2_CLASSES))


def replace_path(dst, src):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve(strict=True))


def write_pickle(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        path.unlink()
    with path.open("wb") as f:
        pickle.dump(payload, f)


def load_pickle(path):
    with Path(path).open("rb") as f:
        return pickle.load(f)


def find_extra_support_entry(extra_support_paths, target_class):
    target_norm = normalize_class_name(target_class)
    for extra_path in extra_support_paths:
        extra_path = Path(extra_path)
        extra_support = load_pickle(extra_path)
        for extra_class, payload in extra_support.items():
            if normalize_class_name(extra_class) == target_norm:
                return extra_path, payload
    return None, None


def build_eval_bundle(
        support_path,
        out_dir,
        val_classes,
        dataset_flag,
        extra_support_paths=None):
    support_path = Path(support_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    extra_support_paths = [Path(path) for path in (extra_support_paths or [])]

    support = load_pickle(support_path)
    base_support_class_count = len(support)
    support_classes = list(support.keys())
    all_classes = list(dict.fromkeys(list(val_classes) + support_classes))
    normalized_map = build_normalized_mapping(all_classes)

    normalized_val_classes = [normalized_map[c] for c in val_classes]
    normalized_support_classes = {normalized_map[c] for c in support_classes}
    initial_missing_support = [
        cls for cls, norm_cls in zip(val_classes, normalized_val_classes)
        if norm_cls not in normalized_support_classes
    ]
    filled_support = {}
    for missing_class in initial_missing_support:
        extra_path, payload = find_extra_support_entry(
            extra_support_paths, missing_class)
        if payload is None:
            continue
        support[missing_class] = payload
        normalized_support_classes.add(normalize_class_name(missing_class))
        filled_support[missing_class] = str(extra_path)

    support_classes = list(support.keys())
    missing_support = [
        cls for cls in val_classes
        if normalize_class_name(cls) not in normalized_support_classes
    ]

    norm_path = out_dir / "normalized_class_dict.pkl"
    neg_path = out_dir / "Neg_supports_v2.pkl"
    pca_path = out_dir / "7_25_pca_meta_DINOv2_256.pkl"
    bundled_support = (
        out_dir
        / "DOTA2_1024_500"
        / "ss_train"
        / "Step5_3_Prepare_Visual_Text_DINOv2_support.pkl"
    )

    write_pickle(norm_path, normalized_map)
    write_pickle(
        neg_path,
        {
            "all_support": {},
            "cls_tree": {},
            "parent_mapping": {},
            "neg_dict": {
                dataset_flag: {class_name: [] for class_name in normalized_val_classes}
            },
        },
    )
    write_pickle(
        pca_path,
        {
            "note": (
                "Eval-time placeholder. Flex_Rtmdet_v3_1_formal.predict "
                "loads pca_meta but does not read fields from it."
            )
        },
    )
    if filled_support:
        write_pickle(bundled_support, support)
    else:
        replace_path(bundled_support, support_path)

    summary = {
        "support_path": str(support_path),
        "out_dir": str(out_dir),
        "dataset_flag": dataset_flag,
        "val_class_count": len(val_classes),
        "base_support_class_count": base_support_class_count,
        "support_class_count": len(support_classes),
        "val_classes": list(val_classes),
        "support_classes": support_classes,
        "filled_support_classes": filled_support,
        "missing_support_classes": missing_support,
        "extra_support_paths": [str(path) for path in extra_support_paths],
        "normalized_class_dict": str(norm_path),
        "neg_support_data": str(neg_path),
        "pca_meta_pth": str(pca_path),
        "bundled_support": str(bundled_support),
    }
    (out_dir / "eval_bundle_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + os.linesep)
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--support-pkl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config", default="")
    parser.add_argument("--val-classes", default="")
    parser.add_argument("--dataset-flag", default="Data1_DOTA2")
    parser.add_argument("--extra-support-pkl", action="append", default=[])
    return parser.parse_args()


def main():
    args = parse_args()
    if args.val_classes:
        val_classes = [item for item in args.val_classes.split(",") if item]
    else:
        val_classes = load_config_val_classes(args.config)
    summary = build_eval_bundle(
        support_path=args.support_pkl,
        out_dir=args.out_dir,
        val_classes=val_classes,
        dataset_flag=args.dataset_flag,
        extra_support_paths=args.extra_support_pkl,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
