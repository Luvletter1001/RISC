import json
import pickle
from pathlib import Path

import numpy as np

from M_Tools.analysis.build_focus_ovd_eval_bundle import build_eval_bundle


def test_build_eval_bundle_writes_init_pkls_and_reports_missing_support(tmp_path):
    support_path = tmp_path / "source_support.pkl"
    support = {
        "small-vehicle": {
            "texts": ["small vehicle"],
            "text_embeds": np.zeros((2, 768), dtype=np.float32),
            "visual_embeds": np.zeros((3, 1024), dtype=np.float32),
            "confidence_scores": np.ones(3, dtype=np.float32),
        },
        "tennis-court": {
            "texts": ["tennis court"],
            "text_embeds": np.zeros((2, 768), dtype=np.float32),
            "visual_embeds": np.zeros((3, 1024), dtype=np.float32),
            "confidence_scores": np.ones(3, dtype=np.float32),
        },
    }
    with support_path.open("wb") as f:
        pickle.dump(support, f)

    out_dir = tmp_path / "bundle"
    summary = build_eval_bundle(
        support_path=support_path,
        out_dir=out_dir,
        val_classes=["airport", "small-vehicle", "tennis-court"],
        dataset_flag="Data1_DOTA2",
    )

    assert summary["support_class_count"] == 2
    assert summary["missing_support_classes"] == ["airport"]

    with (out_dir / "normalized_class_dict.pkl").open("rb") as f:
        norm = pickle.load(f)
    assert norm["small-vehicle"] == "Small_Vehicle"
    assert norm["small vehicle"] == "Small_Vehicle"
    assert norm["Small_Vehicle"] == "Small_Vehicle"

    with (out_dir / "Neg_supports_v2.pkl").open("rb") as f:
        neg = pickle.load(f)
    assert neg["neg_dict"]["Data1_DOTA2"]["Small_Vehicle"] == []

    assert (out_dir / "7_25_pca_meta_DINOv2_256.pkl").exists()
    linked_support = (
        out_dir
        / "DOTA2_1024_500"
        / "ss_train"
        / "Step5_3_Prepare_Visual_Text_DINOv2_support.pkl"
    )
    assert linked_support.exists()
    assert json.loads((out_dir / "eval_bundle_summary.json").read_text()) == summary


def test_build_eval_bundle_fills_missing_support_from_extra_sources(tmp_path):
    support_path = tmp_path / "source_support.pkl"
    with support_path.open("wb") as f:
        pickle.dump({
            "small-vehicle": {
                "texts": ["small vehicle"],
                "text_embeds": np.zeros((2, 768), dtype=np.float32),
                "visual_embeds": np.zeros((3, 1024), dtype=np.float32),
                "confidence_scores": np.ones(3, dtype=np.float32),
            },
        }, f)

    extra_path = tmp_path / "extra_support.pkl"
    with extra_path.open("wb") as f:
        pickle.dump({
            "Airport": {
                "texts": ["airport"],
                "text_embeds": np.ones((2, 768), dtype=np.float32),
                "visual_embeds": np.ones((3, 1024), dtype=np.float32),
                "confidence_scores": np.ones(3, dtype=np.float32),
            },
        }, f)

    out_dir = tmp_path / "bundle"
    summary = build_eval_bundle(
        support_path=support_path,
        out_dir=out_dir,
        val_classes=["airport", "small-vehicle"],
        dataset_flag="Data1_DOTA2",
        extra_support_paths=[extra_path],
    )

    assert summary["missing_support_classes"] == []
    assert summary["filled_support_classes"] == {"airport": str(extra_path)}
    assert summary["base_support_class_count"] == 1
    assert summary["support_class_count"] == 2

    bundled_support = (
        out_dir
        / "DOTA2_1024_500"
        / "ss_train"
        / "Step5_3_Prepare_Visual_Text_DINOv2_support.pkl"
    )
    with bundled_support.open("rb") as f:
        merged = pickle.load(f)
    assert set(merged) == {"airport", "small-vehicle"}


def test_build_eval_bundle_replaces_existing_support_symlink_when_filling(tmp_path):
    support_path = tmp_path / "source_support.pkl"
    with support_path.open("wb") as f:
        pickle.dump({
            "small-vehicle": {
                "texts": ["small vehicle"],
                "text_embeds": np.zeros((2, 768), dtype=np.float32),
                "visual_embeds": np.zeros((3, 1024), dtype=np.float32),
                "confidence_scores": np.ones(3, dtype=np.float32),
            },
        }, f)
    extra_path = tmp_path / "extra_support.pkl"
    with extra_path.open("wb") as f:
        pickle.dump({
            "Airport": {
                "texts": ["airport"],
                "text_embeds": np.ones((2, 768), dtype=np.float32),
                "visual_embeds": np.ones((3, 1024), dtype=np.float32),
                "confidence_scores": np.ones(3, dtype=np.float32),
            },
        }, f)

    out_dir = tmp_path / "bundle"
    build_eval_bundle(
        support_path=support_path,
        out_dir=out_dir,
        val_classes=["airport", "small-vehicle"],
        dataset_flag="Data1_DOTA2",
    )
    bundled_support = (
        out_dir
        / "DOTA2_1024_500"
        / "ss_train"
        / "Step5_3_Prepare_Visual_Text_DINOv2_support.pkl"
    )
    assert bundled_support.is_symlink()

    build_eval_bundle(
        support_path=support_path,
        out_dir=out_dir,
        val_classes=["airport", "small-vehicle"],
        dataset_flag="Data1_DOTA2",
        extra_support_paths=[extra_path],
    )

    assert not bundled_support.is_symlink()
    with bundled_support.open("rb") as f:
        merged = pickle.load(f)
    assert set(merged) == {"airport", "small-vehicle"}
