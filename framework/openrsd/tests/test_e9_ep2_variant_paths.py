from M_Tools.analysis.run_e9_scale_counterfactual_probe import (
    build_override_model_rows,
    ep2_variant_filename,
)


def test_ep2_variant_filename_encodes_probe_metadata():
    name = ep2_variant_filename(
        model_name="p4_s3c",
        case_index=7,
        tile_img_id="P0001__1024__0___0",
        variant_name="neutral_same_scale",
    )

    assert name.startswith("ep2case-p4_s3c-0007-P0001-1024-0-0")
    assert "__ep2ctx-P0001-1024-0-0__" in name
    assert "__ep2obj-p4_s3c-0007__" in name
    assert "__ep2ctrl-neutral_same_scale__" in name
    assert "__ep2step-01" in name
    assert name.endswith(".jpg")


def test_build_override_model_rows_separates_actual_and_case_source_model():
    rows = build_override_model_rows(
        override_model_name="p4_ep2",
        override_config="M_configs/Diagnostics/p4_ep2.py",
        override_checkpoint="work_dirs/p4.pth",
        case_model_name="faahead_lsknet",
    )

    assert rows == [{
        "model": "p4_ep2",
        "config": "M_configs/Diagnostics/p4_ep2.py",
        "checkpoint": "work_dirs/p4.pth",
        "case_model": "faahead_lsknet",
    }]
