from copy import deepcopy
import json

import pytest

from M_Tools.analysis.ovd_orbit_p0_g0 import (
    G0SealError,
    canonical_json_bytes,
    load_canonical_json,
    validate_authority,
)


_SHA256 = "a" * 64


def canonical_authority():
    """Return a minimal valid, in-memory G0 authority record."""
    return {
        "schema": "ovd-orbit-p0-g0-authority-v1",
        "forbidden_scene_id": "P0148",
        "candidate_scene_plan_sha256": _SHA256,
        "object_inventory_sha256": _SHA256,
        "checkpoint": {"path": "checkpoints/p0.pth", "sha256": _SHA256},
        "resolved_config": {"path": "configs/p0.py", "sha256": _SHA256},
        "code": {
            "commit": "0123456789abcdef",
            "files": [{"path": "M_Tools/analysis/export.py", "sha256": _SHA256}],
        },
        "vocabulary": {
            "base": [{"name": "base-a", "provenance": "base-source"}],
            "novel": [{"name": "novel-a", "provenance": "novel-source"}],
        },
        "prompt_families": [
            {"name": "prompt-a", "sha256": _SHA256},
            {"name": "prompt-b", "sha256": _SHA256},
            {"name": "prompt-c", "sha256": _SHA256},
        ],
        "primary_prompt_family": "prompt-a",
        "native_temperature": {"rule_id": "native-v1", "sha256": _SHA256},
        "render_contract": {"kind": "lossless-square-c4", "sha256": _SHA256},
        "eligibility_policy": {
            "schema": "canonical-object-inventory-v1",
            "min_size": 16.0,
            "max_size": 256.0,
            "max_overlap": 0.25,
            "sha256": _SHA256,
        },
        "bootstrap": {"seed": 17, "repetitions": 1000},
    }


def test_validate_authority_seals_plan_shaped_combined_vocabulary_and_primary():
    sealed = validate_authority(canonical_authority())

    assert sealed.base_classes == ("base-a",)
    assert sealed.novel_classes == ("novel-a",)
    assert sealed.vocabulary == ("base-a", "novel-a")
    assert sealed.primary_prompt_family == "prompt-a"
    assert sealed.raw["vocabulary"]["base"][0]["name"] == "base-a"


def test_validate_authority_rejects_overlapping_base_and_novel_classes():
    authority = canonical_authority()
    authority["vocabulary"]["novel"][0]["name"] = "base-a"

    with pytest.raises(G0SealError, match="disjoint"):
        validate_authority(authority)


def test_validate_authority_rejects_blank_class_provenance():
    authority = deepcopy(canonical_authority())
    authority["vocabulary"]["base"][0]["provenance"] = ""

    with pytest.raises(G0SealError, match="provenance"):
        validate_authority(authority)


def test_validate_authority_accepts_finite_integer_eligibility_thresholds():
    authority = canonical_authority()
    authority["eligibility_policy"].update(
        min_size=0,
        max_size=10 ** 400,
        max_overlap=0,
    )

    assert validate_authority(authority).raw["eligibility_policy"]["max_size"] == 10 ** 400


@pytest.mark.parametrize(
    "add_unknown_key",
    [
        lambda authority: authority.update(unexpected="value"),
        lambda authority: authority["checkpoint"].update(unexpected="value"),
        lambda authority: authority["resolved_config"].update(unexpected="value"),
        lambda authority: authority["code"].update(unexpected="value"),
        lambda authority: authority["code"]["files"][0].update(unexpected="value"),
        lambda authority: authority["vocabulary"].update(unexpected="value"),
        lambda authority: authority["vocabulary"]["base"][0].update(unexpected="value"),
        lambda authority: authority["prompt_families"][0].update(unexpected="value"),
        lambda authority: authority["native_temperature"].update(unexpected="value"),
        lambda authority: authority["render_contract"].update(unexpected="value"),
        lambda authority: authority["eligibility_policy"].update(unexpected="value"),
        lambda authority: authority["bootstrap"].update(unexpected="value"),
    ],
)
def test_validate_authority_rejects_unknown_keys_at_every_mapping_level(
        add_unknown_key):
    authority = canonical_authority()
    add_unknown_key(authority)

    with pytest.raises(G0SealError, match="unknown"):
        validate_authority(authority)


def test_load_canonical_json_returns_only_a_mapping(tmp_path):
    mapping_path = tmp_path / "authority.json"
    mapping_path.write_bytes(b'{"a":1}\n')

    assert load_canonical_json(mapping_path, label="authority") == {"a": 1}

    non_mapping_path = tmp_path / "list.json"
    non_mapping_path.write_bytes(b'[]\n')
    with pytest.raises(G0SealError, match="authority"):
        load_canonical_json(non_mapping_path, label="authority")


def test_load_canonical_json_includes_label_for_noncanonical_bytes(tmp_path):
    path = tmp_path / "candidate.json"
    path.write_bytes(b'{"z":1,"a":2}\n')

    with pytest.raises(G0SealError, match="candidate scene plan"):
        load_canonical_json(path, label="candidate scene plan")


def test_canonical_json_bytes_materializes_frozen_authority_raw_deterministically():
    authority = canonical_authority()
    sealed = validate_authority(authority)

    first = canonical_json_bytes(sealed.raw)
    second = canonical_json_bytes(sealed.raw)

    assert first == second
    assert json.loads(first) == authority


def test_load_canonical_json_labels_missing_files(tmp_path):
    missing_path = tmp_path / "missing-authority.json"

    with pytest.raises(G0SealError) as error:
        load_canonical_json(missing_path, label="authority file")

    assert "authority file" in str(error.value)
    assert "missing" in str(error.value)
