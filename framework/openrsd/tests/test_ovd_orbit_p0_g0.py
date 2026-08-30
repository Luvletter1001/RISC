from copy import deepcopy
import json

import pytest

import M_Tools.analysis.ovd_orbit_p0_g0 as g0
from M_Tools.analysis.ovd_orbit_p0_g0 import (
    G0SealError,
    canonical_json_bytes,
    load_canonical_json,
    validate_authority,
)


_SHA256 = "a" * 64
_ANNOTATION_SHA256 = "b" * 64


def canonical_candidate_plan():
    """Return a minimal valid in-memory candidate-scene plan."""
    return {
        "schema": "ovd-orbit-p0-candidate-plan-v1",
        "records": [
            {
                "scene_id": "P0001",
                "split": "train",
                "image_path": "images/P0001.png",
                "image_sha256": _SHA256,
                "annotation_path": "annotations/P0001.json",
                "annotation_sha256": _ANNOTATION_SHA256,
            },
        ],
    }


def canonical_object_rows():
    """Return four canonical object rows covering every eligibility outcome."""
    shared = {
        "annotation_sha256": _ANNOTATION_SHA256,
        "box": [1.0, 2.0, 3.0, 4.0, 5.0],
        "class_name": "base-a",
        "overlap": 0.0,
        "scene_id": "P0001",
    }
    return [
        {**shared, "object_id": "00-eligible", "size": 16.0},
        {**shared, "object_id": "01-below", "size": 15.9},
        {**shared, "object_id": "02-above", "size": 256.1},
        {
            **shared,
            "object_id": "03-overlap",
            "overlap": 0.251,
            "size": 16.0,
        },
    ]


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


def test_validate_authority_requires_p0148_as_the_forbidden_scene():
    authority = canonical_authority()
    authority["forbidden_scene_id"] = "P0001"

    with pytest.raises(G0SealError, match="P0148"):
        validate_authority(authority)


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


def test_candidate_validation_indexes_scenes_without_aliasing_input():
    plan = canonical_candidate_plan()

    candidates = g0.validate_candidate_plan(plan)
    plan["records"][0]["split"] = "changed-after-validation"

    assert list(candidates) == ["P0001"]
    assert candidates["P0001"]["split"] == "train"


def test_candidate_validation_rejects_forbidden_scene_p0148():
    plan = canonical_candidate_plan()
    plan["records"][0]["scene_id"] = "P0148"

    with pytest.raises(G0SealError, match="P0148"):
        g0.validate_candidate_plan(plan)


def test_candidate_validation_rejects_cross_split_scene_leakage():
    plan = canonical_candidate_plan()
    leaked_record = deepcopy(plan["records"][0])
    leaked_record["split"] = "validation"
    plan["records"].append(leaked_record)

    with pytest.raises(G0SealError, match="leakage"):
        g0.validate_candidate_plan(plan)


def test_candidate_validation_rejects_duplicate_scene_identity():
    plan = canonical_candidate_plan()
    plan["records"].append(deepcopy(plan["records"][0]))

    with pytest.raises(G0SealError, match="unique"):
        g0.validate_candidate_plan(plan)


def test_load_object_rows_skips_blanks_and_requires_canonical_json_objects(tmp_path):
    row = canonical_object_rows()[0]
    path = tmp_path / "objects.jsonl"
    path.write_bytes(b"\n" + canonical_json_bytes(row) + b"\n")

    loaded = g0.load_object_rows(path)
    assert isinstance(loaded, tuple)
    assert dict(loaded[0]) == {**row, "box": tuple(row["box"])}

    path.write_text("\n  \n", encoding="utf-8")
    with pytest.raises(G0SealError, match="blank"):
        g0.load_object_rows(path)

    path.write_text('{"size":16,"object_id":"x"}\n', encoding="utf-8")
    with pytest.raises(G0SealError, match="line 1"):
        g0.load_object_rows(path)

    path.write_bytes(canonical_json_bytes({"object_id": "x", "size": 1}))
    with pytest.raises(G0SealError) as error:
        g0.load_object_rows(path)
    assert "line 1" in str(error.value)
    assert "scene_id" in str(error.value)

    path.write_bytes(canonical_json_bytes(row) + b'{"size":1e999}\n')
    with pytest.raises(G0SealError, match="line 2"):
        g0.load_object_rows(path)

    path.write_bytes(canonical_json_bytes(row) + b'{"text":"\\ud800"}\n')
    with pytest.raises(G0SealError, match="line 2"):
        g0.load_object_rows(path)


def test_load_object_rows_returns_sorted_recursively_frozen_rows(tmp_path):
    row_z = deepcopy(canonical_object_rows()[0])
    row_a = deepcopy(canonical_object_rows()[0])
    row_z["object_id"] = "z"
    row_a["object_id"] = "a"
    path = tmp_path / "objects.jsonl"
    path.write_bytes(canonical_json_bytes(row_z) + canonical_json_bytes(row_a))

    loaded = g0.load_object_rows(path)

    assert isinstance(loaded, tuple)
    assert [row["object_id"] for row in loaded] == ["a", "z"]
    with pytest.raises(TypeError):
        loaded[0]["object_id"] = "changed"
    with pytest.raises(TypeError):
        loaded[0]["box"][0] = 99.0

    validated = g0.validate_object_rows(
        loaded,
        g0.validate_candidate_plan(canonical_candidate_plan()),
        validate_authority(canonical_authority()),
    )
    assert [row["object_id"] for row in validated] == ["a", "z"]


def test_validated_rows_select_one_eligible_object_and_record_reason_precedence():
    authority = validate_authority(canonical_authority())
    candidates = g0.validate_candidate_plan(canonical_candidate_plan())
    rows = canonical_object_rows()

    validated = g0.validate_object_rows(tuple(rows), candidates, authority)
    decisions, eligible = g0.select_eligible_objects(
        (row for row in reversed(validated)), authority)

    assert isinstance(validated, tuple)
    assert isinstance(decisions, tuple)
    assert isinstance(eligible, tuple)
    assert dict(eligible[0]) == {**rows[0], "box": tuple(rows[0]["box"])}
    assert [(row["object_id"], row["eligible"], row["exclusion_reason"])
            for row in decisions] == [
                ("00-eligible", True, None),
                ("01-below", False, "below_min_size"),
                ("02-above", False, "above_max_size"),
                ("03-overlap", False, "overlap_exceeds_max"),
            ]
    for decision in decisions:
        assert set(decision) == {
            "annotation_sha256",
            "box",
            "class_name",
            "object_id",
            "overlap",
            "scene_id",
            "size",
            "eligible",
            "exclusion_reason",
        }
    with pytest.raises(TypeError):
        validated[0] = rows[0]
    with pytest.raises(TypeError):
        decisions[0] = decisions[0]
    with pytest.raises(TypeError):
        validated[0]["box"][0] = 99.0
    assert validated[0]["box"][0] == 1.0
    with pytest.raises(TypeError):
        decisions[0]["eligible"] = False
    assert decisions[0]["eligible"] is True
    with pytest.raises(TypeError):
        eligible[0]["object_id"] = "changed"
    assert eligible[0]["object_id"] == "00-eligible"


def test_object_validation_accepts_a_one_shot_generator_and_sorts_its_tuple():
    authority = validate_authority(canonical_authority())
    candidates = g0.validate_candidate_plan(canonical_candidate_plan())
    rows = canonical_object_rows()

    validated = g0.validate_object_rows(
        (row for row in reversed(rows)), candidates, authority)

    assert isinstance(validated, tuple)
    assert [row["object_id"] for row in validated] == [
        "00-eligible",
        "01-below",
        "02-above",
        "03-overlap",
    ]


def test_selection_rejects_a_raw_unvalidated_row_before_reading_its_box():
    raw_row = canonical_object_rows()[0]
    raw_row["box"] = None

    with pytest.raises(G0SealError, match="validated"):
        g0.select_eligible_objects(
            (raw_row,), validate_authority(canonical_authority()))


def test_selection_rejects_duplicate_validated_row_identity():
    authority = validate_authority(canonical_authority())
    validated = g0.validate_object_rows(
        canonical_object_rows(),
        g0.validate_candidate_plan(canonical_candidate_plan()),
        authority,
    )

    with pytest.raises(G0SealError, match="duplicate"):
        g0.select_eligible_objects((validated[0], validated[0]), authority)


def test_selection_rejects_a_tampered_validated_row_marker():
    authority = validate_authority(canonical_authority())
    validated = g0.validate_object_rows(
        canonical_object_rows(),
        g0.validate_candidate_plan(canonical_candidate_plan()),
        authority,
    )
    tampered = dict(validated[0])
    tampered["size"] = 17.0

    with pytest.raises(G0SealError, match="marker"):
        g0.select_eligible_objects((tampered,), authority)


def test_object_validation_rejects_unknown_class():
    rows = canonical_object_rows()
    rows[0]["class_name"] = "unknown-class"

    with pytest.raises(G0SealError, match="class_name"):
        g0.validate_object_rows(
            rows,
            g0.validate_candidate_plan(canonical_candidate_plan()),
            validate_authority(canonical_authority()),
        )


def test_object_validation_rejects_duplicate_scene_and_object_identity():
    rows = canonical_object_rows()
    rows.append(deepcopy(rows[0]))

    with pytest.raises(G0SealError, match="unique"):
        g0.validate_object_rows(
            rows,
            g0.validate_candidate_plan(canonical_candidate_plan()),
            validate_authority(canonical_authority()),
        )


def test_object_validation_rejects_annotation_hash_mismatch():
    rows = canonical_object_rows()
    rows[0]["annotation_sha256"] = "c" * 64

    with pytest.raises(G0SealError, match="annotation_sha256"):
        g0.validate_object_rows(
            rows,
            g0.validate_candidate_plan(canonical_candidate_plan()),
            validate_authority(canonical_authority()),
        )


def test_object_validation_rejects_malformed_box():
    rows = canonical_object_rows()
    rows[0]["box"] = [1.0, 2.0, 3.0, 4.0]

    with pytest.raises(G0SealError, match="box"):
        g0.validate_object_rows(
            rows,
            g0.validate_candidate_plan(canonical_candidate_plan()),
            validate_authority(canonical_authority()),
        )


def test_object_validation_rejects_nonfinite_size():
    rows = canonical_object_rows()
    rows[0]["size"] = float("inf")

    with pytest.raises(G0SealError, match="size"):
        g0.validate_object_rows(
            rows,
            g0.validate_candidate_plan(canonical_candidate_plan()),
            validate_authority(canonical_authority()),
        )
