from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

import M_Tools.analysis.ovd_orbit_p0_g0 as g0
import M_Tools.analysis.prepare_ovd_orbit_p0_g0_seal as cli
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


def scope_authority():
    """Return an eight-class authority suitable for synthetic scope tests."""
    authority = canonical_authority()
    authority["object_inventory_sha256"] = _ANNOTATION_SHA256
    authority["vocabulary"] = {
        "base": [
            {"name": f"base-{index}", "provenance": "synthetic-base"}
            for index in range(3)
        ],
        "novel": [
            {"name": f"novel-{index}", "provenance": "synthetic-novel"}
            for index in range(5)
        ],
    }
    authority["code"]["files"] = [
        {"path": "M_Tools/analysis/z_synthetic.py", "sha256": "e" * 64},
        {"path": "M_Tools/analysis/a_synthetic.py", "sha256": "f" * 64},
    ]
    return authority


def scope_candidate_plan(scene_count=80):
    """Return synthetic, non-asset candidate records for the requested scope."""
    records = []
    for index in range(1, scene_count + 1):
        scene_id = f"scene-{index:03d}"
        records.append({
            "scene_id": scene_id,
            "split": "train" if index % 2 else "validation",
            "image_path": f"synthetic/{scene_id}.png",
            "image_sha256": _SHA256,
            "annotation_path": f"synthetic/{scene_id}.json",
            "annotation_sha256": _ANNOTATION_SHA256,
        })
    return {"schema": "ovd-orbit-p0-candidate-plan-v1", "records": records}


def scope_object_rows(scene_count=80, objects_per_scene=10):
    """Return valid synthetic object rows evenly spanning all eight classes."""
    classes = [*(f"base-{index}" for index in range(3)),
               *(f"novel-{index}" for index in range(5))]
    rows = []
    for scene_index in range(1, scene_count + 1):
        scene_id = f"scene-{scene_index:03d}"
        for object_index in range(objects_per_scene):
            rows.append({
                "annotation_sha256": _ANNOTATION_SHA256,
                "box": [float(object_index), 2.0, 3.0, 4.0, 5.0],
                "class_name": classes[((scene_index - 1) * objects_per_scene
                                        + object_index) % len(classes)],
                "object_id": f"{scene_id}:{object_index}",
                "overlap": 0.0,
                "scene_id": scene_id,
                "size": 16.0,
            })
    return rows


def scope_asset_hashes():
    """Return the complete observed-asset mapping required by Task 4."""
    return {
        "checkpoint": _SHA256,
        "resolved_config": _SHA256,
        "code:M_Tools/analysis/a_synthetic.py": "f" * 64,
        "code:M_Tools/analysis/z_synthetic.py": "e" * 64,
    }


def scope_candidate_asset_hashes(scene_count=80):
    """Return the declared source-byte hashes for every synthetic candidate."""
    return {
        f"scene-{index:03d}": {
            "image_sha256": _SHA256,
            "annotation_sha256": _ANNOTATION_SHA256,
        }
        for index in range(1, scene_count + 1)
    }


def test_view_ids_are_the_five_canonical_c4_entries():
    assert g0.VIEW_IDS == ("rot000_a", "rot000_b", "rot090", "rot180", "rot270")


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


def test_build_g0_artifacts_seals_deterministic_five_view_package():
    authority = validate_authority(scope_authority())
    candidate_plan = scope_candidate_plan()
    object_rows = scope_object_rows()
    input_plan = deepcopy(candidate_plan)
    input_rows = deepcopy(object_rows)

    package = g0.build_g0_artifacts(
        authority=authority,
        candidate_plan=candidate_plan,
        object_rows=object_rows,
        candidate_plan_sha256=_SHA256,
        object_inventory_sha256=_ANNOTATION_SHA256,
        asset_hashes=scope_asset_hashes(),
        candidate_asset_hashes=scope_candidate_asset_hashes(),
    )

    assert g0.VIEW_IDS == ("rot000_a", "rot000_b", "rot090", "rot180", "rot270")
    assert set(package) == {
        "input_manifest.json",
        "object_eligibility.jsonl",
        "object_view_plan.jsonl",
        "seal_diagnostics.json",
        "receipt.json",
        "result.md",
    }
    assert all(isinstance(value, bytes) for value in package.values())
    assert candidate_plan == input_plan
    assert object_rows == input_rows

    views = [json.loads(line) for line in package["object_view_plan.jsonl"].splitlines()]
    first = [row for row in views if row["object_id"] == "scene-001:0"]
    assert [row["view_id"] for row in first] == list(g0.VIEW_IDS)
    assert first[0]["render_digest"] == first[1]["render_digest"]
    assert first[0]["view_id"] != first[1]["view_id"]

    manifest = json.loads(package["input_manifest.json"])
    assert manifest["schema"] == "ovd-orbit-p0-g0-input-manifest-v1"
    assert manifest["status"] == "G0_INPUTS_SEALED_NO_FORWARD"
    assert manifest["candidate_plan_sha256"] == _SHA256
    assert manifest["object_inventory_sha256"] == _ANNOTATION_SHA256
    assert manifest["asset_hashes"] == scope_asset_hashes()
    assert manifest["declared_asset_hashes"] == scope_asset_hashes()
    assert manifest["declared_assets"] == {
        "checkpoint": {"path": "checkpoints/p0.pth", "sha256": _SHA256},
        "resolved_config": {"path": "configs/p0.py", "sha256": _SHA256},
    }
    assert manifest["code_identity"] == {
        "commit": "0123456789abcdef",
        "files": [
            {"path": "M_Tools/analysis/a_synthetic.py", "sha256": "f" * 64},
            {"path": "M_Tools/analysis/z_synthetic.py", "sha256": "e" * 64},
        ],
    }
    assert manifest["observed_asset_hashes"] == scope_asset_hashes()
    assert manifest["declared_candidate_assets"] == scope_candidate_asset_hashes()
    assert manifest["observed_candidate_assets"] == scope_candidate_asset_hashes()
    assert manifest["candidate_scene_counts_by_split"] == {
        "train": 40,
        "validation": 40,
    }
    assert list(manifest["candidate_scene_counts_by_split"]) == [
        "train", "validation"]
    assert manifest["leakage_count"] == 0
    assert manifest["leakage_status"] == "pass"
    assert manifest["view_plan_sha256"] == hashlib.sha256(
        package["object_view_plan.jsonl"]).hexdigest()

    diagnostics = json.loads(package["seal_diagnostics.json"])
    assert diagnostics["g0_scope_ready"] is True
    assert diagnostics["strict_ovd_ready"] is True
    assert diagnostics["eligible_scene_count"] == 80
    assert diagnostics["eligible_object_count"] == 800
    assert diagnostics["supported_class_count"] == 8
    assert diagnostics["candidate_scene_count"] == 80
    assert diagnostics["candidate_scene_counts_by_split"] == {
        "train": 40,
        "validation": 40,
    }
    assert list(diagnostics["candidate_scene_counts_by_split"]) == [
        "train", "validation"]
    assert diagnostics["leakage_count"] == 0
    assert diagnostics["leakage_status"] == "pass"
    for field in (
            "candidate_scene_counts_by_split", "leakage_count", "leakage_status"):
        assert manifest[field] == diagnostics[field]

    receipt = json.loads(package["receipt.json"])
    assert receipt["status"] == "G0_INPUTS_SEALED_NO_FORWARD"
    assert set(receipt["artifact_sha256"]) == set(package) - {"receipt.json"}
    for name, digest in receipt["artifact_sha256"].items():
        assert digest == hashlib.sha256(package[name]).hexdigest()

    assert b"_validated_object_sha256" not in package["object_eligibility.jsonl"]
    assert b"_validated_object_sha256" not in package["object_view_plan.jsonl"]
    result_text = package["result.md"].decode("utf-8").lower()
    assert "未执行前向计算" in result_text
    assert not any(term in result_text for term in ("model", "logits", "detection", "ap"))

    reversed_package = g0.build_g0_artifacts(
        authority=authority,
        candidate_plan={
            "schema": candidate_plan["schema"],
            "records": list(reversed(candidate_plan["records"])),
        },
        object_rows=reversed(object_rows),
        candidate_plan_sha256=_SHA256,
        object_inventory_sha256=_ANNOTATION_SHA256,
        asset_hashes={
            "code:M_Tools/analysis/z_synthetic.py": "e" * 64,
            "resolved_config": _SHA256,
            "checkpoint": _SHA256,
            "code:M_Tools/analysis/a_synthetic.py": "f" * 64,
        },
        candidate_asset_hashes=dict(reversed(
            list(scope_candidate_asset_hashes().items()))),
    )
    assert reversed_package == package


@pytest.mark.parametrize(
    ("field", "bad_digest"),
    [
        ("candidate_plan_sha256", "1" * 64),
        ("object_inventory_sha256", "2" * 64),
    ],
)
def test_build_g0_artifacts_rejects_supplied_digest_mismatches_before_sealing(
        field, bad_digest):
    arguments = {
        "authority": validate_authority(scope_authority()),
        "candidate_plan": scope_candidate_plan(),
        "object_rows": scope_object_rows(),
        "candidate_plan_sha256": _SHA256,
        "object_inventory_sha256": _ANNOTATION_SHA256,
        "asset_hashes": scope_asset_hashes(),
        "candidate_asset_hashes": scope_candidate_asset_hashes(),
    }
    arguments[field] = bad_digest

    with pytest.raises(G0SealError, match=field):
        g0.build_g0_artifacts(**arguments)


def test_build_g0_artifacts_rejects_authority_observed_hash_mismatch():
    arguments = {
        "authority": validate_authority(scope_authority()),
        "candidate_plan": scope_candidate_plan(),
        "object_rows": scope_object_rows(),
        "candidate_plan_sha256": _SHA256,
        "object_inventory_sha256": _ANNOTATION_SHA256,
        "asset_hashes": {
            **scope_asset_hashes(),
            "checkpoint": "c" * 64,
        },
        "candidate_asset_hashes": scope_candidate_asset_hashes(),
    }

    with pytest.raises(G0SealError, match="asset_hashes.checkpoint"):
        g0.build_g0_artifacts(**arguments)


@pytest.mark.parametrize(
    ("scene_count", "objects_per_scene", "trim_objects"),
    [(79, 10, 0), (80, 10, 1)],
)
def test_build_g0_artifacts_returns_fail_stop_package_below_scope(
        scene_count, objects_per_scene, trim_objects):
    rows = scope_object_rows(scene_count, objects_per_scene)
    if trim_objects:
        rows = rows[:-trim_objects]

    package = g0.build_g0_artifacts(
        authority=validate_authority(scope_authority()),
        candidate_plan=scope_candidate_plan(scene_count),
        object_rows=rows,
        candidate_plan_sha256=_SHA256,
        object_inventory_sha256=_ANNOTATION_SHA256,
        asset_hashes=scope_asset_hashes(),
        candidate_asset_hashes=scope_candidate_asset_hashes(scene_count),
    )

    diagnostics = json.loads(package["seal_diagnostics.json"])
    receipt = json.loads(package["receipt.json"])
    result_text = package["result.md"].decode("utf-8").lower()
    assert diagnostics["g0_scope_ready"] is False
    assert receipt["status"] == "P0_INPUT_FAIL_STOP"
    assert "strict ovd" not in result_text


def _write_fake_asset(path, contents):
    path.write_bytes(contents)
    return str(path), hashlib.sha256(contents).hexdigest()


def write_cli_scope_inputs(
        tmp_path, *, checkpoint_contents=b"checkpoint", scene_count=80,
        objects_per_scene=10, trim_objects=0):
    """Write canonical scope inputs and tiny opaque byte assets only."""
    authority = scope_authority()
    checkpoint_path, checkpoint_hash = _write_fake_asset(
        tmp_path / "checkpoint.bin", checkpoint_contents)
    config_path, config_hash = _write_fake_asset(
        tmp_path / "resolved-config.py", b"config bytes")
    code_a_path, code_a_hash = _write_fake_asset(
        tmp_path / "code-a.py", b"code a")
    code_z_path, code_z_hash = _write_fake_asset(
        tmp_path / "code-z.py", b"code z")
    authority["checkpoint"] = {"path": checkpoint_path, "sha256": checkpoint_hash}
    authority["resolved_config"] = {"path": config_path, "sha256": config_hash}
    authority["code"]["files"] = [
        {"path": code_z_path, "sha256": code_z_hash},
        {"path": code_a_path, "sha256": code_a_hash},
    ]

    candidate_plan = scope_candidate_plan(scene_count)
    for record in candidate_plan["records"]:
        scene_id = record["scene_id"]
        image_path, image_hash = _write_fake_asset(
            tmp_path / f"{scene_id}.image.bin",
            f"image:{scene_id}".encode("utf-8"))
        annotation_path, annotation_hash = _write_fake_asset(
            tmp_path / f"{scene_id}.annotation.bin",
            f"annotation:{scene_id}".encode("utf-8"))
        record["image_path"] = image_path
        record["image_sha256"] = image_hash
        record["annotation_path"] = annotation_path
        record["annotation_sha256"] = annotation_hash
    object_rows = scope_object_rows(scene_count, objects_per_scene)
    if trim_objects:
        object_rows = object_rows[:-trim_objects]
    for row in object_rows:
        row["annotation_sha256"] = candidate_plan["records"][
            int(row["scene_id"].split("-")[1]) - 1]["annotation_sha256"]
    candidate_bytes = canonical_json_bytes(candidate_plan)
    object_bytes = b"".join(canonical_json_bytes(row) for row in object_rows)
    authority["candidate_scene_plan_sha256"] = hashlib.sha256(
        candidate_bytes).hexdigest()
    authority["object_inventory_sha256"] = hashlib.sha256(object_bytes).hexdigest()

    authority_path = tmp_path / "authority.json"
    candidate_path = tmp_path / "candidate.json"
    object_path = tmp_path / "objects.jsonl"
    authority_path.write_bytes(canonical_json_bytes(authority))
    candidate_path.write_bytes(candidate_bytes)
    object_path.write_bytes(object_bytes)
    return authority_path, candidate_path, object_path, authority


def test_verify_authority_assets_streams_declared_opaque_files(tmp_path):
    authority_path, _, _, authority_raw = write_cli_scope_inputs(tmp_path)
    authority = validate_authority(
        load_canonical_json(authority_path, label="authority"))

    observed = g0.verify_authority_assets(authority)

    expected = {
        "checkpoint": authority_raw["checkpoint"]["sha256"],
        "resolved_config": authority_raw["resolved_config"]["sha256"],
        "code:" + authority_raw["code"]["files"][1]["path"]:
            authority_raw["code"]["files"][1]["sha256"],
        "code:" + authority_raw["code"]["files"][0]["path"]:
            authority_raw["code"]["files"][0]["sha256"],
    }
    assert observed == expected
    assert list(observed) == [
        "checkpoint",
        "resolved_config",
        "code:" + authority_raw["code"]["files"][1]["path"],
        "code:" + authority_raw["code"]["files"][0]["path"],
    ]


def test_verify_candidate_assets_streams_declared_image_and_annotation_files(
        tmp_path):
    _, candidate_path, _, _ = write_cli_scope_inputs(tmp_path)
    candidate_plan = load_canonical_json(
        candidate_path, label="candidate scene plan")
    candidates = g0.validate_candidate_plan(candidate_plan)

    observed = g0.verify_candidate_assets(candidates)

    expected = {
        scene_id: {
            "image_sha256": candidates[scene_id]["image_sha256"],
            "annotation_sha256": candidates[scene_id]["annotation_sha256"],
        }
        for scene_id in sorted(candidates)
    }
    assert observed == expected
    assert list(observed) == sorted(candidates)
    assert set(observed["scene-001"]) == {
        "image_sha256", "annotation_sha256"}
    with pytest.raises(TypeError):
        observed["scene-001"] = {}
    with pytest.raises(TypeError):
        observed["scene-001"]["image_sha256"] = "changed"


def test_cli_publishes_success_once_and_refuses_overwrite(tmp_path):
    authority_path, candidate_path, object_path, _ = write_cli_scope_inputs(tmp_path)
    output_dir = tmp_path / "g0-seal"
    args = [
        "--authority-json", str(authority_path),
        "--candidate-scene-plan", str(candidate_path),
        "--object-inventory", str(object_path),
        "--output-dir", str(output_dir),
    ]

    assert cli.main(args) == 0
    assert {path.name for path in output_dir.iterdir()} == {
        "input_manifest.json",
        "object_eligibility.jsonl",
        "object_view_plan.jsonl",
        "seal_diagnostics.json",
        "receipt.json",
        "result.md",
    }
    receipt_bytes = (output_dir / "receipt.json").read_bytes()
    assert json.loads(receipt_bytes)["status"] == "G0_INPUTS_SEALED_NO_FORWARD"

    with pytest.raises(FileExistsError):
        cli.main(args)
    assert (output_dir / "receipt.json").read_bytes() == receipt_bytes


def test_cli_below_scope_publishes_full_package_and_returns_stop(tmp_path):
    authority_path, candidate_path, object_path, _ = write_cli_scope_inputs(
        tmp_path, scene_count=79)
    output_dir = tmp_path / "g0-seal-below-scope"

    assert cli.main([
        "--authority-json", str(authority_path),
        "--candidate-scene-plan", str(candidate_path),
        "--object-inventory", str(object_path),
        "--output-dir", str(output_dir),
    ]) == 2
    assert {path.name for path in output_dir.iterdir()} == {
        "input_manifest.json",
        "object_eligibility.jsonl",
        "object_view_plan.jsonl",
        "seal_diagnostics.json",
        "receipt.json",
        "result.md",
    }
    receipt = json.loads((output_dir / "receipt.json").read_bytes())
    diagnostics = json.loads((output_dir / "seal_diagnostics.json").read_bytes())
    assert receipt["status"] == "P0_INPUT_FAIL_STOP"
    assert diagnostics["g0_scope_ready"] is False
    assert (output_dir / "object_view_plan.jsonl").read_bytes()


def test_cli_hash_mismatch_publishes_only_fail_stop_artifacts(tmp_path):
    authority_path, candidate_path, object_path, authority = write_cli_scope_inputs(
        tmp_path, checkpoint_contents=b"observed checkpoint")
    authority["checkpoint"]["sha256"] = hashlib.sha256(
        b"declared checkpoint").hexdigest()
    authority_path.write_bytes(canonical_json_bytes(authority))
    output_dir = tmp_path / "g0-seal-failure"

    assert cli.main([
        "--authority-json", str(authority_path),
        "--candidate-scene-plan", str(candidate_path),
        "--object-inventory", str(object_path),
        "--output-dir", str(output_dir),
    ]) == 2

    assert {path.name for path in output_dir.iterdir()} == {
        "receipt.json", "seal_diagnostics.json", "result.md",
    }
    receipt = json.loads((output_dir / "receipt.json").read_bytes())
    assert receipt["schema"] == "ovd-orbit-p0-g0-receipt-v1"
    assert receipt["status"] == "P0_INPUT_FAIL_STOP"
    assert "artifact_sha256" not in receipt
    assert "checkpoint" in receipt["error"]
    result = (output_dir / "result.md").read_text(encoding="utf-8")
    assert "输入封存失败" in result
    assert "模型" in result and "GPU" in result and "指标" in result
    assert not (output_dir / "object_view_plan.jsonl").exists()


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("image_path", None),
        ("annotation_path", b"modified annotation"),
    ],
)
def test_cli_candidate_asset_failures_publish_only_fail_stop_artifacts(
        tmp_path, field, replacement):
    authority_path, candidate_path, object_path, _ = write_cli_scope_inputs(tmp_path)
    candidate_plan = json.loads(candidate_path.read_bytes())
    asset_path = Path(candidate_plan["records"][0][field])
    if replacement is None:
        asset_path.unlink()
    else:
        asset_path.write_bytes(replacement)
    output_dir = tmp_path / f"g0-seal-{field}-failure"

    assert cli.main([
        "--authority-json", str(authority_path),
        "--candidate-scene-plan", str(candidate_path),
        "--object-inventory", str(object_path),
        "--output-dir", str(output_dir),
    ]) == 2

    assert {path.name for path in output_dir.iterdir()} == {
        "receipt.json", "seal_diagnostics.json", "result.md",
    }
    receipt = json.loads((output_dir / "receipt.json").read_bytes())
    assert receipt["status"] == "P0_INPUT_FAIL_STOP"
    assert "scene-001" in receipt["error"]
    assert "hash" in receipt["error"]
    assert not (output_dir / "object_view_plan.jsonl").exists()


def test_cli_rejects_a_file_hash_that_does_not_match_loaded_input_snapshot(
        tmp_path, monkeypatch):
    authority_path, candidate_path, object_path, authority = write_cli_scope_inputs(
        tmp_path)
    replacement_plan = scope_candidate_plan()
    replacement_plan["records"][0]["split"] = "replacement"
    replacement_hash = hashlib.sha256(
        canonical_json_bytes(replacement_plan)).hexdigest()
    authority["candidate_scene_plan_sha256"] = replacement_hash
    authority_path.write_bytes(canonical_json_bytes(authority))
    original_sha256_file = g0.sha256_file

    def replacement_hash_after_load(path):
        if Path(path) == candidate_path:
            return replacement_hash
        return original_sha256_file(path)

    monkeypatch.setattr(g0, "sha256_file", replacement_hash_after_load)
    output_dir = tmp_path / "g0-seal-snapshot-failure"

    assert cli.main([
        "--authority-json", str(authority_path),
        "--candidate-scene-plan", str(candidate_path),
        "--object-inventory", str(object_path),
        "--output-dir", str(output_dir),
    ]) == 2
    assert {path.name for path in output_dir.iterdir()} == {
        "receipt.json", "seal_diagnostics.json", "result.md",
    }
    assert "candidate" in json.loads(
        (output_dir / "receipt.json").read_bytes())["error"]


@pytest.mark.parametrize(
    ("disappearing_input", "error_fragment"),
    [
        ("candidate", "candidate scene plan input rehash"),
        ("object", "object inventory input rehash"),
    ],
)
def test_cli_post_load_input_rehash_oserror_publishes_fail_stop_artifacts(
        tmp_path, monkeypatch, disappearing_input, error_fragment):
    authority_path, candidate_path, object_path, _ = write_cli_scope_inputs(tmp_path)
    disappearing_path = {
        "candidate": candidate_path,
        "object": object_path,
    }[disappearing_input]
    original_sha256_file = g0.sha256_file

    def disappear_before_rehash(path):
        if Path(path) == disappearing_path:
            disappearing_path.unlink()
        return original_sha256_file(path)

    monkeypatch.setattr(g0, "sha256_file", disappear_before_rehash)
    output_dir = tmp_path / f"g0-seal-{disappearing_input}-rehash-failure"

    assert cli.main([
        "--authority-json", str(authority_path),
        "--candidate-scene-plan", str(candidate_path),
        "--object-inventory", str(object_path),
        "--output-dir", str(output_dir),
    ]) == 2
    assert {path.name for path in output_dir.iterdir()} == {
        "receipt.json", "seal_diagnostics.json", "result.md",
    }
    receipt = json.loads((output_dir / "receipt.json").read_bytes())
    assert receipt["status"] == "P0_INPUT_FAIL_STOP"
    assert error_fragment in receipt["error"]
    assert not (output_dir / "object_view_plan.jsonl").exists()


def test_cli_source_has_no_framework_or_accelerator_operations():
    source = Path(cli.__file__).read_text(encoding="utf-8")

    assert "torch" not in source
    assert "M_AD" not in source
    assert "cuda" not in source
    assert "model" not in source


def test_g0_docs_record_input_seal_without_forward_or_p0_metrics():
    """All user-facing P0 docs must preserve the G0 input-only boundary."""
    repository_root = Path(__file__).resolve().parents[3]
    doc_paths = (
        repository_root / "docs/research/ovd_orbit_p0/README.md",
        repository_root / "docs/research/ovd_orbit_p0/p0_protocol.md",
        repository_root / "docs/research/ovd_orbit_p0/progress.md",
    )
    required_terms = (
        "G0_INPUTS_SEALED_NO_FORWARD",
        "P0_INPUT_FAIL_STOP",
        "rot000_a",
        "rot000_b",
    )

    for doc_path in doc_paths:
        assert doc_path.is_file()
        text = doc_path.read_text(encoding="utf-8")
        normalized = " ".join(text.lower().split())

        for term in required_terms:
            assert term in text
        assert "no actual model" in normalized
        assert "no gpu" in normalized
        assert "no p0 metric" in normalized
        assert "no receipt based on real project assets" in normalized
        assert "no live-p0 receipt was published in this implementation task" in normalized
        assert "receipt publication" not in normalized
        assert "minimal failure package" in normalized
        assert "valid-but-below-g0-scope" in normalized
        assert "full diagnostic package" in normalized
        assert "cli exits 2 in both forms" in normalized
        assert "neither form authorizes a forward" in normalized
        assert "prevents the g0 seal from proceeding" not in normalized
        assert "before it can construct the g0 seal" not in normalized

    design_path = repository_root / (
        "docs/superpowers/specs/2026-08-31-ovd-orbit-p0-g0-input-seal-design.md")
    assert design_path.is_file()
    design = " ".join(design_path.read_text(encoding="utf-8").lower().split())
    assert "**status:** implemented and cpu-tested with synthetic fixtures;" in design
    assert "no real project assets were executed" in design
    assert "no g0 success status is claimed" in design
    assert "implementation has not started" not in design
    assert "--object-inventory" in design
    assert "v1 does not parse raw annotations" in design
    assert "candidate annotation files are only stream-hash verified and bound" in design
    assert "native annotation adapter is intentionally out of v1 scope" in design
    assert "human-readable summary" in design
    assert "narrative summary, not a table" in design
    assert "human-readable table" not in design
