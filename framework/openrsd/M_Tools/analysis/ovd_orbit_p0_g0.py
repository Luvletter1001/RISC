"""Pure-Python authority validation for the OVD object-orbit P0 G0 seal."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import ctypes
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from types import MappingProxyType
from typing import Any, Mapping


_AUTHORITY_SCHEMA = "ovd-orbit-p0-g0-authority-v1"
_ELIGIBILITY_SCHEMA = "canonical-object-inventory-v1"
_CANDIDATE_PLAN_SCHEMA = "ovd-orbit-p0-candidate-plan-v1"
_RENDER_CONTRACT_KIND = "lossless-square-c4"
_FORBIDDEN_SCENE_ID = "P0148"
_HEX_DIGITS = frozenset("0123456789abcdef")
VIEW_IDS = ("rot000_a", "rot000_b", "rot090", "rot180", "rot270")
_CANDIDATE_PLAN_FIELDS = frozenset({"schema", "records"})
_CANDIDATE_RECORD_FIELDS = (
    "scene_id",
    "split",
    "image_path",
    "image_sha256",
    "annotation_path",
    "annotation_sha256",
)
_OBJECT_ROW_FIELDS = (
    "annotation_sha256",
    "box",
    "class_name",
    "object_id",
    "overlap",
    "scene_id",
    "size",
)
_VALIDATION_MARKER_FIELD = "_validated_object_sha256"
_VALIDATED_OBJECT_ROW_FIELDS = frozenset(
    (*_OBJECT_ROW_FIELDS, _VALIDATION_MARKER_FIELD))
_CANDIDATE_ASSET_HASH_FIELDS = ("image_sha256", "annotation_sha256")
_PROTOCOL_THRESHOLD_VALUES = {
    "g0": {
        "min_scenes": 80,
        "min_objects": 800,
        "min_supported_classes": 8,
        "min_novel_objects": 300,
        "min_novel_classes": 5,
    },
    "g1": {
        "identity_p99": 0.0001,
        "noise_multiplier": 1.25,
        "aggregation_relative_difference": 0.25,
    },
    "g2": {
        "er_acc": 0.03,
        "smd_margin": 0.20,
        "er_js": 0.01,
        "min_passing_conditions": 2,
        "min_prompt_intervals": 2,
    },
    "g3": {
        "min_model_families": 2,
        "did_closed_smd": 0.20,
        "did_shift_smd": 0.20,
        "replication_ratio_min": 0.33,
        "replication_ratio_max": 3.0,
    },
    "g4": {
        "min_optimizer_steps": 100,
        "max_optimizer_steps": 250,
    },
}
_INTEGER_PROTOCOL_THRESHOLD_FIELDS = frozenset({
    ("g0", "min_scenes"),
    ("g0", "min_objects"),
    ("g0", "min_supported_classes"),
    ("g0", "min_novel_objects"),
    ("g0", "min_novel_classes"),
    ("g2", "min_passing_conditions"),
    ("g2", "min_prompt_intervals"),
    ("g3", "min_model_families"),
    ("g4", "min_optimizer_steps"),
    ("g4", "max_optimizer_steps"),
})
_FLOAT_PROTOCOL_THRESHOLD_FIELDS = frozenset({
    ("g1", "identity_p99"),
    ("g1", "noise_multiplier"),
    ("g1", "aggregation_relative_difference"),
    ("g2", "er_acc"),
    ("g2", "smd_margin"),
    ("g2", "er_js"),
    ("g3", "did_closed_smd"),
    ("g3", "did_shift_smd"),
    ("g3", "replication_ratio_min"),
    ("g3", "replication_ratio_max"),
})


class G0SealError(ValueError):
    """Raised when a G0 authority record violates the sealed contract."""


@dataclass(frozen=True)
class SealedAuthority:
    """Validated immutable view of a canonical G0 authority record."""

    forbidden_scene_id: str
    vocabulary: tuple[str, ...]
    base_classes: tuple[str, ...]
    novel_classes: tuple[str, ...]
    primary_prompt_family: str
    raw: Mapping[str, Any]


class _CanonicalJsonMapping(dict[str, Any]):
    """Mapping loaded from one canonical JSON byte snapshot."""

    def __init__(self, value: Mapping[str, Any], source_sha256: str) -> None:
        super().__init__(value)
        self.source_sha256 = source_sha256


class _LoadedObjectRows(tuple):
    """Immutable object rows paired with their source-byte digest."""

    def __new__(
            cls, rows: Iterable[Mapping[str, Any]], source_sha256: str) -> _LoadedObjectRows:
        result = super().__new__(cls, rows)
        result.source_sha256 = source_sha256
        return result


def canonical_json_bytes(value: Any) -> bytes:
    """Encode a JSON value with the one canonical G0 representation."""
    try:
        text = json.dumps(
            _json_native(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return text.encode("utf-8") + b"\n"
    except (TypeError, UnicodeEncodeError, ValueError) as error:
        raise G0SealError("value must be canonical JSON") from error


def _json_native(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    return value


def sha256_bytes(value: bytes) -> str:
    """Return the lowercase SHA-256 digest for one bytes value."""
    if not isinstance(value, bytes):
        raise G0SealError("sha256_bytes requires bytes")
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path | str) -> str:
    """Stream a file into SHA-256 without deserializing its contents."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _authority_asset_items(
        authority: SealedAuthority) -> tuple[tuple[str, Mapping[str, Any]], ...]:
    """Return the declared opaque assets in their stable receipt-key order."""
    authority = _require_sealed_authority(authority)
    raw = authority.raw
    code_assets = sorted(
        ((f"code:{entry['path']}", entry) for entry in raw["code"]["files"]),
        key=lambda item: item[0],
    )
    return (
        ("checkpoint", raw["checkpoint"]),
        ("resolved_config", raw["resolved_config"]),
        *code_assets,
    )


def verify_authority_assets(authority: SealedAuthority) -> dict[str, str]:
    """Stream and compare every authority-declared opaque asset byte-for-byte."""
    authority = _revalidate_sealed_authority(authority)
    observed: dict[str, str] = {}
    for asset_name, asset in _authority_asset_items(authority):
        try:
            observed_hash = sha256_file(asset["path"])
        except (OSError, ValueError) as error:
            raise G0SealError(f"{asset_name} hash mismatch") from error
        if observed_hash != asset["sha256"]:
            raise G0SealError(f"{asset_name} hash mismatch")
        observed[asset_name] = observed_hash
    return observed


def _declared_candidate_asset_hashes(
        candidates: Mapping[str, Mapping[str, str]]) -> dict[str, dict[str, str]]:
    candidates = _require_mapping(candidates, "candidates")
    declared: dict[str, dict[str, str]] = {}
    for scene_id in sorted(candidates):
        scene_id = _require_nonempty_string(scene_id, "candidate scene_id")
        candidate = _require_mapping(candidates[scene_id], f"candidates[{scene_id}]")
        declared[scene_id] = {
            field: _require_sha256(
                _require_field(candidate, field, f"candidates[{scene_id}]"),
                f"candidates[{scene_id}].{field}")
            for field in _CANDIDATE_ASSET_HASH_FIELDS
        }
    return declared


def verify_candidate_assets(
        candidates: Mapping[str, Mapping[str, str]]) -> Mapping[str, Mapping[str, str]]:
    """Stream and compare every candidate's declared image and annotation bytes."""
    candidates = _require_mapping(candidates, "candidates")
    declared = _declared_candidate_asset_hashes(candidates)
    observed: dict[str, Mapping[str, str]] = {}
    for scene_id, expected in declared.items():
        candidate = _require_mapping(candidates[scene_id], f"candidates[{scene_id}]")
        scene_observed: dict[str, str] = {}
        for kind, path_field, hash_field in (
                ("image", "image_path", "image_sha256"),
                ("annotation", "annotation_path", "annotation_sha256")):
            try:
                observed_hash = sha256_file(_require_nonempty_string(
                    _require_field(candidate, path_field, f"candidates[{scene_id}]"),
                    f"candidates[{scene_id}].{path_field}"))
            except (OSError, ValueError) as error:
                raise G0SealError(
                    f"candidate {scene_id} {kind} hash mismatch") from error
            if observed_hash != expected[hash_field]:
                raise G0SealError(
                    f"candidate {scene_id} {kind} hash mismatch")
            scene_observed[hash_field] = observed_hash
        observed[scene_id] = MappingProxyType(scene_observed)
    return MappingProxyType(observed)


def _reject_json_constant(value: str) -> None:
    raise G0SealError("JSON constants must be finite")


def load_canonical_json(path: Path | str, *, label: str) -> Mapping[str, Any]:
    """Load JSON only when its input bytes equal the canonical encoding."""
    label = _require_nonempty_string(label, "label")
    source = Path(path)
    try:
        raw = source.read_bytes()
    except OSError as error:
        raise G0SealError(f"{label} could not read {source}") from error
    try:
        value = json.loads(raw.decode("utf-8"), parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, G0SealError) as error:
        raise G0SealError(f"{label} must contain valid canonical JSON") from error
    if canonical_json_bytes(value) != raw:
        raise G0SealError(f"{label} bytes must be canonical JSON")
    return _CanonicalJsonMapping(
        _require_mapping(value, label), sha256_bytes(raw))


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise G0SealError(f"{name} must be a mapping")
    return value


def _require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise G0SealError(f"{name} must be a list")
    if not value:
        raise G0SealError(f"{name} must be nonempty")
    return value


def _require_nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise G0SealError(f"{name} must be a nonempty string")
    return value


def _require_field(mapping: Mapping[str, Any], name: str, context: str) -> Any:
    if name not in mapping:
        raise G0SealError(f"{context}.{name} is required")
    return mapping[name]


def _reject_unknown_keys(
        mapping: Mapping[str, Any], allowed: frozenset[str], context: str) -> None:
    unknown = sorted(repr(key) for key in mapping if key not in allowed)
    if unknown:
        raise G0SealError(f"{context} has unknown keys: {', '.join(unknown)}")


def _require_sha256(value: Any, name: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in _HEX_DIGITS for character in value)):
        raise G0SealError(f"{name} must be a lowercase 64-hex SHA-256")
    return value


def _validate_asset(mapping: Mapping[str, Any], context: str) -> None:
    _reject_unknown_keys(mapping, frozenset({"path", "sha256"}), context)
    _require_nonempty_string(_require_field(mapping, "path", context),
                             f"{context}.path")
    _require_sha256(_require_field(mapping, "sha256", context),
                    f"{context}.sha256")


def _validate_class_group(vocabulary: Mapping[str, Any], field: str) -> tuple[str, ...]:
    classes = _require_list(_require_field(vocabulary, field, "vocabulary"),
                            f"vocabulary.{field}")
    names: list[str] = []
    for index, entry in enumerate(classes):
        context = f"vocabulary.{field}[{index}]"
        class_mapping = _require_mapping(entry, context)
        _reject_unknown_keys(class_mapping, frozenset({"name", "provenance"}),
                             context)
        name = _require_nonempty_string(
            _require_field(class_mapping, "name", context), f"{context}.name")
        _require_nonempty_string(
            _require_field(class_mapping, "provenance", context),
            f"{context}.provenance")
        names.append(name)
    if len(names) != len(set(names)):
        raise G0SealError(f"vocabulary.{field} class names must be unique")
    return tuple(names)


def _validate_vocabulary(authority: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    vocabulary = _require_mapping(
        _require_field(authority, "vocabulary", "authority"), "vocabulary")
    _reject_unknown_keys(vocabulary, frozenset({"base", "novel"}), "vocabulary")
    return (_validate_class_group(vocabulary, "base"),
            _validate_class_group(vocabulary, "novel"))


def _validate_code(authority: Mapping[str, Any]) -> None:
    code = _require_mapping(_require_field(authority, "code", "authority"), "code")
    _reject_unknown_keys(code, frozenset({"commit", "files"}), "code")
    _require_nonempty_string(_require_field(code, "commit", "code"), "code.commit")
    files = _require_list(_require_field(code, "files", "code"), "code.files")
    paths: set[str] = set()
    for index, entry in enumerate(files):
        context = f"code.files[{index}]"
        asset = _require_mapping(entry, context)
        _validate_asset(asset, context)
        path = asset["path"]
        if path in paths:
            raise G0SealError("code.files paths must be unique")
        paths.add(path)


def _validate_prompts(authority: Mapping[str, Any]) -> tuple[str, ...]:
    families = _require_list(
        _require_field(authority, "prompt_families", "authority"),
        "prompt_families")
    if len(families) != 3:
        raise G0SealError("prompt_families must contain exactly 3 entries")
    names: list[str] = []
    for index, entry in enumerate(families):
        context = f"prompt_families[{index}]"
        family = _require_mapping(entry, context)
        _reject_unknown_keys(family, frozenset({"name", "sha256"}), context)
        names.append(_require_nonempty_string(
            _require_field(family, "name", context), f"{context}.name"))
        _require_sha256(_require_field(family, "sha256", context),
                        f"{context}.sha256")
    if len(names) != len(set(names)):
        raise G0SealError("prompt family names must be unique")
    primary = _require_nonempty_string(
        _require_field(authority, "primary_prompt_family", "authority"),
        "primary_prompt_family")
    if primary not in names:
        raise G0SealError("primary_prompt_family must name a prompt family")
    return tuple(names)


def _validate_text_embedding_hashes(
        authority: Mapping[str, Any], prompt_names: tuple[str, ...]) -> None:
    records = _require_list(
        _require_field(authority, "text_embedding_hashes", "authority"),
        "text_embedding_hashes")
    if len(records) != 3:
        raise G0SealError("text_embedding_hashes must contain exactly 3 entries")
    families: list[str] = []
    for index, entry in enumerate(records):
        context = f"text_embedding_hashes[{index}]"
        record = _require_mapping(entry, context)
        _reject_unknown_keys(record, frozenset({"prompt_family", "sha256"}),
                             context)
        families.append(_require_nonempty_string(
            _require_field(record, "prompt_family", context),
            f"{context}.prompt_family"))
        _require_sha256(_require_field(record, "sha256", context),
                        f"{context}.sha256")
    if tuple(families) != prompt_names:
        raise G0SealError(
            "text_embedding_hashes prompt_family sequence must match prompt_families")


def _validate_oracle_mouth(authority: Mapping[str, Any]) -> None:
    oracle_mouth = _require_mapping(
        _require_field(authority, "oracle_mouth", "authority"), "oracle_mouth")
    _reject_unknown_keys(
        oracle_mouth,
        frozenset({
            "adapter_type",
            "carrier_source_identity_schema",
            "definition_sha256",
        }),
        "oracle_mouth")
    _require_nonempty_string(
        _require_field(oracle_mouth, "adapter_type", "oracle_mouth"),
        "oracle_mouth.adapter_type")
    _require_nonempty_string(
        _require_field(
            oracle_mouth, "carrier_source_identity_schema", "oracle_mouth"),
        "oracle_mouth.carrier_source_identity_schema")
    _require_sha256(
        _require_field(oracle_mouth, "definition_sha256", "oracle_mouth"),
        "oracle_mouth.definition_sha256")


def _require_finite_real(value: Any, name: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise G0SealError(f"{name} must be a finite real number")
    if isinstance(value, float) and not math.isfinite(value):
        raise G0SealError(f"{name} must be a finite real number")
    return value


def _validate_protocol_threshold_bundle(authority: Mapping[str, Any]) -> None:
    bundle = _require_mapping(
        _require_field(authority, "protocol_threshold_bundle", "authority"),
        "protocol_threshold_bundle")
    _reject_unknown_keys(
        bundle, frozenset(_PROTOCOL_THRESHOLD_VALUES),
        "protocol_threshold_bundle")
    for section, expected_values in _PROTOCOL_THRESHOLD_VALUES.items():
        context = f"protocol_threshold_bundle.{section}"
        values = _require_mapping(
            _require_field(bundle, section, "protocol_threshold_bundle"), context)
        _reject_unknown_keys(values, frozenset(expected_values), context)
        for field, expected_value in expected_values.items():
            value = _require_finite_real(
                _require_field(values, field, context), f"{context}.{field}")
            if ((section, field) in _INTEGER_PROTOCOL_THRESHOLD_FIELDS
                    and type(value) is not int):
                raise G0SealError(f"{context}.{field} must be an integer")
            if ((section, field) in _FLOAT_PROTOCOL_THRESHOLD_FIELDS
                    and type(value) is not float):
                raise G0SealError(f"{context}.{field} must be a float")
            if value != expected_value:
                raise G0SealError(
                    f"{context}.{field} must match the P0-v1 value")
    if (bundle["g3"]["replication_ratio_min"]
            > bundle["g3"]["replication_ratio_max"]):
        raise G0SealError(
            "protocol_threshold_bundle.g3 replication ratio bounds are inverted")
    if (bundle["g4"]["min_optimizer_steps"]
            > bundle["g4"]["max_optimizer_steps"]):
        raise G0SealError(
            "protocol_threshold_bundle.g4 optimizer step bounds are inverted")
    declared_sha256 = _require_sha256(
        _require_field(
            authority, "protocol_threshold_bundle_sha256", "authority"),
        "protocol_threshold_bundle_sha256")
    try:
        observed_sha256 = sha256_bytes(canonical_json_bytes(bundle))
    except G0SealError as error:
        raise G0SealError(
            "protocol_threshold_bundle must be canonical JSON") from error
    if declared_sha256 != observed_sha256:
        raise G0SealError("protocol_threshold_bundle_sha256 mismatch")


def _validate_eligibility_policy(authority: Mapping[str, Any]) -> None:
    eligibility_policy = _require_mapping(
        _require_field(authority, "eligibility_policy", "authority"),
        "eligibility_policy")
    _reject_unknown_keys(
        eligibility_policy,
        frozenset({"schema", "min_size", "max_size", "max_overlap", "sha256"}),
        "eligibility_policy")
    schema = _require_field(eligibility_policy, "schema", "eligibility_policy")
    if schema != _ELIGIBILITY_SCHEMA:
        raise G0SealError(
            f"eligibility_policy.schema must be {_ELIGIBILITY_SCHEMA}")
    minimum = _require_finite_real(
        _require_field(eligibility_policy, "min_size", "eligibility_policy"),
        "eligibility_policy.min_size")
    maximum = _require_finite_real(
        _require_field(eligibility_policy, "max_size", "eligibility_policy"),
        "eligibility_policy.max_size")
    overlap = _require_finite_real(
        _require_field(eligibility_policy, "max_overlap", "eligibility_policy"),
        "eligibility_policy.max_overlap")
    if minimum < 0 or maximum < minimum:
        raise G0SealError(
            "eligibility_policy requires 0 <= min_size <= max_size")
    if overlap < 0:
        raise G0SealError("eligibility_policy.max_overlap must be nonnegative")
    _require_sha256(
        _require_field(eligibility_policy, "sha256", "eligibility_policy"),
        "eligibility_policy.sha256")


def _validate_bootstrap(authority: Mapping[str, Any]) -> None:
    bootstrap = _require_mapping(
        _require_field(authority, "bootstrap", "authority"), "bootstrap")
    _reject_unknown_keys(bootstrap, frozenset({"seed", "repetitions"}),
                         "bootstrap")
    for field in ("seed", "repetitions"):
        value = _require_field(bootstrap, field, "bootstrap")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise G0SealError(f"bootstrap.{field} must be a strictly positive integer")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def validate_authority(authority: Mapping[str, Any]) -> SealedAuthority:
    """Validate and seal every predeclared P0 G0 scientific authority choice."""
    authority = _require_mapping(authority, "authority")
    _reject_unknown_keys(
        authority,
        frozenset({
            "schema",
            "forbidden_scene_id",
            "candidate_scene_plan_sha256",
            "object_inventory_sha256",
            "checkpoint",
            "resolved_config",
            "code",
            "vocabulary",
            "prompt_families",
            "text_embedding_hashes",
            "primary_prompt_family",
            "native_temperature",
            "oracle_mouth",
            "protocol_threshold_bundle",
            "protocol_threshold_bundle_sha256",
            "render_contract",
            "eligibility_policy",
            "bootstrap",
        }),
        "authority")
    schema = _require_field(authority, "schema", "authority")
    if schema != _AUTHORITY_SCHEMA:
        raise G0SealError(f"authority.schema must be {_AUTHORITY_SCHEMA}")

    forbidden_scene_id = _require_nonempty_string(
        _require_field(authority, "forbidden_scene_id", "authority"),
        "forbidden_scene_id")
    if forbidden_scene_id != _FORBIDDEN_SCENE_ID:
        raise G0SealError(
            f"authority.forbidden_scene_id must be {_FORBIDDEN_SCENE_ID}")
    _require_sha256(
        _require_field(authority, "candidate_scene_plan_sha256", "authority"),
        "candidate_scene_plan_sha256")
    _require_sha256(
        _require_field(authority, "object_inventory_sha256", "authority"),
        "object_inventory_sha256")
    _validate_asset(
        _require_mapping(_require_field(authority, "checkpoint", "authority"),
                         "checkpoint"),
        "checkpoint")
    _validate_asset(
        _require_mapping(_require_field(authority, "resolved_config", "authority"),
                         "resolved_config"),
        "resolved_config")
    _validate_code(authority)

    base_classes, novel_classes = _validate_vocabulary(authority)
    if not set(base_classes).isdisjoint(novel_classes):
        raise G0SealError("vocabulary.base and vocabulary.novel must be disjoint")

    prompt_names = _validate_prompts(authority)
    _validate_text_embedding_hashes(authority, prompt_names)
    primary_prompt_family = authority["primary_prompt_family"]

    native_temperature = _require_mapping(
        _require_field(authority, "native_temperature", "authority"),
        "native_temperature")
    _reject_unknown_keys(native_temperature, frozenset({"rule_id", "sha256"}),
                         "native_temperature")
    _require_nonempty_string(
        _require_field(native_temperature, "rule_id", "native_temperature"),
        "native_temperature.rule_id")
    _require_sha256(
        _require_field(native_temperature, "sha256", "native_temperature"),
        "native_temperature.sha256")

    render_contract = _require_mapping(
        _require_field(authority, "render_contract", "authority"),
        "render_contract")
    _reject_unknown_keys(render_contract, frozenset({"kind", "sha256"}),
                         "render_contract")
    if _require_field(render_contract, "kind", "render_contract") != _RENDER_CONTRACT_KIND:
        raise G0SealError(f"render_contract.kind must be {_RENDER_CONTRACT_KIND}")
    _require_sha256(
        _require_field(render_contract, "sha256", "render_contract"),
        "render_contract.sha256")

    _validate_eligibility_policy(authority)
    _validate_bootstrap(authority)
    _validate_oracle_mouth(authority)
    _validate_protocol_threshold_bundle(authority)

    return SealedAuthority(
        forbidden_scene_id=forbidden_scene_id,
        vocabulary=base_classes + novel_classes,
        base_classes=base_classes,
        novel_classes=novel_classes,
        primary_prompt_family=primary_prompt_family,
        raw=_freeze(authority),
    )


def validate_candidate_plan(value: Mapping[str, Any]) -> Mapping[str, Mapping[str, str]]:
    """Validate a candidate plan and index its sealed records by scene ID."""
    plan = _require_mapping(value, "candidate plan")
    _reject_unknown_keys(plan, _CANDIDATE_PLAN_FIELDS, "candidate plan")
    if _require_field(plan, "schema", "candidate plan") != _CANDIDATE_PLAN_SCHEMA:
        raise G0SealError(
            f"candidate plan.schema must be {_CANDIDATE_PLAN_SCHEMA}")
    records = _require_list(
        _require_field(plan, "records", "candidate plan"), "candidate plan.records")

    indexed: dict[str, Mapping[str, str]] = {}
    for index, value in enumerate(records):
        context = f"candidate plan.records[{index}]"
        record = _require_mapping(value, context)
        _reject_unknown_keys(record, frozenset(_CANDIDATE_RECORD_FIELDS), context)
        canonical = {
            field: _require_nonempty_string(
                _require_field(record, field, context), f"{context}.{field}")
            for field in _CANDIDATE_RECORD_FIELDS
        }
        for field in ("image_sha256", "annotation_sha256"):
            _require_sha256(canonical[field], f"{context}.{field}")
        scene_id = canonical["scene_id"]
        if scene_id == _FORBIDDEN_SCENE_ID:
            raise G0SealError(
                f"candidate plan must exclude forbidden scene {_FORBIDDEN_SCENE_ID}")
        previous = indexed.get(scene_id)
        if previous is not None:
            if previous["split"] != canonical["split"]:
                raise G0SealError(
                    f"candidate plan scene {scene_id} has split leakage")
            raise G0SealError(
                f"candidate plan scene {scene_id} must have a unique identity")
        indexed[scene_id] = MappingProxyType(canonical)
    return MappingProxyType(indexed)


def load_object_rows(path: Path | str) -> tuple[Mapping[str, Any], ...]:
    """Load sorted, recursively frozen canonical UTF-8 JSON-object rows."""
    source = Path(path)
    try:
        raw = source.read_bytes()
    except OSError as error:
        raise G0SealError(f"object rows could not read {source}") from error
    lines = raw.splitlines(keepends=True)

    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        try:
            value = json.loads(
                raw_line.decode("utf-8"), parse_constant=_reject_json_constant)
            canonical = canonical_json_bytes(value)
        except (UnicodeDecodeError, UnicodeEncodeError, json.JSONDecodeError,
                G0SealError) as error:
            raise G0SealError(
                f"object rows line {line_number} must contain valid canonical JSON") from error
        if not isinstance(value, dict):
            raise G0SealError(f"object rows line {line_number} must be a JSON object")
        if canonical != raw_line:
            raise G0SealError(
                f"object rows line {line_number} must be canonical JSON")
        context = f"object rows line {line_number}"
        for field in ("scene_id", "object_id"):
            _require_nonempty_string(
                _require_field(value, field, context), f"{context}.{field}")
        rows.append({key: item for key, item in value.items()})
    if not rows:
        raise G0SealError("object rows file must not be blank")
    rows.sort(key=lambda row: (row["scene_id"], row["object_id"]))
    return _LoadedObjectRows(
        (_freeze(row) for row in rows), sha256_bytes(raw))


def _canonical_object_row(
        row: Mapping[str, Any], context: str) -> dict[str, Any]:
    """Copy one validated object row into its fixed primitive-dict shape."""
    return {
        "annotation_sha256": row["annotation_sha256"],
        "box": list(row["box"]),
        "class_name": row["class_name"],
        "object_id": row["object_id"],
        "overlap": row["overlap"],
        "scene_id": row["scene_id"],
        "size": row["size"],
    }


def _object_validation_marker(row: Mapping[str, Any], context: str) -> str:
    """Derive the internal marker from the canonical public object fields."""
    fields = {
        field: _require_field(row, field, context)
        for field in _OBJECT_ROW_FIELDS
    }
    try:
        return sha256_bytes(canonical_json_bytes(fields))
    except G0SealError as error:
        raise G0SealError(
            f"{context} validated object marker cannot be computed") from error


def _require_sealed_authority(authority: Any) -> SealedAuthority:
    if not isinstance(authority, SealedAuthority):
        raise G0SealError("authority must be a SealedAuthority")
    return authority


def _revalidate_sealed_authority(authority: Any) -> SealedAuthority:
    """Materialize and validate a sealed authority at a public trust boundary."""
    authority = _require_sealed_authority(authority)
    raw = _json_native(authority.raw)
    return validate_authority(_require_mapping(raw, "authority.raw"))


def _require_iterable(rows: Any, name: str) -> Iterable[Mapping[str, Any]]:
    try:
        return iter(rows)
    except TypeError as error:
        raise G0SealError(f"{name} must be iterable") from error


def validate_object_rows(
        rows: Iterable[Mapping[str, Any]],
        candidates: Mapping[str, Mapping[str, str]],
        authority: SealedAuthority) -> tuple[Mapping[str, Any], ...]:
    """Validate, sort, and recursively freeze canonical object rows."""
    authority = _revalidate_sealed_authority(authority)
    candidates = _require_mapping(candidates, "candidates")

    validated: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    for index, value in enumerate(_require_iterable(rows, "object rows")):
        context = f"object rows[{index}]"
        row = _require_mapping(value, context)
        _reject_unknown_keys(row, frozenset(_OBJECT_ROW_FIELDS), context)
        for field in ("annotation_sha256", "class_name", "object_id", "scene_id"):
            _require_nonempty_string(
                _require_field(row, field, context), f"{context}.{field}")
        annotation_sha256 = _require_sha256(
            row["annotation_sha256"], f"{context}.annotation_sha256")
        scene_id = row["scene_id"]
        try:
            candidate = _require_mapping(candidates[scene_id], f"candidates[{scene_id}]")
            expected_annotation_sha256 = _require_sha256(
                _require_field(candidate, "annotation_sha256", f"candidates[{scene_id}]"),
                f"candidates[{scene_id}].annotation_sha256")
        except KeyError as error:
            raise G0SealError(f"{context}.scene_id must name a candidate scene") from error
        if annotation_sha256 != expected_annotation_sha256:
            raise G0SealError(
                f"{context}.annotation_sha256 must match its candidate")
        if row["class_name"] not in authority.vocabulary:
            raise G0SealError(f"{context}.class_name must be in authority vocabulary")

        box = _require_field(row, "box", context)
        if not isinstance(box, (list, tuple)) or len(box) != 5:
            raise G0SealError(
                f"{context}.box must be a list or frozen tuple of 5 finite real numbers")
        for box_index, coordinate in enumerate(box):
            _require_finite_real(coordinate, f"{context}.box[{box_index}]")
        size = _require_finite_real(
            _require_field(row, "size", context), f"{context}.size")
        overlap = _require_finite_real(
            _require_field(row, "overlap", context), f"{context}.overlap")
        if size < 0:
            raise G0SealError(f"{context}.size must be nonnegative")
        if overlap < 0:
            raise G0SealError(f"{context}.overlap must be nonnegative")

        identity = (scene_id, row["object_id"])
        if identity in identities:
            raise G0SealError("object rows must have unique (scene_id, object_id) identities")
        identities.add(identity)
        canonical = _canonical_object_row(row, context)
        canonical[_VALIDATION_MARKER_FIELD] = _object_validation_marker(
            canonical, context)
        validated.append(canonical)
    validated.sort(key=lambda row: (row["scene_id"], row["object_id"]))
    return tuple(_freeze(row) for row in validated)


def _eligibility_policy(authority: SealedAuthority) -> Mapping[str, Any]:
    authority = _require_sealed_authority(authority)
    return _require_mapping(authority.raw["eligibility_policy"], "eligibility_policy")


def _protocol_threshold_bundle(authority: SealedAuthority) -> Mapping[str, Any]:
    authority = _require_sealed_authority(authority)
    return _require_mapping(
        authority.raw["protocol_threshold_bundle"], "protocol_threshold_bundle")


def select_eligible_objects(
        rows: Iterable[Mapping[str, Any]],
        authority: SealedAuthority) -> tuple[
            tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...]]:
    """Apply the sealed eligibility policy and return frozen decision and eligible rows."""
    authority = _revalidate_sealed_authority(authority)
    policy = _eligibility_policy(authority)
    minimum = policy["min_size"]
    maximum = policy["max_size"]
    max_overlap = policy["max_overlap"]

    ordered_rows: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    for index, value in enumerate(_require_iterable(rows, "object rows")):
        context = f"object rows[{index}]"
        row = _require_mapping(value, context)
        if _VALIDATION_MARKER_FIELD not in row:
            raise G0SealError(f"{context} must be a validated object row")
        _reject_unknown_keys(row, _VALIDATED_OBJECT_ROW_FIELDS, context)
        marker = _require_sha256(
            row[_VALIDATION_MARKER_FIELD],
            f"{context}.{_VALIDATION_MARKER_FIELD}")
        if marker != _object_validation_marker(row, context):
            raise G0SealError(f"{context} validated object marker mismatch")
        canonical = _canonical_object_row(row, context)
        identity = (canonical["scene_id"], canonical["object_id"])
        if identity in identities:
            raise G0SealError(
                "selection rows must not contain duplicate (scene_id, object_id) identities")
        identities.add(identity)
        ordered_rows.append(canonical)
    ordered_rows.sort(key=lambda row: (row["scene_id"], row["object_id"]))

    decisions: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for row in ordered_rows:
        if row["size"] < minimum:
            exclusion_reason: str | None = "below_min_size"
        elif row["size"] > maximum:
            exclusion_reason = "above_max_size"
        elif row["overlap"] > max_overlap:
            exclusion_reason = "overlap_exceeds_max"
        else:
            exclusion_reason = None
        decision = _canonical_object_row(row, "decision row")
        decision["eligible"] = exclusion_reason is None
        decision["exclusion_reason"] = exclusion_reason
        decisions.append(decision)
        if exclusion_reason is None:
            eligible.append(_canonical_object_row(row, "eligible row"))
    return (tuple(_freeze(row) for row in decisions),
            tuple(_freeze(row) for row in eligible))


def _canonical_view_source(row: Mapping[str, Any], context: str) -> dict[str, Any]:
    """Copy the public fields required to produce one frozen view record."""
    row = _require_mapping(row, context)
    scene_id = _require_nonempty_string(
        _require_field(row, "scene_id", context), f"{context}.scene_id")
    object_id = _require_nonempty_string(
        _require_field(row, "object_id", context), f"{context}.object_id")
    class_name = _require_nonempty_string(
        _require_field(row, "class_name", context), f"{context}.class_name")
    box = _require_field(row, "box", context)
    if not isinstance(box, (list, tuple)) or len(box) != 5:
        raise G0SealError(f"{context}.box must contain 5 finite real numbers")
    canonical_box = []
    for index, coordinate in enumerate(box):
        canonical_box.append(_require_finite_real(
            coordinate, f"{context}.box[{index}]"))
    return {
        "scene_id": scene_id,
        "object_id": object_id,
        "class_name": class_name,
        "box": canonical_box,
    }


def build_view_plan(
        rows: Iterable[Mapping[str, Any]], *,
        render_contract_sha256: str) -> tuple[Mapping[str, Any], ...]:
    """Expand eligible objects into the five canonical frozen C4 view records."""
    render_contract_sha256 = _require_sha256(
        render_contract_sha256, "render_contract_sha256")
    views: list[dict[str, Any]] = []
    for index, value in enumerate(_require_iterable(rows, "eligible rows")):
        row = _canonical_view_source(value, f"eligible rows[{index}]")
        digest_payload = {
            "scene_id": row["scene_id"],
            "object_id": row["object_id"],
            "box": row["box"],
            "render_contract_sha256": render_contract_sha256,
        }
        identity_digest = sha256_bytes(canonical_json_bytes(digest_payload))
        for view_id in VIEW_IDS:
            render_digest = identity_digest
            if view_id not in ("rot000_a", "rot000_b"):
                render_digest = sha256_bytes(canonical_json_bytes({
                    **digest_payload,
                    "view_id": view_id,
                }))
            views.append({
                "scene_id": row["scene_id"],
                "object_id": row["object_id"],
                "class_name": row["class_name"],
                "box": row["box"],
                "view_id": view_id,
                "render_digest": render_digest,
            })
    view_order = {view_id: index for index, view_id in enumerate(VIEW_IDS)}
    views.sort(key=lambda row: (
        row["scene_id"], row["object_id"], view_order[row["view_id"]]))
    return tuple(_freeze(row) for row in views)


def _canonical_jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) for row in rows)


def _declared_authority_asset_hashes(
        authority: SealedAuthority) -> dict[str, str]:
    return {
        name: _require_sha256(asset["sha256"], f"authority.{name}.sha256")
        for name, asset in _authority_asset_items(authority)
    }


def _validated_asset_hashes(
        asset_hashes: Mapping[str, str], authority: SealedAuthority) -> dict[str, str]:
    asset_hashes = _require_mapping(asset_hashes, "asset_hashes")
    expected = _declared_authority_asset_hashes(authority)
    expected_set = frozenset(expected)
    _reject_unknown_keys(asset_hashes, expected_set, "asset_hashes")
    missing = sorted(expected_set.difference(asset_hashes))
    if missing:
        raise G0SealError(f"asset_hashes is missing: {', '.join(missing)}")
    observed: dict[str, str] = {}
    for name, expected_hash in expected.items():
        observed_hash = _require_sha256(
            asset_hashes[name], f"asset_hashes.{name}")
        if observed_hash != expected_hash:
            raise G0SealError(
                f"asset_hashes.{name} must match its authority declared hash")
        observed[name] = observed_hash
    return observed


def _validated_candidate_asset_hashes(
        candidate_asset_hashes: Mapping[str, Mapping[str, str]],
        candidates: Mapping[str, Mapping[str, str]]) -> Mapping[str, Mapping[str, str]]:
    candidate_asset_hashes = _require_mapping(
        candidate_asset_hashes, "candidate_asset_hashes")
    declared = _declared_candidate_asset_hashes(candidates)
    expected_scenes = frozenset(declared)
    _reject_unknown_keys(
        candidate_asset_hashes, expected_scenes, "candidate_asset_hashes")
    missing = sorted(expected_scenes.difference(candidate_asset_hashes))
    if missing:
        raise G0SealError(
            f"candidate_asset_hashes is missing: {', '.join(missing)}")

    observed: dict[str, Mapping[str, str]] = {}
    for scene_id, expected_hashes in declared.items():
        context = f"candidate_asset_hashes[{scene_id}]"
        values = _require_mapping(candidate_asset_hashes[scene_id], context)
        _reject_unknown_keys(values, frozenset(_CANDIDATE_ASSET_HASH_FIELDS), context)
        row: dict[str, str] = {}
        for field in _CANDIDATE_ASSET_HASH_FIELDS:
            observed_hash = _require_sha256(
                _require_field(values, field, context), f"{context}.{field}")
            if observed_hash != expected_hashes[field]:
                raise G0SealError(
                    f"{context}.{field} must match its declared hash")
            row[field] = observed_hash
        observed[scene_id] = MappingProxyType(row)
    return MappingProxyType(observed)


def _decision_artifact_rows(
        decisions: Iterable[Mapping[str, Any]],
        candidates: Mapping[str, Mapping[str, str]]) -> tuple[Mapping[str, Any], ...]:
    """Attach sealed candidate splits while excluding internal validation markers."""
    emitted: list[dict[str, Any]] = []
    for index, value in enumerate(decisions):
        decision = _require_mapping(value, f"decision rows[{index}]")
        row = _canonical_object_row(decision, f"decision rows[{index}]")
        row["eligible"] = decision["eligible"]
        row["exclusion_reason"] = decision["exclusion_reason"]
        row["split"] = candidates[row["scene_id"]]["split"]
        emitted.append(row)
    emitted.sort(key=lambda row: (row["scene_id"], row["object_id"], row["split"]))
    return tuple(_freeze(row) for row in emitted)


def _sorted_values(values: Mapping[str, Any]) -> dict[str, Any]:
    return {name: values[name] for name in sorted(values)}


def _candidate_scene_counts_by_split(
        candidates: Mapping[str, Mapping[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates.values():
        split = candidate["split"]
        counts[split] = counts.get(split, 0) + 1
    return _sorted_values(counts)


def _seal_diagnostics(
        *, authority: SealedAuthority,
        candidates: Mapping[str, Mapping[str, str]],
        decisions: Iterable[Mapping[str, Any]],
        eligible: Iterable[Mapping[str, Any]],
        view_count: int) -> dict[str, Any]:
    eligible_rows = tuple(eligible)
    decision_rows = tuple(decisions)
    class_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    exclusion_counts = {
        "eligible": 0,
        "below_min_size": 0,
        "above_max_size": 0,
        "overlap_exceeds_max": 0,
    }
    for decision in decision_rows:
        reason = decision["exclusion_reason"]
        exclusion_counts["eligible" if reason is None else reason] += 1
    for row in eligible_rows:
        class_name = row["class_name"]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        split = candidates[row["scene_id"]]["split"]
        split_counts[split] = split_counts.get(split, 0) + 1

    eligible_scene_count = len({row["scene_id"] for row in eligible_rows})
    novel_classes = set(authority.novel_classes)
    novel_object_count = sum(
        count for class_name, count in class_counts.items()
        if class_name in novel_classes)
    novel_class_count = sum(
        1 for class_name in class_counts if class_name in novel_classes)
    g0_thresholds = _require_mapping(
        _protocol_threshold_bundle(authority)["g0"],
        "protocol_threshold_bundle.g0")
    strict_ovd_ready = (
        novel_object_count >= g0_thresholds["min_novel_objects"]
        and novel_class_count >= g0_thresholds["min_novel_classes"]
    )
    supported_class_count = len(class_counts)
    g0_scope_ready = (
        eligible_scene_count >= g0_thresholds["min_scenes"]
        and len(eligible_rows) >= g0_thresholds["min_objects"]
        and supported_class_count >= g0_thresholds["min_supported_classes"]
    )
    return {
        "schema": "ovd-orbit-p0-g0-seal-diagnostics-v1",
        "candidate_scene_count": len(candidates),
        "validated_object_count": len(decision_rows),
        "eligible_scene_count": eligible_scene_count,
        "eligible_object_count": len(eligible_rows),
        "supported_class_count": supported_class_count,
        "view_count": view_count,
        "candidate_scene_counts_by_split": _candidate_scene_counts_by_split(
            candidates),
        "eligible_object_counts_by_class": _sorted_values(class_counts),
        "eligible_object_counts_by_split": _sorted_values(split_counts),
        "object_counts_by_exclusion": _sorted_values(exclusion_counts),
        "novel_object_count": novel_object_count,
        "novel_class_count": novel_class_count,
        "leakage_count": 0,
        "leakage_status": "pass",
        "g0_scope_ready": g0_scope_ready,
        "strict_ovd_ready": strict_ovd_ready,
    }


def _input_manifest(
        *, authority: SealedAuthority, status: str,
        candidate_plan_sha256: str, object_inventory_sha256: str,
        asset_hashes: Mapping[str, str],
        candidate_asset_hashes: Mapping[str, Mapping[str, str]],
        candidates: Mapping[str, Mapping[str, str]], diagnostics: Mapping[str, Any],
        view_plan_sha256: str) -> dict[str, Any]:
    raw = authority.raw
    prompt_hashes = {
        entry["name"]: entry["sha256"]
        for entry in raw["prompt_families"]
    }
    code_files = sorted(
        ({"path": entry["path"], "sha256": entry["sha256"]}
         for entry in raw["code"]["files"]),
        key=lambda entry: (entry["path"], entry["sha256"]),
    )
    return {
        "schema": "ovd-orbit-p0-g0-input-manifest-v1",
        "status": status,
        "authority_canonical_sha256": sha256_bytes(canonical_json_bytes(raw)),
        "candidate_plan_sha256": candidate_plan_sha256,
        "object_inventory_sha256": object_inventory_sha256,
        "asset_hashes": dict(asset_hashes),
        "declared_asset_hashes": _declared_authority_asset_hashes(authority),
        "observed_asset_hashes": dict(asset_hashes),
        "declared_candidate_assets": _declared_candidate_asset_hashes(candidates),
        "observed_candidate_assets": {
            scene_id: dict(candidate_asset_hashes[scene_id])
            for scene_id in sorted(candidate_asset_hashes)
        },
        "declared_assets": {
            "checkpoint": {
                "path": raw["checkpoint"]["path"],
                "sha256": raw["checkpoint"]["sha256"],
            },
            "resolved_config": {
                "path": raw["resolved_config"]["path"],
                "sha256": raw["resolved_config"]["sha256"],
            },
        },
        "code_identity": {
            "commit": raw["code"]["commit"],
            "files": code_files,
        },
        "candidate_scene_counts_by_split": diagnostics[
            "candidate_scene_counts_by_split"],
        "leakage_count": diagnostics["leakage_count"],
        "leakage_status": diagnostics["leakage_status"],
        "vocabulary_sha256": sha256_bytes(canonical_json_bytes(raw["vocabulary"])),
        "prompt_family_sha256": _sorted_values(prompt_hashes),
        "text_embedding_hashes": [
            {
                "prompt_family": entry["prompt_family"],
                "sha256": entry["sha256"],
            }
            for entry in raw["text_embedding_hashes"]
        ],
        "native_temperature_sha256": raw["native_temperature"]["sha256"],
        "oracle_mouth": {
            "adapter_type": raw["oracle_mouth"]["adapter_type"],
            "carrier_source_identity_schema": raw["oracle_mouth"][
                "carrier_source_identity_schema"],
            "definition_sha256": raw["oracle_mouth"]["definition_sha256"],
        },
        "protocol_threshold_bundle_sha256": raw[
            "protocol_threshold_bundle_sha256"],
        "render_contract_sha256": raw["render_contract"]["sha256"],
        "counts": {
            name: diagnostics[name]
            for name in (
                "candidate_scene_count",
                "validated_object_count",
                "eligible_scene_count",
                "eligible_object_count",
                "supported_class_count",
                "view_count",
            )
        },
        "view_plan_sha256": view_plan_sha256,
    }


def _result_markdown(status: str, diagnostics: Mapping[str, Any]) -> bytes:
    """Return the Chinese count-only, no-forward human summary."""
    return (
        "# P0 G0 输入封存摘要\n\n"
        f"状态：{status}\n\n"
        f"候选场景数：{diagnostics['candidate_scene_count']}\n"
        f"合格场景数：{diagnostics['eligible_scene_count']}\n"
        f"合格对象数：{diagnostics['eligible_object_count']}\n"
        f"支持类别数：{diagnostics['supported_class_count']}\n"
        f"视图条目数：{diagnostics['view_count']}\n\n"
        "未执行前向计算；本文件仅记录输入封存计数。\n"
    ).encode("utf-8")


def build_failure_artifacts(error: G0SealError) -> dict[str, bytes]:
    """Build the minimal fail-stop package without exposing input contents."""
    if not isinstance(error, G0SealError):
        raise TypeError("error must be a G0SealError")
    message = str(error)
    status = "P0_INPUT_FAIL_STOP"
    diagnostics = {
        "schema": "ovd-orbit-p0-g0-seal-diagnostics-v1",
        "status": status,
        "error": message,
    }
    return {
        "receipt.json": canonical_json_bytes({
            "schema": "ovd-orbit-p0-g0-receipt-v1",
            "status": status,
            "error": message,
        }),
        "seal_diagnostics.json": canonical_json_bytes(diagnostics),
        "result.md": (
            "# P0 G0 输入封存失败\n\n"
            f"输入封存失败：{message}\n\n"
            "未运行模型、GPU 或指标计算。\n"
        ).encode("utf-8"),
    }


def _validated_artifact_items(
        artifacts: Mapping[str, bytes]) -> tuple[tuple[str, bytes], ...]:
    if not isinstance(artifacts, Mapping):
        raise G0SealError("artifacts must be a mapping")
    items = tuple(artifacts.items())
    if not items:
        raise G0SealError("artifacts must not be empty")
    names: set[str] = set()
    validated: list[tuple[str, bytes]] = []
    for name, payload in items:
        if not isinstance(name, str) or not name:
            raise G0SealError("artifact names must be nonempty strings")
        path = Path(name)
        if (path.is_absolute() or path.name != name or name in {".", ".."}
                or "/" in name or "\\" in name or "\x00" in name):
            raise G0SealError(f"unsafe artifact name: {name!r}")
        if name in names:
            raise G0SealError(f"duplicate artifact name: {name!r}")
        if not isinstance(payload, bytes):
            raise G0SealError(f"artifact {name!r} must be nonempty bytes")
        if not payload and name != "object_view_plan.jsonl":
            raise G0SealError(f"artifact {name!r} must be nonempty bytes")
        names.add(name)
        validated.append((name, payload))
    return tuple(validated)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_noreplace(source: Path, target: Path) -> None:
    """Use Linux renameat2 so a concurrent target cannot be overwritten."""
    if os.name != "posix":
        raise G0SealError("no-replace directory publication is unavailable")
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = libc.renameat2
    except (AttributeError, OSError) as error:
        raise G0SealError("no-replace directory publication is unavailable") from error
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(target),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, os.strerror(error_number), str(target))
    raise OSError(error_number, os.strerror(error_number), str(target))


def _raise_if_output_exists(output_dir: Path) -> None:
    try:
        output_dir.lstat()
    except FileNotFoundError:
        return
    raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), str(output_dir))


def publish_artifacts(output_dir: Path | str, artifacts: Mapping[str, bytes]) -> None:
    """Atomically publish a nonempty byte package without replacing an output."""
    items = _validated_artifact_items(artifacts)
    target = Path(output_dir)
    if not target.name or target.name in {".", ".."}:
        raise G0SealError("output_dir must name a new directory")
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    _raise_if_output_exists(target)

    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=parent))
    try:
        for name, payload in items:
            destination = temporary / name
            with destination.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        _fsync_directory(temporary)
        _rename_directory_noreplace(temporary, target)
        _fsync_directory(parent)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def build_g0_artifacts(
        *, authority: SealedAuthority, candidate_plan: Mapping[str, Any],
        object_rows: Iterable[Mapping[str, Any]], candidate_plan_sha256: str,
        object_inventory_sha256: str,
        asset_hashes: Mapping[str, str],
        candidate_asset_hashes: Mapping[str, Mapping[str, str]]) -> dict[str, bytes]:
    """Seal deterministic G0 inputs without reading assets or executing a forward pass."""
    authority = _revalidate_sealed_authority(authority)
    candidate_plan_sha256 = _require_sha256(
        candidate_plan_sha256, "candidate_plan_sha256")
    object_inventory_sha256 = _require_sha256(
        object_inventory_sha256, "object_inventory_sha256")
    if candidate_plan_sha256 != authority.raw["candidate_scene_plan_sha256"]:
        raise G0SealError(
            "candidate_plan_sha256 must match authority.candidate_scene_plan_sha256")
    if object_inventory_sha256 != authority.raw["object_inventory_sha256"]:
        raise G0SealError(
            "object_inventory_sha256 must match authority.object_inventory_sha256")
    asset_hashes = _validated_asset_hashes(asset_hashes, authority)
    candidates = validate_candidate_plan(candidate_plan)
    candidate_asset_hashes = _validated_candidate_asset_hashes(
        candidate_asset_hashes, candidates)
    validated_rows = validate_object_rows(object_rows, candidates, authority)
    decisions, eligible = select_eligible_objects(validated_rows, authority)
    render_contract_sha256 = authority.raw["render_contract"]["sha256"]
    views = build_view_plan(
        eligible, render_contract_sha256=render_contract_sha256)
    decision_rows = _decision_artifact_rows(decisions, candidates)
    diagnostics = _seal_diagnostics(
        authority=authority,
        candidates=candidates,
        decisions=decisions,
        eligible=eligible,
        view_count=len(views),
    )
    status = (
        "G0_INPUTS_SEALED_NO_FORWARD"
        if diagnostics["g0_scope_ready"] else "P0_INPUT_FAIL_STOP"
    )
    diagnostics["status"] = status

    view_bytes = _canonical_jsonl_bytes(views)
    artifacts = {
        "input_manifest.json": canonical_json_bytes(_input_manifest(
            authority=authority,
            status=status,
            candidate_plan_sha256=candidate_plan_sha256,
            object_inventory_sha256=object_inventory_sha256,
            asset_hashes=asset_hashes,
            candidate_asset_hashes=candidate_asset_hashes,
            candidates=candidates,
            diagnostics=diagnostics,
            view_plan_sha256=sha256_bytes(view_bytes),
        )),
        "object_eligibility.jsonl": _canonical_jsonl_bytes(decision_rows),
        "object_view_plan.jsonl": view_bytes,
        "seal_diagnostics.json": canonical_json_bytes(diagnostics),
        "result.md": _result_markdown(status, diagnostics),
    }
    artifacts["receipt.json"] = canonical_json_bytes({
        "schema": "ovd-orbit-p0-g0-receipt-v1",
        "status": status,
        "artifact_sha256": {
            name: sha256_bytes(artifacts[name]) for name in sorted(artifacts)
        },
    })
    return artifacts


__all__ = [
    "G0SealError",
    "SealedAuthority",
    "canonical_json_bytes",
    "sha256_bytes",
    "sha256_file",
    "verify_authority_assets",
    "verify_candidate_assets",
    "load_canonical_json",
    "validate_authority",
    "validate_candidate_plan",
    "load_object_rows",
    "validate_object_rows",
    "select_eligible_objects",
    "VIEW_IDS",
    "build_view_plan",
    "build_g0_artifacts",
    "build_failure_artifacts",
    "publish_artifacts",
]
