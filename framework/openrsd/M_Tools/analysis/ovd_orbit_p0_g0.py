"""Pure-Python authority validation for the OVD object-orbit P0 G0 seal."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


_AUTHORITY_SCHEMA = "ovd-orbit-p0-g0-authority-v1"
_ELIGIBILITY_SCHEMA = "canonical-object-inventory-v1"
_RENDER_CONTRACT_KIND = "lossless-square-c4"
_HEX_DIGITS = frozenset("0123456789abcdef")


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
    except (TypeError, ValueError) as error:
        raise G0SealError("value must be canonical JSON") from error
    return text.encode("utf-8") + b"\n"


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
    return _require_mapping(value, label)


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
    for index, entry in enumerate(files):
        _validate_asset(_require_mapping(entry, f"code.files[{index}]"),
                        f"code.files[{index}]")


def _validate_prompts(authority: Mapping[str, Any]) -> None:
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


def _require_finite_real(value: Any, name: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise G0SealError(f"{name} must be a finite real number")
    if isinstance(value, float) and not math.isfinite(value):
        raise G0SealError(f"{name} must be a finite real number")
    return value


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
            "primary_prompt_family",
            "native_temperature",
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

    _validate_prompts(authority)
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

    return SealedAuthority(
        forbidden_scene_id=forbidden_scene_id,
        vocabulary=base_classes + novel_classes,
        base_classes=base_classes,
        novel_classes=novel_classes,
        primary_prompt_family=primary_prompt_family,
        raw=_freeze(authority),
    )


__all__ = [
    "G0SealError",
    "SealedAuthority",
    "canonical_json_bytes",
    "sha256_bytes",
    "sha256_file",
    "load_canonical_json",
    "validate_authority",
]
