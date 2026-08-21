#!/usr/bin/env python3
"""Publish a CPU-only diagnostic bundle for a DOTA-v2 fixed-Q dump."""

import argparse
import contextlib
import ctypes
import errno
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
import uuid
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from dotav2_q600_diagnostics import (  # noqa: E402
    NON_MANIFEST_PAYLOADS,
    DiagnosticError,
    attach_refinement,
    build_diagnostics,
    build_manifest,
    cap_error_examples,
    decompose_ground_truth,
    evaluate_records,
    extract_reference_points,
    load_case_manifest,
    load_class_protocol,
    prepare_records,
    read_metric_record,
    render_payloads,
    summarize_calibration,
    summarize_capacity,
    summarize_cases,
    summarize_gt_evidence,
    summarize_queries,
    summarize_strata,
    validate_metric_parity,
)
from validate_dotav2_q600_dump import load_cpu, validate_records  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the fail-closed analyzer command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dump', type=Path)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--official-metrics-json', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--training-metrics-json', type=Path)
    parser.add_argument('--training-step', type=int, default=24)
    parser.add_argument('--iou-threshold', type=float, default=.5)
    parser.add_argument('--sample-size', type=int, default=500000)
    parser.add_argument('--sample-seed', type=int, default=20260722)
    parser.add_argument('--max-error-examples', type=int, default=100)
    parser.add_argument('--case-dump', type=Path)
    parser.add_argument('--case-manifest', type=Path)
    parser.add_argument('--allow-noncanonical', action='store_true')
    parser.add_argument('--expected-records', type=int, default=13833)
    parser.add_argument('--queries-per-image', type=int, default=600)
    parser.add_argument('--num-classes', type=int, default=18)
    return parser


def _positive_integer(value, description: str) -> int:
    if type(value) is not int or value <= 0:
        raise DiagnosticError('{} must be a positive integer'.format(
            description))
    return value


def _nonnegative_integer(value, description: str) -> int:
    if type(value) is not int or value < 0:
        raise DiagnosticError('{} must be a nonnegative integer'.format(
            description))
    return value


def validate_mode(args: argparse.Namespace) -> bool:
    """Validate mode gates and return whether this is a canonical run."""
    try:
        values = vars(args)
    except TypeError as error:
        raise DiagnosticError('missing analyzer argument namespace') from error
    fields = (
        'dump', 'config', 'checkpoint', 'official_metrics_json',
        'output_dir', 'training_metrics_json', 'training_step',
        'iou_threshold', 'sample_size', 'sample_seed',
        'max_error_examples', 'case_dump', 'case_manifest',
        'allow_noncanonical', 'expected_records', 'queries_per_image',
        'num_classes',
    )
    required = {
        'dump', 'config', 'checkpoint', 'official_metrics_json',
        'output_dir',
    }
    missing = [
        field for field in fields
        if field not in values or (
            field in required and values[field] is None)
    ]
    if missing:
        raise DiagnosticError(
            'missing required analyzer field(s): {}'.format(
                ', '.join(missing)))
    _positive_integer(args.expected_records, 'expected records')
    _positive_integer(args.queries_per_image, 'queries per image')
    _positive_integer(args.num_classes, 'number of classes')
    _nonnegative_integer(args.sample_size, 'sample size')
    _nonnegative_integer(args.max_error_examples, 'max error examples')
    if type(args.sample_seed) is not int:
        raise DiagnosticError('sample seed must be an integer')
    if (type(args.iou_threshold) not in (int, float) or
            not 0.0 <= args.iou_threshold <= 1.0):
        raise DiagnosticError('IoU threshold must be in [0, 1]')
    if (args.case_dump is None) != (args.case_manifest is None):
        raise DiagnosticError(
            'case dump and case manifest must be provided together')

    defaults_match = (
        args.expected_records == 13833 and
        args.queries_per_image == 600 and
        args.num_classes == 18 and
        args.iou_threshold == .5
    )
    if not args.allow_noncanonical and not defaults_match:
        raise DiagnosticError(
            'noncanonical counts or IoU require --allow-noncanonical')
    canonical = not args.allow_noncanonical
    if canonical:
        if args.training_metrics_json is None:
            raise DiagnosticError(
                'canonical mode requires training metrics JSON')
        if type(args.training_step) is not int or args.training_step != 24:
            raise DiagnosticError('canonical training step must equal 24')
        if os.environ.get('CUDA_VISIBLE_DEVICES', ''):
            raise DiagnosticError(
                'canonical mode requires CUDA_VISIBLE_DEVICES to be unset')
    elif type(args.training_step) is not int:
        raise DiagnosticError('training step must be an integer')
    return canonical


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_no_replace(source: Path, destination: Path) -> None:
    """Atomically rename a directory without replacing an existing path."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, 'renameat2', None)
    if renameat2 is None:
        raise DiagnosticError(
            'atomic no-replace publication is unavailable')
    renameat2.argtypes = (
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise DiagnosticError(
            'output directory already exists: {}'.format(destination))
    raise OSError(error_number, os.strerror(error_number), str(destination))


def _preserve_failed_directory(temporary: Path, output: Path) -> Path:
    while True:
        failed = output.parent / '{}.{}.failed'.format(
            output.name, uuid.uuid4().hex)
        if not _path_occupied(failed):
            os.rename(temporary, failed)
            _fsync_directory(output.parent)
            return failed


def _path_occupied(path: Path) -> bool:
    return os.path.lexists(str(path))


def _materialize_expected_failure(directory, error, command) -> None:
    """Replace only this invocation's temporary contents with failure JSON."""
    directory = Path(directory)
    for child in directory.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()
    _write_bytes(directory, 'failure.json', _failure_bytes(error, command))
    _fsync_directory(directory)


@contextlib.contextmanager
def atomic_output_directory(output_dir, failure_command=None):
    """Yield a sibling temporary directory and atomically publish it."""
    output = Path(output_dir)
    if _path_occupied(output):
        raise DiagnosticError(
            'output directory already exists: {}'.format(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    if _path_occupied(output):
        raise DiagnosticError(
            'output directory already exists: {}'.format(output))
    temporary = Path(tempfile.mkdtemp(
        prefix=output.name + '.', dir=str(output.parent)))
    published = False
    try:
        yield temporary
        if _path_occupied(output):
            raise DiagnosticError(
                'output directory already exists: {}'.format(output))
        _rename_no_replace(temporary, output)
        published = True
        _fsync_directory(output.parent)
    except BaseException as original_error:
        preservation_source = output if published else temporary
        materialization_error = None
        if (isinstance(original_error, DiagnosticError) and
                failure_command is not None and
                _path_occupied(preservation_source)):
            try:
                _materialize_expected_failure(
                    preservation_source, original_error, failure_command)
            except BaseException as error:
                materialization_error = error
        if _path_occupied(preservation_source):
            try:
                _preserve_failed_directory(preservation_source, output)
            except BaseException as preservation_error:
                raise preservation_error from (
                    materialization_error or original_error)
        if materialization_error is not None:
            raise materialization_error from original_error
        raise


def _set_cpu_threads() -> None:
    try:
        if torch.get_num_threads() != 1:
            torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        if (torch.get_num_threads() != 1 or
                torch.get_num_interop_threads() != 1):
            raise RuntimeError('thread settings did not take effect')
    except (RuntimeError, TypeError, ValueError) as error:
        raise DiagnosticError(
            'failed to set torch CPU thread counts to 1') from error


def _file_provenance(path) -> dict:
    source = Path(path)
    try:
        resolved = source.expanduser().resolve(strict=True)
        metadata = resolved.stat()
        if not stat.S_ISREG(metadata.st_mode):
            raise DiagnosticError(
                'input must be a regular readable file: {}'.format(source))
        digest = hashlib.sha256()
        byte_count = 0
        with resolved.open('rb') as stream:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                byte_count += len(chunk)
    except DiagnosticError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise DiagnosticError(
            'input must be a regular readable file: {}'.format(source)
        ) from error
    return {
        'path': str(resolved),
        'sha256': digest.hexdigest(),
        'bytes': byte_count,
    }


def _verify_file_provenance(path, expected) -> None:
    """Reject path, target, byte-count, or SHA256 changes after loading."""
    try:
        actual = _file_provenance(path)
    except DiagnosticError as error:
        raise DiagnosticError(
            'input changed after provenance hash: {}'.format(path)) from error
    if actual != expected:
        raise DiagnosticError(
            'input changed after provenance hash: {}'.format(path))


def _load_verified_metric(path, step=None):
    """Resolve one metric source, then bind parsed values to its hash."""
    resolved_record = read_metric_record(path, step=step)
    source_path = Path(resolved_record.source)
    provenance = _file_provenance(source_path)
    record = read_metric_record(Path(provenance['path']), step=step)
    _verify_file_provenance(source_path, provenance)
    return record, provenance


def _command_for(args) -> list:
    command = getattr(args, '_command', None)
    if command is not None:
        return [str(value) for value in command]
    command = [
        sys.executable, str(Path(__file__).resolve()), str(args.dump),
        '--config', str(args.config),
        '--checkpoint', str(args.checkpoint),
        '--official-metrics-json', str(args.official_metrics_json),
        '--output-dir', str(args.output_dir),
    ]
    if args.training_metrics_json is not None:
        command.extend([
            '--training-metrics-json', str(args.training_metrics_json)])
    command.extend([
        '--training-step', str(args.training_step),
        '--iou-threshold', str(args.iou_threshold),
        '--sample-size', str(args.sample_size),
        '--sample-seed', str(args.sample_seed),
        '--max-error-examples', str(args.max_error_examples),
    ])
    if args.case_dump is not None:
        command.extend(['--case-dump', str(args.case_dump)])
    if args.case_manifest is not None:
        command.extend(['--case-manifest', str(args.case_manifest)])
    if args.allow_noncanonical:
        command.append('--allow-noncanonical')
    command.extend([
        '--expected-records', str(args.expected_records),
        '--queries-per-image', str(args.queries_per_image),
        '--num-classes', str(args.num_classes),
    ])
    return command


def _environment() -> dict:
    return {
        'cwd': str(Path.cwd().resolve()),
        'python': sys.version,
        'torch': str(torch.__version__),
        'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS'),
        'torch_num_threads': torch.get_num_threads(),
        'torch_num_interop_threads': torch.get_num_interop_threads(),
    }


def _write_bytes(directory: Path, name: str, content: bytes) -> None:
    if type(content) is not bytes:
        raise DiagnosticError('published payloads must be bytes')
    with (directory / name).open('xb') as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


_OUTCOMES = (
    'tp',
    'duplicate_fp',
    'semantic_fp',
    'localization_background_fp',
    'empty_tile_fp',
    'ignored_prediction',
)
_FALSE_OUTCOMES = _OUTCOMES[1:5]


def _region_summary(class_results, endpoint) -> dict:
    row_count = 0
    tp_count = 0
    fp_count = 0
    outcome_counts = {outcome: 0 for outcome in _OUTCOMES}
    for result in class_results:
        end = len(result.scores) if endpoint is None else int(
            getattr(result, endpoint))
        outcomes = np.asarray(result.outcomes[:end])
        row_count += end
        tp_count += int(np.sum(result.tp[:end]))
        fp_count += int(np.sum(result.fp[:end]))
        for outcome in _OUTCOMES:
            outcome_counts[outcome] += int(np.sum(outcomes == outcome))
    return {
        'row_count': row_count,
        'tp_count': tp_count,
        'fp_count': fp_count,
        'ignored_prediction_count': outcome_counts['ignored_prediction'],
        **outcome_counts,
        'shares': {
            outcome: (outcome_counts[outcome] / fp_count
                      if fp_count else None)
            for outcome in _FALSE_OUTCOMES
        },
    }


def _refinement_summary(evidence) -> dict:
    fields = (
        'prior_center_distance',
        'prior_scale_change',
        'prior_aspect_change',
        'prior_angle_change',
    )
    summary = {}
    for field in fields:
        values = np.asarray([
            float(getattr(row, field)) for row in evidence
            if getattr(row, field) is not None
        ], dtype=np.float64)
        summary[field] = {
            'count': int(len(values)),
            'mean': float(values.mean()) if len(values) else None,
            'median': float(np.median(values)) if len(values) else None,
        }
    return summary


def _group_summary(protocol, bundle, names) -> dict:
    name_set = set(names)
    class_ids = [
        class_id for class_id, name in enumerate(protocol.classes)
        if name in name_set
    ]
    included = [
        bundle.class_results[class_id] for class_id in class_ids
        if bundle.class_results[class_id].num_gts > 0
    ]
    mean_ap = (
        float(np.mean([result.ap for result in included]))
        if included else None)
    oracle_map = (
        float(np.mean([result.oracle_ap for result in included]))
        if included else None)
    return {
        'class_ids': class_ids,
        'class_names': [protocol.classes[class_id] for class_id in class_ids],
        'num_gts': int(sum(result.num_gts for result in included)),
        'map': mean_ap,
        'oracle_map': oracle_map,
        'headroom': (
            oracle_map - mean_ap if included else None),
    }


def _class_region(result, endpoint) -> dict:
    end = len(result.scores) if endpoint is None else int(
        getattr(result, endpoint))
    outcomes = np.asarray(result.outcomes[:end])
    return {
        outcome: int(np.sum(outcomes == outcome)) for outcome in _OUTCOMES
    }


def _per_class_rows(protocol, bundle, evidence, calibration) -> list:
    state_names = (
        'geometry_miss', 'semantic_miss', 'ownership_miss',
        'evaluator_reachable',
    )
    correlation = {
        row['class_id']: row['tp_score_iou_spearman']
        for row in calibration['per_class']
    }
    base_names = set(protocol.base_classes)
    rows = []
    for result in bundle.class_results:
        class_id = int(result.class_id)
        all_region = _class_region(result, None)
        last_region = _class_region(result, 'last_tp_end')
        support_region = _class_region(result, 'ap_support_end')
        state_counts = {
            state: sum(
                row.class_id == class_id and row.state == state
                for row in evidence)
            for state in state_names
        }
        row = {
            'class_id': class_id,
            'class_name': protocol.classes[class_id],
            'group': ('base' if protocol.classes[class_id] in base_names
                      else 'novel'),
            'num_gts': int(result.num_gts),
            'prediction_rows': len(result.scores),
            'tp_count': int(np.sum(result.tp)),
            'fp_count': int(np.sum(result.fp)),
            'recall': (float(result.recall[-1])
                       if len(result.recall) else 0.0),
            'ap': float(result.ap),
            'oracle_ap': float(result.oracle_ap),
            'oracle_headroom': float(result.oracle_ap - result.ap),
            'last_tp_end': int(result.last_tp_end),
            'ap_support_end': int(result.ap_support_end),
            **state_counts,
            'tp_score_iou_spearman': correlation[class_id],
        }
        for prefix, region in (
                ('all', all_region),
                ('through_last_tp', last_region),
                ('ap_support', support_region)):
            for outcome in _OUTCOMES:
                row['{}_{}'.format(prefix, outcome)] = region[outcome]
        rows.append(row)
    return rows


def _error_examples(bundle):
    for record_index, record in enumerate(bundle.records):
        for query_id, outcome in enumerate(bundle.row_outcomes[record_index]):
            outcome = str(outcome)
            if outcome not in _FALSE_OUTCOMES:
                continue
            yield {
                'outcome': outcome,
                'img_id': str(record.img_id),
                'query_id': query_id,
                'score': float(record.scores[query_id]),
                'class_id': int(record.labels[query_id]),
                'matched_gt': int(bundle.row_matched_gt[
                    record_index][query_id]),
                'assigned_iou': float(bundle.row_assigned_iou[
                    record_index][query_id]),
                'same_label_iou': float(bundle.row_same_label_iou[
                    record_index][query_id]),
            }


def _load_checkpoint(path, expected_queries, canonical):
    try:
        checkpoint = torch.load(path, map_location='cpu')
        try:
            references = extract_reference_points(
                checkpoint, expected_queries=expected_queries,
                canonical=canonical)
        finally:
            del checkpoint
        return references
    except DiagnosticError:
        raise
    except MemoryError:
        raise
    except Exception as error:
        raise DiagnosticError('failed to load checkpoint on CPU') from error


def _load_validated_dump(path, expected_records, queries, classes):
    try:
        raw = load_cpu(path)
        try:
            integrity = validate_records(
                raw, expected_records=expected_records,
                queries_per_image=queries, num_classes=classes)
        except AssertionError as error:
            raise DiagnosticError(
                'dump record validation failed: {}'.format(error)) from error
        prepared = prepare_records(
            raw, queries_per_image=queries, num_classes=classes)
        del raw
        expected_rows = expected_records * queries
        actual_rows = sum(len(record.scores) for record in prepared)
        if len(prepared) != expected_records:
            raise DiagnosticError(
                'dump record count mismatch: expected {}, got {}'.format(
                    expected_records, len(prepared)))
        if actual_rows != expected_rows:
            raise DiagnosticError(
                'dump prediction row count mismatch: expected {}, got {}'.
                format(expected_rows, actual_rows))
        expected_integrity = {
            'records': expected_records,
            'unique_image_ids': expected_records,
            'prediction_rows': expected_rows,
            'all_cpu_finite': True,
        }
        if (not isinstance(integrity, Mapping) or
                any(integrity.get(key) != value
                    for key, value in expected_integrity.items())):
            raise DiagnosticError(
                'dump validator integrity is inconsistent with prepared '
                'records')
        return prepared, integrity
    except DiagnosticError:
        raise
    except MemoryError:
        raise
    except Exception as error:
        raise DiagnosticError('failed to load prediction dump on CPU') from error


def _validate_canonical_case_records(case_raw, queries, classes):
    """Apply an optimized-interpreter-safe 2xQ canonical case gate."""
    try:
        try:
            integrity = validate_records(
                case_raw, expected_records=2,
                queries_per_image=queries, num_classes=classes)
        except AssertionError as error:
            raise DiagnosticError(
                'case dump validation failed: {}'.format(error)) from error
        prepared = prepare_records(
            case_raw, queries_per_image=queries, num_classes=classes)
        actual_rows = sum(len(record.scores) for record in prepared)
        if len(prepared) != 2:
            raise DiagnosticError(
                'canonical case record count mismatch: expected 2, got {}'.
                format(len(prepared)))
        if actual_rows != 2 * queries:
            raise DiagnosticError(
                'canonical case prediction row count mismatch: expected {}, '
                'got {}'.format(2 * queries, actual_rows))
        expected_integrity = {
            'records': 2,
            'unique_image_ids': 2,
            'prediction_rows': 2 * queries,
            'all_cpu_finite': True,
        }
        if (not isinstance(integrity, Mapping) or
                any(integrity.get(key) != value
                    for key, value in expected_integrity.items())):
            raise DiagnosticError(
                'canonical case validator integrity is inconsistent with '
                'prepared records')
        return prepared
    except DiagnosticError:
        raise
    except MemoryError:
        raise
    except Exception as error:
        raise DiagnosticError(
            'failed to validate canonical case dump') from error


def _failure_bytes(error, command) -> bytes:
    message = ' '.join(str(error).splitlines())
    payload = {
        'schema_version': 1,
        'canonical': False,
        'error': message,
        'command': command,
    }
    try:
        return (json.dumps(
            payload, ensure_ascii=False, sort_keys=True, indent=2,
            allow_nan=False) + '\n').encode('utf-8')
    except (TypeError, ValueError) as serialization_error:
        raise DiagnosticError(
            'failure metadata must be JSON-safe') from serialization_error


def run(args: argparse.Namespace):
    """Orchestrate existing CPU diagnostics and atomically publish outputs."""
    canonical = validate_mode(args)
    output = Path(args.output_dir)
    if _path_occupied(output):
        raise DiagnosticError(
            'output directory already exists: {}'.format(output))
    command = _command_for(args)
    with atomic_output_directory(
            output, failure_command=command) as temporary:
        try:
            _set_cpu_threads()
            config_provenance = _file_provenance(args.config)
            protocol = load_class_protocol(args.config, canonical=canonical)
            _verify_file_provenance(args.config, config_provenance)
            try:
                small_vehicle_class_id = protocol.classes.index(
                    'small-vehicle')
            except ValueError as error:
                raise DiagnosticError(
                    'class protocol must contain exactly small-vehicle') from error
            if protocol.classes.count('small-vehicle') != 1:
                raise DiagnosticError(
                    'class protocol must contain exactly small-vehicle')

            official, official_provenance = _load_verified_metric(
                args.official_metrics_json)
            training = None
            training_provenance = None
            if args.training_metrics_json is not None:
                training, training_provenance = _load_verified_metric(
                    args.training_metrics_json, step=args.training_step)

            provenance = {
                'config': config_provenance,
                'checkpoint': _file_provenance(args.checkpoint),
                'dump': _file_provenance(args.dump),
                'official_metrics': official_provenance,
            }
            if training is not None:
                provenance['training_metrics'] = training_provenance
            if args.case_dump is not None:
                provenance['case_dump'] = _file_provenance(args.case_dump)
                provenance['case_manifest'] = _file_provenance(
                    args.case_manifest)

            references = _load_checkpoint(
                args.checkpoint, args.queries_per_image, canonical)
            _verify_file_provenance(
                args.checkpoint, provenance['checkpoint'])
            prepared, integrity = _load_validated_dump(
                args.dump, args.expected_records, args.queries_per_image,
                args.num_classes)
            _verify_file_provenance(args.dump, provenance['dump'])
            integrity.update({
                'ordinary_gt': int(sum(len(row.gt_boxes)
                                       for row in prepared)),
                'ignored_gt': int(sum(len(row.ignored_boxes)
                                      for row in prepared)),
                'classes': args.num_classes,
                'queries': args.queries_per_image,
            })

            bundle = evaluate_records(
                prepared, num_classes=args.num_classes,
                iou_threshold=args.iou_threshold)
            parity = validate_metric_parity(
                bundle.mean_ap, official, training)
            evidence = decompose_ground_truth(
                prepared, iou_threshold=args.iou_threshold)
            attached = attach_refinement(evidence, prepared, references)
            gt_decomposition = summarize_gt_evidence(attached)
            gt_decomposition['refinement'] = _refinement_summary(attached)
            image_gt_counts = [len(row.gt_boxes) for row in prepared]
            class_image_gt_counts = {
                class_id: [
                    int(np.sum(row.gt_labels == class_id))
                    for row in prepared
                ]
                for class_id in range(args.num_classes)
            }
            capacity = summarize_capacity(
                image_gt_counts, class_image_gt_counts,
                args.queries_per_image)
            calibration = summarize_calibration(
                bundle, sample_size=args.sample_size,
                seed=args.sample_seed,
                iou_threshold=args.iou_threshold)
            queries = summarize_queries(
                bundle, attached, query_count=args.queries_per_image,
                num_classes=args.num_classes)
            strata = summarize_strata(
                attached, protocol, small_vehicle_class_id)

            if args.case_dump is None:
                cases = summarize_cases(None, None, small_vehicle_class_id)
            else:
                case_manifest = load_case_manifest(args.case_manifest)
                _verify_file_provenance(
                    args.case_manifest, provenance['case_manifest'])
                try:
                    case_raw = load_cpu(args.case_dump)
                    _verify_file_provenance(
                        args.case_dump, provenance['case_dump'])
                    if canonical:
                        _validate_canonical_case_records(
                            case_raw, args.queries_per_image,
                            args.num_classes)
                    cases = summarize_cases(
                        case_raw, case_manifest, small_vehicle_class_id)
                    del case_raw
                except DiagnosticError:
                    raise
                except MemoryError:
                    raise
                except Exception as error:
                    raise DiagnosticError(
                        'failed to load optional case dump on CPU') from error

            gt_classes = [
                result for result in bundle.class_results
                if result.num_gts > 0
            ]
            oracle_map = float(np.mean([
                result.oracle_ap for result in gt_classes
            ])) if gt_classes else 0.0
            warnings = []
            if parity.get('replay_delta_warning'):
                warnings.append(
                    'replay_delta_warning: official and training metrics differ')
            if cases[0]['case_gate'] == 'not_run':
                warnings.append(
                    'case studies not_run; diagnostic bundle is incomplete')
            fp_all = _region_summary(bundle.class_results, None)
            fp_last = _region_summary(
                bundle.class_results, 'last_tp_end')
            fp_support = _region_summary(
                bundle.class_results, 'ap_support_end')
            base = _group_summary(
                protocol, bundle, protocol.base_classes)
            novel = _group_summary(
                protocol, bundle, protocol.novel_classes)
            summary = build_diagnostics(
                canonical=canonical,
                integrity=integrity,
                parity=parity,
                reconstructed_map=bundle.mean_ap,
                oracle_map=oracle_map,
                gt_decomposition=gt_decomposition,
                capacity=capacity,
                fp_all=fp_all,
                fp_last_tp=fp_last,
                fp_ap_support=fp_support,
                calibration=calibration,
                query_summary=queries,
                base_summary=base,
                novel_summary=novel,
                case_gate=cases,
                warnings=warnings,
            )
            tables = {
                'per_class.csv': _per_class_rows(
                    protocol, bundle, attached, calibration),
                'strata.csv': strata,
                'per_query.csv': queries['rows'],
                'case_studies.csv': cases,
                'error_examples.csv': cap_error_examples(
                    _error_examples(bundle), args.max_error_examples),
            }
            payloads = render_payloads(summary, tables)
            manifest = build_manifest(
                payloads, provenance, command, _environment())
            for name in NON_MANIFEST_PAYLOADS:
                _write_bytes(temporary, name, payloads[name])
            _write_bytes(temporary, 'manifest.json', manifest)
            _fsync_directory(temporary)
        except DiagnosticError as error:
            _write_bytes(
                temporary, 'failure.json', _failure_bytes(error, command))
            _fsync_directory(temporary)
            raise


def main(argv=None) -> int:
    """Parse arguments and publish a diagnostic bundle."""
    try:
        raw_argv = list(sys.argv[1:] if argv is None else argv)
        args = build_parser().parse_args(raw_argv)
        args._command = [
            sys.executable, str(Path(__file__).resolve()), *raw_argv]
        run(args)
    except DiagnosticError as error:
        print(' '.join(str(error).splitlines()), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
