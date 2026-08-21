from __future__ import annotations

from typing import Iterable, Mapping

from experiments.rotation_semantic_attractor.src.utils.status import ExperimentStatus, can_be_done_full, count_statuses


def scientific_rows(rows: Iterable[Mapping], *, include_smoke: bool = False) -> list[Mapping]:
    selected = []
    for row in rows:
        status = row.get("status")
        if status == ExperimentStatus.DONE_FULL:
            family = row.get("experiment_family") or row.get("family") or "generic"
            if can_be_done_full(str(family), row):
                selected.append(row)
        elif include_smoke and status == ExperimentStatus.DONE_SMOKE:
            selected.append(row)
    return selected


def status_summary_counts(rows: Iterable[Mapping]) -> dict[str, int]:
    return count_statuses(rows)


def render_stage_decomposition_section(rows: list[Mapping]) -> str:
    lines = ["## Stage Decomposition", ""]
    if not rows:
        return "\n".join(lines + ["No stage decomposition rows available.", ""])
    lines.extend(
        [
            "| model_name | family | architecture | dense logits | query logits | pre-NMS | post-NMS | status |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        query = row.get("query_logits_status", "")
        if query == ExperimentStatus.NOT_APPLICABLE:
            query_text = "query logits: N/A by architecture"
        else:
            query_text = str(query)
        lines.append(
            "| {model} | {family} | {arch} | {dense} | {query} | {pre} | {post} | {status} |".format(
                model=row.get("model_name", ""),
                family=row.get("model_family", ""),
                arch=row.get("architecture_type", ""),
                dense=row.get("dense_logits_status", ""),
                query=query_text,
                pre=row.get("pre_nms_status", ""),
                post=row.get("post_nms_status", ""),
                status=row.get("status", ""),
            )
        )
    return "\n".join(lines) + "\n"
