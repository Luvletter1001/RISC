from experiments.rotation_semantic_attractor.scripts import build_report_helpers as helpers
from experiments.rotation_semantic_attractor.src.utils.status import ExperimentStatus


def test_report_filters_proxy_and_schema_rows_from_scientific_tables():
    rows = [
        {"model_name": "full", "status": ExperimentStatus.DONE_FULL, "value": 1},
        {"model_name": "smoke", "status": ExperimentStatus.DONE_SMOKE, "value": 2},
        {"model_name": "proxy", "status": ExperimentStatus.SMOKE_PROXY, "value": 3},
        {"model_name": "schema", "status": ExperimentStatus.SCHEMA_ONLY, "value": 4},
        {"model_name": "na", "status": ExperimentStatus.NOT_APPLICABLE, "value": 5},
    ]

    main_rows = helpers.scientific_rows(rows, include_smoke=False)
    assert [row["model_name"] for row in main_rows] == ["full"]

    smoke_rows = helpers.scientific_rows(rows, include_smoke=True)
    assert [row["model_name"] for row in smoke_rows] == ["full", "smoke"]


def test_status_summary_counts_failed_only_as_failures():
    rows = [
        {"status": ExperimentStatus.NOT_APPLICABLE},
        {"status": ExperimentStatus.NOT_SELECTED_IN_THIS_SMOKE},
        {"status": ExperimentStatus.UNSUPPORTED_BY_CURRENT_CODE},
        {"status": ExperimentStatus.SMOKE_PROXY},
        {"status": ExperimentStatus.FAILED},
    ]

    counts = helpers.status_summary_counts(rows)

    assert counts["failure_count"] == 1
    assert counts["not_applicable_count"] == 1
    assert counts["not_selected_count"] == 1
    assert counts["unsupported_count"] == 1
    assert counts["proxy_count"] == 1


def test_report_text_uses_not_applicable_language_for_closed_set_query_logits():
    stage_rows = [
        {
            "model_name": "rotated_retinanet_msrr",
            "model_family": "closed_set",
            "architecture_type": "dense_head",
            "query_logits_status": ExperimentStatus.NOT_APPLICABLE,
            "query_logits_reason": "closed-set dense/head detector does not use query logits",
            "status": ExperimentStatus.DONE_SMOKE,
        }
    ]

    text = helpers.render_stage_decomposition_section(stage_rows)

    assert "query_logits_not_available_for_closed_set" not in text
    assert "query logits: N/A by architecture" in text
