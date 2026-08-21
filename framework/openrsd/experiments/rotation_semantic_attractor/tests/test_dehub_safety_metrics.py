from experiments.rotation_semantic_attractor.src.metrics.dehub_safety import (
    class_distribution_from_histograms,
    js_kl_divergence,
)


def test_js_kl_divergence_is_zero_for_identical_distributions():
    js, kl = js_kl_divergence({"small-vehicle": 0.5, "ship": 0.5}, {"small-vehicle": 0.5, "ship": 0.5})
    assert js == 0.0
    assert kl == 0.0


def test_class_distribution_from_histograms_normalizes_counts():
    dist = class_distribution_from_histograms([
        {"class_histogram": '{"small-vehicle": 3, "ship": 1}'},
        {"class_histogram": {"ship": 2}},
    ])
    assert dist["small-vehicle"] == 0.5
    assert dist["ship"] == 0.5
