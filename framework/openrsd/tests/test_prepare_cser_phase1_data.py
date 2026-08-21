from M_Tools.analysis.prepare_cser_phase1_data import (
    select_source_stems,
    select_support_classes,
)


def test_select_source_stems_requires_image_and_pkl_and_excludes_heldout():
    image_names = ["a.png", "b.png", "c.png", "only_image.png"]
    archive_names = [
        "a.pkl", "b.pkl", "nested/c.pkl", "only_pkl.pkl", "README.txt"]
    heldout = {"b"}

    selected = select_source_stems(image_names, archive_names, heldout)

    assert selected == ["a", "c"]


def test_select_source_stems_is_deterministic_and_suffix_strict():
    image_names = ["z.png", "a.jpg", "a.png", "z.png.bak"]
    archive_names = ["z.pkl", "a.pickle", "a.pkl.bak"]

    selected = select_source_stems(image_names, archive_names, set())

    assert selected == ["z"]


def test_select_support_classes_preserves_requested_dota1_order():
    support = {
        "ship": {"visual_embeds": [1]},
        "helipad": {"visual_embeds": [2]},
        "plane": {"visual_embeds": [3]},
    }

    selected = select_support_classes(support, ["plane", "ship"])

    assert list(selected) == ["plane", "ship"]
    assert selected["plane"] is support["plane"]
