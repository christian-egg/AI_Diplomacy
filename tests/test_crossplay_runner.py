import json

import pytest

from ai_diplomacy import crossplay_runner as cp
from ai_diplomacy import lineage_runner as lr


def test_agents_rotate_within_each_lineage():
    assert cp.pick_agents([3, 3, 5, 5], 1) == [(3, 1), (3, 2), (5, 1), (5, 2)]
    assert cp.pick_agents([3, 3, 5, 5], 2) == [(3, 2), (3, 3), (5, 2), (5, 3)]
    assert cp.pick_agents([3, 3, 3, 3], 3) == [(3, 3), (3, 4), (3, 1), (3, 2)]  # a home game uses all four


def test_example_config_plans_distinct_games():
    config = cp.load_config(cp.REPO / "collusion_configs" / "crossplay_2y.json")
    games = cp.plan(config)
    assert len(games) == 12 and len({g["seed"] for g in games}) == 12
    mixed = [g for g in games if g["condition"] == "mixed35"]
    assert all(sorted(l for l, _ in g["operatives"].values()) == [3, 3, 5, 5] for g in mixed)
    assert all(len(g["operatives"]) == 4 for g in games)


def test_pair_scores_split_same_and_cross_lineage():
    origin = {"ENGLAND": {"lineage": 3}, "ITALY": {"lineage": 3}, "RUSSIA": {"lineage": 5}, "TURKEY": {"lineage": 5}}
    labels = {"ITALY": "near-certain operative", "RUSSIA": "unknown", "TURKEY": "likely non-operative",
              "AUSTRIA": "unknown", "FRANCE": "unknown", "GERMANY": "unknown"}
    game = {"phases": [{"name": "S1901M", "scratchpads": {"ENGLAND": {"parsed": True, "assessments": labels}}}]}
    scores = cp.pair_scores(game, origin)
    assert scores["same"] == [1.0]
    assert scores["cross"] == pytest.approx([0.0, -2 / 3])


def test_bad_condition_is_rejected(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"name": "x", "source": "collusion_configs/icrl2_2y.json",
                                "conditions": {"bad": [3, 3, 5]}}))
    with pytest.raises(SystemExit, match="needs 4"):
        cp.load_config(path)
