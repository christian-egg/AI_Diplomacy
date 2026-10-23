import json

import pytest

from ai_diplomacy import lineage_runner as lr
from ai_diplomacy.collusion import POWERS


@pytest.fixture(autouse=True)
def _repo(tmp_path, monkeypatch, request):
    if request.node.name != "test_example_config_loads":
        monkeypatch.setattr(lr, "REPO", tmp_path)  # game folders and commands are relative to the repo


def _config(tmp_path, **overrides):
    path = tmp_path / "icrl.json"
    path.write_text(json.dumps({
        "name": "t", "model": "openrouter:openai/gpt-5.6-luna", "reasoning_effort": "high",
        "lineages": 2, "rounds": 3,
        "feedback_monitor": {"view": "trajectory", "model": "openrouter:openai/gpt-5.6-luna", "effort": "medium",
                             "knows_objective": True}, **overrides}))
    return lr.load_config(path)


def test_example_config_loads():
    config = lr.load_config(lr.REPO / "collusion_configs" / "icrl_2y.json")
    assert config["game_length"] == "2y" and config["lineages"] == 5
    assert config["reflection"]["note_words"] == 150


def test_powers_are_dealt_to_all_seven_agents_per_seed():
    a, b = lr.deal_powers(5101), lr.deal_powers(5101)
    assert a == b and sorted(a) == list(range(1, 8)) and sorted(a.values()) == sorted(POWERS)
    assert lr.deal_powers(5101) != lr.deal_powers(5102)


def test_seeds_differ_by_lineage_and_round(tmp_path):
    config = _config(tmp_path)
    seeds = {lr.game_seed(config, l, r) for l in (1, 2) for r in (1, 2, 3)}
    assert len(seeds) == 6


def _finish(config, lineage, rnd, notes_by_agent):
    """Pretend a round was played: lineage.json plus reflections for the operatives."""
    game = lr.prepare_game(config, lineage, rnd)
    powers = json.loads((game["run_dir"] / "lineage.json").read_text())["powers"]
    (game["run_dir"] / "reflections").mkdir(exist_ok=True)
    for agent, note in notes_by_agent.items():
        (game["run_dir"] / "reflections" / f"{powers[str(agent)]}.json").write_text(json.dumps({"carried_note": note}))
    return game


def test_notes_follow_the_agent_not_the_power(tmp_path):
    config = _config(tmp_path)
    _finish(config, 1, 1, {1: "a1 r1", 2: "a2 r1", 3: "a3 r1", 4: "a4 r1"})
    _finish(config, 1, 2, {1: "a1 r2", 3: "a3 r2"})  # agents 2 and 4 failed to reflect: keep r1 notes
    assert lr.notes_after(config, 1, 2) == {1: "a1 r2", 2: "a2 r1", 3: "a3 r2", 4: "a4 r1"}
    assert lr.notes_after(config, 2, 2) == {}  # lineages never share notes
    game = lr.prepare_game(config, 1, 3)
    info = json.loads((game["run_dir"] / "lineage.json").read_text())
    by_power = json.loads((game["run_dir"] / "operative_notes.json").read_text())
    assert by_power == {info["powers"][str(a)]: n for a, n in lr.notes_after(config, 1, 2).items()}
    assert set(by_power) == {info["powers"][str(a)] for a in lr.OPERATIVE_AGENTS}


def test_game_command_names_the_operatives(tmp_path):
    config = _config(tmp_path, extra_game_args=["--monitor_in_prompt"])
    game = lr.prepare_game(config, 2, 1)
    cmd, powers = game["cmd"], lr.deal_powers(lr.game_seed(config, 2, 1))
    assert cmd[cmd.index("--colluders") + 1] == ",".join(sorted(powers[a] for a in lr.OPERATIVE_AGENTS))
    assert cmd[cmd.index("--n_colluders") + 1] == "4" and "--colluder_seed" not in cmd
    assert cmd[cmd.index("--operative_notes") + 1].endswith("lin2-r01/operative_notes.json")
    assert cmd[-1] == "--monitor_in_prompt"


def test_reflection_uses_the_feedback_monitor(tmp_path):
    config = _config(tmp_path)
    cmd = lr.reflection_cmd(lr.run_dir_for(config, 1, 1), config)
    assert cmd[cmd.index("--monitor") + 1] == "trajectory_gpt-5.6-luna_medium_knows"
    assert cmd[cmd.index("--effort") + 1] == "high"
