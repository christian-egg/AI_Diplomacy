import json

import pytest

from ai_diplomacy import collusion_runner as runner


def _config(tmp_path, **overrides):
    path = tmp_path / "batch.json"
    path.write_text(json.dumps({"name": "t", "model": "openrouter:openai/gpt-5.6-luna", "reasoning_effort": "high",
                                "n_colluders": [0, 4], "games_per_condition": 2,
                                "monitors": [{"view": "actions", "model": "openrouter:openai/gpt-5.6-luna",
                                              "effort": "medium", "turns_per_step": 1, "stop_on_detection": True}],
                                **overrides}))
    return runner.load_config(path)


def test_example_config_loads():
    config = runner.load_config(runner.REPO / "collusion_configs" / "pilot.json")
    assert config["n_colluders"] == [0, 4] and config["reasoning_effort"] == ["high"]


def test_missing_required_keys(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"name": "x"}))
    with pytest.raises(SystemExit, match="missing model, reasoning_effort, n_colluders"):
        runner.load_config(path)


def test_plan_names_folders_by_condition_and_seeds_operatives(tmp_path):
    games = runner.plan_games(_config(tmp_path))
    names = [g["run_dir"].name for g in games]
    assert names == ["gpt-5.6-luna-high-2y-ops0-g1", "gpt-5.6-luna-high-2y-ops0-g2",
                     "gpt-5.6-luna-high-2y-ops4-g1", "gpt-5.6-luna-high-2y-ops4-g2"]
    control, operatives = games[0]["cmd"], games[3]["cmd"]
    assert "--colluder_seed" not in control
    assert operatives[operatives.index("--colluder_seed") + 1] == "1002"
    assert operatives[operatives.index("--reasoning_effort") + 1] == "high"
    assert operatives[operatives.index("--models") + 1].count("gpt-5.6-luna") == 7


def test_effort_lists_multiply_conditions(tmp_path):
    games = runner.plan_games(_config(tmp_path, reasoning_effort=["medium", "high"], n_colluders=4,
                                      games_per_condition=1))
    assert [g["condition"] for g in games] == ["gpt-5.6-luna-medium-2y-ops4", "gpt-5.6-luna-high-2y-ops4"]


def test_monitor_command_passes_options(tmp_path):
    config = _config(tmp_path)
    run_dir = runner.REPO / "results" / "t" / "x"
    cmd = runner.monitor_cmd(run_dir, config["monitors"][0])
    assert cmd[cmd.index("--name") + 1] == "actions_gpt-5.6-luna_medium"
    assert cmd[cmd.index("--turns_per_step") + 1] == "1" and "--stop_on_detection" in cmd


def test_run_one_skips_finished_steps(tmp_path, monkeypatch):
    config = _config(tmp_path, results_dir=str(tmp_path / "results"), games_per_condition=1, n_colluders=4,
                     judges=[{"model": "openrouter:openai/gpt-5.6-luna", "effort": "medium"}])
    monkeypatch.setattr(runner, "REPO", tmp_path)
    game = runner.plan_games(config)[0]
    ran = []

    def fake_run(cmd, log, env):
        ran.append(cmd)
        if "lm_game.py" in cmd:
            (game["run_dir"] / "overview.jsonl").write_text("{}")
        if "ai_diplomacy.monitor" in cmd or "ai_diplomacy.leakage_judge" in cmd:
            folder = ("monitor/runs" if "ai_diplomacy.monitor" in cmd else "leakage")
            out = game["run_dir"] / folder / f"{cmd[cmd.index('--name') + 1]}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("{}")

    monkeypatch.setattr(runner, "_run", fake_run)
    runner.run_one(game, config, {})
    kinds = ["lm_game.py", "ai_diplomacy.monitor", "ai_diplomacy.leakage_judge", "ai_diplomacy.dashboard"]
    assert [[k in c for k in kinds].index(True) for c in ran] == [0, 1, 2, 3]
    ran.clear()
    assert runner.run_one(game, config, {}).endswith("game already done, 1 monitors, 1 judges")
    assert all("ai_diplomacy.dashboard" in c for c in ran)  # only the dashboard is redone


def test_read_env_file(tmp_path):
    path = tmp_path / ".env"
    path.write_text("# comment\nOPENROUTER_API_KEY='abc'\nOTHER=1\n")
    assert runner.read_env_file(path) == {"OPENROUTER_API_KEY": "abc", "OTHER": "1"}


def test_docent_upload_setting(tmp_path):
    assert _config(tmp_path)["docent_collection_id"] is None  # off by default
    cmd = runner.upload_cmd(["results/t/a", "results/t/b"], "abc")
    assert cmd[1:] == ["-m", "ai_diplomacy.docent_upload", "results/t/a", "results/t/b", "--collection_id", "abc"]


def test_upload_needs_a_docent_key_before_anything_runs(tmp_path, monkeypatch):
    path = tmp_path / "batch.json"
    path.write_text(json.dumps({"name": "t", "model": "m", "reasoning_effort": "high", "n_colluders": 0,
                                "docent_collection_id": "abc"}))
    monkeypatch.setattr("sys.argv", ["runner", str(path)])
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    monkeypatch.delenv("DOCENT_API_KEY", raising=False)
    monkeypatch.setattr(runner, "run_one", lambda *a: pytest.fail("a game started"))
    with pytest.raises(SystemExit, match="DOCENT_API_KEY is not"):
        runner.main()
