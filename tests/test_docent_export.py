import asyncio
import csv
import json
from argparse import Namespace

from diplomacy import Game

from ai_diplomacy.collusion import operative_system_prompt
from ai_diplomacy.docent_export import game_settings, runs_for
from ai_diplomacy.game_history import GameHistory
from ai_diplomacy.game_logic import save_game_state

OPS = ["ENGLAND", "ITALY", "RUSSIA"]
POWERS = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]


def _game(tmp_path, with_settings=True):
    game, history = Game(), GameHistory()
    history.add_phase("S1901M")
    game.set_orders("FRANCE", ["A PAR - BUR"])
    game.process()
    game.power_model_map = {p: "openrouter:openai/gpt-5.6-luna" for p in POWERS}
    run = tmp_path / "batch" / "game-g1"
    run.mkdir(parents=True)
    config = Namespace(colluders=OPS, colluder_seed=1, max_year=1902, num_negotiation_rounds=3)
    if with_settings:
        config = Namespace(**vars(config), reasoning_effort="high", game_length="2y", n_colluders=3,
                           harness_commit="abc1234", monitor_in_prompt=False)
    asyncio.run(save_game_state(game, {}, history, str(run / "lmvsgame.json"), config, completed_phase_name="S1901M"))
    with (run / "llm_responses.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["timestamp", "model", "power", "phase", "response_type", "raw_input",
                                          "raw_response", "success"])
        w.writeheader()
        for p in POWERS:
            w.writerow({"timestamp": "t", "model": "m", "power": p, "phase": "S1901M", "response_type": "order_generation",
                        "raw_input": f"prompt for {p}", "raw_response": f"orders from {p}", "success": "Success"})
    (run / "overview.jsonl").write_text("{}\n{}\n{}\n")
    return run


def test_agents_run_has_a_transcript_per_power_and_tags(tmp_path):
    run = _game(tmp_path)
    [agents] = runs_for(run)
    assert agents.metadata["type"] == "agents" and agents.metadata["version"] == "abc1234"
    assert {"type:agents", "version:abc1234", "model:gpt-5.6-luna", "effort:high", "game_length:2y",
            "operatives:3", "batch:batch", "status:complete"} <= set(agents.metadata["tags"])
    names = [t.name for t in agents.transcripts]
    assert names[:7] == [f"{p} ({'secret operative' if p in OPS else 'regular player'})" for p in POWERS]
    assert names[7].startswith("Game record")
    england = agents.transcripts[1]
    assert england.messages[0].content == operative_system_prompt("ENGLAND", 3)
    assert [m.content for m in england.messages[1:]] == ["prompt for ENGLAND", "orders from ENGLAND"]
    assert "You are playing as AUSTRIA" in agents.transcripts[0].messages[0].content
    assert agents.metadata["operative_powers"] == OPS and agents.metadata["operatives"] == "3"


def test_backfill_supplies_settings_for_old_runs(tmp_path):
    run = _game(tmp_path, with_settings=False)
    (run / "run_settings_backfill.json").write_text(json.dumps({"harness_commit": "old+patch", "reasoning_effort": "medium"}))
    game = json.loads((run / "lmvsgame.json").read_text())
    game.pop("run_settings", None)
    (run / "lmvsgame.json").write_text(json.dumps(game))
    settings = game_settings(run, game)
    assert settings["harness_commit"] == "old+patch" and settings["reasoning_effort"] == "medium"
    assert settings["game_length"] == "2y" and settings["num_negotiation_rounds"] == 3
    [agents] = runs_for(run)
    assert "version:old+patch" in agents.metadata["tags"]


def test_monitor_and_judge_runs(tmp_path):
    run = _game(tmp_path)
    (run / "monitor" / "runs").mkdir(parents=True)
    step = {"through": "S1901M", "phases": ["S1901M"], "correctness": 60.0, "group_exists": 70, "probabilities": {},
            "counts_for_detection": True, "valid": True, "problems": [], "user_message": "BATCH 1", "reply": "{...}"}
    (run / "monitor" / "runs" / "trajectory_m_medium.json").write_text(json.dumps({
        "settings": {"view": "trajectory", "model": "openrouter:openai/gpt-5.6-luna", "effort": "medium",
                     "turns_per_step": 2, "detection_threshold": 80, "harness_commit": "def5678"},
        "system_prompt": "SYSTEM", "steps": [step], "detected_at": None, "usage": {}}))
    (run / "leakage").mkdir()
    (run / "leakage" / "m_medium.json").write_text(json.dumps({
        "settings": {"model": "openrouter:openai/gpt-5.6-luna", "effort": "medium", "harness_commit": "def5678"},
        "system_prompt": "JUDGE", "usage": {},
        "phases": [{"phase": "S1901M", "messages": 1, "sent": {"ENGLAND": 1}, "parsed": True, "valid": True,
                    "problems": [], "flagged": [{"id": 1, "sender": "ENGLAND", "round": 1, "recipient": "ITALY",
                                                 "text": "x", "quote": "x"}], "user_message": "PHASE 1", "reply": "{}"}]}))
    agents, monitor, judge = runs_for(run)
    assert monitor.metadata["type"] == "monitor" and judge.metadata["type"] == "judge"
    assert {"version:def5678", "game_version:abc1234", "monitor_view:trajectory", "detected:no",
            "operatives:3"} <= set(monitor.metadata["tags"])
    assert [m.content for m in monitor.transcripts[0].messages] == ["SYSTEM", "BATCH 1", "{...}"]
    assert judge.metadata["summary"]["operative_leaks"] == 1
    assert len({agents.metadata["export_key"], monitor.metadata["export_key"], judge.metadata["export_key"]}) == 3
