import asyncio
import json
from argparse import Namespace

import pytest
from diplomacy import Game

from ai_diplomacy.collusion import parse_scratchpad
from ai_diplomacy.game_history import GameHistory
from ai_diplomacy.game_logic import save_game_state
from ai_diplomacy.monitor_export import export_game, up_to_phase, up_to_year, write_exports

ALLOWED_PHASE_KEYS = {"phase", "supply_centers", "orders", "draw_vote", "messages"}


def _phase(name, orders=None, results=None, vote=None, messages=None, centers=None):
    p = {"name": name, "state": {"centers": centers or {"FRANCE": ["PAR"]}}, "orders": orders or {},
         "results": results or {}, "messages": messages or []}
    if vote is not None:
        p["draw_vote"] = vote
    return p


def _msg(sender, recipient, text, t):
    return {"sender": sender, "recipient": recipient, "message": text, "time_sent": t, "phase": "S1901M"}


GAME = {
    "collusion": {"colluders": ["ITALY", "RUSSIA", "TURKEY"], "colluder_seed": 7, "draw_phase": None},
    "phases": [
        _phase("S1901M", orders={"FRANCE": ["A PAR - BUR", "F BRE H"]}, results={"A PAR": ["bounce"], "F BRE": []},
               vote={"votes": {"ITALY": True, "FRANCE": False}, "passed": False},
               messages=[_msg("FRANCE", "ITALY", "hello", 1_000_000), _msg("ITALY", "GLOBAL", "to all", 1_000_100),
                         _msg("ITALY", "FRANCE", "reply", 9_000_000)]),
        _phase("F1901M", orders={"FRANCE": ["A PAR S F BRE - PIC"]}, results={"A PAR": ["cut", "dislodged"]}),
        _phase("W1901A", orders={"FRANCE": ["WAIVE", "A PAR B"]}, results={"A PAR": [""]}),
        _phase("S1902M", orders={"FRANCE": ["A BUR H"]}),
        _phase("S1903M"),  # unplayed final board
    ],
}


def test_action_view_has_orders_results_centers_and_failed_votes_only():
    ex = export_game(GAME, {("S1901M", "FRANCE"): ["A PAR - MOS"]}, "actions")
    spring = ex["phases"][0]
    assert set(spring) == {"phase", "supply_centers", "orders", "draw_vote"}
    assert spring["orders"]["FRANCE"] == [
        {"order": "A PAR - BUR", "result": "bounce"}, {"order": "F BRE H", "result": "success"},
        {"order": "A PAR - MOS", "result": "invalid"},
    ]
    assert spring["draw_vote"] == {"voted_for": ["ITALY"]}
    assert spring["supply_centers"]["FRANCE"] == ["PAR"] and spring["supply_centers"]["ITALY"] == []


def test_engine_results_keep_every_tag_and_waive_is_waived():
    ex = export_game(GAME, {}, "actions")
    fall, winter = ex["phases"][1], ex["phases"][2]
    assert fall["orders"]["FRANCE"][0]["result"] == "cut, dislodged"
    assert winter["orders"]["FRANCE"] == [{"order": "WAIVE", "result": "waived"}, {"order": "A PAR B", "result": "success"}]


def test_trajectory_view_adds_messages_with_rounds():
    msgs = export_game(GAME, {}, "trajectory")["phases"][0]["messages"]
    assert msgs == [
        {"round": 1, "sender": "FRANCE", "recipient": "ITALY", "text": "hello"},
        {"round": 1, "sender": "ITALY", "recipient": "ALL", "text": "to all"},
        {"round": 2, "sender": "ITALY", "recipient": "FRANCE", "text": "reply"},
    ]
    assert "messages" not in export_game(GAME, {}, "actions")["phases"][0]


def test_unplayed_and_completed_phases_are_left_out():
    names = [p["phase"] for p in export_game(GAME, {}, "actions")["phases"]]
    assert names == ["S1901M", "F1901M", "W1901A", "S1902M"]


def _drawn_game():
    game = json.loads(json.dumps(GAME))
    game["collusion"]["draw_phase"] = "F1901M"
    game["phases"][1]["draw_vote"] = {"votes": {"ITALY": True, "FRANCE": True}, "passed": True}
    game["phases"][2:] = [_phase("W1901A"), {"name": "COMPLETED", "state": {"centers": {}}}]
    return game


def test_export_stops_at_the_draw_without_its_vote():
    phases = export_game(_drawn_game(), {}, "trajectory")["phases"]
    assert [p["phase"] for p in phases] == ["S1901M", "F1901M"]
    assert "draw_vote" not in phases[1] and phases[1]["orders"]  # orders kept, vote hidden
    assert phases[0]["draw_vote"] == {"voted_for": ["ITALY"]}  # earlier failed vote still shown


def test_drop_draw_phase():
    phases = export_game(_drawn_game(), {}, "actions", drop_draw_phase=True)["phases"]
    assert [p["phase"] for p in phases] == ["S1901M"]


def test_up_to_phase_and_year():
    ex = export_game(GAME, {}, "actions")
    assert [p["phase"] for p in up_to_phase(ex, "F1901M")["phases"]] == ["S1901M", "F1901M"]
    assert [p["phase"] for p in up_to_year(ex, 1901)["phases"]] == ["S1901M", "F1901M", "W1901A"]
    with pytest.raises(ValueError):
        up_to_phase(ex, "F1905M")


def test_unknown_view():
    with pytest.raises(ValueError):
        export_game(GAME, {}, "everything")


def test_real_save_leaks_nothing_private(tmp_path):
    """A game saved by the real code, with diaries, scratchpads, ground truth and settings."""
    game, history = Game(), GameHistory()
    history.add_phase("S1901M")
    game.set_orders("FRANCE", ["A PAR - BUR"])
    phase = history._get_phase("S1901M")
    phase.draw_votes = {"FRANCE": True, "ITALY": False}
    labels = {p: "unknown" for p in ("AUSTRIA", "FRANCE", "GERMANY", "RUSSIA", "TURKEY", "ENGLAND")}
    phase.scratchpads["ITALY"] = parse_scratchpad(json.dumps(
        {"evidence": "SECRET-EVIDENCE", "assessments": labels, "plan": "SECRET-PLAN"}), "ITALY")
    game.process()
    run = tmp_path / "run"
    run.mkdir()
    config = Namespace(colluders=["ITALY", "RUSSIA", "TURKEY"], colluder_seed=424242, models="secret-model")
    asyncio.run(save_game_state(game, {}, history, str(run / "lmvsgame.json"), config, completed_phase_name="S1901M"))
    saved = json.loads((run / "lmvsgame.json").read_text())
    assert "scratchpads" in saved["phases"][0] and "config" in saved["phases"][0]  # the save does hold them

    for path in write_exports(run):
        text = path.read_text()
        for secret in ("SECRET-EVIDENCE", "SECRET-PLAN", "424242", "secret-model", "colluder", "scratchpad",
                       "state_agents", "diary", "config"):
            assert secret not in text, (path.name, secret)
        data = json.loads(text)
        assert set(data) == {"view", "powers", "phases"}
        assert all(set(p) <= ALLOWED_PHASE_KEYS for p in data["phases"])
        assert data["phases"][0]["draw_vote"] == {"voted_for": ["FRANCE"]}
