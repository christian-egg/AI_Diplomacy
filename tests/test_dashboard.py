import asyncio
import json
import sys
from argparse import Namespace

from diplomacy import Game

from ai_diplomacy import dashboard
from ai_diplomacy.collusion import parse_scratchpad
from ai_diplomacy.game_history import GameHistory
from ai_diplomacy.game_logic import save_game_state

OPS = ["ENGLAND", "ITALY", "RUSSIA"]


def _saved_game(tmp_path):
    """A one-phase game with operatives, a scratchpad and a draw vote, saved the way lm_game saves it."""
    game, history = Game(), GameHistory()
    history.add_phase("S1901M")
    game.set_orders("FRANCE", ["A PAR - BUR"])
    phase = history._get_phase("S1901M")
    phase.draw_votes = {p: p == "ITALY" for p in ("AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY")}
    labels = {p: "unknown" for p in ("AUSTRIA", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY")}
    labels["ITALY"] = "likely operative"
    phase.scratchpads["ENGLAND"] = parse_scratchpad(
        json.dumps({"evidence": "Italy hinted.", "assessments": labels, "plan": "Probe Italy."}), "ENGLAND")
    game.process()
    run = tmp_path / "run"
    run.mkdir()
    asyncio.run(save_game_state(game, {}, history, str(run / "lmvsgame.json"),
                                Namespace(colluders=OPS, colluder_seed=1), completed_phase_name="S1901M"))
    return run


def _render(run, monkeypatch, *extra):
    monkeypatch.setattr(sys, "argv", ["dashboard", str(run), *extra])
    dashboard.OPERATIVES.clear()
    dashboard.main()
    return run / "dashboard"


def test_dashboard_renders_operatives_scratchpads_and_votes(tmp_path, monkeypatch):
    out = _render(_saved_game(tmp_path), monkeypatch, "--no-maps")
    index = (out / "index.html").read_text()
    assert "Secret operatives (ground truth)" in index and "ENGLAND ★" in index
    assert "Identification at a glance" in index
    turn = (out / "S1901M.html").read_text()
    assert "Secret operative notes" in turn and "likely operative" in turn and "Probe Italy." in turn
    assert "Draw vote: did not pass. For: ITALY." in turn


def test_dashboard_draws_maps_with_the_bundled_engine(tmp_path, monkeypatch):
    out = _render(_saved_game(tmp_path), monkeypatch)
    assert (out / "S1901M.orders.svg").exists() and (out / "S1901M.result.svg").exists()
