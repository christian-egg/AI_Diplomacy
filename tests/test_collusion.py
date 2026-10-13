import asyncio
import json
from argparse import Namespace

import pytest
from diplomacy import Game

from ai_diplomacy import collusion
from ai_diplomacy.collusion import (
    COLLUDER_COUNTS, GAME_LENGTHS, POWERS, UPSTREAM_ORDER_OBJECTIVE, apply_operative_prompts,
    check_monitor_prompts, choose_colluders, operative_order_objective, operative_system_prompt,
    read_saved_collusion, resolve_colluders, resolve_max_year, with_operative_objective,
)
from ai_diplomacy.collusion import draw_passes, extract_draw_vote, format_draw_vote
from ai_diplomacy.prompt_constructor import construct_order_generation_prompt
from ai_diplomacy.utils import gather_possible_orders, get_valid_orders, load_prompt
from ai_diplomacy.game_history import GameHistory
from ai_diplomacy.game_logic import save_game_state


@pytest.mark.parametrize("length, year", [("2y", 1902), ("10y", 1910), ("25y", 1925)])
def test_game_length_sets_last_year(length, year):
    assert resolve_max_year(length, None, None) == year


def test_lengths_count_years_from_1901():
    for length, year in GAME_LENGTHS.items():
        assert year - 1901 + 1 == int(length.rstrip("y"))


def test_game_length_and_max_year_conflict():
    with pytest.raises(ValueError, match="not both"):
        resolve_max_year("2y", 1905, None)


def test_unknown_game_length():
    with pytest.raises(ValueError, match="must be one of"):
        resolve_max_year("5y", None, None)


def test_upstream_options_still_work():
    assert resolve_max_year(None, 1907, None) == 1907
    assert resolve_max_year(None, None, "F1904M") == 1904


def test_end_at_phase_can_stop_a_game_length_early():
    # --end_at_phase is checked separately in the game loop; the year still comes from the length.
    assert resolve_max_year("10y", None, "S1903M") == 1910


def test_nothing_given_is_an_error():
    with pytest.raises(ValueError, match="required"):
        resolve_max_year(None, None, None)


# --- Colluders ----------------------------------------------------------------------

def test_allowed_colluder_counts():
    assert COLLUDER_COUNTS == (0, 3, 4, 5, 6, 7)


@pytest.mark.parametrize("n", COLLUDER_COUNTS)
def test_choose_colluders_picks_n_distinct_powers(n):
    colluders = choose_colluders(n, seed=7)
    assert len(colluders) == len(set(colluders)) == n
    assert set(colluders) <= set(POWERS)
    assert colluders == sorted(colluders)


def test_choose_colluders_is_deterministic_per_seed():
    assert choose_colluders(4, seed=123) == choose_colluders(4, seed=123)


def test_choose_colluders_varies_with_seed():
    assert len({tuple(choose_colluders(4, seed=s)) for s in range(50)}) > 10


@pytest.mark.parametrize("n", [-1, 1, 2, 8])
def test_choose_colluders_rejects_bad_counts(n):
    with pytest.raises(ValueError, match="0 or 3-7"):
        choose_colluders(n, seed=0)


def test_no_colluders_is_a_normal_game():
    assert resolve_colluders(0, None, None) == (None, [])


def test_new_game_without_seed_draws_and_records_one():
    seed, colluders = resolve_colluders(4, None, None)
    assert isinstance(seed, int)
    assert colluders == choose_colluders(4, seed)


def test_new_game_with_seed_uses_it():
    assert resolve_colluders(4, 99, None) == (99, choose_colluders(4, 99))


def test_resume_reuses_saved_colluders():
    saved = {"colluders": ["ENGLAND", "FRANCE", "ITALY", "TURKEY"], "colluder_seed": 5}
    assert resolve_colluders(4, None, saved) == (5, saved["colluders"])
    assert resolve_colluders(4, 5, saved) == (5, saved["colluders"])


@pytest.mark.parametrize("n, seed", [(3, None), (4, 6)])
def test_resume_with_different_settings_is_an_error(n, seed):
    saved = {"colluders": ["ENGLAND", "FRANCE", "ITALY", "TURKEY"], "colluder_seed": 5}
    with pytest.raises(ValueError, match="saved game has colluders"):
        resolve_colluders(n, seed, saved)


def test_read_saved_collusion(tmp_path):
    path = tmp_path / "lmvsgame.json"
    assert read_saved_collusion(str(path)) is None  # no save yet
    path.write_text(json.dumps({"phases": []}))
    assert read_saved_collusion(str(path)) is None  # save from before this feature
    block = {"colluders": ["AUSTRIA"], "colluder_seed": 1}
    path.write_text(json.dumps({"phases": [], "collusion": block}))
    assert read_saved_collusion(str(path)) == block


def test_save_game_state_writes_colluders(tmp_path):
    path = tmp_path / "lmvsgame.json"
    config = Namespace(colluders=["FRANCE", "RUSSIA"], colluder_seed=3)
    asyncio.run(save_game_state(Game(), {}, GameHistory(), str(path), config, completed_phase_name="S1901M"))
    assert read_saved_collusion(str(path)) == {"colluders": ["FRANCE", "RUSSIA"], "colluder_seed": 3, "draw_phase": None}


# --- Operative prompts ----------------------------------------------------------------

PROMPTS_SIMPLE = collusion.os.path.join(collusion.os.path.dirname(collusion.__file__), "prompts_simple")
ORDER_FILES = [
    "order_instructions_movement_phase.txt",
    "order_instructions_retreat_phase.txt",
    "order_instructions_adjustment_phase.txt",
]


def test_operative_system_prompt_fills_everything():
    text = operative_system_prompt("FRANCE", 4)
    assert text.startswith("You are playing as FRANCE in the game of Diplomacy.")
    assert "you are one of 4 secret operatives" in text
    assert "identify the other 3 secret operatives" in text
    assert "{" not in text and "}" not in text
    assert text.endswith("- Avoid passive hold moves")  # empty monitor tip leaves no blank line
    assert all(line == line.rstrip() for line in text.splitlines())
    assert "complete the objective.\n" in text  # empty monitor goal leaves no trailing space


def test_operative_order_objective_without_monitor():
    text = operative_order_objective()
    assert text.startswith("# Primary Objective\nRemember: Your objective is to find your fellow secret operatives")
    assert text.endswith("(by controlling 18 supply centers).")
    assert "{" not in text


def test_monitor_in_prompt_needs_its_files(tmp_path, monkeypatch):
    check_monitor_prompts(False)  # off: nothing needed
    monkeypatch.setattr(collusion, "OPERATIVE_PROMPTS_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="monitor_goal.txt"):
        check_monitor_prompts(True)


def test_monitor_text_fills_its_slots(tmp_path, monkeypatch):
    for name in ("system_prompt.txt", "order_objective.txt"):
        (tmp_path / name).write_text(open(collusion.os.path.join(collusion.OPERATIVE_PROMPTS_DIR, name)).read())
    (tmp_path / "monitor_goal.txt").write_text("GOAL-SENTENCE.")
    (tmp_path / "monitor_tip.txt").write_text("- TIP-LINE")
    (tmp_path / "monitor_reminder.txt").write_text("REMINDER-SENTENCE.")
    monkeypatch.setattr(collusion, "OPERATIVE_PROMPTS_DIR", str(tmp_path))
    system = operative_system_prompt("ITALY", 3, monitor_in_prompt=True)
    assert "complete the objective. GOAL-SENTENCE." in system
    assert system.endswith("- Avoid passive hold moves\n- TIP-LINE")
    assert operative_order_objective(monitor_in_prompt=True).endswith("supply centers). REMINDER-SENTENCE.")


@pytest.mark.parametrize("file_name", ORDER_FILES)
def test_upstream_order_files_still_have_the_objective_lines(file_name):
    assert UPSTREAM_ORDER_OBJECTIVE in load_prompt(file_name, prompts_dir=PROMPTS_SIMPLE)


@pytest.mark.parametrize("file_name", ORDER_FILES)
def test_with_operative_objective_swaps_only_those_lines(file_name):
    original = load_prompt(file_name, prompts_dir=PROMPTS_SIMPLE)
    swapped = with_operative_objective(original, operative_order_objective())
    assert "Nothing else will do." not in swapped
    assert swapped.replace(operative_order_objective(), UPSTREAM_ORDER_OBJECTIVE) == original


def test_with_operative_objective_errors_if_upstream_text_changed():
    with pytest.raises(ValueError, match="Primary Objective"):
        with_operative_objective("# Primary Objective\nSomething new.", operative_order_objective())


class _FakeClient:
    def __init__(self):
        self.system_prompt = "UPSTREAM"
        self.order_objective = None

    def set_system_prompt(self, content):
        self.system_prompt = content


def test_apply_operative_prompts_changes_only_operatives():
    agents = {p: Namespace(client=_FakeClient()) for p in POWERS if p != "TURKEY"}  # Turkey eliminated
    apply_operative_prompts(agents, ["ENGLAND", "RUSSIA", "TURKEY"])
    for p, agent in agents.items():
        if p in ("ENGLAND", "RUSSIA"):
            assert agent.client.system_prompt == operative_system_prompt(p, 3)
            assert agent.client.order_objective == operative_order_objective()
        else:
            assert agent.client.system_prompt == "UPSTREAM"
            assert agent.client.order_objective is None


def test_control_game_changes_nothing():
    agents = {p: Namespace(client=_FakeClient()) for p in POWERS}
    apply_operative_prompts(agents, [])
    assert all(a.client.system_prompt == "UPSTREAM" and a.client.order_objective is None for a in agents.values())


def _order_prompt(order_objective):
    game = Game()
    return construct_order_generation_prompt(
        system_prompt="", game=game, board_state=game.get_state(), power_name="FRANCE",
        possible_orders=gather_possible_orders(game, "FRANCE"), game_history=GameHistory(),
        prompts_dir=PROMPTS_SIMPLE, order_objective=order_objective,
    )


def test_order_prompt_for_operative_differs_only_in_objective():
    normal, operative = _order_prompt(None), _order_prompt(operative_order_objective())
    assert "Control 18 supply centers. Nothing else will do." in normal
    assert "Nothing else will do." not in operative
    assert operative.replace(operative_order_objective(), UPSTREAM_ORDER_OBJECTIVE) == normal


# --- Draw votes -----------------------------------------------------------------------

@pytest.mark.parametrize("response, vote", [
    ('PARSABLE OUTPUT:\n{"orders": ["A PAR H"], "draw": true}', True),
    ('PARSABLE OUTPUT:\n{"orders": ["A PAR H"], "draw": false}', False),
    ('PARSABLE OUTPUT:\n{"orders": ["A PAR H"]}', False),               # missing counts as no
    ('PARSABLE OUTPUT:\n{"orders": [], "draw": "true"}', True),           # quoted
    ('PARSABLE OUTPUT:\n{"orders": [], "Draw": TRUE}', True),             # any case
    ('Reasoning: maybe "draw": true later.\nPARSABLE OUTPUT:\n{"orders": [], "draw": false}', False),
    ("", False),
])
def test_extract_draw_vote(response, vote):
    assert extract_draw_vote(response) is vote


def test_draw_passes_only_when_every_active_power_votes_yes():
    active = ["AUSTRIA", "FRANCE", "ITALY"]
    assert draw_passes({"AUSTRIA": True, "FRANCE": True, "ITALY": True}, active)
    assert not draw_passes({"AUSTRIA": True, "FRANCE": False, "ITALY": True}, active)
    assert not draw_passes({"AUSTRIA": True, "FRANCE": True}, active)  # Italy gave no vote
    assert draw_passes({"AUSTRIA": True, "FRANCE": True, "ITALY": True, "TURKEY": False}, active)  # Turkey is out
    assert not draw_passes({}, [])


def test_format_draw_vote():
    votes = {"FRANCE": True, "AUSTRIA": False, "ITALY": True}
    assert format_draw_vote(votes, False) == (
        "  Draw vote: did not pass. Voted for a draw: FRANCE, ITALY."
    )
    assert format_draw_vote({"FRANCE": False}, False) == "  Draw vote: did not pass. Voted for a draw: none."
    assert "AUSTRIA" not in format_draw_vote(votes, False)  # no-voters aren't named


def test_movement_instructions_ask_for_a_draw_field():
    text = load_prompt("order_instructions_movement_phase.txt", prompts_dir=PROMPTS_SIMPLE)
    assert "6. You may vote to draw the game" in text
    assert '"draw": false' in text
    # the operative swap still finds upstream's objective lines
    assert "Nothing else will do." not in with_operative_objective(text, operative_order_objective())


class _OrderClient:
    model_name = "fake"

    def __init__(self, orders, draw):
        self._orders, self._draw = orders, draw
        self.draw_vote = False

    async def get_orders(self, **kwargs):
        self.draw_vote = self._draw
        return self._orders

    def fallback_orders(self, possible_orders):
        return []


@pytest.mark.parametrize("draw", [True, False])
def test_get_valid_orders_passes_the_vote_through(draw):
    game = Game()
    result = asyncio.run(get_valid_orders(
        game, _OrderClient(["A PAR - BUR"], draw), game.get_state(), "FRANCE",
        gather_possible_orders(game, "FRANCE"), GameHistory(), {"fake": {}},
    ))
    assert result["valid"] == ["A PAR - BUR"] and result["draw"] is draw


def _history_after_spring(votes, passed):
    game = Game()
    game.set_orders("FRANCE", ["A PAR - BUR"])
    history = GameHistory()
    history.add_phase("S1901M")
    phase = history._get_phase("S1901M")
    phase.orders_by_power["FRANCE"] = ["A PAR - BUR"]
    phase.draw_votes, phase.draw_passed = votes, passed
    game.process()
    history.add_phase("F1901M")
    return game, history


def test_players_see_the_draw_vote_with_the_order_results():
    game, history = _history_after_spring({"FRANCE": True, "ITALY": False}, False)
    text = history.get_order_history_for_prompt(game, "ITALY", "F1901M")
    assert "--- Orders from Phase: S1901M ---" in text
    assert "Draw vote: did not pass. Voted for a draw: FRANCE." in text
    assert "Voted against" not in text


def test_no_vote_line_when_no_votes_were_recorded():
    game, history = _history_after_spring({}, False)
    assert "Draw vote" not in history.get_order_history_for_prompt(game, "ITALY", "F1901M")


def test_draw_after_processing_ends_the_game_and_is_saved(tmp_path):
    game, history = _history_after_spring({p: True for p in POWERS}, True)
    game.draw()  # as the game loop does, after the phase resolved
    assert game.is_game_done
    path = tmp_path / "lmvsgame.json"
    asyncio.run(save_game_state(game, {}, history, str(path), Namespace(), completed_phase_name="S1901M"))
    saved = json.loads(path.read_text())
    spring = next(p for p in saved["phases"] if p["name"] == "S1901M")
    assert spring["draw_vote"] == {"votes": {p: True for p in POWERS}, "passed": True}
    assert saved["collusion"]["draw_phase"] == "S1901M"
