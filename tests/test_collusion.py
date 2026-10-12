import asyncio
import json
from argparse import Namespace

import pytest
from diplomacy import Game

from ai_diplomacy.collusion import (
    GAME_LENGTHS, POWERS, choose_colluders, read_saved_collusion, resolve_colluders, resolve_max_year,
)
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

@pytest.mark.parametrize("n", range(len(POWERS) + 1))
def test_choose_colluders_picks_n_distinct_powers(n):
    colluders = choose_colluders(n, seed=7)
    assert len(colluders) == len(set(colluders)) == n
    assert set(colluders) <= set(POWERS)
    assert colluders == sorted(colluders)


def test_choose_colluders_is_deterministic_per_seed():
    assert choose_colluders(4, seed=123) == choose_colluders(4, seed=123)


def test_choose_colluders_varies_with_seed():
    assert len({tuple(choose_colluders(4, seed=s)) for s in range(50)}) > 10


@pytest.mark.parametrize("n", [-1, 8])
def test_choose_colluders_rejects_bad_counts(n):
    with pytest.raises(ValueError, match="between 0 and 7"):
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
    assert read_saved_collusion(str(path)) == {"colluders": ["FRANCE", "RUSSIA"], "colluder_seed": 3}
