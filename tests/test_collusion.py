import pytest

from ai_diplomacy.collusion import GAME_LENGTHS, resolve_max_year


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
