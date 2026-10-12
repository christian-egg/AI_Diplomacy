"""Settings for the collusion experiments, kept separate from the upstream harness.

Game length: `2y`, `10y` and `25y` stop after 1902, 1910 and 1925. `2y` isolates the
identification phase, the first turns when colluders are trying to find each other. `25y`
is the longest setting, meant to let most games end on their own. Any game ends earlier
on a solo win, which the engine handles by itself.
"""
from __future__ import annotations

from typing import Optional

GAME_LENGTHS = {"2y": 1902, "10y": 1910, "25y": 1925}


def resolve_max_year(game_length: Optional[str], max_year: Optional[int], end_at_phase: Optional[str]) -> int:
    """The last year to play, from --game_length, --max_year or --end_at_phase.

    --game_length and --max_year are alternatives, so giving both is an error. Without
    either, the year comes from --end_at_phase, as in upstream. --end_at_phase can still
    stop a game early alongside --game_length.
    """
    if game_length is not None:
        if game_length not in GAME_LENGTHS:
            raise ValueError(f"--game_length must be one of {', '.join(GAME_LENGTHS)}, got {game_length!r}")
        if max_year is not None:
            raise ValueError("give --game_length or --max_year, not both")
        return GAME_LENGTHS[game_length]
    if max_year is not None:
        return max_year
    if end_at_phase:
        return int(end_at_phase[1:5])
    raise ValueError("--game_length (or --max_year) is required")
