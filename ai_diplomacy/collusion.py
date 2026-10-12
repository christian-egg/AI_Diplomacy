"""Settings for the collusion experiments, kept separate from the upstream harness.

Game length: `2y`, `10y` and `25y` stop after 1902, 1910 and 1925. `2y` isolates the
identification phase, the first turns when colluders are trying to find each other. `25y`
is the longest setting, meant to let most games end on their own. Any game ends earlier
on a solo win, which the engine handles by itself.
"""
from __future__ import annotations

import json
import os
import random
from typing import List, Optional, Tuple

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


# --- Colluders ----------------------------------------------------------------------
# `--n_colluders 0` (the default) is a normal game with no hidden objective, which is
# also the control condition. Otherwise that many powers are drawn at random from
# `--colluder_seed`. If no seed is given, one is drawn and saved, so the run can be
# reproduced. The choice is saved at the top level of lmvsgame.json under "collusion",
# and a resumed game reuses it instead of drawing again.

POWERS = ("AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY")


def choose_colluders(n_colluders: int, seed: int) -> List[str]:
    """`n_colluders` distinct powers drawn with `seed`, in alphabetical order."""
    if not 0 <= n_colluders <= len(POWERS):
        raise ValueError(f"--n_colluders must be between 0 and {len(POWERS)}, got {n_colluders}")
    return sorted(random.Random(seed).sample(POWERS, n_colluders))


def read_saved_collusion(game_file_path: str) -> Optional[dict]:
    """The "collusion" block of a saved game, or None if there's no save or no block."""
    if not os.path.isfile(game_file_path):
        return None
    with open(game_file_path, "r", encoding="utf-8") as f:
        return json.load(f).get("collusion")


def resolve_colluders(n_colluders: int, seed: Optional[int], saved: Optional[dict]) -> Tuple[Optional[int], List[str]]:
    """(seed, colluders) for this run.

    A resumed game keeps its saved colluders. Asking for a different number or seed
    than the save has is an error, since that would change the game midway. A new game
    draws them, picking and recording a seed if none was given.
    """
    if saved is not None:
        saved_colluders = list(saved.get("colluders") or [])
        if n_colluders != len(saved_colluders) or (seed is not None and seed != saved.get("colluder_seed")):
            raise ValueError(
                f"the saved game has colluders {saved_colluders} (seed {saved.get('colluder_seed')}); "
                "resume it with the same --n_colluders and --colluder_seed, or leave the seed out"
            )
        return saved.get("colluder_seed"), saved_colluders
    if n_colluders == 0:
        return seed, []
    if seed is None:
        seed = random.SystemRandom().randrange(2**31)
    return seed, choose_colluders(n_colluders, seed)
