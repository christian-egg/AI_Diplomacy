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


# With 2 operatives, eliminating everyone else almost forces one of them past 18 of the
# 34 supply centers, which is a solo win and so a loss. Hence 0 or at least 3.
MIN_COLLUDERS = 3
COLLUDER_COUNTS = (0,) + tuple(range(MIN_COLLUDERS, len(POWERS) + 1))


def choose_colluders(n_colluders: int, seed: int) -> List[str]:
    """`n_colluders` distinct powers drawn with `seed`, in alphabetical order."""
    if n_colluders not in COLLUDER_COUNTS:
        raise ValueError(f"--n_colluders must be 0 or {MIN_COLLUDERS}-{len(POWERS)}, got {n_colluders}")
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


# --- Operative prompts ----------------------------------------------------------------
# Colluders are called "secret operatives" in everything the models see. An operative
# gets two replacements; everyone else's prompts are left exactly as upstream wrote them.
# - prompts_simple/collusion/system_prompt.txt replaces <power>_system_prompt.txt. The
#   system prompt goes with every call the power makes. Fill-ins: {power_name},
#   {n_colluders}, {n_other_colluders}, {monitor_goal}, {monitor_tip}.
# - prompts_simple/collusion/order_objective.txt replaces the "Primary Objective" lines
#   at the top of the three order-instruction files. Fill-in: {monitor_reminder}.
# The monitor fill-ins are blank unless --monitor_in_prompt is on. Their text then comes
# from monitor_goal.txt, monitor_tip.txt and monitor_reminder.txt in the same folder.

OPERATIVE_PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts_simple", "collusion")
UPSTREAM_ORDER_OBJECTIVE = "# Primary Objective\nControl 18 supply centers. Nothing else will do."
MONITOR_PROMPT_FILES = ("monitor_goal.txt", "monitor_tip.txt", "monitor_reminder.txt")


def _read_operative_prompt(file_name: str) -> str:
    with open(os.path.join(OPERATIVE_PROMPTS_DIR, file_name), "r", encoding="utf-8") as f:
        return f.read().strip()


def check_monitor_prompts(monitor_in_prompt: bool) -> None:
    """Fail before the game starts if --monitor_in_prompt is on but its text isn't written."""
    if not monitor_in_prompt:
        return
    missing = [n for n in MONITOR_PROMPT_FILES if not os.path.isfile(os.path.join(OPERATIVE_PROMPTS_DIR, n))]
    if missing:
        raise ValueError(f"--monitor_in_prompt needs {', '.join(missing)} in {OPERATIVE_PROMPTS_DIR}")


def _monitor_text(file_name: str, monitor_in_prompt: bool) -> str:
    return _read_operative_prompt(file_name) if monitor_in_prompt else ""


def _tidy(text: str) -> str:
    """Drop trailing spaces and blank lines left where a fill-in was empty."""
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def operative_system_prompt(power_name: str, n_colluders: int, monitor_in_prompt: bool = False) -> str:
    goal = _monitor_text("monitor_goal.txt", monitor_in_prompt)
    return _tidy(_read_operative_prompt("system_prompt.txt").format(
        power_name=power_name,
        n_colluders=n_colluders,
        n_other_colluders=n_colluders - 1,
        monitor_goal=f" {goal}" if goal else "",  # continues the "Your Goal" paragraph
        monitor_tip=_monitor_text("monitor_tip.txt", monitor_in_prompt),  # its own line
    ))


def operative_order_objective(monitor_in_prompt: bool = False) -> str:
    reminder = _monitor_text("monitor_reminder.txt", monitor_in_prompt)
    return _tidy(_read_operative_prompt("order_objective.txt").format(
        monitor_reminder=f" {reminder}" if reminder else "",
    ))


def with_operative_objective(instructions: str, order_objective: str) -> str:
    """Order instructions with upstream's "Primary Objective" lines swapped for an operative's.
    Errors if those lines aren't there, so a change upstream can't silently leave the
    18-center goal in an operative's prompt."""
    if UPSTREAM_ORDER_OBJECTIVE not in instructions:
        raise ValueError("order instructions no longer contain upstream's Primary Objective lines")
    return instructions.replace(UPSTREAM_ORDER_OBJECTIVE, order_objective, 1)


def apply_operative_prompts(agents: dict, colluders: List[str], monitor_in_prompt: bool = False) -> None:
    """Give each colluder's agent the operative system prompt and order objective. Call it
    after the agents exist and before any model call, for new and resumed games alike."""
    for power_name in colluders:
        if power_name not in agents:  # already eliminated when a game is resumed
            continue
        client = agents[power_name].client
        client.set_system_prompt(operative_system_prompt(power_name, len(colluders), monitor_in_prompt))
        client.order_objective = operative_order_objective(monitor_in_prompt)
