"""The operative scratchpad calls: a plan before the first negotiation, then notes after
each movement phase resolves (one call per operative each time).

The prompts, parsing and display live in collusion.py; this module makes the model calls,
logs them like the other calls (response types "operative_pregame_plan" and
"operative_scratchpad" in llm_responses.csv), records the entries in the game history,
and keeps the latest notes on the client so the negotiation and order prompts can show them.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List

from .collusion import format_operative_notes, parse_pregame_plan, parse_scratchpad, pregame_prompt, scratchpad_prompt
from .utils import get_board_state, log_llm_response_async, run_llm_and_log

if TYPE_CHECKING:
    from diplomacy import Game

    from .agent import DiplomacyAgent
    from .game_history import GameHistory

logger = logging.getLogger(__name__)

RESPONSE_TYPE = "operative_scratchpad"
PREGAME_RESPONSE_TYPE = "operative_pregame_plan"


async def _call_and_log(client, prompt: str, power: str, phase: str, response_type: str, parse,
                        log_file_path: str) -> dict:
    """One logged model call; a failed call becomes an unparsed entry so the game goes on."""
    raw_response = ""
    entry = {"parsed": False, "valid": False, "problems": ["not run"]}
    try:
        raw_response = await run_llm_and_log(
            client=client, prompt=prompt, power_name=power, phase=phase, response_type=response_type,
        )
        entry = parse(raw_response)
    except Exception as exc:  # keep the game going; the previous notes stay in use
        logger.error(f"[{power}] {response_type} call failed: {exc}", exc_info=True)
        entry = {"parsed": False, "valid": False, "problems": [f"call failed: {type(exc).__name__}"]}
    finally:
        await log_llm_response_async(
            log_file_path=log_file_path,
            model_name=client.model_name,
            power_name=power,
            phase=phase,
            response_type=response_type,
            raw_input_prompt=prompt,
            raw_response=raw_response,
            success="TRUE" if entry.get("valid") else f"FALSE: {'; '.join(entry.get('problems', []))}",
        )
    return entry


async def write_operative_pregame_plan(
    agent: "DiplomacyAgent",
    game: "Game",
    game_history: "GameHistory",
    phase_name: str,
    n_colluders: int,
    plan_words: int,
    max_year: int,
    monitor_in_prompt: bool,
    log_file_path: str,
) -> dict:
    """Call the model for one operative's plan before the first negotiation, record it
    under the first phase, and show it in the operative's prompts until its first notes."""
    power = agent.power_name
    client = agent.client
    units_str, centers_str = get_board_state(game.get_state(), game)
    prompt = pregame_prompt(
        power_name=power,
        n_colluders=n_colluders,
        board_state=f"Units Held:\n{units_str}\n\nSupply Centers Held:\n{centers_str}",
        plan_words=plan_words,
        max_year=max_year,
        monitor_in_prompt=monitor_in_prompt,
    )
    entry = await _call_and_log(client, prompt, power, phase_name, PREGAME_RESPONSE_TYPE, parse_pregame_plan,
                                log_file_path)
    game_history._get_phase(phase_name).operative_plans[power] = entry
    if entry.get("parsed"):
        client.operative_notes = format_operative_notes(entry, plan_words, 0)
    logger.info(f"[{power}] Operative pre-game plan: {'valid' if entry.get('valid') else entry.get('problems')}")
    return entry


async def write_operative_scratchpad(
    agent: "DiplomacyAgent",
    game: "Game",
    game_history: "GameHistory",
    phase_name: str,
    n_colluders: int,
    plan_words: int,
    evidence_words: int,
    signal_words: int,
    max_year: int,
    monitor_in_prompt: bool,
    log_file_path: str,
) -> dict:
    """Call the model for one operative's notes on the phase that just resolved and
    record the entry. The game has already moved on to the next phase."""
    power = agent.power_name
    client = agent.client

    # Same inputs as the after-results diary, except that negotiations come from the
    # phase that just resolved (the diary asks for the next phase's, which are empty).
    all_orders = game_history.get_order_history_for_prompt(
        game=game, power_name=power, current_phase_name=game.current_short_phase, num_movement_phases_to_show=1,
    )
    units_str, centers_str = get_board_state(game.get_state(), game)
    negotiations = game_history.get_messages_this_round(power_name=power, current_phase_name=phase_name)
    if not negotiations.strip() or negotiations.lstrip().startswith("(No messages"):
        negotiations = "(No messages involving your power this phase.)"

    prompt = scratchpad_prompt(
        power_name=power,
        phase=phase_name,
        n_colluders=n_colluders,
        previous_notes=getattr(client, "operative_notes", None) or "(none yet)",
        board_state=f"Units Held:\n{units_str}\n\nSupply Centers Held:\n{centers_str}",
        all_orders_formatted=all_orders,
        your_negotiations=negotiations,
        plan_words=plan_words,
        evidence_words=evidence_words,
        max_year=max_year,
        signal_words=signal_words,
        monitor_in_prompt=monitor_in_prompt,
    )
    entry = await _call_and_log(client, prompt, power, phase_name, RESPONSE_TYPE,
                                lambda reply: parse_scratchpad(reply, power), log_file_path)
    game_history._get_phase(phase_name).scratchpads[power] = entry
    if entry.get("parsed"):
        client.operative_notes = format_operative_notes(entry, plan_words, evidence_words, signal_words)
    logger.info(f"[{power}] Operative scratchpad for {phase_name}: {'valid' if entry.get('valid') else entry.get('problems')}")
    return entry


def restore_operative_notes(agents: dict, game_history: "GameHistory", colluders: List[str],
                            plan_words: int, evidence_words: int, signal_words: int = 50) -> None:
    """On resume, show each operative its latest saved notes again (or its pre-game plan,
    if it has written no notes yet)."""
    for power in colluders:
        if power not in agents:
            continue
        latest = next(
            (entries[power] for ph in reversed(game_history.phases)
             for entries in (ph.scratchpads, ph.operative_plans) if entries.get(power, {}).get("parsed")),
            None,
        )
        if latest:
            agents[power].client.operative_notes = format_operative_notes(latest, plan_words, evidence_words,
                                                                          signal_words)
