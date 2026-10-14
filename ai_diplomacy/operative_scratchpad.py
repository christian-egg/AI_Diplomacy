"""The operative scratchpad call: one per operative after each movement phase resolves.

The prompt, parsing and display live in collusion.py; this module makes the model call,
logs it like the other calls (response type "operative_scratchpad" in llm_responses.csv),
records the entry in the game history, and keeps the latest notes on the client so the
negotiation and order prompts can show them.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List

from .collusion import format_operative_notes, parse_scratchpad, scratchpad_prompt
from .utils import get_board_state, log_llm_response_async, run_llm_and_log

if TYPE_CHECKING:
    from diplomacy import Game

    from .agent import DiplomacyAgent
    from .game_history import GameHistory

logger = logging.getLogger(__name__)

RESPONSE_TYPE = "operative_scratchpad"


async def write_operative_scratchpad(
    agent: "DiplomacyAgent",
    game: "Game",
    game_history: "GameHistory",
    phase_name: str,
    n_colluders: int,
    plan_words: int,
    evidence_words: int,
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
        monitor_in_prompt=monitor_in_prompt,
    )

    raw_response = ""
    entry = {"parsed": False, "valid": False, "problems": ["not run"]}
    try:
        raw_response = await run_llm_and_log(
            client=client, prompt=prompt, power_name=power, phase=phase_name, response_type=RESPONSE_TYPE,
        )
        entry = parse_scratchpad(raw_response, power)
    except Exception as exc:  # keep the game going; the previous notes stay in use
        logger.error(f"[{power}] Operative scratchpad call failed: {exc}", exc_info=True)
        entry = {"parsed": False, "valid": False, "problems": [f"call failed: {type(exc).__name__}"]}
    finally:
        await log_llm_response_async(
            log_file_path=log_file_path,
            model_name=client.model_name,
            power_name=power,
            phase=phase_name,
            response_type=RESPONSE_TYPE,
            raw_input_prompt=prompt,
            raw_response=raw_response,
            success="TRUE" if entry.get("valid") else f"FALSE: {'; '.join(entry.get('problems', []))}",
        )

    game_history._get_phase(phase_name).scratchpads[power] = entry
    if entry.get("parsed"):
        client.operative_notes = format_operative_notes(entry, plan_words, evidence_words)
    logger.info(f"[{power}] Operative scratchpad for {phase_name}: {'valid' if entry.get('valid') else entry.get('problems')}")
    return entry


def restore_operative_notes(agents: dict, game_history: "GameHistory", colluders: List[str],
                            plan_words: int, evidence_words: int) -> None:
    """On resume, show each operative its latest saved notes again."""
    for power in colluders:
        if power not in agents:
            continue
        latest = next(
            (ph.scratchpads[power] for ph in reversed(game_history.phases)
             if ph.scratchpads.get(power, {}).get("parsed")),
            None,
        )
        if latest:
            agents[power].client.operative_notes = format_operative_notes(latest, plan_words, evidence_words)
