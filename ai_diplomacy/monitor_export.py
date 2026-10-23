"""What each monitor may see of a finished game, exported from its saved files.

    uv run python -m ai_diplomacy.monitor_export results/luna_high_2y_ops4
    uv run python -m ai_diplomacy.monitor_export results/luna_high_2y_ops4 --drop_draw_phase

Writes <run>/monitor/actions.json and <run>/monitor/trajectory.json. The monitor itself
(prompt, model, scoring) comes later. Decided with the user on 2026-10-14:

- Action-only view (the hosting platform's view), per phase:
  - each power's supply centers at the start of the phase;
  - every order with its result: the engine's adjudication, and "invalid" for orders the
    engine rejected;
  - failed draw votes (only the powers that voted for a draw, as players see them).
- Trajectory view (the deployer's view): the same, plus every message (private and
  public) with its negotiation round.
- Negotiations view (added 2026-10-23): only the messages, with their rounds, for each
  movement phase that had any. No supply centers, orders, results or draw votes, so it
  shows how much a monitor gets from the text alone.
- If a draw passes, the export stops at that phase. It keeps the phase's orders and
  results, but not its vote and nothing after it. --drop_draw_phase removes that phase
  too, in case it gives too much away.
- Never exported:
  - diaries, goals and relationships;
  - order reasoning;
  - operatives' scratchpads;
  - the "collusion" block (operatives, seed);
  - model names, settings and timestamps;
  - how many operatives there are (that belongs in the monitor's prompt).

The monitor will read a game one turn at a time, so up_to_phase() and up_to_year() cut an
export after a given phase or year.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from .collusion import POWERS
from .dashboard import GLOBAL, load_invalid_orders, with_rounds

VIEWS = ("actions", "trajectory", "negotiations")


def _phase_results(phase: dict) -> dict[str, str]:
    """order -> result tag, from the engine's per-unit adjudication where it has one.
    (order_results keeps only the first tag, e.g. "cut" for "cut, dislodged".)"""
    engine = phase.get("results") or {}
    fallback = {e["order"]: e.get("result", "") for by_type in (phase.get("order_results") or {}).values()
                for entries in by_type.values() for e in entries}
    out = {}
    for orders in (phase.get("orders") or {}).values():
        for order in orders:
            if order.upper() == "WAIVE":
                out[order] = "waived"
                continue
            tags = engine.get(" ".join(order.split()[:2]))
            if tags is not None:
                tags = [t for t in tags if t]  # a successful build is recorded as [""]
                out[order] = ", ".join(tags) if tags else "success"
            else:
                out[order] = fallback.get(order) or "success"
    return out


def _played(phase: dict) -> bool:
    return phase["name"] != "COMPLETED" and bool(any((phase.get("orders") or {}).values()) or phase.get("messages"))


def export_game(game: dict, invalid_orders: dict, view: str, drop_draw_phase: bool = False) -> dict:
    """The export for one view. invalid_orders maps (phase, power) -> rejected orders."""
    if view not in VIEWS:
        raise ValueError(f"view must be one of {VIEWS}, got {view!r}")
    draw_phase = (game.get("collusion") or {}).get("draw_phase")
    phases_out = []
    for phase in game["phases"]:
        name = phase["name"]
        if draw_phase and name == draw_phase and drop_draw_phase:
            break
        if not _played(phase):
            continue
        messages = [
            {"round": rnd, "sender": snd, "recipient": "ALL" if rec == GLOBAL else rec, "text": text}
            for rnd, snd, rec, text in with_rounds(phase.get("messages") or [])
        ]
        if view == "negotiations":
            if messages:
                phases_out.append({"phase": name, "messages": messages})
            if draw_phase and name == draw_phase:
                break
            continue
        results = _phase_results(phase)
        orders = {}
        for power in POWERS:
            entries = [{"order": o, "result": results[o]} for o in (phase.get("orders") or {}).get(power, [])]
            entries += [{"order": o, "result": "invalid"} for o in invalid_orders.get((name, power), [])]
            if entries:
                orders[power] = entries
        out = {
            "phase": name,
            "supply_centers": {p: list(phase["state"]["centers"].get(p, [])) for p in POWERS},
            "orders": orders,
        }
        vote = phase.get("draw_vote")
        if vote and not vote.get("passed"):  # the passing vote is never shown
            out["draw_vote"] = {"voted_for": [p for p in POWERS if vote.get("votes", {}).get(p)]}
        if view == "trajectory":
            out["messages"] = messages
        phases_out.append(out)
        if draw_phase and name == draw_phase:
            break  # nothing after the phase in which the draw passed
    return {"view": view, "powers": list(POWERS), "phases": phases_out}


def up_to_phase(export: dict, phase_name: str) -> dict:
    """The export cut after `phase_name` (inclusive)."""
    names = [p["phase"] for p in export["phases"]]
    if phase_name not in names:
        raise ValueError(f"{phase_name} is not in this export")
    return {**export, "phases": export["phases"][: names.index(phase_name) + 1]}


def up_to_year(export: dict, year: int) -> dict:
    """The export cut at the end of `year`, i.e. after its winter builds."""
    return {**export, "phases": [p for p in export["phases"] if int(p["phase"][1:5]) <= year]}


def write_exports(run_dir: Path, drop_draw_phase: bool = False, out_dir: Optional[Path] = None) -> list[Path]:
    game = json.loads((run_dir / "lmvsgame.json").read_text())
    invalid = load_invalid_orders(run_dir)
    out_dir = out_dir or run_dir / "monitor"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for view in VIEWS:
        path = out_dir / f"{view}.json"
        path.write_text(json.dumps(export_game(game, invalid, view, drop_draw_phase), indent=1))
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="run folder containing lmvsgame.json")
    parser.add_argument("--drop_draw_phase", action="store_true",
                        help="leave out the phase in which a draw passed (default: keep its orders)")
    parser.add_argument("--out", type=Path, help="output folder (default: <run_dir>/monitor)")
    args = parser.parse_args()
    for path in write_exports(args.run_dir, args.drop_draw_phase, args.out):
        data = json.loads(path.read_text())
        print(f"wrote {path} ({len(data['phases'])} phases)")


if __name__ == "__main__":
    main()
