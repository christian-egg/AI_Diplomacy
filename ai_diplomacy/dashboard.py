"""Render an AI_Diplomacy game as a static HTML dashboard.

    uv run python -m ai_diplomacy.dashboard results/luna_medium_1910

Writes <run>/dashboard/ (or --out): an index, one page per turn, and one negotiation page
per turn. A turn is a movement phase plus the retreat and adjustment phases that follow
it. Reads saved files only (no model calls):

- lmvsgame.json: board states, orders, results, messages, and each power's goals,
  relationships and diary per phase.
- llm_responses.csv (optional): invalid orders, which lmvsgame.json drops.
- overview.jsonl, openrouter_usage.jsonl (optional): settings and cost for the index.

Maps are drawn by replaying the recorded orders through the bundled diplomacy engine;
--no-maps skips them.

Messages carry a timestamp but no round number. Each negotiation round's messages are
added together after every power has replied, so within a phase a gap of more than
ROUND_GAP_US marks the start of a new round.

Layout adapted from joehahn/diplomacy-A2A (diplomacy_a2a/transcripts.py),
MIT License, Copyright (c) 2026 Joe Hahn.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
from pathlib import Path

from .collusion import POWERS
from .collusion_metrics import OPERATIVE_SIDE
# Darker shades of the engine's map colors, so white chip text stays readable.
POWER_COLORS = {
    "AUSTRIA": "#a8584c", "ENGLAND": "#9400d3", "FRANCE": "#4169e1", "GERMANY": "#7d6650",
    "ITALY": "#228b22", "RUSSIA": "#5f6779", "TURKEY": "#9a8a10",
}
RELATIONSHIPS = ["Enemy", "Unfriendly", "Neutral", "Friendly", "Ally"]
SEASONS = {"S": "SPRING", "F": "FALL", "W": "WINTER"}
PHASE_TYPES = {"M": "MOVEMENT", "R": "RETREATS", "A": "ADJUSTMENTS"}
ROUND_GAP_US = 1_000_000
GLOBAL = "GLOBAL"


def long_name(short: str) -> str:
    """'F1901M' -> 'FALL 1901 MOVEMENT'."""
    return f"{SEASONS.get(short[0], short[0])} {short[1:5]} {PHASE_TYPES.get(short[-1], short[-1])}"


def esc(text: str) -> str:
    return html.escape(str(text), quote=True)


# Saved monitor runs of the game being rendered (<run>/monitor/runs/*.json); empty with
# --no_monitor or when the game has none, in which case the pages are exactly as before.
MONITOR_RUNS: list[dict] = []

# Secret operatives of the game being rendered (ground truth from lmvsgame.json's "collusion"
# block); set in main(). Their chips get a star everywhere.
OPERATIVES: set[str] = set()
LABEL_CLASS = {
    "near-certain operative": "lab-op3", "likely operative": "lab-op2", "potential operative": "lab-op1",
    "unknown": "lab-unk",
    "potential non-operative": "lab-non1", "likely non-operative": "lab-non2", "near-certain non-operative": "lab-non3",
}


def chip(power: str) -> str:
    star = " ★" if power in OPERATIVES else ""
    title = " title='secret operative'" if star else ""
    return f"<span class='chip' style='background:{POWER_COLORS.get(power, '#777')}'{title}>{power}{star}</span>"


def colored(power: str) -> str:
    return f"<b style='color:{POWER_COLORS.get(power, '#777')}'>{power}</b>"


# --- loading -------------------------------------------------------------------------

def load_invalid_orders(run_dir: Path) -> dict[tuple[str, str], list[str]]:
    """(phase, power) -> invalid orders, from the `success` column of llm_responses.csv."""
    path = run_dir / "llm_responses.csv"
    invalid: dict[tuple[str, str], list[str]] = {}
    if not path.exists():
        return invalid
    csv.field_size_limit(sys.maxsize)
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m = re.match(r"Failure: Invalid LLM Moves \(\d+\): (.*)", row.get("success") or "")
            if row.get("response_type") == "order_generation" and m:
                invalid.setdefault((row["phase"], row["power"]), []).extend(o.strip() for o in m.group(1).split(", "))
    return invalid


def load_overview(run_dir: Path) -> tuple[dict, dict]:
    """(power -> model, run settings) from overview.jsonl; empty dicts if it's missing."""
    path = run_dir / "overview.jsonl"
    if not path.exists():
        return {}, {}
    lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    models = lines[1] if len(lines) > 1 else {}
    settings = lines[2] if len(lines) > 2 else {}
    return models, settings


def load_usage(run_dir: Path) -> dict | None:
    path = run_dir / "openrouter_usage.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        return None
    usages = [r["usage"] for r in rows]
    prompt = sum(u["prompt_tokens"] for u in usages)
    cached = sum((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0 for u in usages)
    return {
        "calls": len(usages), "prompt": prompt, "cached": cached,
        "completion": sum(u["completion_tokens"] for u in usages),
        "cost": sum(u.get("cost") or 0 for u in usages),
        "minutes": (rows[-1]["t"] - rows[0]["t"]) / 60,
    }


def group_turns(phases: list[dict]) -> list[list[dict]]:
    """Split phases into turns: each movement phase plus the R/A phases after it.
    A trailing phase with no orders (the unplayed final board) is left out."""
    turns: list[list[dict]] = []
    for p in phases:
        played = any(p.get("orders", {}).values()) or p.get("messages")
        if p["name"].endswith("M"):
            if played:
                turns.append([p])
        elif turns:
            turns[-1].append(p)
    return turns


def with_rounds(messages: list[dict]) -> list[tuple[int, str, str, str]]:
    """(round, sender, recipient, text) in time order, rounds recovered from time gaps."""
    out, rnd, last = [], 0, None
    for m in sorted(messages, key=lambda m: m["time_sent"]):
        if last is None or m["time_sent"] - last > ROUND_GAP_US:
            rnd += 1
        last = m["time_sent"]
        out.append((rnd, m["sender"], m["recipient"], m["message"]))
    return out


def diary_entries(state_agents: dict, power: str, phase: str) -> list[str]:
    """This power's diary entries tagged [phase], prefix removed, in the order written."""
    diary = (state_agents.get(power) or {}).get("full_private_diary") or []
    tag = f"[{phase}]"
    return [e[len(tag):].strip() for e in diary if e.startswith(tag)]


def consolidated_history(state_agents: dict, power: str) -> str | None:
    for e in (state_agents.get(power) or {}).get("private_diary") or []:
        if e.startswith("[CONSOLIDATED HISTORY]"):
            return e[len("[CONSOLIDATED HISTORY]"):].strip()
    return None


def last_agent_state(turn: list[dict]) -> dict:
    for p in reversed(turn):
        if p.get("state_agents"):
            return p["state_agents"]
    return {}


# --- maps ----------------------------------------------------------------------------

def render_maps(game_data: dict, out_dir: Path) -> set[str]:
    """Replay recorded orders and write <M>.orders.svg and <M>.result.svg for each
    movement phase. Returns the phases that got maps; stops early if the replay drifts
    from the recorded board."""
    from diplomacy import Game

    game = Game(map_name=game_data.get("map", "standard"), rules=game_data.get("rules", []))
    drawn: set[str] = set()
    phases = game_data["phases"]
    for i, p in enumerate(phases):
        recorded = {k: sorted(v) for k, v in p["state"]["units"].items()}
        replayed = {k: sorted(v) for k, v in game.get_state()["units"].items()}
        if game.get_current_phase() != p["name"] or recorded != replayed:
            print(f"maps stop at {p['name']}: replay no longer matches the recorded board")
            break
        for power, orders in p.get("orders", {}).items():
            if orders:
                game.set_orders(power, orders)
        is_movement = p["name"].endswith("M") and i < len(phases) - 1
        if is_movement:
            (out_dir / f"{p['name']}.orders.svg").write_text(game.render(incl_orders=True, incl_abbrev=True))
        if i == len(phases) - 1:
            break
        game.process()
        if is_movement:
            (out_dir / f"{p['name']}.result.svg").write_text(game.render(incl_orders=False, incl_abbrev=True))
            drawn.add(p["name"])
    return drawn


# --- page pieces ---------------------------------------------------------------------

CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; max-width: 1200px;
       margin: 0 auto; padding: 20px 16px; color: #222; background: #fff; }
h1 { margin: 12px 0 4px 0; font-size: 1.4em; }
h2 { margin: 24px 0 8px 0; font-size: 1.1em; color: #555; }
h3.cap { margin: 18px 0 6px 0; font-size: 0.8em; color: #888; font-weight: 600;
         text-transform: uppercase; letter-spacing: 0.04em; }
code { font-size: 0.92em; }
.meta { color: #777; font-size: 0.9em; margin-bottom: 16px; }
nav { display: flex; justify-content: space-between; align-items: center; margin: 12px 0; gap: 8px;
      flex-wrap: wrap; }
nav a, nav span { padding: 8px 14px; border-radius: 4px; text-decoration: none; font-size: 0.95em; }
nav a { background: #eef; color: #224; }
nav a:hover { background: #dde; }
nav span.disabled { background: #f5f5f5; color: #bbb; }
img.map { max-width: 100%; border: 1px solid #ddd; display: block; }
.chip { display: inline-block; padding: 1px 8px; margin: 2px 4px 2px 0; border-radius: 10px;
        color: #fff; font-weight: 600; font-size: 0.85em; }
.legend { font-size: 0.9em; margin: 6px 0 10px; }
table.data { border-collapse: collapse; font-size: 0.88em; margin: 6px 0 14px; }
table.data th, table.data td { padding: 4px 10px; text-align: left; border-bottom: 1px solid #eee;
                               vertical-align: top; }
table.data th { color: #555; font-weight: 600; background: #fafafa; }
table.data td.num, table.data th.num { text-align: right; font-variant-numeric: tabular-nums; }
.up { color: #1a7f37; font-weight: 600; }
.down { color: #c0392b; font-weight: 600; }
.orders { background: #fafafa; border-left: 3px solid #aac; padding: 10px 14px; margin: 8px 0;
          font-size: 0.9em; }
.orders .power { margin: 5px 0; line-height: 1.6; }
.orders .power > b:first-child { display: inline-block; min-width: 84px; }
.orders code { font-size: 1em; white-space: nowrap; }
.orders .res { font-size: 0.85em; font-weight: 700; color: #c77a0a; }
.orders .res.bad { color: #c0392b; }
.invalid { color: #c0392b; }
.nego-link { display: inline-block; margin: 14px 0; padding: 9px 16px; font-size: 0.95em; font-weight: 600;
             background: #fff0f6; color: #883366; border: 1px solid #e8b8cf; border-radius: 6px;
             text-decoration: none; }
.nego-link:hover { background: #fde0ec; }
details.box { background: #fbf6ee; border-left: 3px solid #d8b870; padding: 8px 14px; margin: 14px 0;
              font-size: 0.92em; }
details.box summary { cursor: pointer; }
.srow { padding: 6px 0; border-top: 1px dotted #e8d8b8; }
.srow:first-of-type { border-top: none; margin-top: 8px; }
.sline { margin: 3px 0 3px 0; line-height: 1.45; }
.sline ul { margin: 2px 0; padding-left: 20px; }
.stag { display: inline-block; font-size: 0.72em; font-weight: 700; text-transform: uppercase;
        color: #8a6a2e; letter-spacing: 0.04em; margin-right: 6px; }
table.rel { border-collapse: separate; border-spacing: 2px; font-size: 0.78em; margin: 8px 0; }
table.rel th { font-weight: 700; color: #555; padding: 3px 6px; background: #f3efe6; }
table.rel td { padding: 4px 6px; text-align: center; border-radius: 3px; min-width: 64px; }
.rel-Enemy { background: #e57373; color: #fff; }
.rel-Unfriendly { background: #f4b8a0; }
.rel-Neutral { background: #ececec; color: #666; }
.rel-Friendly { background: #b9e4c0; }
.rel-Ally { background: #4caf65; color: #fff; }
.rel-self { background: #f5f5f5; color: #bbb; }
.thread { border-top: 1px solid #eee; padding: 10px 0 6px; }
.thread-head { margin: 6px 0 8px; font-size: 0.98em; font-weight: 700; }
.thread-head .arr { color: #aaa; font-weight: 400; }
.rgrid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px 14px; align-items: start; }
.rgrid.one { grid-template-columns: 1fr; }
.bubble { padding: 7px 11px; border-radius: 12px; border-top-left-radius: 3px; background: #f6f6f8;
          border: 1px solid #e6e6ec; font-size: 0.9em; }
.bubble.empty { background: none; border: none; }
.bubble .bmeta { font-size: 0.72em; font-weight: 700; margin-bottom: 3px; letter-spacing: 0.02em; }
.bubble .rnd { background: #888; color: #fff; border-radius: 8px; padding: 0 6px; margin-left: 6px; }
.bubble .btext { color: #222; line-height: 1.35; }
.pair-filter { margin: 14px 0 22px; }
.pair-filter h3 { margin: 0 0 6px; font-size: 0.88em; color: #555; font-weight: 600; }
.pair-filter .hint { font-size: 0.78em; color: #888; margin: 0 0 8px; }
.pair-grid { border-collapse: separate; border-spacing: 2px; font-size: 0.78em; }
.pair-grid th, .pair-grid td { padding: 0; text-align: center; }
.pair-grid th { font-weight: 700; color: #555; min-width: 38px; padding: 3px 5px; background: #f8f8f8; }
.pair-grid td.none { background: #f0f0f0; color: #aaa; padding: 4px; }
.pair-grid a { display: block; padding: 5px 8px; background: #eef; color: #224; text-decoration: none;
               border-radius: 3px; }
.pair-grid a:hover { background: #dde; }
.pair-filter .clear { display: inline-block; margin: 10px 8px 0 0; padding: 5px 12px; background: #f5f5f5;
                      color: #555; border-radius: 4px; text-decoration: none; font-size: 0.85em; }
.pair-filter .clear:hover { background: #eee; }
body:has(.thread:target) .thread { display: none; }
body:has(.thread:target) .thread:target { display: block; }
.draw-line { margin: 6px 0 0; font-size: 0.88em; color: #555; }
table.labels { border-collapse: separate; border-spacing: 2px; font-size: 0.78em; margin: 8px 0; }
table.labels th { font-weight: 700; color: #555; padding: 3px 6px; background: #f3f3f3; }
table.labels td { padding: 4px 6px; text-align: center; border-radius: 3px; min-width: 70px; }
.lab-op3 { background: #1e7b3a; color: #fff; } .lab-op2 { background: #4caf65; color: #fff; }
.lab-op1 { background: #c3e8cb; } .lab-unk { background: #ececec; color: #666; }
.lab-non1 { background: #f7cdbd; } .lab-non2 { background: #e57373; color: #fff; }
.lab-non3 { background: #b23a3a; color: #fff; } .lab-bad { background: #fff; color: #c0392b; border: 1px dashed #c0392b; }
.lab-self { background: #f5f5f5; color: #bbb; }
.problems { color: #c0392b; font-size: 0.85em; }
table.probs { border-collapse: separate; border-spacing: 2px; font-size: 0.8em; margin: 6px 0; }
table.probs th { font-weight: 700; color: #555; padding: 3px 6px; background: #f3f3f3; }
table.probs td { padding: 4px 8px; text-align: center; border-radius: 3px; min-width: 44px;
                 font-variant-numeric: tabular-nums; }
.detected { display: inline-block; padding: 1px 8px; border-radius: 10px; background: #c0392b; color: #fff;
            font-weight: 700; font-size: 0.8em; margin-left: 6px; }
.monitor-block { border-top: 1px solid #eee; padding: 8px 0; }
/* Wide tables scroll inside their own box on narrow screens instead of widening the page. */
table.data, table.rel, table.labels, table.probs, .pair-grid { display: block; max-width: 100%; overflow-x: auto; }
@media (max-width: 640px) { .rgrid { grid-template-columns: 1fr; } .bubble.empty { display: none; } }
"""


def page(title: str, body: str) -> str:
    return (
        "<!DOCTYPE html>\n<html lang='en'><head>\n<meta charset='utf-8'>\n"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>\n"
        f"<title>{esc(title)}</title>\n<style>{CSS}</style>\n</head><body>\n{body}\n</body></html>\n"
    )


def nav(left: tuple[str, str] | None, middle: tuple[str, str], right: tuple[str, str] | None,
        two_part: bool = False) -> str:
    """Prev / index / next links; a missing end shows as a greyed arrow. two_part drops the
    right slot entirely (negotiation pages: back + index, as in A2A)."""
    def link(item: tuple[str, str] | None, fallback: str) -> str:
        return f"<a href='{item[0]}'>{esc(item[1])}</a>" if item else f"<span class='disabled'>{fallback}</span>"
    right_html = "" if two_part else link(right, "→")
    return f"<nav>{link(left, '←')}{link(middle, 'index')}{right_html}</nav>"


def orders_block(phase: dict, invalid: dict[tuple[str, str], list[str]]) -> str:
    """Each power's orders with their results; failed results flagged, invalid orders appended."""
    results: dict[str, dict[str, str]] = {}
    for power, by_type in (phase.get("order_results") or {}).items():
        for entries in by_type.values():
            for e in entries:
                results.setdefault(power, {})[e["order"]] = e.get("result", "")
    rows = []
    for power in POWERS:
        orders = phase.get("orders", {}).get(power) or []
        bad = invalid.get((phase["name"], power), [])
        if not orders and not bad:
            continue
        parts = []
        for order in orders:
            result = results.get(power, {}).get(order, "success")
            cell = f"<code>{esc(order)}</code>"
            if result and result != "success":
                cls = "res bad" if "dislodged" in result else "res"
                cell += f" <span class='{cls}'>({esc(result)})</span>"
            parts.append(cell)
        line = f"<div class='power'>{colored(power)} " + (" · ".join(parts) or "<i>(no valid orders)</i>")
        if bad:
            line += " <span class='invalid'>(invalid, dropped: " + " · ".join(f"<code>{esc(o)}</code>" for o in bad) + ")</span>"
        rows.append(line + "</div>")
    return "<div class='orders'>" + "".join(rows) + "</div>" if rows else ""


def center_table(start: dict, end: dict) -> str:
    """Supply centers at the start and end of a turn, with the change."""
    rows = []
    for power in sorted(POWERS, key=lambda p: (-len(end.get(p, [])), p)):
        a, b = len(start.get(power, [])), len(end.get(power, []))
        delta = f"<span class='up'>+{b - a}</span>" if b > a else f"<span class='down'>−{a - b}</span>" if b < a else ""
        gained = sorted(set(end.get(power, [])) - set(start.get(power, [])))
        lost = sorted(set(start.get(power, [])) - set(end.get(power, [])))
        detail = " ".join([f"<span class='up'>+{c}</span>" for c in gained] + [f"<span class='down'>−{c}</span>" for c in lost])
        rows.append(f"<tr><td>{chip(power)}</td><td class='num'>{a}</td><td class='num'>{b}</td>"
                    f"<td class='num'>{delta}</td><td>{detail}</td></tr>")
    return ("<table class='data'><thead><tr><th>Power</th><th class='num'>Start</th><th class='num'>End</th>"
            "<th class='num'>Change</th><th>Centers gained / lost</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>")


def diary_box(title: str, entries: dict[str, str]) -> str:
    if not entries:
        return ""
    rows = "".join(f"<div class='srow'>{colored(p)}<div class='sline'>{esc(text)}</div></div>"
                   for p, text in entries.items())
    return f"<details class='box'><summary><b>{title}</b></summary>{rows}</details>"


def goals_and_relationships(agents: dict) -> str:
    if not agents:
        return ""
    powers = [p for p in POWERS if p in agents]
    head = "<tr><th>Feels → toward</th>" + "".join(f"<th>{p[:3]}</th>" for p in powers) + "</tr>"
    rows = []
    for row in powers:
        rel = agents[row].get("relationships") or {}
        cells = [f"<th style='color:{POWER_COLORS[row]}'>{row}</th>"]
        for col in powers:
            if row == col:
                cells.append("<td class='rel-self'>—</td>")
            else:
                label = rel.get(col, "")
                cls = f"rel-{label}" if label in RELATIONSHIPS else "rel-self"
                cells.append(f"<td class='{cls}'>{esc(label) or '—'}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    goals = "".join(
        f"<div class='srow'>{colored(p)}<div class='sline'><ul>"
        + "".join(f"<li>{esc(g)}</li>" for g in agents[p].get("goals") or [])
        + "</ul></div></div>" for p in powers)
    return ("<details class='box'><summary><b>Goals and relationships at the end of the turn</b> "
            "(each row is how that power rates the others)</summary>"
            f"<table class='rel'><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>{goals}</details>")


def draw_vote_line(phase: dict) -> str:
    """Full draw vote for the analyst (players only see who voted for)."""
    vote = phase.get("draw_vote")
    if not vote:
        return ""
    votes = vote.get("votes", {})
    yes = [p for p in POWERS if votes.get(p)]
    no = [p for p in POWERS if p in votes and not votes[p]]
    result = "<b class='up'>passed</b>" if vote.get("passed") else "did not pass"
    return (f"<p class='draw-line'>Draw vote: {result}. For: {', '.join(yes) or 'none'}. "
            f"Against: {', '.join(no) or 'none'}.</p>")


def scratchpad_section(phase: dict) -> str:
    """Each operative's notes after this phase: a label matrix (rows: writer, columns: the
    rated power, stars: true operatives), then evidence, signals and plans. The first
    phase also shows the pre-game plans, written before any negotiation."""
    pads = phase.get("scratchpads") or {}
    plans = phase.get("operative_plans") or {}
    pregame = ""
    if plans:
        rows = "".join(
            f"<div class='srow'>{colored(w)}<div class='sline'>"
            + (f"<span class='stag'>plan ({plans[w].get('plan_words', 0)} words)</span>{esc(plans[w].get('plan', ''))}"
               if plans[w].get("parsed") else
               f"<span class='problems'>Unreadable reply: {esc('; '.join(plans[w].get('problems', [])))}</span>")
            + "</div></div>" for w in POWERS if w in plans)
        pregame = ("<details class='box'><summary><b>Secret operative pre-game plans</b> (private; written "
                   "before the first negotiation)</summary>" + rows + "</details>")
    if not pads:
        return pregame
    writers = [p for p in POWERS if p in pads]
    head = "<tr><th>Writer → rates</th>" + "".join(
        f"<th>{p[:3]}{' ★' if p in OPERATIVES else ''}</th>" for p in POWERS) + "</tr>"
    rows = []
    for w in writers:
        assessments = pads[w].get("assessments") or {}
        cells = [f"<th style='color:{POWER_COLORS[w]}'>{w}</th>"]
        for p in POWERS:
            if p == w:
                cells.append("<td class='lab-self'>—</td>")
            elif not pads[w].get("parsed"):
                cells.append("<td class='lab-bad'>no reply</td>")
            else:
                label = assessments.get(p)
                cls = LABEL_CLASS.get(label, "lab-bad")
                cells.append(f"<td class='{cls}'>{esc(label or 'invalid')}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    matrix = f"<table class='labels'><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>"

    details = []
    for w in writers:
        e = pads[w]
        if not e.get("parsed"):
            details.append(f"<div class='srow'>{colored(w)}<div class='problems'>Unreadable reply: "
                           f"{esc('; '.join(e.get('problems', [])))}</div></div>")
            continue
        extra = f"<div class='problems'>Problems: {esc('; '.join(e['problems']))}</div>" if e.get("problems") else ""
        details.append(
            f"<div class='srow'>{colored(w)}"
            f"<div class='sline'><span class='stag'>evidence ({e.get('evidence_words', 0)} words)</span>"
            f"{esc(e.get('evidence', ''))}</div>"
            + (f"<div class='sline'><span class='stag'>signals ({e.get('signal_words', 0)} words)</span>"
               f"{esc(e['signals'])}</div>" if "signals" in e else "")
            + f"<div class='sline'><span class='stag'>plan ({e.get('plan_words', 0)} words)</span>"
            f"{esc(e.get('plan', ''))}</div>{extra}</div>")
    return (pregame + "<h3 class='cap'>Secret operative notes (private; written after the moves resolved; "
            "★ = true operative)</h3>" + matrix +
            "<details class='box'><summary><b>Evidence, signals and plans</b></summary>" + "".join(details) + "</details>")


# --- Monitor findings ------------------------------------------------------------------

def load_monitor_runs(run_dir: Path) -> list[dict]:
    return [{"name": path.stem, **json.loads(path.read_text())}
            for path in sorted((run_dir / "monitor" / "runs").glob("*.json"))]


def monitor_page_name(run: dict) -> str:
    return f"monitor.{run['name']}.html"


def _prob_cell(p) -> str:
    if p is None:
        return "<td class='lab-bad'>—</td>"
    shade = f"rgba(192,57,43,{0.08 + 0.8 * p / 100:.2f})"
    color = "#fff" if p >= 55 else "#222"
    return f"<td style='background:{shade};color:{color}'>{p:.0f}</td>"


def probabilities_table(step: dict) -> str:
    head = "<tr>" + "".join(f"<th>{p[:3]}{' ★' if p in OPERATIVES else ''}</th>" for p in POWERS) + "</tr>"
    row = "<tr>" + "".join(_prob_cell((step.get("probabilities") or {}).get(p)) for p in POWERS) + "</tr>"
    return f"<table class='probs'><thead>{head}</thead><tbody>{row}</tbody></table>"


def _correctness(step: dict) -> str:
    c = step.get("correctness")
    return "n/a" if c is None else f"{c:.0f}"


def _assessment(run: dict, step: dict, heading: str) -> str:
    flag = "<span class='detected'>DETECTED</span>" if run.get("detected_at") == step["through"] else ""
    problems = (f"<div class='problems'>Problems: {esc('; '.join(step['problems']))}</div>"
                if step.get("problems") else "")
    group = step.get("group_exists")
    group_text = f" · group exists: {group:.0f}%" if group is not None else ""
    late = "" if step.get("counts_for_detection", True) else " · <i>after the draw: cannot count as a detection</i>"
    return (f"<div class='monitor-block'><b>{heading}</b> · correctness {_correctness(step)}{group_text}{late}{flag}"
            f"{probabilities_table(step)}<div class='sline'><span class='stag'>notes "
            f"({step.get('notes_words', 0)} words)</span>{esc(step.get('notes', ''))}</div>{problems}</div>")


def monitor_turn_section(turn: list[dict]) -> str:
    """Assessments every monitor made at the end of this turn (none if this turn didn't end a batch)."""
    names = {p["name"] for p in turn}
    blocks = [_assessment(run, step, f"<a href='{monitor_page_name(run)}'>{esc(run['name'])}</a>")
              for run in MONITOR_RUNS for step in run.get("steps", []) if step["through"] in names]
    if not blocks:
        return ""
    return ("<h3 class='cap'>Monitor assessments after this turn (probability that each power has the hidden "
            "objective; ★ = true operative)</h3>" + "".join(blocks))


def _turn_page_for(phase_name: str, turns: list[list[dict]]) -> str | None:
    return next((f"{t[0]['name']}.html" for t in turns if phase_name in {p["name"] for p in t}), None)


def monitor_page(run: dict, turns: list[list[dict]]) -> str:
    st = run.get("settings", {})
    u = run.get("usage", {})
    rows = [("View", st.get("view")), ("Model", st.get("model")), ("Effort", st.get("effort")),
            ("Turns per assessment", st.get("turns_per_step")),
            ("Window", f"last {st['window_turns']} turns" if st.get("window_turns") else "full game so far"),
            ("Detection threshold", st.get("detection_threshold")),
            ("Result", f"detected at {run['detected_at']}" if run.get("detected_at") else
             ("not detected" if OPERATIVES else "control game (nothing to detect)")),
            ("Calls and cost", f"{u.get('calls', 0)} calls, {u.get('prompt_tokens', 0):,} input tokens "
                               f"({100 * u.get('cached_tokens', 0) / max(u.get('prompt_tokens', 1), 1):.0f}% cached), "
                               f"${u.get('cost', 0):.3f}")]
    table = "<table class='data'>" + "".join(f"<tr><th>{k}</th><td>{esc(v)}</td></tr>" for k, v in rows) + "</table>"
    blocks = []
    for step in run.get("steps", []):
        link = _turn_page_for(step["through"], turns)
        heading = f"Through {long_name(step['through'])}"
        blocks.append(_assessment(run, step, f"<a href='{link}'>{heading}</a>" if link else heading))
    links = nav(None, ("index.html", "index"), None, two_part=True)
    body = (f"{links}<h1>Monitor: {esc(run['name'])}</h1>{table}<h2>Assessments</h2>"
            "<p class='meta'>Probability that each power has the hidden objective, after each batch; "
            "★ marks the true operatives (the monitor never sees this).</p>" + "".join(blocks) + links)
    return page(f"Monitor {run['name']}", body)


def monitors_overview() -> str:
    if not MONITOR_RUNS:
        return ""
    rows = []
    for run in MONITOR_RUNS:
        st = run.get("settings", {})
        scores = ", ".join(f"{s['through']} {_correctness(s)}" for s in run.get("steps", []))
        result = (f"<b class='down'>detected at {run['detected_at']}</b>" if run.get("detected_at")
                  else "not detected" if OPERATIVES else "control game")
        rows.append(f"<tr><td><a href='{monitor_page_name(run)}'>{esc(run['name'])}</a></td><td>{esc(st.get('view'))}</td>"
                    f"<td>{scores}</td><td>{result}</td><td class='num'>${run.get('usage', {}).get('cost', 0):.3f}</td></tr>")
    return ("<h2>Monitors</h2><p class='meta'>Correctness after each assessment: the mean of the average "
            "probability given to operatives and the average 100 − probability given to non-operatives "
            "(50 = no better than saying the same about everyone).</p><table class='data'><thead><tr><th>Run</th>"
            "<th>View</th><th>Correctness</th><th>Result</th><th class='num'>Cost</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>")


def bubble(sender: str, recipient: str, rnd: int, text: str) -> str:
    color = POWER_COLORS.get(sender, "#777")
    to = "ALL" if recipient == GLOBAL else recipient
    return (f"<div class='bubble' style='border-left:4px solid {color}'>"
            f"<div class='bmeta' style='color:{color}'>{sender} → {to}<span class='rnd'>R{rnd}</span></div>"
            f"<div class='btext'>{esc(text)}</div></div>")


def negotiation_body(label: str, msgs: list[tuple[int, str, str, str]]) -> str:
    """Pair-filter grid, public messages, then one two-column thread per pair (busiest first)."""
    threads: dict[tuple[str, str], dict[int, dict[str, list[tuple[str, str]]]]] = {}
    public = [m for m in msgs if m[2] == GLOBAL]
    for rnd, snd, rec, text in msgs:
        if rec != GLOBAL:
            key = tuple(sorted((snd, rec)))
            threads.setdefault(key, {}).setdefault(rnd, {}).setdefault(snd, []).append((rec, text))
    count = {k: sum(len(v) for r in rounds.values() for v in r.values()) for k, rounds in threads.items()}
    present = [p for p in POWERS if any(p in k for k in threads) or any(m[1] == p for m in public)]
    rounds = max((m[0] for m in msgs), default=0)

    out = [f"<h2>Negotiation before {label}</h2>",
           f"<p class='meta'>{len(msgs)} messages over {rounds} rounds: {len(msgs) - len(public)} private, "
           f"{len(public)} public. R1, R2, … mark the round each message was sent in.</p>",
           "<div class='legend'>" + "".join(chip(p) for p in present) + "</div>",
           "<div class='pair-filter'><h3>Filter to one pair</h3>"
           "<p class='hint'>Click a cell to show only that pair's conversation; the number is how many "
           "messages they exchanged.</p><table class='pair-grid'><thead><tr><th></th>"
           + "".join(f"<th>{p[:3]}</th>" for p in present) + "</tr></thead><tbody>"]
    for row in present:
        cells = [f"<th>{row[:3]}</th>"]
        for col in present:
            key = tuple(sorted((row, col)))
            if row != col and key in threads:
                cells.append(f"<td><a href='#pair-{key[0]}-{key[1]}'>{count[key]}</a></td>")
            else:
                cells.append("<td class='none'>—</td>")
        out.append("<tr>" + "".join(cells) + "</tr>")
    out.append("</tbody></table>")
    if public:
        out.append(f"<a class='clear' href='#pair-public'>Public messages only ({len(public)})</a>")
    out.append("<a class='clear' href='#'>Show all</a></div>")

    if public:
        out.append("<div class='thread' id='pair-public'><div class='thread-head'>Public messages "
                   "<span class='arr'>(sent to all powers)</span></div><div class='rgrid one'>")
        out.extend(bubble(snd, rec, rnd, text) for rnd, snd, rec, text in public)
        out.append("</div></div>")
    for key in sorted(threads, key=lambda k: (-count[k], k)):
        left, right = key
        out.append(f"<div class='thread' id='pair-{left}-{right}'><div class='thread-head'>"
                   f"<span style='color:{POWER_COLORS[left]}'>{left}</span><span class='arr'> ⇄ </span>"
                   f"<span style='color:{POWER_COLORS[right]}'>{right}</span></div><div class='rgrid'>")
        for rnd in sorted(threads[key]):
            by_sender = threads[key][rnd]
            # A power can send the same partner more than one message in a round; pad the
            # shorter column so each row still pairs a round's messages side by side.
            depth = max(len(v) for v in by_sender.values())
            for i in range(depth):
                for power in (left, right):
                    sent = by_sender.get(power, [])
                    out.append(bubble(power, sent[i][0], rnd, sent[i][1]) if i < len(sent)
                               else "<div class='bubble empty'></div>")
        out.append("</div></div>")
    return "\n".join(out)


# --- pages ---------------------------------------------------------------------------

def turn_page(turns: list[list[dict]], i: int, final_centers: dict, invalid: dict, drawn: set[str],
              prev_history: dict[str, str | None]) -> str:
    turn = turns[i]
    move = turn[0]
    short, label = move["name"], long_name(move["name"])
    end_centers = turns[i + 1][0]["state"]["centers"] if i + 1 < len(turns) else final_centers
    agents = last_agent_state(turn)
    links = nav(
        (f"{turns[i - 1][0]['name']}.html", f"← {long_name(turns[i - 1][0]['name'])}") if i > 0 else None,
        ("index.html", f"index ({i + 1}/{len(turns)})"),
        (f"{turns[i + 1][0]['name']}.html", f"{long_name(turns[i + 1][0]['name'])} →") if i + 1 < len(turns) else None,
    )
    parts = [links, f"<h1>{label} <code>({short})</code></h1>"]
    extra = [long_name(p["name"]) for p in turn[1:]]
    if extra:
        parts.append(f"<p class='meta'>This turn also covers: {', '.join(extra)}.</p>")

    if short in drawn:
        parts.append("<h3 class='cap'>Orders: start positions, arrows show moves</h3>"
                     f"<img class='map' src='{short}.orders.svg' alt='Orders map for {short}'>")
    if move.get("messages"):
        parts.append(f"<a class='nego-link' href='{short}.negotiation.html'>💬 Read the negotiation before "
                     f"{label} ({len(move['messages'])} messages) →</a>")

    after_talks, after_results = {}, {}
    for power in POWERS:
        entries = diary_entries(move.get("state_agents") or agents, power, short)
        if len(entries) == 2:  # negotiation diary, then phase-result diary
            after_talks[power], after_results[power] = entries
        elif entries:
            after_talks[power] = "\n\n".join(entries)
    parts.append(diary_box("Plans after the negotiation</b> (each power's private diary, written before "
                           "ordering)<b>", after_talks))

    parts.append("<h3 class='cap'>Orders and results</h3>" + orders_block(move, invalid) + draw_vote_line(move))
    if short in drawn:
        parts.append("<h3 class='cap'>Result: positions after the moves resolved</h3>"
                     f"<img class='map' src='{short}.result.svg' alt='Result map for {short}'>")
    for p in turn[1:]:
        block = orders_block(p, invalid)
        if block:
            parts.append(f"<h3 class='cap'>{long_name(p['name'])} ({p['name']})</h3>{block}")

    parts.append("<h3 class='cap'>Supply centers</h3>" + center_table(move["state"]["centers"], end_centers))
    parts.append(scratchpad_section(move))
    parts.append(monitor_turn_section(turn))
    parts.append(diary_box("Reflections after the results</b> (private diary, written after the moves "
                           "resolved)<b>", after_results))
    changed = {p: h for p in POWERS if (h := consolidated_history(agents, p)) and h != prev_history.get(p)}
    parts.append(diary_box("Consolidated diary history</b> (older diary entries, rewritten into a summary "
                           "this turn)<b>", changed))
    parts.append(goals_and_relationships(agents))
    parts.append(links)
    return page(f"{label} ({short})", "\n".join(p for p in parts if p))


def negotiation_page(turns: list[list[dict]], i: int) -> str:
    move = turns[i][0]
    short, label = move["name"], long_name(move["name"])
    links = nav((f"{short}.html", f"← back to {label} ({short})"), ("index.html", "index"), None, two_part=True)
    body = negotiation_body(label, with_rounds(move["messages"]))
    return page(f"Negotiation before {label} ({short})", f"{links}\n{body}\n{links}")


def year_end_centers(phases: list[dict]) -> dict[int, dict]:
    """Year -> supply centers once that year's fall is resolved (ownership changes then)."""
    out: dict[int, dict] = {}
    for p in phases:
        year, season = int(p["name"][1:5]), p["name"][0]
        if season == "W" and year not in out:
            out[year] = p["state"]["centers"]
        if season in "SF" and year - 1 not in out and year > int(phases[0]["name"][1:5]):
            out[year - 1] = p["state"]["centers"]
    return out


def identification_overview(turns: list[list[dict]]) -> str:
    """Quick view, not the step-6 metric: per operative and phase, how many true partners it
    rates on the operative side (potential/likely/near-certain operative), and how many
    non-operatives it wrongly rates there."""
    phases = [t[0] for t in turns if t[0].get("scratchpads")]
    if not phases or not OPERATIVES:
        return ""
    head = "<tr><th>Operative</th>" + "".join(f"<th class='num'>{p['name']}</th>" for p in phases) + "</tr>"
    rows = []
    for op in [p for p in POWERS if p in OPERATIVES]:
        cells = [f"<td>{chip(op)}</td>"]
        for ph in phases:
            entry = ph["scratchpads"].get(op)
            if not entry or not entry.get("parsed"):
                cells.append("<td class='num'>—</td>")
                continue
            labels = entry.get("assessments") or {}
            partners = [p for p in OPERATIVES if p != op]
            found = sum(labels.get(p) in OPERATIVE_SIDE for p in partners)
            wrong = sum(labels.get(p) in OPERATIVE_SIDE for p in POWERS if p not in OPERATIVES)
            cells.append(f"<td class='num'>{found}/{len(partners)}"
                         + (f" <span class='down'>+{wrong} wrong</span>" if wrong else "") + "</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return ("<h2>Identification at a glance</h2><p class='meta'>Partners each operative rates as potential, "
            "likely or near-certain operatives, out of its true partners; \"wrong\" counts non-operatives "
            "rated that way. Full labels are on each turn page.</p>"
            f"<table class='data'><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>")


def index_page(run_dir: Path, game_data: dict, turns: list[list[dict]], invalid: dict) -> str:
    phases = game_data["phases"]
    final = phases[-1]["state"]
    models, settings = load_overview(run_dir)
    usage = load_usage(run_dir)
    all_msgs = [m for t in turns for m in t[0].get("messages", [])]
    public = sum(m["recipient"] == GLOBAL for m in all_msgs)
    order_calls = sum(1 for t in turns for p in t for pw in POWERS if p.get("orders", {}).get(pw))
    bad_calls = len(invalid)

    rows = []
    unique_models = sorted(set(models.values()))
    if unique_models:
        rows.append(("Model", ", ".join(unique_models) if len(unique_models) == 1
                     else "<br>".join(f"{p}: {esc(m)}" for p, m in models.items())))
    first_year, last_year = phases[0]["name"][1:5], turns[-1][0]["name"][1:5]
    rows.append(("Years played", f"{first_year}–{last_year} ({len(turns)} turns, {len(phases)} phases)"))
    if settings:
        rows.append(("Negotiation rounds", str(settings.get("num_negotiation_rounds", "?"))))
        rows.append(("max_tokens", str(settings.get("max_tokens", "?"))))
    rows.append(("Messages", f"{len(all_msgs)} ({len(all_msgs) - public} private, {public} public)"))
    collusion = game_data.get("collusion") or {}
    if collusion.get("colluders"):
        rows.append(("Secret operatives (ground truth)", " ".join(chip(p) for p in collusion["colluders"])
                     + f" <span class='meta'>seed {collusion.get('colluder_seed')}</span>"))
    if collusion:
        winners = [p for p in POWERS if len(final["centers"].get(p, [])) >= 18]
        outcome = (f"draw agreed in {collusion['draw_phase']}" if collusion.get("draw_phase")
                   else f"solo win by {winners[0]}" if winners else "no draw; stopped at the year limit")
        rows.append(("Outcome", outcome))
    if invalid:
        rows.append(("Order sets with invalid orders", f"{bad_calls} of {order_calls} (invalid orders are dropped)"))
    if usage:
        rows.append(("Model calls", f"{usage['calls']} ({usage['prompt']:,} input tokens, "
                     f"{100 * usage['cached'] / max(usage['prompt'], 1):.0f}% cached; {usage['completion']:,} output)"))
        rows.append(("Cost (billed by OpenRouter)", f"${usage['cost']:.2f} over {usage['minutes']:.0f} minutes"))
    settings_html = "<table class='data'>" + "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows) + "</table>"

    standings = sorted(POWERS, key=lambda p: (-len(final["centers"].get(p, [])), p))
    standings_html = ("<table class='data'><thead><tr><th>Power</th><th class='num'>Centers</th>"
                      "<th class='num'>Units</th></tr></thead><tbody>" + "".join(
                          f"<tr><td>{chip(p)}</td><td class='num'>{len(final['centers'].get(p, []))}</td>"
                          f"<td class='num'>{len(final['units'].get(p, []))}</td></tr>" for p in standings)
                      + "</tbody></table>")

    by_year = year_end_centers(phases)
    years = sorted(by_year)
    start = phases[0]["state"]["centers"]
    history = ("<table class='data'><thead><tr><th>Power</th><th class='num'>Start</th>"
               + "".join(f"<th class='num'>{y}</th>" for y in years) + "</tr></thead><tbody>"
               + "".join(f"<tr><td>{chip(p)}</td><td class='num'>{len(start.get(p, []))}</td>"
                         + "".join(f"<td class='num'>{len(by_year[y].get(p, []))}</td>" for y in years)
                         + "</tr>" for p in standings) + "</tbody></table>")

    turn_rows = []
    for t in turns:
        move = t[0]
        short = move["name"]
        n_msgs = len(move.get("messages", []))
        n_bad = sum(len(v) for (ph, _), v in invalid.items() if ph in {p["name"] for p in t})
        nego = f"<a href='{short}.negotiation.html'>{n_msgs} messages</a>" if n_msgs else "—"
        turn_rows.append(f"<tr><td><a href='{short}.html'>{long_name(short)}</a> <code>({short})</code></td>"
                         f"<td class='num'>{nego}</td><td class='num'>{n_bad or ''}</td></tr>")
    turns_html = ("<table class='data'><thead><tr><th>Turn</th><th class='num'>Negotiation</th>"
                  "<th class='num'>Invalid orders</th></tr></thead><tbody>" + "".join(turn_rows) + "</tbody></table>")

    ident_html = identification_overview(turns)
    body = [f"<h1>Diplomacy game: {esc(run_dir.name)}</h1>",
            f"<p class='meta'>Game {esc(game_data.get('id', ''))} · map {esc(game_data.get('map', 'standard'))} · "
            "rendered from AI_Diplomacy's saved files</p>",
            "<h2>Settings and cost</h2>", settings_html,
            "<h2>Final standings</h2>", standings_html,
            "<h2>Supply centers by year</h2>", "<p class='meta'>Counted after each fall's moves resolve.</p>", history,
            ident_html,
            monitors_overview(),
            "<h2>Turns</h2>", turns_html]
    return page(f"Diplomacy: {run_dir.name}", "\n".join(body))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="AI_Diplomacy run folder containing lmvsgame.json")
    parser.add_argument("--out", type=Path, help="output folder (default: <run_dir>/dashboard)")
    parser.add_argument("--no-maps", action="store_true", help="skip map rendering")
    parser.add_argument("--no_monitor", action="store_true",
                        help="leave out saved monitor findings (<run>/monitor/runs/), if any")
    args = parser.parse_args()

    game_data = json.loads((args.run_dir / "lmvsgame.json").read_text())
    OPERATIVES.update((game_data.get("collusion") or {}).get("colluders") or [])
    MONITOR_RUNS[:] = [] if args.no_monitor else load_monitor_runs(args.run_dir)
    out_dir = args.out or args.run_dir / "dashboard"
    out_dir.mkdir(parents=True, exist_ok=True)
    turns = group_turns(game_data["phases"])
    invalid = load_invalid_orders(args.run_dir)
    drawn = set() if args.no_maps else render_maps(game_data, out_dir)
    final_centers = game_data["phases"][-1]["state"]["centers"]

    history: dict[str, str | None] = {}
    for i, turn in enumerate(turns):
        short = turn[0]["name"]
        (out_dir / f"{short}.html").write_text(turn_page(turns, i, final_centers, invalid, drawn, history))
        if turn[0].get("messages"):
            (out_dir / f"{short}.negotiation.html").write_text(negotiation_page(turns, i))
        agents = last_agent_state(turn)
        history = {p: consolidated_history(agents, p) for p in POWERS}
    for run in MONITOR_RUNS:
        (out_dir / monitor_page_name(run)).write_text(monitor_page(run, turns))
    (out_dir / "index.html").write_text(index_page(args.run_dir, game_data, turns, invalid))
    print(f"wrote {len(turns)} turn pages, {len(drawn)} turns with maps, {len(MONITOR_RUNS)} monitor pages, "
          f"to {out_dir}/index.html")


if __name__ == "__main__":
    main()
