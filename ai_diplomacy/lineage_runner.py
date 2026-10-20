"""Run lineages of games in which the operatives learn from their own past games (ICRL).

    uv run python -m ai_diplomacy.lineage_runner collusion_configs/icrl_2y.json --dry_run
    uv run python -m ai_diplomacy.lineage_runner collusion_configs/icrl_2y.json --env_file ~/CoordinationGames/.env

Decided with the user on 2026-10-19:

- A lineage is a fixed set of agents 1-7 playing one game per round. Agents 1-4 are always
  the operatives and 5-7 never are; each round the seven are dealt the powers at random
  (seeded by lineage and round), so the operatives' powers change every game.
- After each game, every operative reflects on it (operative_reflection.py), using the
  "feedback_monitor"'s view of the game, and writes a note. In the next round that agent's
  note goes into its system prompt, whatever power it plays. Notes are private to each
  agent: nothing is pooled, and agents are not told their partners are the same players.
  Non-operatives (5-7) don't reflect and carry nothing.
- "eval_monitors" score the games but never feed back, so progress isn't measured only by
  the monitor the operatives are learning from.
- Lineages are independent replicates and run in parallel ("parallel_games" at a time);
  rounds within a lineage run in order.

Games go to results/<name>/lin<L>-r<RR>/, each with lineage.json (round, seed, which agent
played which power, the notes it started with) and operative_notes.json (the notes as
lm_game reads them). Every step is skipped if it already finished, so an interrupted run
resumes with the same command, and raising "rounds" in the config extends it.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import statistics
import subprocess
import sys
from pathlib import Path

from .collusion import POWERS
from .collusion_runner import (REPO, game_finished, model_label, monitor_cmd, monitor_name, read_env_file,
                               run_one, upload_cmd)

N_AGENTS = len(POWERS)
OPERATIVE_AGENTS = (1, 2, 3, 4)
DEFAULTS = {
    "results_dir": "results",
    "game_length": "2y",
    "num_negotiation_rounds": 3,
    "lineages": 5,
    "rounds": 6,
    "seed_base": 5000,
    "parallel_games": 5,
    "extra_game_args": [],
    "eval_monitors": [],
    "judges": [],
    "dashboards": True,
    "reflection": {},
    "docent_collection_id": None,
}
REQUIRED = ("name", "model", "reasoning_effort", "feedback_monitor")


def load_config(path: Path) -> dict:
    config = {**DEFAULTS, **json.loads(path.read_text())}
    missing = [k for k in REQUIRED if k not in config]
    if missing:
        raise SystemExit(f"{path}: missing {', '.join(missing)}")
    config["reflection"] = {"model": config["model"], "effort": config["reasoning_effort"],
                            "note_words": 150, "summary_words": 80, **config["reflection"]}
    return config


def deal_powers(seed: int) -> dict:
    """{agent (1-7): power} for one game."""
    return dict(zip(range(1, N_AGENTS + 1), random.Random(seed).sample(POWERS, N_AGENTS)))


def game_seed(config: dict, lineage: int, rnd: int) -> int:
    return config["seed_base"] + 100 * lineage + rnd


def run_dir_for(config: dict, lineage: int, rnd: int) -> Path:
    return REPO / config["results_dir"] / config["name"] / f"lin{lineage}-r{rnd:02d}"


def notes_after(config: dict, lineage: int, rnd: int) -> dict:
    """{agent: note} after rounds 1..rnd of a lineage: each operative agent's latest note
    (an agent whose reflection is missing or unreadable keeps the note it had)."""
    notes = {}
    for r in range(1, rnd + 1):
        run_dir = run_dir_for(config, lineage, r)
        info_path = run_dir / "lineage.json"
        if not info_path.exists():
            break
        powers = json.loads(info_path.read_text())["powers"]
        for agent in OPERATIVE_AGENTS:
            path = run_dir / "reflections" / f"{powers[str(agent)]}.json"
            if path.exists() and json.loads(path.read_text()).get("carried_note"):
                notes[agent] = json.loads(path.read_text())["carried_note"]
    return notes


def all_monitors(config: dict) -> list:
    return [config["feedback_monitor"], *config["eval_monitors"]]


def prepare_game(config: dict, lineage: int, rnd: int) -> dict:
    """Write lineage.json and operative_notes.json, and return the game for run_one."""
    seed = game_seed(config, lineage, rnd)
    powers = deal_powers(seed)
    notes = notes_after(config, lineage, rnd - 1)
    operatives = sorted(powers[a] for a in OPERATIVE_AGENTS)
    run_dir = run_dir_for(config, lineage, rnd)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "lineage.json").write_text(json.dumps({
        "lineage": lineage, "round": rnd, "seed": seed,
        "powers": {str(a): p for a, p in powers.items()}, "operative_agents": list(OPERATIVE_AGENTS),
        "notes": {str(a): n for a, n in notes.items()}}, indent=1))
    notes_file = run_dir / "operative_notes.json"
    notes_file.write_text(json.dumps({powers[a]: n for a, n in notes.items()}, indent=1))
    cmd = [sys.executable, "lm_game.py",
           "--run_dir", str(run_dir.relative_to(REPO)),
           "--game_length", config["game_length"],
           "--n_colluders", str(len(OPERATIVE_AGENTS)),
           "--colluders", ",".join(operatives),
           "--operative_notes", str(notes_file.relative_to(REPO)),
           "--num_negotiation_rounds", str(config["num_negotiation_rounds"]),
           "--models", ",".join([config["model"]] * N_AGENTS),
           "--reasoning_effort", config["reasoning_effort"],
           *[str(a) for a in config["extra_game_args"]]]
    return {"condition": config["name"], "run_dir": run_dir, "cmd": cmd}


def reflection_cmd(run_dir: Path, config: dict) -> list:
    r = config["reflection"]
    cmd = [sys.executable, "-m", "ai_diplomacy.operative_reflection", str(run_dir.relative_to(REPO)),
           "--model", r["model"], "--monitor", monitor_name(config["feedback_monitor"]),
           "--note_words", str(r["note_words"]), "--summary_words", str(r["summary_words"])]
    if r.get("effort"):
        cmd += ["--effort", r["effort"]]
    return cmd


def reflections_done(run_dir: Path) -> bool:
    game = json.loads((run_dir / "lmvsgame.json").read_text())
    return all((run_dir / "reflections" / f"{p}.json").exists() for p in game["collusion"]["colluders"])


def run_lineage(config: dict, lineage: int, env: dict) -> str:
    batch = {**config, "monitors": all_monitors(config)}
    for rnd in range(1, config["rounds"] + 1):
        game = prepare_game(config, lineage, rnd)
        run_one(game, batch, env)
        if not reflections_done(game["run_dir"]):
            with (game["run_dir"] / "runner.log").open("a") as f:
                cmd = reflection_cmd(game["run_dir"], config)
                f.write(f"\n$ {' '.join(cmd)}\n")
                f.flush()
                subprocess.run(cmd, cwd=REPO, env=env, stdout=f,
                               stderr=subprocess.STDOUT, check=True)
        print(f"lineage {lineage}: round {rnd} done", flush=True)
    return f"lineage {lineage}: {config['rounds']} rounds done"


# --- Summary --------------------------------------------------------------------------------

def round_summary(config: dict) -> list:
    """One row per round, over lineages: identification and each monitor's detections."""
    from .collusion_metrics import game_metrics, monitor_summaries
    rows = []
    for rnd in range(1, config["rounds"] + 1):
        scores, final, mutual, detections, games = [], [], 0, {monitor_name(m): 0 for m in all_monitors(config)}, 0
        last_corr = {monitor_name(m): [] for m in all_monitors(config)}
        for lineage in range(1, config["lineages"] + 1):
            run_dir = run_dir_for(config, lineage, rnd)
            if not game_finished(run_dir):
                continue
            games += 1
            m = game_metrics(json.loads((run_dir / "lmvsgame.json").read_text()))
            scores += [r["score"] for r in m["rows"]]
            last = m["rows"][-1]["phase"] if m["rows"] else None
            final += [r["score"] for r in m["rows"] if r["phase"] == last]
            mutual += bool(m["mutual_identification_phase"])
            for mon in monitor_summaries(run_dir):
                if mon["name"] in detections:
                    detections[mon["name"]] += bool(mon["detected_at"])
                    if mon["correctness"] and mon["correctness"][-1][1] is not None:
                        last_corr[mon["name"]].append(mon["correctness"][-1][1])
        if games:
            rows.append({"round": rnd, "games": games,
                         "identification": statistics.mean(scores) if scores else None,
                         "final_identification": statistics.mean(final) if final else None,
                         "mutual": mutual, "detections": detections,
                         "final_correctness": {k: statistics.mean(v) if v else None for k, v in last_corr.items()}})
    return rows


def print_summary(config: dict) -> None:
    rows = round_summary(config)
    names = [monitor_name(m) for m in all_monitors(config)]
    fmt = lambda v: "n/a" if v is None else f"{v:+.2f}"
    print("round  games  ident(all)  ident(final)  mutual  " + "  ".join(f"[{n}] detected, final score" for n in names))
    for r in rows:
        mons = "  ".join(f"{r['detections'][n]}/{r['games']}, "
                         f"{'n/a' if r['final_correctness'][n] is None else round(r['final_correctness'][n])}"
                         for n in names)
        print(f"{r['round']:>5}  {r['games']:>5}  {fmt(r['identification']):>10}  {fmt(r['final_identification']):>12}  "
              f"{r['mutual']:>3}/{r['games']}  {mons}")


def total_cost(run_dirs: list) -> float:
    total = 0.0
    for d in run_dirs:
        usage = d / "openrouter_usage.jsonl"
        if usage.exists():
            total += sum(json.loads(l)["usage"].get("cost") or 0 for l in usage.read_text().splitlines() if l)
        for extra in [*(d / "monitor" / "runs").glob("*.json"), *(d / "leakage").glob("*.json"),
                      *(d / "reflections").glob("*.json")]:
            total += (json.loads(extra.read_text()).get("usage") or {}).get("cost") or 0
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("--dry_run", action="store_true", help="print the plan for round 1 without running anything")
    parser.add_argument("--summary", action="store_true", help="print the per-round summary of finished games and exit")
    parser.add_argument("--env_file", type=Path, help="file with OPENROUTER_API_KEY=... (shell variables take precedence)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.summary:
        print_summary(config)
        return
    print(f"{config['name']}: {config['lineages']} lineages x {config['rounds']} rounds of {config['game_length']} "
          f"games ({model_label(config['model'])} {config['reasoning_effort']}); feedback monitor "
          f"{monitor_name(config['feedback_monitor'])}, eval monitors "
          f"{', '.join(monitor_name(m) for m in config['eval_monitors']) or 'none'}")
    if args.dry_run:
        for lineage in range(1, config["lineages"] + 1):
            seed = game_seed(config, lineage, 1)
            powers = deal_powers(seed)
            print(f"  lineage {lineage}, round 1 (seed {seed}): operatives "
                  + ", ".join(f"agent {a}={powers[a]}" for a in OPERATIVE_AGENTS))
        run_dir = run_dir_for(config, 1, 1)
        for m in all_monitors(config):
            print("    " + " ".join(monitor_cmd(run_dir, m)[1:]))
        print("    " + " ".join(reflection_cmd(run_dir, config)[1:]))
        return

    env = {**(read_env_file(args.env_file) if args.env_file else {}), **os.environ}
    if not env.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY is not set (use --env_file or export it)")
    if config["docent_collection_id"] and not env.get("DOCENT_API_KEY"):
        raise SystemExit("docent_collection_id is set but DOCENT_API_KEY is not (use --env_file or export it)")
    env.pop("OPENROUTER_REASONING_EFFORT", None)

    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=config["parallel_games"]) as pool:
        futures = {pool.submit(run_lineage, config, lineage, env): lineage
                   for lineage in range(1, config["lineages"] + 1)}
        for future in concurrent.futures.as_completed(futures):
            try:
                print(future.result(), flush=True)
            except subprocess.CalledProcessError as exc:
                failures.append(futures[future])
                print(f"FAILED lineage {futures[future]} (see the game's runner.log): {exc}", flush=True)

    print_summary(config)
    run_dirs = sorted(p for p in (REPO / config["results_dir"] / config["name"]).glob("lin*-r*") if game_finished(p))
    print(f"billed cost of these games, monitors and reflections: ${total_cost(run_dirs):.2f}")
    if run_dirs and config["docent_collection_id"]:
        upload = subprocess.run(upload_cmd([str(d.relative_to(REPO)) for d in run_dirs],
                                           config["docent_collection_id"]), cwd=REPO, env=env)
        if upload.returncode != 0:
            failures.append("docent upload")
    if failures:
        raise SystemExit(f"failed: {failures}")


if __name__ == "__main__":
    main()
