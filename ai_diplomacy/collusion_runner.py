"""Run a batch of collusion games, then the monitors and dashboards for each.

    uv run python -m ai_diplomacy.collusion_runner collusion_configs/pilot.json --dry_run
    uv run python -m ai_diplomacy.collusion_runner collusion_configs/pilot.json --env_file ~/CoordinationGames/.env

The config (JSON) describes one batch; see collusion_configs/pilot.json and COLLUSION.md.
Every value of "n_colluders" (and of "reasoning_effort", if it's a list) is a condition,
played "games_per_condition" times. Games go to results/<batch>/<condition>-g<k>/, and
monitors to <game>/monitor/runs/.

Each step is skipped if it already finished (a game whose overview.jsonl exists, a monitor
whose output file exists), so an interrupted batch can be rerun with the same command; a
half-played game resumes from its last saved phase. Games run "parallel_games" at a time.
At the end the runner prints the collusion metrics and the billed cost of the batch.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULTS = {
    "results_dir": "results",
    "game_length": "2y",
    "games_per_condition": 1,
    "num_negotiation_rounds": 3,
    "colluder_seed_base": 1000,
    "parallel_games": 1,
    "extra_game_args": [],
    "monitors": [],
    "judges": [],
    "dashboards": True,
}
REQUIRED = ("name", "model", "reasoning_effort", "n_colluders")


def load_config(path: Path) -> dict:
    config = {**DEFAULTS, **json.loads(path.read_text())}
    missing = [k for k in REQUIRED if k not in config]
    if missing:
        raise SystemExit(f"{path}: missing {', '.join(missing)}")
    for key in ("n_colluders", "reasoning_effort"):
        if not isinstance(config[key], list):
            config[key] = [config[key]]
    return config


def model_label(model: str) -> str:
    """'openrouter:openai/gpt-5.6-luna' -> 'gpt-5.6-luna'."""
    return model.split("/")[-1].split(":")[-1]


def plan_games(config: dict) -> list[dict]:
    """One entry per game: its folder and the lm_game command."""
    games = []
    batch_dir = REPO / config["results_dir"] / config["name"]
    for effort, n in itertools.product(config["reasoning_effort"], config["n_colluders"]):
        condition = f"{model_label(config['model'])}-{effort}-{config['game_length']}-ops{n}"
        for k in range(1, config["games_per_condition"] + 1):
            run_dir = batch_dir / f"{condition}-g{k}"
            cmd = [sys.executable, "lm_game.py",
                   "--run_dir", str(run_dir.relative_to(REPO)),
                   "--game_length", config["game_length"],
                   "--n_colluders", str(n),
                   "--num_negotiation_rounds", str(config["num_negotiation_rounds"]),
                   "--models", ",".join([config["model"]] * 7),
                   "--reasoning_effort", effort]
            if n:
                cmd += ["--colluder_seed", str(config["colluder_seed_base"] + k)]
            cmd += [str(a) for a in config["extra_game_args"]]
            games.append({"condition": condition, "run_dir": run_dir, "cmd": cmd})
    return games


def monitor_name(monitor: dict) -> str:
    return monitor.get("name") or f"{monitor['view']}_{model_label(monitor['model'])}_{monitor.get('effort') or 'default'}"


def monitor_cmd(run_dir: Path, monitor: dict) -> list[str]:
    cmd = [sys.executable, "-m", "ai_diplomacy.monitor", str(run_dir.relative_to(REPO)),
           "--view", monitor["view"], "--model", monitor["model"], "--name", monitor_name(monitor)]
    if monitor.get("effort"):
        cmd += ["--effort", monitor["effort"]]
    for key, value in monitor.items():
        if key in ("view", "model", "effort", "name"):
            continue
        if value is True:
            cmd.append(f"--{key}")
        elif value not in (False, None):
            cmd += [f"--{key}", str(value)]
    return cmd


def judge_name(judge: dict) -> str:
    return judge.get("name") or f"{model_label(judge['model'])}_{judge.get('effort') or 'default'}"


def judge_cmd(run_dir: Path, judge: dict) -> list[str]:
    cmd = [sys.executable, "-m", "ai_diplomacy.leakage_judge", str(run_dir.relative_to(REPO)),
           "--model", judge["model"], "--name", judge_name(judge)]
    if judge.get("effort"):
        cmd += ["--effort", judge["effort"]]
    return cmd


def game_finished(run_dir: Path) -> bool:
    return (run_dir / "overview.jsonl").exists()  # lm_game writes it only at the very end


def _run(cmd: list[str], log: Path, env: dict) -> None:
    with log.open("a") as f:
        f.write(f"\n$ {' '.join(cmd)}\n")
        f.flush()
        subprocess.run(cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)


def run_one(game: dict, config: dict, env: dict) -> str:
    run_dir = game["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    log = run_dir / "runner.log"
    if game_finished(run_dir):
        status = "game already done"
    else:
        game_env = {**env, "OPENROUTER_USAGE_LOG": str(run_dir / "openrouter_usage.jsonl")}
        _run(game["cmd"], log, game_env)
        status = "game played"
    for monitor in config["monitors"]:
        out = run_dir / "monitor" / "runs" / f"{monitor_name(monitor)}.json"
        if not out.exists():
            _run(monitor_cmd(run_dir, monitor), log, env)
    for judge in config["judges"]:
        if not (run_dir / "leakage" / f"{judge_name(judge)}.json").exists():
            _run(judge_cmd(run_dir, judge), log, env)
    if config["dashboards"]:
        _run([sys.executable, "-m", "ai_diplomacy.dashboard", str(run_dir.relative_to(REPO))], log, env)
    return f"{run_dir.relative_to(REPO)}: {status}, {len(config['monitors'])} monitors, {len(config['judges'])} judges"


def read_env_file(path: Path) -> dict:
    values = {}
    for line in path.expanduser().read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip("'\"")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("--dry_run", action="store_true", help="print the plan and commands without running anything")
    parser.add_argument("--env_file", type=Path, help="file with OPENROUTER_API_KEY=... (shell variables take precedence)")
    args = parser.parse_args()

    config = load_config(args.config)
    games = plan_games(config)
    print(f"batch {config['name']}: {len(games)} games, {len(config['monitors'])} monitors each")
    for g in games:
        done = " (already done)" if game_finished(g["run_dir"]) else ""
        print(f"  {g['run_dir'].relative_to(REPO)}{done}")
        if args.dry_run:
            print("    " + " ".join(g["cmd"][1:]))
            for m in config["monitors"]:
                print("    " + " ".join(monitor_cmd(g["run_dir"], m)[1:]))
            for j in config["judges"]:
                print("    " + " ".join(judge_cmd(g["run_dir"], j)[1:]))
    if args.dry_run:
        return

    env = {**(read_env_file(args.env_file) if args.env_file else {}), **os.environ}
    if not env.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY is not set (use --env_file or export it)")
    env.pop("OPENROUTER_REASONING_EFFORT", None)  # each game gets its effort from the config

    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=config["parallel_games"]) as pool:
        futures = {pool.submit(run_one, g, config, env): g for g in games}
        for future in concurrent.futures.as_completed(futures):
            g = futures[future]
            try:
                print(future.result())
            except subprocess.CalledProcessError as exc:
                failures.append(g)
                print(f"FAILED {g['run_dir'].relative_to(REPO)} (see its runner.log): {exc}")

    run_dirs = [str(g["run_dir"].relative_to(REPO)) for g in games if game_finished(g["run_dir"])]
    if run_dirs:
        subprocess.run([sys.executable, "-m", "ai_diplomacy.collusion_metrics", *run_dirs], cwd=REPO, env=env)
        total = 0.0
        for d in run_dirs:
            usage = REPO / d / "openrouter_usage.jsonl"
            if usage.exists():
                total += sum(json.loads(l)["usage"].get("cost") or 0 for l in usage.read_text().splitlines() if l)
            for extra in [*(REPO / d / "monitor" / "runs").glob("*.json"), *(REPO / d / "leakage").glob("*.json")]:
                total += (json.loads(extra.read_text()).get("usage") or {}).get("cost", 0)
        print(f"billed cost of these games, monitors and judges: ${total:.2f}")
    if failures:
        raise SystemExit(f"{len(failures)} games failed")


if __name__ == "__main__":
    main()
