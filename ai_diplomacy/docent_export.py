"""Convert saved collusion-game results into Docent AgentRuns. No network calls.

Three kinds of run, told apart by the tag type:agents, type:monitor or type:judge:

- agents: one per game. One transcript per power holds its model calls in order (each
  call is a separate request with the same system prompt; the user message is the prompt
  as logged in llm_responses.csv, the assistant message the raw reply). A last transcript
  is the game as the trajectory monitor sees it (supply centers, messages, orders), for
  reading the game in order.
- monitor: one per saved monitor run (<run>/monitor/runs/*.json): the monitor's
  conversation, with each assessment's scores in the message metadata.
- judge: one per saved leakage-judge run (<run>/leakage/*.json).

Every run carries tags such as version:<harness commit>, model:..., effort:...,
operatives:..., and each tag is also a top-level metadata field, for filtering. The
version comes from the run's own files (run_settings, or run_settings_backfill.json for
runs made before run_settings existed). Monitor and judge runs made before they saved their
prompts are rebuilt with the prompt files as they were at the run's recorded commit.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

from .collusion import GAME_LENGTHS, POWERS, operative_system_prompt
from .dashboard import load_invalid_orders
from .monitor_export import export_game

EXPORT_SCHEMA = "ai-diplomacy-docent/v1"
REPO = Path(__file__).resolve().parent.parent
PROMPTS_SIMPLE = Path(__file__).resolve().parent / "prompts_simple"
LENGTH_BY_YEAR = {year: name for name, year in GAME_LENGTHS.items()}


# --- Settings and tags ------------------------------------------------------------------

def _model_label(model: str) -> str:
    return model.split("/")[-1].split(":")[-1]


def game_settings(run_dir: Path, game: dict) -> dict:
    """The condition a game ran under, from run_settings, the backfill file and upstream's config."""
    settings = dict(game.get("run_settings") or {})
    backfill = run_dir / "run_settings_backfill.json"
    if backfill.exists():
        for key, value in json.loads(backfill.read_text()).items():
            settings.setdefault(key, value)
    config = next((p["config"] for p in game["phases"] if p.get("config")), {})
    for key in ("max_year", "num_negotiation_rounds", "max_tokens", "simple_prompts"):
        settings.setdefault(key, config.get(key))
    if not settings.get("models"):
        overview = run_dir / "overview.jsonl"
        lines = overview.read_text().splitlines() if overview.exists() else []
        models = json.loads(lines[1]) if len(lines) > 1 else {}
        if not models and config.get("models"):
            names = [m.strip() for m in config["models"].split(",")]
            models = dict(zip(POWERS, names * (7 if len(names) == 1 else 1)))
        settings["models"] = models
    settings.setdefault("n_colluders", len((game.get("collusion") or {}).get("colluders") or []))
    settings.setdefault("monitor_in_prompt", False)
    if not settings.get("game_length"):
        settings["game_length"] = LENGTH_BY_YEAR.get(settings.get("max_year"), f"to{settings.get('max_year')}")
    settings["status"] = "complete" if (run_dir / "overview.jsonl").exists() else "partial"
    return settings


def game_tags(run_dir: Path, settings: dict) -> dict:
    labels = sorted({_model_label(m) for m in (settings.get("models") or {}).values()})
    batch = run_dir.parent.name if run_dir.parent.name != "results" else "none"
    return {
        "model": labels[0] if len(labels) == 1 else ("mixed" if labels else "unknown"),
        "effort": settings.get("reasoning_effort") or "unknown",
        "game_length": settings["game_length"],
        "operatives": str(settings.get("n_colluders", 0)),
        "negotiation_rounds": str(settings.get("num_negotiation_rounds")),
        "monitor_in_prompt": str(bool(settings.get("monitor_in_prompt"))).lower(),
        "status": settings["status"],
        "batch": batch,
    }


def _tag_list(tags: dict) -> list:
    return [f"{k}:{v}" for k, v in tags.items()]


def _metadata(tags: dict, **fields) -> dict:
    clash = set(fields) & set(tags)
    if clash:  # a tag would silently overwrite a field of the same name
        raise ValueError(f"metadata fields clash with tags: {sorted(clash)}")
    return json.loads(json.dumps({**fields, **tags, "tags": _tag_list(tags)}, default=str))


# --- Agents --------------------------------------------------------------------------

def _system_prompt(power: str, operatives: set, settings: dict, saved: dict) -> str:
    """The saved system prompt, or for games saved before it was, a rebuild from the
    current templates (right only if the templates haven't changed since)."""
    if saved.get(power):
        return saved[power]
    if power in operatives:
        return operative_system_prompt(power, len(operatives), settings.get("max_year") or 1925,
                                       bool(settings.get("monitor_in_prompt")))
    return (PROMPTS_SIMPLE / f"{power.lower()}_system_prompt.txt").read_text(encoding="utf-8").strip()


def game_record_text(game: dict, run_dir: Path) -> list:
    """The game phase by phase, as the trajectory monitor sees it (all messages, orders, results)."""
    from .monitor import render_phase
    export = export_game(game, load_invalid_orders(run_dir), "trajectory")
    return [(p["phase"], render_phase(p, "trajectory")) for p in export["phases"]]


def agents_run(run_dir: Path):
    from docent.data_models import AgentRun, Transcript
    from docent.data_models.chat import parse_chat_message

    from .collusion_metrics import coalition_outcome, game_metrics

    game = json.loads((run_dir / "lmvsgame.json").read_text())
    settings = game_settings(run_dir, game)
    collusion = game.get("collusion") or {}
    operatives = set(collusion.get("colluders") or [])
    csv.field_size_limit(sys.maxsize)
    with (run_dir / "llm_responses.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    transcripts = []
    for power in POWERS:
        calls = [r for r in rows if r["power"] == power]
        model = (settings.get("models") or {}).get(power)
        messages = [parse_chat_message({"role": "system", "content": _system_prompt(power, operatives, settings, game.get("system_prompts") or {}),
                                        "metadata": {"provenance": "system prompt, rebuilt from the prompt files"}})]
        for i, r in enumerate(calls):
            meta = {"call_index": i, "phase": r["phase"], "response_type": r["response_type"],
                    "timestamp": r["timestamp"], "independent_call": True}
            messages.append(parse_chat_message({"role": "user", "content": r["raw_input"], "metadata": meta}))
            reply = {"role": "assistant", "content": r["raw_response"], "metadata": {**meta, "success": r["success"]}}
            if model:
                reply["model"] = model
            messages.append(parse_chat_message(reply))
        role = "secret operative" if power in operatives else "regular player"
        transcripts.append(Transcript(
            name=f"{power} ({role})", messages=messages,
            description=(f"{power}'s model calls in order. Each user/assistant pair is a separate request with "
                         "this system prompt; the model saw only what is in that request."),
            metadata={"power": power, "operative": power in operatives, "model": model, "calls": len(calls)}))
    record = [parse_chat_message({"role": "user", "content": text, "metadata": {"phase": phase}})
              for phase, text in game_record_text(game, run_dir)]
    if record:
        transcripts.append(Transcript(
            name="Game record (trajectory-monitor view)", messages=record,
            description="Each phase as the trajectory monitor sees it. No agent saw the game in this form.",
            metadata={"kind": "game_record"}))

    tags = {"type": "agents", "version": settings.get("harness_commit") or "unknown", **game_tags(run_dir, settings)}
    metrics = game_metrics(game)
    metrics.pop("rows", None)
    outcome = coalition_outcome(game, operatives)["outcome"] if operatives else "control"
    tags["outcome"] = outcome
    usage = run_dir / "openrouter_usage.jsonl"
    cost = sum(json.loads(l)["usage"].get("cost") or 0 for l in usage.read_text().splitlines() if l) if usage.exists() else None
    name_bits = [run_dir.name, "agents", f"{tags['model']} {tags['effort']}", tags["game_length"],
                 f"{tags['operatives']} operatives", f"version {tags['version']}"]
    return AgentRun(
        name=("[PARTIAL] " if settings["status"] == "partial" else "") + " | ".join(name_bits),
        description="One transcript per power (its model calls in order), plus the game record.",
        transcripts=transcripts,
        metadata=_metadata(tags, export_schema=EXPORT_SCHEMA, export_key=f"{EXPORT_SCHEMA}:agents:{game['id']}",
                           game_id=game["id"], run_folder=(run_dir.name if tags["batch"] == "none"
                                                           else f"{tags['batch']}/{run_dir.name}"),
                           settings=settings, operative_powers=sorted(operatives), colluder_seed=collusion.get("colluder_seed"),
                           draw_phase=collusion.get("draw_phase"), metrics=metrics, cost_usd=cost))


# --- Monitor and judge ------------------------------------------------------------------

def _prompts_at(commit: Optional[str], folder: str) -> Optional[Callable[[str], str]]:
    """A prompt reader for the files as they were at `commit`, or None to use today's files."""
    if not commit or "+" in commit:
        return None
    def read(name: str) -> str:
        path = f"ai_diplomacy/prompts_simple/{folder}/{name}"
        return subprocess.run(["git", "show", f"{commit}:{path}"], cwd=REPO, capture_output=True, text=True,
                              check=True).stdout.strip()
    return read


def _monitor_conversation(game: dict, run_dir: Path, result: dict) -> tuple:
    """(system prompt, [user message per step]) — saved ones if present, else rebuilt."""
    steps = result.get("steps", [])
    if result.get("system_prompt") and all("user_message" in s for s in steps):
        return result["system_prompt"], [s["user_message"] for s in steps]
    from . import monitor
    st = result["settings"]
    export = export_game(game, load_invalid_orders(run_dir), st["view"], st.get("drop_draw_phase", False))
    all_batches = monitor.batches(monitor.group_turns(export["phases"]), st["turns_per_step"])
    reader = _prompts_at(st.get("harness_commit"), "monitor")
    original = monitor._prompt
    if reader:
        monitor._prompt = reader
    try:
        replies = [s.get("reply", "") for s in steps]
        users = []
        system = None
        for k in range(len(steps)):
            msgs = monitor.build_messages(all_batches[:k + 1], replies[:k], st["view"], st["turns_per_step"],
                                          st.get("notes_words", 100), st.get("window_turns"),
                                          **({"knows_objective": True} if st.get("knows_objective") else {}))
            system = system or msgs[0]["content"]
            users.append(msgs[-1]["content"])
    finally:
        monitor._prompt = original
    return system, users


def monitor_runs(run_dir: Path, game: dict, settings: dict) -> list:
    from docent.data_models import AgentRun, Transcript
    from docent.data_models.chat import parse_chat_message

    operatives = sorted((game.get("collusion") or {}).get("colluders") or [])
    runs = []
    for path in sorted((run_dir / "monitor" / "runs").glob("*.json")):
        result = json.loads(path.read_text())
        st = result["settings"]
        system, users = _monitor_conversation(game, run_dir, result)
        messages = [parse_chat_message({"role": "system", "content": system})]
        for step, user in zip(result["steps"], users):
            meta = {k: step.get(k) for k in ("through", "phases", "correctness", "group_exists", "probabilities",
                                             "counts_for_detection", "valid", "problems")}
            messages.append(parse_chat_message({"role": "user", "content": user, "metadata": {"through": step["through"]}}))
            messages.append(parse_chat_message({"role": "assistant", "content": step.get("reply", ""),
                                                "model": st.get("model"), "metadata": meta}))
        gt = game_tags(run_dir, settings)
        detected = "control" if not operatives else ("yes" if result.get("detected_at") else "no")
        tags = {"type": "monitor", "version": st.get("harness_commit") or "unknown",
                "game_version": settings.get("harness_commit") or "unknown",
                "monitor_view": st["view"], "monitor_model": _model_label(st.get("model") or "unknown"),
                "monitor_effort": st.get("effort") or "default", "turns_per_step": str(st.get("turns_per_step")),
                "window": f"{st['window_turns']}" if st.get("window_turns") else "full",
                "knows_objective": str(bool(st.get("knows_objective"))).lower(),
                "detection_threshold": str(st.get("detection_threshold")), "detected": detected,
                "game_model": gt["model"], "game_effort": gt["effort"], "game_length": gt["game_length"],
                "operatives": gt["operatives"], "batch": gt["batch"]}
        scores = ", ".join(f"{s['through']} {s['correctness']:.0f}" for s in result["steps"] if s.get("correctness") is not None)
        runs.append(AgentRun(
            name=f"{run_dir.name} | monitor {st['view']} {tags['monitor_model']} {tags['monitor_effort']} | "
                 f"correctness {scores} | detected {detected}",
            description="The monitor's conversation. Correctness, detection and the operatives are research "
                        "metadata; the monitor never saw them.",
            transcripts=[Transcript(name=f"monitor ({st['view']})", messages=messages,
                                    description="System prompt, then one batch of turns and the reply per step.")],
            metadata=_metadata(tags, export_schema=EXPORT_SCHEMA,
                               export_key=f"{EXPORT_SCHEMA}:monitor:{game['id']}:{path.stem}", game_id=game["id"],
                               monitor_name=path.stem, settings=st, operative_powers=operatives,
                               detected_at=result.get("detected_at"), usage=result.get("usage"),
                               steps=[{k: s.get(k) for k in ("through", "correctness", "group_exists",
                                                             "probabilities", "counts_for_detection")}
                                      for s in result["steps"]])))
    return runs


def judge_runs(run_dir: Path, game: dict, settings: dict) -> list:
    from docent.data_models import AgentRun, Transcript
    from docent.data_models.chat import parse_chat_message

    from .leakage_judge import leakage_summary, phase_request, system_prompt

    operatives = set((game.get("collusion") or {}).get("colluders") or [])
    export = None
    runs = []
    for path in sorted((run_dir / "leakage").glob("*.json")):
        result = json.loads(path.read_text())
        st = result.get("settings") or {}
        if not all("user_message" in ph for ph in result["phases"]):
            export = export or {p["phase"]: p for p in export_game(game, {}, "trajectory")["phases"]}
        system = result.get("system_prompt") or system_prompt()
        messages = [parse_chat_message({"role": "system", "content": system})]
        for ph in result["phases"]:
            user = ph.get("user_message") or phase_request(export[ph["phase"]])
            messages.append(parse_chat_message({"role": "user", "content": user, "metadata": {"phase": ph["phase"]}}))
            messages.append(parse_chat_message({"role": "assistant", "content": ph.get("reply", ""), "model": st.get("model"),
                                                "metadata": {"phase": ph["phase"], "flagged": ph["flagged"],
                                                             "valid": ph["valid"], "problems": ph["problems"]}}))
        gt = game_tags(run_dir, settings)
        summary = leakage_summary(result, operatives)
        tags = {"type": "judge", "version": st.get("harness_commit") or "unknown",
                "game_version": settings.get("harness_commit") or "unknown",
                "judge_model": _model_label(st.get("model") or "unknown"), "judge_effort": st.get("effort") or "default",
                "game_model": gt["model"], "game_effort": gt["effort"], "game_length": gt["game_length"],
                "operatives": gt["operatives"], "batch": gt["batch"]}
        runs.append(AgentRun(
            name=f"{run_dir.name} | leakage judge {tags['judge_model']} {tags['judge_effort']} | "
                 f"{summary['operative_leaks']} operative leaks, {summary['false_positives']} false positives",
            description="The leakage judge's calls, one per phase. Which powers are operatives is research metadata.",
            transcripts=[Transcript(name="leakage judge", messages=messages,
                                    description="Each user/assistant pair is a separate request.")],
            metadata=_metadata(tags, export_schema=EXPORT_SCHEMA,
                               export_key=f"{EXPORT_SCHEMA}:judge:{game['id']}:{path.stem}", game_id=game["id"],
                               judge_name=path.stem, settings=st, operative_powers=sorted(operatives), summary=summary,
                               usage=result.get("usage"))))
    return runs


def runs_for(run_dir: Path) -> list:
    """Every AgentRun for one game folder: the agents, then its monitor and judge runs."""
    game = json.loads((run_dir / "lmvsgame.json").read_text())
    settings = game_settings(run_dir, game)
    return [agents_run(run_dir), *monitor_runs(run_dir, game, settings), *judge_runs(run_dir, game, settings)]
