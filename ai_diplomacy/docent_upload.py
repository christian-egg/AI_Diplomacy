"""Upload collusion-game results to a Docent collection. No model calls.

    uv run python -m ai_diplomacy.docent_upload results/pilot/* --collection_id <id> --env_file ~/CoordinationGames/.env
    uv run python -m ai_diplomacy.docent_upload results/*/ --collection_id <id> --dry_run

For each game folder this uploads the agents run and every saved monitor and judge run
(see docent_export.py), each tagged with its kind (type:agents|monitor|judge), version
and condition. Everything is built and checked before anything is uploaded. A receipt
(docent_upload.json in the game folder) records each uploaded run by its export key, so
re-running skips what is already in that collection and adds only new monitor or judge
runs. Needs DOCENT_API_KEY (from --env_file or the shell).
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .collusion_runner import read_env_file
from .docent_export import runs_for

RECEIPT = "docent_upload.json"


def _receipts(folder: Path) -> list:
    path = folder / RECEIPT
    return json.loads(path.read_text()) if path.exists() else []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="game folders containing lmvsgame.json")
    parser.add_argument("--collection_id", required=True)
    parser.add_argument("--env_file", type=Path, help="file with DOCENT_API_KEY=... (shell variables take precedence)")
    parser.add_argument("--dry_run", action="store_true", help="build and list the runs without uploading")
    args = parser.parse_args()

    folders = [p.resolve() for p in args.runs if (p / "lmvsgame.json").exists()]
    planned = [(folder, run) for folder in folders for run in runs_for(folder)]  # build everything first
    done = {(r["collection_id"], r["export_key"]) for folder in folders for r in _receipts(folder)}
    todo = [(f, r) for f, r in planned if (args.collection_id, r.metadata["export_key"]) not in done]
    print(f"{len(planned)} runs from {len(folders)} games; {len(planned) - len(todo)} already uploaded")
    for _, run in todo:
        print(f"  {run.metadata['type']:<7} {len(run.model_dump_json()) / 1e6:5.1f} MB  {run.name[:100]}")
    if args.dry_run or not todo:
        return

    env = {**(read_env_file(args.env_file) if args.env_file else {}), **os.environ}
    if not env.get("DOCENT_API_KEY"):
        raise SystemExit("DOCENT_API_KEY is not set (use --env_file or export it)")
    from docent import Docent
    client = Docent(api_key=env["DOCENT_API_KEY"])

    uploaded = 0
    for folder, run in todo:
        client.add_agent_runs(args.collection_id, [run])
        for tag in run.metadata["tags"]:
            client.tag_transcript(args.collection_id, run.id, tag)
        receipts = _receipts(folder)
        receipts.append({"collection_id": args.collection_id, "export_key": run.metadata["export_key"],
                         "agent_run_id": run.id, "type": run.metadata["type"],
                         "uploaded_utc": datetime.now(timezone.utc).isoformat()})
        (folder / RECEIPT).write_text(json.dumps(receipts, indent=1))
        uploaded += 1
        print(f"uploaded {run.metadata['type']}: {run.name[:100]}")
    print(f"{uploaded} uploaded to collection {args.collection_id}")


if __name__ == "__main__":
    main()
