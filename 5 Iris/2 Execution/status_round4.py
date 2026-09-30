"""Read-only progress for the seven declared Iris campaigns, no framework import."""
import json
import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "3 Connectome"))
from flybrain.paths import repo_path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--brief", action="store_true", help="Compact counts and full-connectome epoch progress")
parser.add_argument("--campaigns", default="5 Iris/5 Outputs/campaigns",
                    help="Repository-relative directory containing round4-* campaigns")
options = parser.parse_args()
inventory = json.loads((REPO / "5 Iris/1 Configuration/inventory.json").read_text())
summary = {"time_utc": datetime.now(timezone.utc).isoformat(), "campaigns": []}
for name in ("core", "full", "controls", "lif", "local", "transfer", "null"):
    folder = repo_path(options.campaigns) / f"round4-{name}"
    path = folder / "campaign.json"
    if not path.exists():
        summary["campaigns"].append({"name": name, "state": "not_started"})
        continue
    state = json.loads(path.read_text())
    row = {"name": name, "completed": state["completed"], "planned": state["planned"],
           "errors": state["errors"], "active": []}
    if options.brief and name != "full":
        row.pop("active")
        summary["campaigns"].append(row)
        continue
    completed_jobs = {repo_path(record["run"]).resolve().parent for record in state["runs"]}
    histories_by_job = {}
    for history_path in sorted(folder.glob("runs/*/*/history.json")):
        histories_by_job.setdefault(history_path.parent.parent, []).append(history_path)
    row["superseded_partial_attempts"] = sum(
        not (path.parent / "results.json").exists()
        for job, paths in histories_by_job.items()
        for path in (paths if job.resolve() in completed_jobs else paths[:-1])
    )
    for job, paths in histories_by_job.items():
        history_path = paths[-1]
        if job.resolve() in completed_jobs or (history_path.parent / "results.json").exists():
            continue
        try:
            history = json.loads(history_path.read_text())
            config = json.loads((history_path.parent / "config.json").read_text())
        except (ValueError, OSError):
            continue  # A history file can be in the middle of its next write.
        row["active"].append({"run": history_path.parent.name, "epoch": history[-1]["epoch"],
                              "budget": config["epochs"], "seconds": round(history[-1]["elapsed_seconds"], 1)})
    summary["campaigns"].append(row)
summary["completed"] = sum(r.get("completed", 0) for r in summary["campaigns"])
summary["planned"] = inventory.get("total_runs", inventory.get("total_jobs", 6390))
if options.brief:
    for row in summary["campaigns"]:
        if "active" in row:
            row["active_epochs"] = [(run["epoch"], run["budget"]) for run in row.pop("active")]
print(json.dumps(summary, indent=None if options.brief else 2))
