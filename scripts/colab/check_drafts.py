"""Check a window's w0 drafts before a resume uses them (R&D, 2026-10-17, after window 12b's refused critique).

  uv run python scripts/colab/check_drafts.py <window dir> <arm> <work dir> [<critic>]

On the window's own files: the drafts' seal verifies, their batch pins name the plan in pilot-out, the slots are the
frozen screen list, then a critique wave on role-set's real path (this commit, the real gates, the real cards) runs on
them against a stub critic. The stub's verdicts mean nothing; the path is what's proven. Exit 0 when all of it passes."""

import hashlib
import json
import shutil
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
from stub_llama import StubLlama  # noqa: E402

from role_set import waves  # noqa: E402
from role_set.write import WriteConfig, run  # noqa: E402

FROZEN = ["T02-A-01", "T02-C-01", "T02-D-01", "T02-E-01", "T02-B-01", "T02-A-02", "T02-C-08", "T02-D-05"]
ROOT = Path(__file__).resolve().parents[2]
win, arm, work = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
critic_name = sys.argv[4] if len(sys.argv) > 4 else "muse-glimmer-30b"
shutil.rmtree(work, ignore_errors=True)
work.mkdir(parents=True)
with tarfile.open(win / "pilot-out.tar") as tf:
    tf.extractall(work / "out", filter="data")
with tarfile.open(win / "pack.tar") as tf:
    tf.extractall(work / "pack", filter="data")
w0 = work / "out" / "pilot" / "pilot" / arm / "w0" / waves.DRAFTS_FILE
plan = work / "out" / "pilot" / "pilot-plan.json"

header, drafts = waves.read(w0)  # 1. the seal verifies, or this raises
print("seal ok:", w0.name, waves.file_sha256(w0)[:12])
plan_sha = hashlib.sha256(plan.read_bytes()).hexdigest()
assert header["batch_pins"]["plan_sha256"] == plan_sha, ("plan", header["batch_pins"]["plan_sha256"][:12], plan_sha[:12])
print("batch pins name this plan:", plan_sha[:12])
ids = [d["slot"]["id"] for d in drafts]
assert sorted(ids) == sorted(FROZEN), ids
print("slots are the frozen 8:", ids)

listed = json.loads((ROOT / "writers.json").read_text(encoding="utf-8"))
critic = next(e for e in listed["critics"] if e["served_as"] == critic_name)
machine = work / "machine.json"
r = json.loads((win / "pin-result.json").read_text(encoding="utf-8"))
machine.write_text(json.dumps(r["machine"]), encoding="utf-8")
stub = StubLlama(header["writer_pins"]["writer_model"], critic_name)
try:
    cfg = WriteConfig(plan=plan, pack=work / "pack", out=work / "w1", writers_file=ROOT / "writers.json",
                      cards=win / "card-values.json", partition="trial:j1-wave1",
                      machine=machine, server_sha256=listed["llama_server"]["sha256"], wave="critique", drafts=w0,
                      critic_endpoint=stub.url, critic_model=critic_name, critic_gguf_sha256=critic["gguf_sha256"])
    res = run(cfg)
finally:
    stub.close()
print("critique wave ran on the real drafts:", {k: res[k] for k in ("states", "accepted", "dropped", "halted")})
print("critic calls:", res["calls"])
