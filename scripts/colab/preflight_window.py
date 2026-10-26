"""Local preflight for a Colab window: the window's own setup, run on the rig for no units before an A100 is paid for
(R&D, 2026-10-17, after windows 10 and 10c failed in setup steps a local run would have caught).

  python scripts/colab/preflight_window.py --roleset <sha> --rnd <sha> --pack pack.tar --cards card-values.json \\
      --models qwen3.8-27b,muse-glimmer-30b [--work <dir>]

It does what pin_window.py does before its first model download, the same way:
- clones role-set at the sha and installs it with uv into a Python 3.15 venv (the VM's install path), then runs the
  installed `role-set`, not the source tree;
- clones rnd at the sha and imports the logging proxy from that clone (the VM's import path);
- unpacks and verifies the pack, makes the pilot and twins plans, and checks their exclusion-state sha;
- builds each model's one-candidate proxy config from the cards and loads it with the proxy's own loader;
- runs one mock write wave and one mock critique wave through the installed role-set, and the twins with --training.

Exit 0 when all of it passes; any failure prints the step and exits 1.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pin_window import requires_python_allows  # noqa: E402 - the same rule the window applies



ROLE_SET_PYTHON = "3.15.1"  # exact patch ("3.15" resolved to 3.15.0a1 under uv 0.9.5); current Python only (Director, 2026-10-25)
UV_VERSION = "0.13.5"  # the same uv as pin_window.py and the C2 launcher

def run(cmd: list[str], cwd: Path | None = None) -> str:
    out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:4])}... exit {out.returncode}: {(out.stderr or out.stdout)[-1200:]}")
    return out.stdout


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--roleset", required=True)
    ap.add_argument("--rnd", required=True)
    ap.add_argument("--pack", required=True, type=Path)
    ap.add_argument("--cards", required=True, type=Path)
    ap.add_argument("--models", required=True)
    ap.add_argument("--work", type=Path, default=None)
    a = ap.parse_args(argv)
    work = a.work or Path(tempfile.mkdtemp(prefix="preflight-"))
    work.mkdir(parents=True, exist_ok=True)
    steps: list[str] = []
    try:
        rs_src, venv = work / "role-set", work / "rs-venv"
        if not (rs_src / ".git").exists():
            run(["git", "clone", "-q", "https://github.com/mcp-tool-shop-org/role-set", str(rs_src)])
        run(["git", "-C", str(rs_src), "checkout", "-q", a.roleset])
        uv_version = run(["uv", "--version"]).split()[1]  # before anything is built or reused
        if uv_version != UV_VERSION:
            raise RuntimeError(f"REFUSED: uv {uv_version} is on PATH, this preflight needs uv {UV_VERSION} (an older uv resolves 3.15 to a pre-release)")
        requires = tomllib.loads((rs_src / "pyproject.toml").read_text(encoding="utf-8"))["project"]["requires-python"]
        if not requires_python_allows(requires, ROLE_SET_PYTHON):
            raise RuntimeError(
                f"REFUSED: role-set {a.roleset[:12]} requires Python {requires}, but the window builds role-set on CPython "
                f"{ROLE_SET_PYTHON} (current Python only), so that older commit cannot be reproduced"
            )
        py = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        role_set = venv / ("Scripts/role-set.exe" if sys.platform == "win32" else "bin/role-set")
        if not role_set.exists():  # a venv whose install failed has a python but no role-set: rebuild it
            shutil.rmtree(venv, ignore_errors=True)
            run(["uv", "python", "install", ROLE_SET_PYTHON])
            run(["uv", "venv", "-q", "-p", ROLE_SET_PYTHON, str(venv)])
            run(["uv", "pip", "install", "-q", "--python", str(py), "-e", str(rs_src)])
        have = run([str(py), "-c", "import sys; print(sys.version.split()[0])"]).strip()
        if have != ROLE_SET_PYTHON:
            raise RuntimeError(f"REFUSED: the role-set venv is CPython {have}, not {ROLE_SET_PYTHON}")
        steps.append(f"role-set {a.roleset[:12]} installed in {have}")

        rnd_src = work / "rnd"
        if not (rnd_src / ".git").exists():
            run(["git", "clone", "-q", "https://github.com/mcp-tool-shop-org/rnd", str(rnd_src)])
        run(["git", "-C", str(rnd_src), "checkout", "-q", a.rnd])
        sys.path.insert(0, str(rnd_src))
        from mcptoolshop_rnd.localbuilders.proxy import load_candidate_config  # noqa: PLC0415
        steps.append(f"rnd {a.rnd[:12]}: proxy imports from the clone")

        pack = work / "pack"
        shutil.rmtree(pack, ignore_errors=True)
        pack.mkdir()
        with tarfile.open(a.pack) as tf:
            tf.extractall(pack)  # noqa: S202 - our own pack, built from a pinned commit
        run([str(role_set), "pack", "verify", "--pack", str(pack)], cwd=rs_src)
        steps.append("pack verifies")

        domains = rs_src / "roles/domains/naics-2022.json"
        plan, twins = work / "pilot-plan.json", work / "twins-plan.json"
        run([str(role_set), "plan", "--pilot", "--seed", "0", "--domains", str(domains), "--out", str(plan)], cwd=rs_src)
        run([str(role_set), "plan", "--pilot-twins", "--seed", "7", "--domains", str(domains), "--out", str(twins)], cwd=rs_src)
        ex = json.loads(plan.read_text(encoding="utf-8")).get("exclusion_state_sha256")
        if not ex:
            raise RuntimeError("the pilot plan carries no exclusion_state_sha256")
        steps.append(f"plans made (exclusion state {ex[:12]})")

        # The window's own code at the commit under test, not the working copy.
        sys.path.insert(0, str(rs_src / "scripts" / "colab"))
        sys.modules.pop("pin_window", None)
        import pin_window as pw  # noqa: PLC0415

        if Path(pw.__file__).resolve().parent != (rs_src / "scripts" / "colab").resolve():
            raise RuntimeError(f"pin_window came from {pw.__file__}, not the commit under test")

        by_name = {m["served_as"]: m for m in pw.MODELS}
        writers = json.loads((rs_src / "writers.json").read_text(encoding="utf-8"))
        listed = {e["served_as"] for e in writers["writers"]} | {e["served_as"] for e in writers["critics"]}
        for name in [x.strip() for x in a.models.split(",") if x.strip()]:
            if name not in by_name:
                raise RuntimeError(f"{name} isn't in pin_window.MODELS")
            if name not in listed:
                raise RuntimeError(f"{name} isn't in writers.json at {a.roleset[:12]}")
            path, cid, info = pw.j1_card({"served_as": name, "card": by_name[name].get("card")}, a.cards, work / "cards-j1")
            load_candidate_config(path, cid)
            steps.append(f"{name}: card {cid} loads in the proxy (removed {info['removed']})")

        rs = {"role_set": str(role_set), "src": str(rs_src)}
        common = ["--pack", str(pack), "--mock", "--partition", "trial:j1-wave1"]
        w0 = pw.write_wave(rs, common + ["--plan", str(plan)], "write", work / "mock" / "w0")
        if w0["exit"] or not w0.get("result"):
            raise RuntimeError(f"mock write wave: {w0.get('stderr') or w0.get('stdout')}")
        d0 = pw.run_dir(work / "mock" / "w0")
        w1 = pw.write_wave(rs, common + ["--plan", str(plan)], "critique", work / "mock" / "w1", drafts=d0 / "wave-drafts.jsonl")
        if w1["exit"] or not w1.get("result"):
            raise RuntimeError(f"mock critique wave: {w1.get('stderr') or w1.get('stdout')}")
        d1 = pw.run_dir(work / "mock" / "w1")
        t0 = pw.write_wave(rs, common + ["--plan", str(twins)], "write", work / "mock" / "t0",
                           extra=["--training", str(d1 / "accepted.jsonl")])
        if t0["exit"] or not t0.get("result"):
            raise RuntimeError(f"mock twins wave: {t0.get('stderr') or t0.get('stdout')}")
        steps.append(f"mock waves pass: write {w0['result']['states']}, critique accepted {w1['result']['accepted']}, twins {t0['result']['states']}")
    except Exception as e:  # noqa: BLE001 - the preflight reports and fails
        for s in steps:
            print(f"ok   {s}")
        print(f"FAIL {type(e).__name__}: {e}")
        return 1
    for s in steps:
        print(f"ok   {s}")
    print(f"PREFLIGHT PASS ({work})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
