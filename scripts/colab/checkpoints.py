"""Turn a window's checkpoint lines back into files (R&D, 2026-10-17).

  python scripts/colab/checkpoints.py extract <smoke-window.log> <out dir>

Two kinds of line:
- `WAVE_CHECKPOINT <arm> <batch>-w<i> <sha256> <base64 gzip of wave-drafts.jsonl>`, printed when a wave ends. Each file is
  checked against its sha256 and written as <out dir>/<batch>-w<i>/<arm>.jsonl.
- `CALL_CHECKPOINT <arm> <batch>-w<i> <base64 gzip of one call-journal line>`, printed as each call finishes. The lines
  are joined, in log order, into <out dir>/<batch>-w<i>/calls/<arm>.jsonl.

To resume a dropped window: for the waves that finished, upload the last one's directory and set PIN_RESUME_WAVE=<i>
and PIN_RESUME_DIR=<its path on the VM>; for the wave it was in, upload that wave's calls/ directory and set
PIN_RESUME_CALLS_DIR=<its path>. The resumed wave replays those calls and asks the model only for the rest.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import sys
from pathlib import Path


def extract(log: Path, out: Path) -> list[Path]:
    written: list[Path] = []
    calls: dict[Path, list[bytes]] = {}
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("WAVE_CHECKPOINT "):
            _, arm, wave, want, blob = line.split(" ", 4)
            data = gzip.decompress(base64.b64decode(blob))
            if hashlib.sha256(data).hexdigest() != want:
                raise SystemExit(f"{arm} {wave}: checkpoint doesn't match its sha256")
            dest = out / wave / f"{arm}.jsonl"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            written.append(dest)
        elif line.startswith("CALL_CHECKPOINT "):
            parts = line.split(" ", 3)
            if len(parts) != 4:
                continue  # a line the log cut short: its call runs again
            _, arm, wave, blob = parts
            try:
                data = gzip.decompress(base64.b64decode(blob))
            except (ValueError, OSError, EOFError):
                continue
            calls.setdefault(out / wave / "calls" / f"{arm}.jsonl", []).append(data)
    for dest, lines in calls.items():
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"".join(x + b"\n" for x in lines))
        written.append(dest)
    return written


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "extract":
        raise SystemExit(__doc__)
    for p in extract(Path(sys.argv[2]), Path(sys.argv[3])):
        print(p)
