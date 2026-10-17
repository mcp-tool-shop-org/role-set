"""Turn a window's WAVE_CHECKPOINT lines back into drafts files (R&D, 2026-10-17).

  python scripts/colab/checkpoints.py extract <smoke-window.log> <out dir>

Each line is `WAVE_CHECKPOINT <arm> <batch>-w<i> <sha256> <base64 gzip of wave-drafts.jsonl>`. Every file is checked
against its sha256 and written as <out dir>/<batch>-w<i>/<arm>.jsonl. To resume a dropped window after wave i, upload
that wave's directory and set PIN_RESUME_WAVE=<i> and PIN_RESUME_DIR=<its path on the VM>.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import sys
from pathlib import Path


def extract(log: Path, out: Path) -> list[Path]:
    written = []
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("WAVE_CHECKPOINT "):
            continue
        _, arm, wave, want, blob = line.split(" ", 4)
        data = gzip.decompress(base64.b64decode(blob))
        if hashlib.sha256(data).hexdigest() != want:
            raise SystemExit(f"{arm} {wave}: checkpoint doesn't match its sha256")
        dest = out / wave / f"{arm}.jsonl"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        written.append(dest)
    return written


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "extract":
        raise SystemExit(__doc__)
    for p in extract(Path(sys.argv[2]), Path(sys.argv[3])):
        print(p)
