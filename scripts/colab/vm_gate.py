"""The VM health gate (Mike and R&D, 2026-10-18): run by smoke_window.sh right after `colab new`, before anything is
uploaded. One short `colab exec` that prints one line, `VM_GATE {json}`; smoke_window.sh stops a VM whose line says
ok false (or whose exec dies or times out) and asks for a fresh one, up to PIN_GATE_TRIES.

What it covers: VMs that are bad at the start. A first exec that dies (windows 11c and 12e lost 5.7 units that way,
after the upload), a wrong or short card, a card already in use, too little disk or RAM, a slow pipe to Hugging Face.
What it doesn't: drops and hangs later in the run. The resume path and checkpoints.py cover those.

Every threshold is a setting with a default; each reading is printed, so gate.jsonl is a record of the VMs we got."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time

# Night one's download rates (2026-10-17, R&D's measure): each BF16 snapshot's size from the Hugging Face API over the
# time from job.log's "downloading" line to its "converting" line. Muse 242-251 MB/s across windows 12b, 12c, 12f and
# 12h; Gemma 258 (12f); Qwen3.8 294 (12f); Granite 311 (12b); Lightning 311 (12h); Qwen3.6 317 (12c). Median 258 MB/s.
# The floor is half of it: a VM at 129 MB/s adds about 4 minutes per 60 GB model, about 20 minutes across a five-model
# window, which is still inside a cap's 15%; below that the VM is the problem.
HF_MEDIAN_MBPS = 258.0
HF_FLOOR_FRACTION = 0.5


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw not in (None, "") else default


def limits() -> dict:
    return {"gpu": os.environ.get("PIN_GATE_GPU", "1") != "0",
            "name_has": os.environ.get("PIN_GATE_GPU_NAME", "A100"),
            "min_vram_mib": _env_float("PIN_GATE_MIN_VRAM_MIB", 39000),  # a 40 GB A100 passes, as on night one
            "max_used_mib": _env_float("PIN_GATE_MAX_USED_MIB", 500),
            "min_disk_gb": _env_float("PIN_GATE_MIN_DISK_GB", 0),  # smoke_window.sh passes the window's peak + 20
            "min_ram_gb": _env_float("PIN_GATE_MIN_RAM_GB", 50),
            "hf": os.environ.get("PIN_GATE_HF", "1") != "0",
            "min_hf_mbps": _env_float("PIN_GATE_MIN_HF_MBPS", HF_MEDIAN_MBPS * HF_FLOOR_FRACTION),
            "hf_url": os.environ.get("PIN_GATE_HF_URL",
                                     "https://huggingface.co/meta-models/Muse-Glimmer-30B/resolve/main/model-00001-of-00002.safetensors"),
            "hf_bytes": int(_env_float("PIN_GATE_HF_BYTES", 200_000_000))}


def gpu_readings() -> dict:
    q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=30)
    name, total, used = (x.strip() for x in q.stdout.strip().splitlines()[0].split(","))
    p = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
    return {"gpu_name": name, "vram_mib": float(total), "vram_used_mib": float(used),
            "gpu_processes": len([x for x in p.stdout.splitlines() if x.strip()])}


def host_readings() -> dict:
    free = shutil.disk_usage("/content" if os.path.isdir("/content") else "/").free
    avail = 0.0
    with open("/proc/meminfo", encoding="utf-8") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                avail = float(line.split()[1]) / 1e6  # kB to GB
    return {"disk_free_gb": round(free / 1e9, 1), "ram_available_gb": round(avail, 1)}


def hf_read(url: str, nbytes: int) -> float:
    """MB/s for one ranged read of a model file the window downloads, from the same CDN."""

    import urllib.request  # noqa: PLC0415

    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{nbytes - 1}"})
    t0, got = time.monotonic(), 0
    with urllib.request.urlopen(req, timeout=60) as r:
        while chunk := r.read(1 << 20):
            got += len(chunk)
    return round(got / 1e6 / max(time.monotonic() - t0, 1e-6), 1)


def verdict(r: dict, lim: dict) -> list[str]:
    """Why this VM fails, or [] when it passes."""

    why = []
    if lim["gpu"]:
        if lim["name_has"] and lim["name_has"] not in r.get("gpu_name", ""):
            why.append(f"card {r.get('gpu_name')!r} is not {lim['name_has']}")
        if r.get("vram_mib", 0) < lim["min_vram_mib"]:
            why.append(f"VRAM {r.get('vram_mib')} MiB < {lim['min_vram_mib']}")
        if r.get("vram_used_mib", 0) > lim["max_used_mib"] or r.get("gpu_processes", 0):
            why.append(f"card in use: {r.get('vram_used_mib')} MiB used, {r.get('gpu_processes')} processes")
    if r.get("disk_free_gb", 0) < lim["min_disk_gb"]:
        why.append(f"disk {r.get('disk_free_gb')} GB free < {lim['min_disk_gb']}")
    if r.get("ram_available_gb", 0) < lim["min_ram_gb"]:
        why.append(f"RAM {r.get('ram_available_gb')} GB available < {lim['min_ram_gb']}")
    if lim["hf"] and r.get("hf_mbps", 0) < lim["min_hf_mbps"]:
        why.append(f"Hugging Face {r.get('hf_mbps')} MB/s < {lim['min_hf_mbps']}")
    return why


def main() -> None:
    lim = limits()
    r: dict = {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        if lim["gpu"]:
            r.update(gpu_readings())
        r.update(host_readings())
        if lim["hf"]:
            reads = [hf_read(lim["hf_url"], lim["hf_bytes"])]
            if reads[0] < lim["min_hf_mbps"]:  # one short read doesn't decide (R&D): a second before rejecting
                reads.append(hf_read(lim["hf_url"], lim["hf_bytes"]))
            r["hf_reads_mbps"], r["hf_mbps"] = reads, max(reads)
        why = verdict(r, lim)
    except Exception as e:  # noqa: BLE001 - any failure to read the VM rejects it, with the reason recorded
        why = [f"gate could not read the VM: {type(e).__name__}: {str(e)[:200]}"]
    r.update(ok=not why, why=why, limits={k: v for k, v in lim.items() if k != "hf_url"})
    print("VM_GATE " + json.dumps(r, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
