"""Window 1 on a Colab VM: build the pinned llama-server with CUDA 13.4, fetch or convert each Colab model's Q8_0
GGUF at its pinned revision, hash everything, and smoke-serve each model once. No rows are written.

Run by scripts/colab/pin_window.sh, which assigns the VM, runs this under an outer timeout, downloads the results
and stops the VM in a trap, then checks the sessions list is empty. (The VM isn't unassigned from inside: the
results must be downloaded first.)

Prints one `PIN_RESULT {json}` line at the end: the machine line, the build (commit, nvcc, archs, binary sha256),
and per model the source (maker GGUF, or a conversion from the BF16 repo at a revision), the file and its sha256,
the smoke tokens/s and whether the reply carried `reasoning_content`.

Standards: PIN_PER_STEP (every input pinned by commit or revision, every output hashed), ANDON_AUTHORITY (a step
that fails stops the window and is reported; a fallback is recorded, never silent). Compensator: the VM is released
in `finally`; nothing is written outside /content.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

# No implicit Hub token, ever (R&D, from Argon's W-A reply, 2026-10-18). With token=None, huggingface_hub looks in the
# environment, ~/.cache/huggingface/token and, inside Colab, the account's secrets vault (userdata.get("HF_TOKEN")). In W-A
# that lookup timed out after ~10 s before Bolt's public download; with a notebook attached to the runtime it would succeed
# and use whatever token the account holds. Set here, before anything imports huggingface_hub; child processes inherit it.
# An explicit token= (the prefetch, the upload child) is unaffected.
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"


def no_implicit_token() -> None:
    """The same for a huggingface_hub an earlier exec in this kernel already imported: it read its constants then."""

    c = sys.modules.get("huggingface_hub.constants")
    if c is not None:
        c.HF_HUB_DISABLE_IMPLICIT_TOKEN = True


no_implicit_token()

LLAMA_CPP_COMMIT = "620d2225cc436ab0b001e3200e590e3b286ef9ab"  # ggml-org/llama.cpp release v0.7.0, 2026-10-15
# SASS for A100 (80), H100/H200 (90) and B200 (100): one binary for Colab and RunPod, no PTX JIT.
CUDA_ARCHS = "80;90;100"
CUDA_ARCHS_RE = re.compile(r"^[0-9]+(;[0-9]+)*$")


def cuda_archs() -> str:
    raw = os.environ.get("PIN_CUDA_ARCHS")
    if raw is None:
        return CUDA_ARCHS
    if not CUDA_ARCHS_RE.fullmatch(raw):
        raise RuntimeError(f"REFUSED: PIN_CUDA_ARCHS={raw!r} must match ^[0-9]+(;[0-9]+)*$")
    return raw
NVCC_VERSION = "13.4.92"  # nvcc, nvvm, crt and the runtime, one release
# cuBLAS follows its own numbering: the CUDA 13.4.2 toolkit (runtime 13.4.92) ships cuBLAS 13.8.0.4. Pinned exactly; a
# `~=13.4` spec pulled the later standalone 13.8.1.7 in window 2 (2026-10-17).
CUBLAS_VERSION = "13.8.0.4"
CUDA_TOOLKIT = "13.4.2 (redistrib 2026-09-16)"
NVCC_PIP = f"nvidia-cuda-nvcc=={NVCC_VERSION}"
CONTENT = Path(os.environ.get("PIN_CONTENT") or "/content")
WORK = Path(os.environ.get("PIN_WORK") or (CONTENT / "pin"))
HEARTBEAT_S = 300
MODELS = [
    # name, how, repo, file or None (convert), revision (None = the repo's current main, recorded)
    # The maker's Muse GGUF repo has only Q4 builds (window 1, 2026-10-17), so both are converted here: one binary, every
    # GGUF made by us. Revisions are pinned; Lightning's window-1 hash is the reproducibility check.
    {"served_as": "nemotron-3.5-lightning-30b-a3b", "how": "convert", "repo": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
     "revision": "a9904d24bcc1d289a1950fa9d2b978c47cf903b9",
     "expect_sha256": "5bea2069f4a8067659ed8bfb4bce58761f37d308e9689818ef77915d3b446832"},
    # Muse was saved with transformers 5.15.0.dev0 (tokenizer class TokenizersBackend), which llama.cpp's pinned
    # 4.57.6 lacks (window 2): its own env on the latest release, and its tokenizer checked id by id against the GGUF.
    {"served_as": "muse-glimmer-30b", "how": "convert", "repo": "meta-models/Muse-Glimmer-30B",
     "revision": "a4e59da52a7bc87ae7251dd5545c0dd437c44b68", "env": {"transformers": "5.20.1"}, "tokenizer_check": True,
     "expect_sha256": "c321320a81f177362f3ddf3c199edff097ab88d10b19a13eab7f606f5ffc4e74"},
    # The pilot's other arms (R&D, 2026-10-17), at the revisions J2's card-values.json read, so one GGUF serves J1 and J2.
    # Classic tokenizer classes, so llama.cpp's own requirements; each tokenizer is still checked id by id.
    {"served_as": "qwen3.8-27b", "how": "convert", "repo": "Qwen/Qwen3.8-27B",
     "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", "tokenizer_check": True,
     "expect_sha256": "2ab268e0bd3f8e5daf1280e2243ac19abd5e3bb34e8444f5cfff5147fab977d4"},
    # Window 5 pinned it on the A100 with an unproven fit; window 6 (H100) reconverts it, so the hash checks determinism.
    {"served_as": "qwen3.6-35b-a3b", "how": "convert", "repo": "Qwen/Qwen3.6-35B-A3B",
     "revision": "995ad96eacd98c81ed38be0c5b274b04031597b0", "tokenizer_check": True,
     "expect_sha256": "b7c0c8eb16a54af29e87845d187969f99ecbbf79cd47cc198be8890d109dc164",
     # Window 8 and 9's Q6_K was requantized from the Q8_0; retired 2026-10-17 (R&D), kept as history, never checked.
     "retired_q6k_sha256": "61deb1f7862709b62692d76ca510a669affeeb4c63e59de72d7aa901b638a4fe"},
    {"served_as": "granite-4.2-30b", "how": "convert", "repo": "ibm-granite/granite-4.2-30b",
     "revision": "9e668ce1c538387ef24d3644e9b0606647762636", "tokenizer_check": True,
     "expect_sha256": "1b5c47835f6f8b8732550541dc432d0a351e3f706961ff3c750d3ee1987b9b4c"},
    # Gemma 4 was saved with transformers 5.5.0.dev0; its tokenizer_config's list-shaped extra_special_tokens breaks
    # llama.cpp's pinned 4.57.6 (window 5), so it converts in its own env, as Muse does.
    {"served_as": "gemma-4-31b-it", "how": "convert", "repo": "google/gemma-4-31B-it",
     "revision": "842da3794eaa0b77d5f08bae87a17459d91ff475", "env": {"transformers": "5.20.1"}, "tokenizer_check": True,
     "expect_sha256": "e357a45071c2cbbc44b97506dfadc34bac0d5f6592f1d500edfaa9647f01e7bf",
     "retired_q6k_sha256": "3a72fddfa71a94d3dc0b03b804ee1bd7dfa56881d3284afd74e69651f14f206f"},
    # Pinned in window 5; its 32k KV cache doesn't fit the A100, so it serves on the H100 (R&D, 2026-10-17).
    {"served_as": "deepseek-r1-distill-qwen-32b", "how": "convert", "repo": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
     "revision": "711ad2ea6aa40cfca18895e8aca02ab92df1a746", "tokenizer_check": True,
     "expect_sha256": "b791775c3bb0d1ab3cd7a73c6760c4ab915680f3587be8a7d29937ca2b9ad24e",
     "retired_q6k_sha256": "c2cda735fb0f026d33db4734c1466b94cfca9ed8f9a88bab0610bca45687bce5"},
    # The larger critic of critic-signal §10 (rnd 1.1.15.2.9), used text-only: the converter takes llm_config as the text
    # config (NemotronHForCausalLM, latent MoE, one MTP head) and leaves the vision tower out. Saved with transformers
    # 5.12.1 (TokenizersBackend), so Muse's recipe: its own env and the tokenizer checked id by id. The pin comes from the
    # first conversion (PIN_ROLE=convert on the RunPod converter) and is added before any prebuild or serve.
    # Pinned 2026-10-19 from plan 2 (RTX PRO 6000 Server pod): BF16 1,282 s, then llama-quantize Q8_0 run twice from the
    # same BF16 (209425cf...), both 6fa3c010 (the second rc 0, 112 s); one conversion, so reproducibility isn't re-proven.
    {"served_as": "nemotron-3.5-super-vl-120b-a12b", "how": "convert", "repo": "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16",
     "revision": "1f8953f7f0935358e3605150d22bc15df062061c", "env": {"transformers": "5.20.1"}, "tokenizer_check": True,
     "via_bf16": True, "expect_sha256": "6fa3c010ebd015d298d910a2ab6ad4d953155b1cc418b0f54581075aafbeeb02"},
    # NVIDIA's own 24GB quantization-aware GGUF: fetched and hashed, not converted.
    {"served_as": "nemotron-3.5-bolt-74b-a9b", "how": "maker_gguf", "repo": "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF",
     "revision": "e0fecb69cb3e5e4c5e8a056fa2aec98c78e91246", "file": "NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf",
     "card": "nemotron-3.5-bolt-74b-a9b-24gb", "expect_sha256": "91d63363e782c8a3d5f4be42816eba8e0d42dd2530bd07a7d0d2dc440599af75"},
]
# Window 1's llama-server (built 2026-10-17 at LLAMA_CPP_COMMIT, nvcc 13.4.92): every later window runs this file,
# unless PIN_SERVER_SHA256 names another pinned build (e.g. window 16g's G4 sm120 binary).
SERVER_SHA256 = "b4659b3fc74037deb96b2148c455753121fa21dde174d7124547e2bf9ffd37c3"
SERVER_ARCHS = "80;90;100 (SASS only, no PTX)"
G4_SERVER_SHA256 = "39742d8a0354d4577da8dcc9cb3a5b9f03f02593d106eff41297e952d6ebfc95"
G4_SERVER_ARCHS = "80;90;100;120 (sm_120a SASS)"
SERVER_BUILDS = (
    {"commit": LLAMA_CPP_COMMIT, "sha256": SERVER_SHA256, "cuda": "13.4", "archs": SERVER_ARCHS},
    {"commit": LLAMA_CPP_COMMIT, "sha256": G4_SERVER_SHA256, "cuda": "13.4", "archs": G4_SERVER_ARCHS},
)
SERVER_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def server_sha256() -> str:
    raw = os.environ.get("PIN_SERVER_SHA256")
    if raw is None:
        return SERVER_SHA256
    if not SERVER_SHA256_RE.fullmatch(raw):
        raise RuntimeError(f"REFUSED: PIN_SERVER_SHA256={raw!r} must be 64 lowercase hex characters")
    return raw


def server_pin_entry(sha: str | None = None, writers_path: Path | None = None) -> dict:
    """The matched server pin entry (commit, sha256, archs) for `sha` from writers.json or SERVER_BUILDS."""

    want = (sha or server_sha256()).strip().lower()
    candidates = list(SERVER_BUILDS)
    if writers_path is not None and writers_path.is_file():
        data = json.loads(writers_path.read_text(encoding="utf-8"))
        candidates = ([data["llama_server"]] if isinstance(data.get("llama_server"), dict) else []) + list(data.get("llama_server_builds") or [])
    for entry in candidates:
        if (entry.get("sha256") or "").strip().lower() == want:
            return {"commit": entry["commit"], "sha256": want, "archs": entry.get("archs") or ""}
    raise RuntimeError(f"REFUSED: llama-server {want[:12]} is not a pinned build in writers.json")


def stub_mode() -> bool:
    return os.environ.get("PIN_STUB") == "1"


def _disk_target() -> Path:
    for p in (CONTENT, WORK, Path(".")):
        if p.exists():
            return p
    return Path(".")


def _meminfo() -> dict[str, int]:
    p = Path("/proc/meminfo")
    if not p.exists():
        return {"MemAvailable": 0, "MemTotal": 0}
    mem = dict(ln.split(":", 1) for ln in p.read_text().splitlines() if ":" in ln)
    return {k: int(mem.get(k, "0 kB").split()[0]) // 1024 for k in ("MemAvailable", "MemTotal")}


# The window's deadline (R&D, from Argon's golden-path reply, 2026-10-18): smoke_window passes PIN_DEADLINE, the epoch at
# which its own spend reaches the cap less the margin. Past it no wave, download or model setup starts, and a wave under
# way is cut: its role-set process gets SIGTERM, then SIGKILL after 30 s, and the job raises DeadlineReached so the
# finally blocks write pilot-out.tar and pin-result.json before the rig fetches. The call journal keeps every finished
# call, so a cut wave loses only the call in flight and a resume replays the rest. Without this, a cap stop killed the
# exec and only job.log came back (12g).
class DeadlineReached(Exception):  # not a RuntimeError: nothing that falls back on a RuntimeError may swallow it
    pass


CUT_AT: dict | None = None  # the wave cut at the deadline, for pin-result.json


def deadline_left() -> float | None:
    d = os.environ.get("PIN_DEADLINE")
    return None if not d else float(d) - time.time()


def check_deadline(before: str) -> None:
    left = deadline_left()
    if left is not None and left <= 0:
        raise DeadlineReached(f"the window's deadline passed before {before}; nothing new starts")


def _signal_group(proc, sig) -> None:
    """Signal a child started with start_new_session=True and everything it spawned: under `sh -c`, killing the shell
    alone leaves its children holding the output pipes, and the wait after the kill never returns (Argon, v2 re-review)."""

    import signal as _signal  # noqa: PLC0415

    try:
        os.killpg(proc.pid, sig if sig is not None else _signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def heartbeat(msg: str) -> None:
    """A HEARTBEAT line in job.log: through the Tee when the job runs under it, written directly otherwise."""

    say(msg)
    if not isinstance(sys.stdout, Tee) and JOB_LOG.parent.exists():
        with JOB_LOG.open("a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")


def _tree_bytes(root: Path | None) -> int:
    """Bytes under root, hidden files included (the Hub's .incomplete files live in local_dir/.cache)."""

    if root is None or not Path(root).exists():
        return 0
    total = 0
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            try:
                total += os.lstat(os.path.join(dirpath, f)).st_size
            except OSError:
                pass
    return total


DOWNLOAD_FLOOR_MBPS = 20  # the hard timeout's floor rate (R&D, 2026-10-18): bytes / 20 MB/s + 300 s


def download_timeout(nbytes: int | None) -> float:
    return 3600.0 if not nbytes else nbytes / (DOWNLOAD_FLOOR_MBPS * 1e6) + 300


# A conversion's time bound, from the snapshot it reads (R&D, 2026-10-19, after Super-VL-120B hit a flat 3600 s cap).
# Measured: the direct q8_0 path read Super-VL's 249 GB BF16 at ~1.8 GB/min of input (0.9 GB/min of output); 30B models
# ran far faster. The bound allows 1 GB/min of input, never below the old 3600 s. A backstop, not a schedule: a window's
# deadline and a pod's watchdog still bound the cost.
CONVERT_FLOOR_S = 3600
CONVERT_INPUT_GB_PER_MIN = 1.0


def convert_timeout(snapshot_bytes: int | None) -> int:
    if not snapshot_bytes:
        return CONVERT_FLOOR_S
    return max(CONVERT_FLOOR_S, int(snapshot_bytes / 1e9 / CONVERT_INPUT_GB_PER_MIN * 60))


def with_heartbeat(fn, label: str, timeout_s: float, watch: Path | None = None):
    """Run fn (an in-process Hub download) in a worker thread while this thread waits on it, writing a HEARTBEAT line
    each HEARTBEAT_S round only if the download made progress: the bytes under `watch` grew, or free disk fell (a Xet
    download may assemble its chunks elsewhere first). A hung download writes nothing, so the rig's stall on job.log
    (PIN_STALL_S, 600 s) ends the window (R&D, 2026-10-18, after window 13q showed the stall is the only guard against a
    dead kernel). The heartbeat is tied to this wait, never a free-running thread (Argon #5). Past timeout_s the job
    raises: the worker thread can't be killed, but the job's finally runs and the window ends, so a lingering daemon
    thread is acceptable. The same at the window's deadline (R&D on #92): a download can come mid-run (W-A fetched Bolt
    lazily at 12:59), so each round checks PIN_DEADLINE and raises DeadlineReached, and the results are written before
    the watcher's cap kill. fn's own exceptions, DeadlineReached included, are raised here. Known residue: a prefetch
    download's lingering worker keeps the token in its closure until hf_hub_download returns (take_token has already
    cleared _TOKEN, and the window is ending)."""

    import threading as _threading  # noqa: PLC0415

    box: dict = {}

    def run() -> None:
        try:
            box["value"] = fn()
        except BaseException as e:  # noqa: BLE001 - handed to the waiting thread, which raises it
            box["error"] = e

    hb_s = float(os.environ.get("PIN_HEARTBEAT_S") or HEARTBEAT_S)
    t0 = time.time()
    last_t, last_bytes, last_free = t0, _tree_bytes(watch), shutil.disk_usage(_disk_target()).free
    worker = _threading.Thread(target=run, name=f"download-{label}", daemon=True)
    worker.start()
    while True:
        left = deadline_left()
        worker.join(timeout=max(0.001, min(hb_s, t0 + timeout_s - time.time(), *([left] if left is not None else []))))
        if not worker.is_alive():
            break
        left = deadline_left()
        if left is not None and left <= 0:
            raise DeadlineReached(f"{label} was under way at the window's deadline; nothing more is fetched")
        now = time.time()
        if now - t0 >= timeout_s:
            raise RuntimeError(f"{label}: not done after {round(timeout_s)} s (floor {DOWNLOAD_FLOOR_MBPS} MB/s); the window stops")
        nbytes, free = _tree_bytes(watch), shutil.disk_usage(_disk_target()).free
        grew = max(nbytes - last_bytes, last_free - free)
        if grew > 1 << 20:  # progress: a heartbeat with the rate, also data for the gate floor's recalibration
            mbps = grew / 1e6 / max(now - last_t, 0.001)
            heartbeat(f"HEARTBEAT {label} elapsed={round(now - t0)}s bytes={nbytes} mbps={mbps:.1f}")
        last_t, last_bytes, last_free = now, nbytes, free
    if "error" in box:
        raise box["error"]
    return box.get("value")


def run_proc(cmd, *, shell: bool = False, cwd=None, timeout: float = 3600, env=None,
             term_first: bool = False, input: str | None = None) -> subprocess.CompletedProcess:  # noqa: A002
    """Run a subprocess, writing a heartbeat line to job.log every HEARTBEAT_S while the subprocess lives (Argon #5).
    Tied to the subprocess wait loop, never a free-running thread, so it stops when the job ends or hangs in Python."""

    hb_s = float(os.environ.get("PIN_HEARTBEAT_S") or HEARTBEAT_S)
    head = (cmd if isinstance(cmd, str) else " ".join(str(x) for x in cmd)).strip().splitlines()[0][:80]
    # errors="replace" (plan 2, 2026-10-19): llama-quantize printed a non-UTF-8 byte (0xc4) on Super-VL and the strict
    # decode raised after the child had finished, losing its rc and timing. A tool's output is a log, never data here.
    proc = subprocess.Popen(cmd, shell=shell, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            encoding="utf-8", errors="replace", env=env,
                            stdin=subprocess.PIPE if input is not None else None, start_new_session=True)
    send = input  # given to the first communicate only
    t0 = time.time()
    deadline = t0 + timeout
    next_hb = t0 + hb_s
    try:
        while True:
            now = time.time()
            wait = max(0.001, min(deadline - now, next_hb - now))
            try:
                out, err = proc.communicate(input=send, timeout=wait)
                return subprocess.CompletedProcess(cmd, proc.returncode, out, err)
            except subprocess.TimeoutExpired:
                send = None
                now = time.time()
                if now >= deadline:
                    import signal as _signal  # noqa: PLC0415

                    # term_first (the deadline): SIGTERM lets the wave's process end its current write, then SIGKILL.
                    _signal_group(proc, _signal.SIGTERM if term_first else None)
                    try:
                        out, err = proc.communicate(timeout=30)
                    except subprocess.TimeoutExpired:
                        _signal_group(proc, None)
                        try:
                            out, err = proc.communicate(timeout=30)
                        except subprocess.TimeoutExpired:
                            out, err = "", ""
                    raise subprocess.TimeoutExpired(cmd, timeout, output=out, stderr=err) from None
                if proc.poll() is None and now >= next_hb:
                    disk_gb = round(shutil.disk_usage(_disk_target()).free / 1e9, 1)
                    ram_mb = _meminfo()["MemAvailable"]
                    heartbeat(f"HEARTBEAT pid={proc.pid} elapsed={round(now - t0)}s disk_free_gb={disk_gb} ram_free_mb={ram_mb} cmd={head}")
                    next_hb = now + hb_s
    finally:
        if proc.poll() is None:
            _signal_group(proc, None)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass


def sh(cmd, cwd=None, timeout=3600, env=None) -> str:
    out = run_proc(cmd, shell=True, cwd=cwd, timeout=timeout, env=env)
    if out.returncode != 0:
        raise RuntimeError(f"{cmd[:80]}... exit {out.returncode}: {(out.stderr or out.stdout)[-1500:]}")
    return out.stdout.strip()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


JOB_LOG = WORK / "job.log"


class Tee:
    """Everything the window prints, also appended to /content/pin/job.log and flushed per write (mistakes #31): the
    exec's output stream stopped reaching the rig in windows 10d and 12a while the job ran on, so the rig reads the job's
    progress, its checkpoints and its PIN_RESULT line from this file instead."""

    def __init__(self, stream, path: Path):
        self.stream, self.file, self.lock = stream, path.open("a", encoding="utf-8"), threading.Lock()

    def write(self, text: str) -> int:
        with self.lock:
            try:
                self.stream.write(text)
                self.stream.flush()
            except Exception:  # noqa: BLE001 - a dead stream must not stop the file
                pass
            self.file.write(text)
            self.file.flush()
        return len(text)

    def flush(self) -> None:
        with self.lock:
            self.file.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def machine_line(cpu_ok: bool = False) -> dict:
    if cpu_ok and not shutil.which("nvidia-smi"):
        # A CPU runtime converting GGUFs (R&D, 2026-10-18): no card, nothing served; the converter's output is the same
        # bytes on any host, which the pin check proves.
        return {"card": "CPU", "memory": "0 MiB", "driver": "none", "compat": False, "driver_cuda": "none",
                "colab_release_tag": os.environ.get("COLAB_RELEASE_TAG"), "cores": os.cpu_count(),
                "disk_free_gb": round(shutil.disk_usage(_disk_target()).free / 2**30, 1), "runtime": "none", "cpu": True}
    if stub_mode():
        return {"card": "STUB (CPU)", "memory": "0 MiB", "driver": "stub", "compat": False,
                "driver_cuda": "stub", "colab_release_tag": os.environ.get("COLAB_RELEASE_TAG") or "stub-local",
                "cores": os.cpu_count(), "disk_free_gb": round(shutil.disk_usage(_disk_target()).free / 2**30, 1),
                "runtime": "13.4", "stub": True}
    gpu = sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader")
    name, mem, driver = [x.strip() for x in gpu.split(",")]
    return {"card": name, "memory": mem, "driver": driver, "compat": False,
            "driver_cuda": sh("nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | cut -d' ' -f3"),
            "colab_release_tag": os.environ.get("COLAB_RELEASE_TAG"), "cores": os.cpu_count(),
            "disk_free_gb": round(shutil.disk_usage(_disk_target()).free / 2**30, 1),
            # The CUDA runtime the build targets; role-set's load_machine requires it (R&D on #77).
            "runtime": "13.4"}


def build() -> dict:
    archs = cuda_archs()
    say("installing nvcc " + NVCC_PIP)
    # nvcc's own compiler parts (nvvm's cicc, crt) come from their own wheels; the image ships 13.0 ones, which reject
    # 13.4's flags (window 1, 2026-10-17: "invalid option: --simt-only"), so every part is pinned to the same release.
    sh(f"{sys.executable} -m pip install -q 'nvidia-cuda-nvcc=={NVCC_VERSION}' 'nvidia-nvvm=={NVCC_VERSION}' "
       f"'nvidia-cuda-crt=={NVCC_VERSION}' 'nvidia-cuda-runtime=={NVCC_VERSION}' 'nvidia-cuda-cccl' 'nvidia-cublas=={CUBLAS_VERSION}' "
       "cmake ninja", timeout=1200)
    import nvidia  # noqa: PLC0415 - installed just above

    # The CUDA 13 wheels put their tools under nvidia/cu13/ (window 1, 2026-10-17: nvcc 13.4.92 at nvidia/cu13/bin),
    # so search the whole nvidia package. Never the image's own /usr/local/cuda-13.0 nvcc.
    found = sorted(p for root in nvidia.__path__ for p in Path(root).rglob("bin/nvcc") if p.is_file())
    if not found:
        raise RuntimeError("the pip nvcc wheel installed no bin/nvcc under the nvidia package")
    nvcc = found[0]
    version = sh(f"{nvcc} --version | tail -2")
    if "13.4" not in version:
        raise RuntimeError(f"nvcc is not 13.4: {version}")
    say(f"nvcc: {version.splitlines()[-1]}")
    src = WORK / "llama.cpp"
    if not (src / ".git").exists():
        sh(f"git clone -q https://github.com/ggml-org/llama.cpp {src} && git -C {src} checkout -q {LLAMA_CPP_COMMIT}", timeout=900)
    # The wheels form a toolkit under nvidia/cu13 (bin, include, lib) but ship only versioned libraries
    # (libcudart.so.13), so CMake's FindCUDAToolkit needs the unversioned names, and the driver library from the VM.
    cu = nvcc.parent.parent
    for so in (cu / "lib").glob("*.so.13"):
        link = so.with_name(so.name[: -len(".13")])
        if not link.exists():
            link.symlink_to(so.name)
    libcuda = sh("ldconfig -p | grep -m1 'libcuda.so ' | awk '{print $NF}'") or sh("ls /usr/lib64-nvidia/libcuda.so | head -1")
    env = {**os.environ, "CUDACXX": str(nvcc), "PATH": f"{nvcc.parent}:{os.environ['PATH']}",
           "LD_LIBRARY_PATH": f"{cu / 'lib'}:" + os.environ.get("LD_LIBRARY_PATH", "")}
    shutil.rmtree(src / "build", ignore_errors=True)
    sh(f"cmake -S {src} -B {src}/build -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES='{archs}' "
       f"-DCUDAToolkit_ROOT={cu} -DCMAKE_CUDA_COMPILER={nvcc} -DCUDA_cuda_driver_LIBRARY={libcuda} "
       f"-DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release", env=env, timeout=900)
    say("building llama-server and llama-quantize")
    sh(f"cmake --build {src}/build --target llama-server llama-quantize -j {os.cpu_count()}", env=env, timeout=3600)
    server = src / "build" / "bin" / "llama-server"
    return {"commit": LLAMA_CPP_COMMIT, "nvcc": version.splitlines()[-1], "cuda_archs": archs,
            "sha256": sha256(server), "path": str(server), "quantize": str(src / "build" / "bin" / "llama-quantize"),
            "src": str(src)}


def hf_revision(repo: str) -> str:
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}", timeout=60) as r:  # noqa: S310 - fixed host
        return json.loads(r.read())["sha"]


def resources(tag: str) -> dict:
    """Disk and RAM at one point of the window, kept in the result (R&D, window 10b: a VM that runs out of either can
    be reclaimed without a clean error)."""

    du = shutil.disk_usage(_disk_target())
    mem = _meminfo()
    row = {"at": tag, "time": time.strftime("%H:%M:%S", time.gmtime()), "disk_free_gb": round(du.free / 1e9, 1),
           "ram_available_mb": mem["MemAvailable"], "ram_total_mb": mem["MemTotal"]}
    RESOURCES.append(row)
    say(f"resources {row}")
    return row


RESOURCES: list = []


# ---- pre-built GGUFs in the studio's private repo (R&D, 2026-10-18: licences cleared for private derivatives) ----------
# A CPU window (PIN_ROLE=prebuild) converts one model, and uploads it only if its Q8_0 byte-matches the model's
# expect_sha256. An A100 pilot window with PIN_GGUF_REPO and PIN_GGUF_REVISION downloads every model it needs before
# any wave, checks each against the same pin, and then has no token left. The token reaches the VM only as a file the
# rig uploads to RAM-backed /dev/shm, outside /content and every tar (R&D, from Argon): never a command line, never the
# environment, never a persisted file. The job reads it once at its very start, whatever the role, deletes the file and
# keeps the value only in _TOKEN until the one step that uses it is done. It goes to the Hub only as a token= argument
# (an explicit token= isn't written to ~/.cache/huggingface). If a token ever shows up in a persisted file, the receipts
# README's compensator applies (Mike revokes it).
HOSTED: dict = {}  # served_as -> {"path", "sha256"} of a GGUF this window took from the private repo
TOKEN_PATH = Path("/dev/shm/hf.token")
_TOKEN: str | None = None  # the value, between the job's start and the step that uses it
# Licences stated only in a model card (Gemma 4 and Granite 4.2 ship no LICENSE file): one vendored copy of the text,
# pinned, so an upload needs no network for it and the text can't drift (R&D, 2026-10-18). Source:
# https://www.apache.org/licenses/LICENSE-2.0.txt, fetched 2026-10-18.
# openmdw-1.1.txt is the agreement alone ("OpenMDW License Agreement, version 1.1" through "Copyright The Linux
# Foundation and its contributors." plus one LF), cut by R&D from the registry snapshot
# E:/AI/licence-snapshots/2026-10-17/openmdw-1.1.txt (page-text sha256 c709632b, read 2026-10-17).
LICENCE_FALLBACK = {"apache-2.0": ("apache-2.0.txt", "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"),
                    "openmdw-1.1": ("openmdw-1.1.txt", "ac19de4aa4e0216bbf0ca2fc5794490637a6ba786f12afec8fb63f0005c5b27f")}
DISK_MARGIN_GB = 20  # R&D: a window's peak plus this must fit the VM's free disk


def token_file() -> Path:
    return TOKEN_PATH


def read_token_file(needed: bool) -> None:
    """At the job's very start, whatever the role: the file is read into _TOKEN and deleted, and the Hub's environment
    variables are dropped. A window that won't use it (needed False) keeps nothing."""

    global _TOKEN
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):  # never left where a subprocess or a later step could read one
        os.environ.pop(k, None)
    p = token_file()
    if not p.exists():
        _TOKEN = None
        return
    try:
        value = p.read_text(encoding="utf-8").strip() or None
    finally:
        p.unlink(missing_ok=True)
    _TOKEN = value if needed else None
    if not needed:
        say("a token file reached a window that doesn't use it: deleted, not kept")


def take_token() -> str | None:
    """The token, handed to the one step that uses it, and forgotten here."""

    global _TOKEN
    tok, _TOKEN = _TOKEN, None
    return tok


def token_clean() -> dict:
    """Every place a token could still be, read after the prefetch (R&D, from Argon's W-A reply): the /dev/shm file, the
    value in _TOKEN, the Hub's environment variables, the Hub's token cache, and the Colab secret huggingface_hub caches
    after a vault lookup (read by getattr, so a hub version without it reads as clean, not a crash)."""

    home = Path(os.environ.get("HF_HOME") or (Path.home() / ".cache" / "huggingface"))
    auth = sys.modules.get("huggingface_hub.utils._auth")
    return {"file_gone": not token_file().exists(), "value_dropped": _TOKEN is None,
            "env_clear": not ({"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"} & set(os.environ)),
            "no_cached_token": not any((home / f).exists() for f in ("token", "stored_tokens")),
            "no_colab_secret": getattr(auth, "_GOOGLE_COLAB_SECRET", None) is None}


def check_token_clean(result: dict) -> None:
    result["token_clean"] = clean = token_clean()
    say(f"token_clean {clean}")
    left = [k for k, v in clean.items() if not v]
    if left:
        raise RuntimeError(f"a token residue is left before any wave ({', '.join(left)}); the window stops here")


def hosted_name(served_as: str) -> str:
    return f"{served_as}/{served_as}-Q8_0.gguf"


def prefetch_hosted(names: list, by_name: dict, result: dict) -> None:
    """Every Q8_0 the window will serve, from the private repo at its pinned commit, before any wave (R&D: the token is
    gone before a wave runs). Each file must equal its expect_sha256, or the window stops before any model is served."""

    repo, rev = os.environ.get("PIN_GGUF_REPO"), os.environ.get("PIN_GGUF_REVISION")
    if not repo:
        take_token()  # nothing to download: a token kept anyway is forgotten, never carried through the waves
        check_token_clean(result)
        return
    if not rev or len(rev) != 40:
        raise RuntimeError("PIN_GGUF_REPO needs PIN_GGUF_REVISION, the repo's full commit sha")
    token = take_token()
    if token is None:
        raise RuntimeError("PIN_GGUF_REPO is set, but no token file reached the VM")
    from huggingface_hub import HfApi, hf_hub_download  # noqa: PLC0415

    try:
        # Every file first, then the disk check, then the downloads (R&D): the window's whole set plus the margin must fit
        # the free disk, or it stops before any download.
        todo = [n for n in names if by_name[n]["how"] == "convert" and n not in HOSTED]
        sizes = {i.path: i.size for i in HfApi(token=token).get_paths_info(repo, [hosted_name(n) for n in todo], revision=rev)}
        missing = [hosted_name(n) for n in todo if not sizes.get(hosted_name(n))]
        if missing:
            raise RuntimeError(f"{repo}@{rev[:12]} doesn't hold {', '.join(missing)}; nothing downloaded")
        need_gb = sum(sizes[hosted_name(n)] for n in todo) / 1e9
        free_gb = shutil.disk_usage(_disk_target()).free / 1e9
        result["hosted_disk"] = {"need_gb": round(need_gb, 1), "free_gb": round(free_gb, 1), "margin_gb": DISK_MARGIN_GB}
        if need_gb + DISK_MARGIN_GB > free_gb:
            raise RuntimeError(f"the pre-built GGUFs need {need_gb:.1f} GB + {DISK_MARGIN_GB} GB margin; {free_gb:.1f} GB free")
        for n in names:
            spec = by_name[n]
            if spec["how"] != "convert" or n in HOSTED:
                continue
            check_deadline(f"{n}'s hosted download")
            resources(f"{n}: before hosted download")
            say(f"downloading {n}'s pre-built Q8_0 from the studio's repo")
            size = sizes[hosted_name(n)]
            path = Path(with_heartbeat(
                lambda n=n: hf_hub_download(repo, hosted_name(n), revision=rev, token=token, local_dir=WORK / "hosted"),
                f"hosted:{n}", download_timeout(size), watch=WORK / "hosted"))
            got = sha256(path)
            if got != spec["expect_sha256"]:
                path.unlink(missing_ok=True)
                raise RuntimeError(f"{n}: the hosted GGUF {got[:12]} isn't its pin {spec['expect_sha256'][:12]}")
            HOSTED[n] = {"path": path, "sha256": got}
    finally:
        token = None  # noqa: F841 - dropped before any wave runs
    result["hosted"] = {"repo": repo, "revision": rev, "models": {n: v["sha256"] for n, v in HOSTED.items()}}
    resources("after hosted downloads")
    check_token_clean(result)


CARD_LICENCE_SRC = """
def card_licence(card):
    \"\"\"A model card's licence from its YAML front matter: `license:`, or `license_name:` when the licence is
    `other` (Super-VL: other / openmdw-1.1), so two different "other" licences never compare equal.\"\"\"

    if not card.exists():
        return None
    lines = card.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    fields = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        for key in ("license", "license_name"):
            if line.startswith(key + ":"):
                fields[key] = line.split(":", 1)[1].strip().strip("'\\"")
    licence = fields.get("license")
    # A bare "other" names no licence at all: None, so it can never match (R&D on #98).
    return fields.get("license_name") if licence == "other" else licence
"""
exec(CARD_LICENCE_SRC)  # defines card_licence here; the upload child execs the same text


# The upload runs in a fresh interpreter (R&D, 2026-10-18, window 13i): for a model without `env`, converter_python
# pip-installs llama.cpp's requirements into this very interpreter mid-job, which replaced huggingface_hub on disk after
# it was imported, and the upload's later imports failed (ImportError XetAuthorizationError). A child process sees one
# consistent huggingface_hub, whatever is on disk. Every check runs in the child, in this order: the repo is private, its
# card names a licence, the model's card names the same one, the licence files are staged; then the upload.
UPLOAD_CHILD = r"""
import hashlib, json, shutil, sys
from pathlib import Path
cfg = json.loads(sys.stdin.readline())
token = cfg.pop("token")
exec(cfg["card_licence_src"])


def refuse(why):
    raise RuntimeError("REFUSED: " + why)


out = {}
try:
    import huggingface_hub
    from huggingface_hub import HfApi, hf_hub_download
    out["hub_version"] = getattr(huggingface_hub, "__version__", None)
    api = HfApi(token=token)
    token = None
    repo, stage = cfg["repo"], Path(cfg["stage"])
    info = api.repo_info(repo)
    if not info.private:
        refuse(repo + " is public; pre-built GGUFs go only to a private repo (R&D's licence line)")
    card = getattr(info, "card_data", None) or {}
    get = card.get if isinstance(card, dict) else (lambda k: getattr(card, k, None))
    repo_licence = get("license")
    if repo_licence == "other":
        # The same rule as card_licence: an "other" repo is labelled by its license_name (openmdw-1.1), never "other".
        repo_licence = get("license_name")  # a bare "other" names no licence: refused below (R&D on #98)
    if not repo_licence:
        refuse(repo + "'s card names no licence; set it before any model goes in")
    stage.mkdir(parents=True, exist_ok=True)
    shutil.move(cfg["gguf"], stage / cfg["gguf_name"])
    listed = HfApi(token=False).list_repo_files(cfg["upstream"], revision=cfg["revision"])  # the upstream repos are public
    legal = [f for f in listed if "/" not in f and f.upper().startswith(("LICENSE", "LICENCE", "NOTICE"))]
    for f in legal + (["README.md"] if "README.md" in listed else []):
        got = Path(hf_hub_download(cfg["upstream"], f, revision=cfg["revision"], local_dir=cfg["legal_dir"], token=False))
        shutil.copyfile(got, stage / ("UPSTREAM-CARD.md" if f == "README.md" else f))
    licence = card_licence(stage / "UPSTREAM-CARD.md")
    if licence != repo_licence:
        # The repo's label is true only if every file in it is under that licence: Lightning (OpenMDW-1.1) and any
        # other licence go elsewhere, never here.
        refuse(cfg["upstream"] + "'s card licence is " + repr(licence) + "; " + repo + " holds " + repo_licence + " models only")
    if not any(f.upper().startswith(("LICENSE", "LICENCE")) for f in legal):
        # Gemma 4 and Granite 4.2 state Apache-2.0 in the card and ship no LICENSE file: the vendored text, pinned.
        fallback = cfg["fallback"].get(licence)
        if not fallback:
            refuse(cfg["upstream"] + " has no LICENSE file and its card's licence " + repr(licence) + " has no known text")
        data = Path(fallback[0]).read_bytes()
        if hashlib.sha256(data).hexdigest() != fallback[1]:
            refuse("the vendored licence text isn't its pinned sha256")
        (stage / "LICENSE").write_bytes(data)
    (stage / "README.md").write_text(cfg["readme"].replace("{licence}", str(licence)), encoding="utf-8")
    done = api.upload_folder(repo_id=repo, folder_path=str(stage), path_in_repo=cfg["path_in_repo"],
                             commit_message=cfg["commit_message"])
    out.update(ok=True, commit=getattr(done, "oid", None), licence=licence, files=sorted(x.name for x in stage.iterdir()))
except BaseException as e:  # every failure is reported, then a non-zero exit
    out.update(ok=False, error=type(e).__name__ + ": " + str(e)[:1000])
print("UPLOAD_RESULT " + json.dumps(out), flush=True)
sys.exit(0 if out.get("ok") else 1)
"""
UPLOAD_CHILD_ENV: dict = {}  # tests point the child at a stand-in huggingface_hub (PYTHONPATH) here
TOKEN_SHAPE_RE = re.compile(r"hf_[A-Za-z0-9]{30,}")


def conversion_route(spec: dict) -> str:
    """The hosted README's conversion line, true to the route fetch_model took (R&D, 2026-10-19: provenance is true)."""

    if spec.get("via_bf16"):
        return "convert_hf_to_gguf.py --outtype bf16, then llama-quantize Q8_0"
    return "convert_hf_to_gguf.py --outtype q8_0"


def upload_hosted(spec: dict, m: dict, result: dict) -> None:
    """A prebuild window's one model, uploaded to the private repo with its upstream licence files and model card and a
    README naming its source, revision, converter and pin (R&D's licence conditions). Only after its pin matched. The
    work runs in UPLOAD_CHILD; the token goes to it on stdin only, never argv or the environment."""

    repo = os.environ["PIN_GGUF_REPO"]
    token = take_token()
    if token is None:
        raise RuntimeError("a prebuild needs the write token's file; none reached the VM")
    stage = WORK / "upload" / spec["served_as"]
    fallback = {}
    for name, (fname, pinned) in LICENCE_FALLBACK.items():
        here = Path(globals().get("__file__") or (CONTENT / "pin_window.py")).resolve().parent
        found = [x for x in (here / "licences" / fname, CONTENT / fname) if x.exists()]
        if found:
            fallback[name] = [str(found[0]), pinned]
    cfg = {"repo": repo, "stage": str(stage), "gguf": m["path"], "gguf_name": Path(hosted_name(spec["served_as"])).name,
           "upstream": spec["repo"], "revision": m["revision"], "legal_dir": str(WORK / "upstream-legal" / spec["served_as"]),
           "fallback": fallback, "path_in_repo": spec["served_as"], "card_licence_src": CARD_LICENCE_SRC,
           "commit_message": f"{spec['served_as']} Q8_0 {m['gguf_sha256'][:12]} from {spec['repo']}@{m['revision'][:12]}",
           "readme": (f"# {spec['served_as']} Q8_0 (private, studio use only)\n\n"
                      f"- Source: https://huggingface.co/{spec['repo']} at revision {m['revision']}\n"
                      "- Licence: {licence} (upstream LICENSE/NOTICE and model card beside this file)\n"
                      f"- Conversion: llama.cpp {LLAMA_CPP_COMMIT}, {conversion_route(spec)}\n"
                      f"- sha256: {m['gguf_sha256']} (role-set pin_window.MODELS expect_sha256)\n"
                      # e.g. Super-VL (R&D, 2026-10-19): one conversion, so reproducibility isn't re-proven for it.
                      + (f"- Note: {m['readme_note']}\n" if m.get("readme_note") else ""))}
    env = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
    env.update(UPLOAD_CHILD_ENV)
    stdin = json.dumps({**cfg, "token": token}) + "\n"
    token = None  # noqa: F841 - the child has it now; this process keeps none
    try:
        # A time heartbeat, not a progress one: the child's upload shows no byte progress the job can read, so a hung
        # upload heartbeats until the 7200 s timeout (a known gap, R&D 2026-10-18).
        child = run_proc([sys.executable, "-c", UPLOAD_CHILD], input=stdin, env=env, timeout=7200)
    finally:
        stdin = None  # noqa: F841
        shutil.rmtree(stage, ignore_errors=True)  # one model per step: gone before the next download
    out, err = TOKEN_SHAPE_RE.sub("[token]", child.stdout or ""), TOKEN_SHAPE_RE.sub("[token]", child.stderr or "")
    lines = [x for x in out.splitlines() if x.startswith("UPLOAD_RESULT ")]
    res = json.loads(lines[-1].removeprefix("UPLOAD_RESULT ")) if lines else {}
    if child.returncode != 0 or not res.get("ok"):
        raise RuntimeError(f"the upload child exited {child.returncode}: {res.get('error') or err[-1500:]}")
    result["uploaded"] = {"repo": repo, "path": hosted_name(spec["served_as"]), "sha256": m["gguf_sha256"],
                          "commit": res.get("commit"), "licence": res.get("licence"), "files": res.get("files"),
                          "hub_version": res.get("hub_version")}


def fetch_model(spec: dict, built: dict, keep_snapshot: bool = False) -> dict:
    check_deadline(f"{spec['served_as']}'s setup")
    if spec["served_as"] in HOSTED:  # taken from the private repo before any wave, and already checked against its pin
        h = HOSTED[spec["served_as"]]
        out = {"served_as": spec["served_as"], "repo": spec["repo"], "revision": spec.get("revision"), "card": spec.get("card")}
        out.update({"gguf_source": f"pre-built, {os.environ.get('PIN_GGUF_REPO')}@{(os.environ.get('PIN_GGUF_REVISION') or '')[:12]}",
                    "gguf_file": Path(h["path"]).name, "gguf_sha256": h["sha256"], "gguf_bytes": Path(h["path"]).stat().st_size,
                    "path": str(h["path"])})
        return out
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download  # noqa: PLC0415

    rev = spec.get("revision") or hf_revision(spec["repo"])
    out = {"served_as": spec["served_as"], "repo": spec["repo"], "revision": rev, "card": spec.get("card")}
    resources(f"{spec['served_as']}: before download")
    if spec["how"] == "maker_gguf":
        listed = HfApi(token=False).list_repo_files(spec["repo"], revision=rev)  # public: never an implicit token (Q1)
        # An exact `file` when the repo's names overlap (Bolt's mmproj file contains the model file's name).
        files = [f for f in listed if f == spec["file"]] if spec.get("file") else \
            [f for f in listed if f.endswith(".gguf") and spec["pattern"] in f]
        if len(files) != 1:
            raise RuntimeError(f"{spec['repo']}: expected one GGUF for {spec.get('file') or spec['pattern']}, found {files}")
        say(f"downloading {files[0]}")
        path = Path(with_heartbeat(
            lambda: hf_hub_download(spec["repo"], files[0], revision=rev, local_dir=WORK / spec["served_as"], token=False),
            f"maker:{spec['served_as']}", download_timeout(None), watch=WORK / spec["served_as"]))
        out.update({"gguf_source": "maker", "gguf_file": files[0]})
    else:
        say(f"downloading {spec['repo']} (BF16) for conversion")
        snap_dir = WORK / (spec["served_as"] + "-bf16")
        snap = Path(with_heartbeat(
            lambda: snapshot_download(spec["repo"], revision=rev, local_dir=snap_dir, token=False,
                                      allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.py", "*.jinja"]),
            f"snapshot:{spec['served_as']}", download_timeout(None), watch=snap_dir))
        py = converter_python(spec, built)
        path = WORK / f"{spec['served_as']}-Q8_0.gguf"
        timeout = convert_timeout(_tree_bytes(snap))
        out["convert_timeout_s"] = timeout
        if spec.get("via_bf16"):
            # critic-signal §10 (R&D, 2026-10-19): the direct q8_0 path quantizes in single-threaded Python (Super-VL-120B
            # ran 0.9 GB/min and hit the old 3600 s cap at 54.8 of ~127 GB). This route writes BF16 (no quantize maths)
            # and quantizes with the pinned, multithreaded llama-quantize, the Q6_K path's own steps. Same Q8_0 level; the
            # per-tensor types are llama-quantize's, so the receipts carry the GGUF's type table.
            if "llama_quantize" not in out:
                out["llama_quantize"] = build_quantize(Path(built["src"]))
            bf16 = WORK / f"{spec['served_as']}-BF16.gguf"
            say("converting to BF16, then llama-quantize Q8_0 (via_bf16)")
            t0 = time.time()
            sh(f"{py} {built['src']}/convert_hf_to_gguf.py {snap} --outtype bf16 --outfile {bf16}", timeout=timeout)
            out["bf16_seconds"] = round(time.time() - t0)
            resources(f"{spec['served_as']}: after BF16 conversion")
            t1 = time.time()
            sh(f"{out['llama_quantize']['path']} {bf16} {path} Q8_0", timeout=timeout)
            out["quantize_seconds"] = round(time.time() - t1)
            out["bf16_sha256"] = sha256(bf16)
            bf16.unlink(missing_ok=True)
            resources(f"{spec['served_as']}: after Q8_0 quantize, BF16 removed")
            out["route"] = "bf16+llama-quantize"
        else:
            say("converting to Q8_0")
            sh(f"{py} {built['src']}/convert_hf_to_gguf.py {snap} --outtype q8_0 --outfile {path}", timeout=timeout)
        if spec.get("tokenizer_check"):
            out["tokenizer_hf"] = hf_token_ids(py, snap)
        if spec.get("brief_check") and BRIEFS.exists():
            # The pilot's own prompts, encoded by the maker's tokenizer before the snapshot goes (R&D, window 4 ruling).
            briefs = json.loads(BRIEFS.read_text(encoding="utf-8"))
            out["brief_texts"] = briefs["writer"] + briefs["critic"]
            out["brief_hf"] = hf_token_ids(py, snap, out["brief_texts"], "brief-strings-hf.json")
        # The converter's environment, so the Q8_0 hash can be reproduced (R&D on #77).
        freeze = sh(f"{py} -m pip freeze")
        gguf_file = sh(f"{py} -c \"import gguf; print(gguf.__file__)\"")
        out.update({"gguf_source": f"converted at llama.cpp {LLAMA_CPP_COMMIT[:12]} from {spec['repo']}@{rev[:12]}",
                    "gguf_file": path.name, "converter": {"pip_freeze_sha256": hashlib.sha256(freeze.encode()).hexdigest(),
                                                           "pip_freeze": freeze.splitlines(), "gguf_module": gguf_file,
                                                           "python": py, "env": spec.get("env") or "llama.cpp requirements"}})
    if spec["how"] != "maker_gguf":
        resources(f"{spec['served_as']}: after conversion")
        if keep_snapshot:  # a Q6_K may still be needed: it's made from a BF16 conversion of this snapshot
            out.update({"snapshot": str(snap), "converter_py": py})
        else:
            shutil.rmtree(snap, ignore_errors=True)
            resources(f"{spec['served_as']}: after dropping snapshot")
    say(f"hashing {path.name}")
    out.update({"gguf_sha256": sha256(path), "gguf_bytes": path.stat().st_size, "path": str(path)})
    return out


def smoke(built: dict, model: dict) -> dict:
    """One served completion: the server starts, answers, and we see whether reasoning comes back apart."""

    port = 8091
    env = {**os.environ, "LD_LIBRARY_PATH": ":".join(str(p) for p in _cuda_lib_dirs()) + ":" + os.environ.get("LD_LIBRARY_PATH", "")}
    ctx = int(os.environ.get("PIN_SMOKE_CTX") or 4096)  # window 5 serves at the pilot writer's context, to test fit
    # The server's log goes to a file, never an unread pipe (a full pipe would stall it), and its offload lines are
    # read back: peak VRAM alone can't show layers or experts left on the CPU (R&D, window 5).
    server_log = WORK / f"{model['served_as']}-server.log"
    log_fh = open(server_log, "w", encoding="utf-8")  # noqa: SIM115 - closed in the finally below
    proc = subprocess.Popen([built["path"], "-m", model["path"], "--port", str(port), "--host", "127.0.0.1", "-ngl", "999",
                             "--log-verbosity", str(LOAD_LOG_VERBOSITY),
                             "-c", str(ctx), "--parallel", "1", "--jinja", "--reasoning-format", "deepseek"], stdout=log_fh,
                            stderr=subprocess.STDOUT, text=True, env=env, start_new_session=True)
    # The model's card kwargs (rnd card-values.json, passed whole in PIN_CARDS_JSON), so a thinking model thinks as it will
    # in the pilot. Sampling stays at temperature 0 and seed 0: this reply is the reference record, not a pilot call.
    kwargs = _card_kwargs(model)
    # Peak VRAM while the model loads and answers: fit is a report, never an exit (amendment B item 4, F6).
    peak = {"mib": 0}
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            try:
                used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                                      capture_output=True, text=True, timeout=10).stdout.split(",")
                peak["mib"] = max(peak["mib"], int(used[0]))
                peak["total_mib"] = int(used[1])
            except Exception:  # noqa: BLE001 - a missed sample only lowers the reading
                pass
            stop.wait(1)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        for _ in range(180):
            if proc.poll() is not None:  # the server exited (window 5: R1 out of memory, then 15 minutes of polling)
                raise RuntimeError(f"llama-server exited ({proc.returncode}): "
                                   + server_log.read_text(encoding="utf-8", errors="replace")[-800:])
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:  # noqa: S310 - loopback
                    if r.status == 200:
                        break
            except Exception:  # noqa: BLE001 - starting up
                time.sleep(5)
        else:
            raise RuntimeError("llama-server never came up: " + server_log.read_text(encoding="utf-8", errors="replace")[-800:])
        max_tokens = int(os.environ.get("PIN_SMOKE_MAX_TOKENS") or SMOKE_MAX_TOKENS)
        req_body = {"model": model["served_as"], "temperature": 0, "seed": 0, "max_tokens": max_tokens,
                    "logprobs": True, "top_logprobs": 5,
                    "messages": [{"role": "user", "content": "In two sentences, say what a changelog is for."}]}
        if kwargs is not None:
            req_body["chat_template_kwargs"] = kwargs
        body = json.dumps(req_body).encode()
        t0 = time.time()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:  # noqa: S310 - loopback
            reply = json.loads(r.read())
        secs = time.time() - t0
        msg = reply["choices"][0]["message"]
        toks = (reply.get("usage") or {}).get("completion_tokens") or 0
        # The dataminer record (rnd abmine's format): the first check that llama-server's logprobs come back usable.
        steps = ((reply["choices"][0].get("logprobs") or {}).get("content")) or []
        record = {"type": "result", "arm": model["served_as"], "prompt": "pin-smoke-0", "reference": True,
                  "cublas": CUBLAS_VERSION, "cuda_toolkit": CUDA_TOOLKIT, "max_tokens": SMOKE_MAX_TOKENS,
                  "tokens": [s.get("token") for s in steps],
                  "top": [[[t.get("token"), t.get("logprob")] for t in (s.get("top_logprobs") or [])] for s in steps]}
        with open(WORK / "generations.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        wait_for_log(server_log)
        tok = tokenizer_check(port, model) if model.get("tokenizer_hf") else None
        # Amendment E1's thinking split: reasoning comes back apart, and no thinking tag is left in the content.
        content = msg.get("content") or ""
        tags = [tg for tg in ("<think>", "</think>", "<|channel>thought", "<channel|>") if tg in content]
        return {"ok": bool((msg.get("content") or "").strip()), "tokens_per_s": round(toks / secs, 1) if secs else None,
                "completion_tokens": toks, "finish_reason": reply["choices"][0].get("finish_reason"), "tokenizer": tok,
                "reasoning_content": bool(msg.get("reasoning_content")), "content_head": (msg.get("content") or "")[:120],
                "logprob_steps": len(steps), "top5_ok": bool(steps) and all(len(s.get("top_logprobs") or []) == 5 for s in steps),
                "ctx": ctx, "peak_vram_mib": peak["mib"], "total_vram_mib": peak.get("total_mib"), "max_tokens": max_tokens,
                "offload_lines": _offload_lines(server_log), "fit": all_on_gpu(_offload_lines(server_log)),
                "server_log_tail": log_tail(server_log),
                "slots": slots(port, ctx),
                "chat_template_kwargs": kwargs,
                "thinking_split": {"reasoning_nonempty": bool((msg.get("reasoning_content") or "").strip()), "tags_in_content": tags,
                                   "ok": bool((msg.get("reasoning_content") or "").strip()) and not tags}}
    finally:
        stop.set()
        _signal_group(proc, 15)
        proc.wait(timeout=60)
        log_fh.close()


SMOKE_MAX_TOKENS = 2048


# libllama's INFO lines ("offloaded N/M layers to GPU") map to verbosity 4 (TRACE) in common/log.cpp at 620d2225, and
# llama-server's default threshold is 3, so they were dropped: window 7 read no load line and stopped both critics.
# 4 shows them without the debug flood of -v (5).
LOAD_LOG_VERBOSITY = 4


def wait_for_log(server_log: Path, needle: str = "layers to GPU", timeout_s: int = 60) -> None:
    for _ in range(timeout_s):
        if server_log.exists() and needle in server_log.read_text(encoding="utf-8", errors="replace"):
            return
        time.sleep(1)


def log_tail(server_log: Path, n: int = 60) -> list[str]:
    text = server_log.read_text(encoding="utf-8", errors="replace") if server_log.exists() else ""
    return [ln[:240] for ln in text.splitlines() if ln.strip()][-n:]


def _offload_lines(server_log: Path) -> list[str]:
    """llama-server's lines on where the weights went: 'offloaded N/N layers to GPU', any fit or auto-ngl adjustment,
    and any tensor or expert override (CPU_Mapped buffers mean weights left on the CPU)."""

    keys = ("offloaded", "offloading", "fit", "n_gpu_layers", "override", "CPU_Mapped", "CPU_REPACK", "model buffer size")
    text = server_log.read_text(encoding="utf-8", errors="replace") if server_log.exists() else ""
    return [ln.strip()[:240] for ln in text.splitlines() if any(k in ln for k in keys)][:40]


def _card_kwargs(model: dict) -> dict | None:
    """The candidate's chat_template_kwargs from PIN_CARDS_JSON (rnd card-values.json), by its card id or served name;
    None when no cards were passed (windows 2 to 4)."""

    raw = os.environ.get("PIN_CARDS_JSON")
    if not raw:
        return None
    cid = model.get("card") or model["served_as"]
    for c in json.loads(raw).get("candidates", []):
        if c.get("candidate") == cid:
            return dict(c.get("chat_template_kwargs") or {})
    raise RuntimeError(f"no card values for {cid}")
# The tokenizer check (R&D, window 3): a fixed set of strings, encoded by the maker's tokenizer and by the GGUF
# through llama-server's /tokenize; every id must match.
TOKENIZER_STRINGS = [
    "hello", "Hello, world!", "The quick brown fox jumps over the lazy dog.", "  leading spaces", "trailing spaces   ",
    "tab\tseparated\tvalues", "line one\nline two", "line one\r\nline two", "\n\n\nthree newlines", "a" * 40,
    "12345", "3.14159", "-42", "1,000,000", "2026-10-17", "10:30 AM", "$450.00", "50%", "#1", "v1.2.3",
    "(parentheses) [brackets] {braces}", "quotes \"double\" and 'single'", "semi; colon: comma, period.", "a/b\\c|d",
    "email@example.com", "https://example.com/path?q=1&r=2", "snake_case and camelCase", "UPPER lower Mixed",
    "don't won't can't", "e.g. i.e. etc.", "naïve café résumé", "Straße München", "Ελληνικά", "Русский текст",
    "日本語のテキスト", "中文文本", "한국어 텍스트", "العربية", "हिन्दी", "emoji 🙂👍", "math: x² + y² = z²",
    "arrows → ← ↑", "def f(x):\n    return x + 1", "{\"key\": [1, 2, 3]}", "<b>bold</b>", "SELECT * FROM t;",
    "Line 3 says the desk opens at 9.", "Question: Which slot does Ama have?", "Answer: the east room", "",
]


def converter_python(spec: dict, built: dict) -> str:
    """The python a model converts with. Default: this interpreter with llama.cpp's own requirements. A spec with
    `env` gets its own venv, llama.cpp's requirements first, then the pinned overrides exactly (`==`)."""

    env = spec.get("env")
    if not env:
        sh(f"{sys.executable} -m pip install -q -r {built['src']}/requirements/requirements-convert_hf_to_gguf.txt", timeout=1200)
        return sys.executable
    venv = WORK / f"env-{spec['served_as']}"
    if not (venv / "bin" / "python").exists():
        # The Colab image's python has no ensurepip, so `-m venv` fails (window 3, 2026-10-17); virtualenv brings its
        # own pip. Pinned, so the env is rebuilt the same way each window.
        sh(f"{sys.executable} -m pip install -q 'virtualenv==20.29.3'", timeout=600)
        sh(f"{sys.executable} -m virtualenv -q {venv}", timeout=300)
    py = str(venv / "bin" / "python")
    sh(f"{py} -m pip install -q -r {built['src']}/requirements/requirements-convert_hf_to_gguf.txt", timeout=1800)
    pins = " ".join(f"'{k}=={v}'" for k, v in env.items())
    sh(f"{py} -m pip install -q {pins}", timeout=1200)
    for k, v in env.items():
        got = sh(f"{py} -c \"import importlib.metadata as m; print(m.version('{k}'))\"")
        if got != v:
            raise RuntimeError(f"{k} is {got} in the {spec['served_as']} env, not {v}")
    return py


def hf_token_ids(py: str, snap: Path, texts: list | None = None, name: str = "tokenizer-strings.json") -> list:
    """The maker's tokenizer (from the snapshot, in the converter's env) on `texts` (TOKENIZER_STRINGS by default), no
    special tokens."""

    strings = WORK / name
    strings.write_text(json.dumps(TOKENIZER_STRINGS if texts is None else texts, ensure_ascii=False), encoding="utf-8")
    code = ("import json,sys; from transformers import AutoTokenizer; "
            f"t = AutoTokenizer.from_pretrained('{snap}'); "
            f"s = json.load(open('{strings}', encoding='utf-8')); "
            "print(json.dumps([t.encode(x, add_special_tokens=False) for x in s]))")
    return json.loads(sh(f"{py} -c \"{code}\"", timeout=600))


def tokenizer_check(port: int, model: dict) -> dict:
    mism = []
    for i, (text, want) in enumerate(zip(TOKENIZER_STRINGS, model["tokenizer_hf"])):
        req = urllib.request.Request(f"http://127.0.0.1:{port}/tokenize", data=json.dumps({"content": text, "add_special": False}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 - loopback
            got = json.loads(r.read())["tokens"]
        if got != want:
            mism.append({"i": i, "text": text[:40], "hf": want[:12], "gguf": got[:12]})
    return {"strings": len(TOKENIZER_STRINGS), "match": not mism, "mismatches": mism[:10]}


def pin_libs(binary: str) -> list[dict]:
    """The CUDA shared libraries the binary loads, copied and hashed into /content/pin/lib (R&D on #77): writers.json's
    llama_server pins the binary plus this set, because a different cuBLAS can move outputs."""

    lib = WORK / "lib"
    lib.mkdir(exist_ok=True)
    env = {**os.environ, "LD_LIBRARY_PATH": ":".join(str(p) for p in _cuda_lib_dirs()) + ":" + os.environ.get("LD_LIBRARY_PATH", "")}
    out = []
    for line in sh(f"ldd {binary}", env=env).splitlines():
        if "=>" not in line:
            continue
        name, _, rest = line.strip().partition(" => ")
        path = rest.split(" (")[0].strip()
        if not path.startswith("/") or not any(k in name for k in ("cuda", "cublas", "cudart", "nvrtc", "nvJitLink")):
            continue
        if name.startswith("libcuda.so"):  # the host driver: the machine line's driver, never a pin (R&D, window 2)
            continue
        dest = lib / name
        shutil.copy(path, dest)
        out.append({"name": name, "source": path, "sha256": sha256(dest), "bytes": dest.stat().st_size})
    if not out:
        raise RuntimeError("ldd lists no CUDA library: the binary isn't linked as expected")
    sh(f"tar -C {WORK} -cf {WORK}/lib.tar lib")
    return out


def _cuda_lib_dirs() -> list:
    import site

    dirs = []
    for root in site.getsitepackages():
        dirs += [p for p in Path(root, "nvidia").glob("*/lib") if p.is_dir()]
    return dirs


BRIEFS = CONTENT / "brief-strings.json"  # uploaded by the window script: 20 writer briefs, 20 critic prompts
CRITIC_CTX = 32768
CRITIC_PORT = 8092


def start_server(built: dict, model: dict, ctx: int, port: int):
    """llama-server for one model, its log in a file; waits until healthy and fails at once if the server exits."""

    env = {**os.environ, "LD_LIBRARY_PATH": ":".join(str(p) for p in _cuda_lib_dirs()) + ":" + os.environ.get("LD_LIBRARY_PATH", "")}
    server_log = WORK / f"{model['served_as']}-critic-server.log"
    log_fh = open(server_log, "w", encoding="utf-8")  # noqa: SIM115 - closed by stop_server
    # One slot: llama-server's default runs several, and each gets ctx / n_slots, so a long thinking reply would
    # truncate at a quarter of the context (R&D on the critic window).
    proc = subprocess.Popen([built["path"], "-m", model["path"], "--port", str(port), "--host", "127.0.0.1", "-ngl", "999",
                             "--log-verbosity", str(LOAD_LOG_VERBOSITY),
                             "-c", str(ctx), "--parallel", "1", "--jinja", "--reasoning-format", "deepseek"], stdout=log_fh,
                            stderr=subprocess.STDOUT, text=True, env=env, start_new_session=True)
    for _ in range(180):
        if proc.poll() is not None:
            log_fh.close()
            raise RuntimeError(f"llama-server exited ({proc.returncode}): " + server_log.read_text(encoding="utf-8", errors="replace")[-800:])
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:  # noqa: S310 - loopback
                if r.status == 200:
                    return proc, log_fh, server_log
        except Exception:  # noqa: BLE001 - starting up
            time.sleep(5)
    _signal_group(proc, 15)
    log_fh.close()
    raise RuntimeError("llama-server never came up: " + server_log.read_text(encoding="utf-8", errors="replace")[-800:])


def stop_server(proc, log_fh) -> None:
    _signal_group(proc, 15)
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        _signal_group(proc, None)
        proc.wait(timeout=30)
    finally:
        log_fh.close()


def all_on_gpu(lines: list[str]) -> dict:
    """Whether llama-server put every layer on the GPU, read from its 'offloaded N/M layers to GPU' line."""

    import re

    for ln in lines:
        m = re.search(r"offloaded (\d+)/(\d+) layers to GPU", ln)
        if m:
            return {"offloaded": int(m.group(1)), "layers": int(m.group(2)), "ok": m.group(1) == m.group(2)}
    return {"offloaded": None, "layers": None, "ok": False, "why": "no 'offloaded N/M layers to GPU' line in the server log"}


def slots(port: int, want_ctx: int) -> dict:
    """The server's slot count and per-slot context, from /props; a per-slot context under the run's is refused."""

    with urllib.request.urlopen(f"http://127.0.0.1:{port}/props", timeout=30) as r:  # noqa: S310 - loopback
        props = json.loads(r.read())
    per_slot = (props.get("default_generation_settings") or {}).get("n_ctx")
    return {"total_slots": props.get("total_slots"), "n_ctx_per_slot": per_slot,
            "ok": isinstance(per_slot, int) and per_slot >= want_ctx}


def tokenize_all(port: int, texts: list, want: list) -> dict:
    mism = []
    for i, (text, ids) in enumerate(zip(texts, want)):
        req = urllib.request.Request(f"http://127.0.0.1:{port}/tokenize", data=json.dumps({"content": text, "add_special": False}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310 - loopback
            got = json.loads(r.read())["tokens"]
        if got != ids:
            first = next((k for k, (a, b) in enumerate(zip(got, ids)) if a != b), min(len(got), len(ids)))
            mism.append({"i": i, "first_difference_at": first, "hf_len": len(ids), "gguf_len": len(got),
                         "hf": ids[max(0, first - 3):first + 5], "gguf": got[max(0, first - 3):first + 5]})
    return {"strings": len(texts), "match": not mism, "mismatches": mism[:10]}


def reference(port: int, model: dict) -> dict:
    """The replay fingerprint at window 4's exact request (no kwargs, temperature 0, seed 0, 2048 tokens)."""

    body = json.dumps({"model": model["served_as"], "temperature": 0, "seed": 0, "max_tokens": SMOKE_MAX_TOKENS,
                       "logprobs": True, "top_logprobs": 5,
                       "messages": [{"role": "user", "content": "In two sentences, say what a changelog is for."}]}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:  # noqa: S310 - loopback
        reply = json.loads(r.read())
    steps = ((reply["choices"][0].get("logprobs") or {}).get("content")) or []
    record = {"type": "result", "arm": model["served_as"], "prompt": "pin-smoke-0", "reference": True, "rerecorded": True,
              "cublas": CUBLAS_VERSION, "cuda_toolkit": CUDA_TOOLKIT, "max_tokens": SMOKE_MAX_TOKENS,
              "tokens": [s.get("token") for s in steps],
              "top": [[[t.get("token"), t.get("logprob")] for t in (s.get("top_logprobs") or [])] for s in steps]}
    with open(WORK / "generations.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return {"steps": len(steps), "finish_reason": reply["choices"][0].get("finish_reason"),
            "top5_ok": bool(steps) and all(len(s.get("top_logprobs") or []) == 5 for s in steps)}


def role_set_env(commit: str) -> dict:
    """role-set at a pinned commit, in its own Python 3.12 (role-set needs <3.13; Colab's python is 3.13)."""

    src, venv = CONTENT / "role-set", CONTENT / "rs-venv"
    if not (src / ".git").exists():
        sh(f"git clone -q https://github.com/mcp-tool-shop-org/role-set {src} && git -C {src} checkout -q {commit}", timeout=600)
    uv = shutil.which("uv")
    if not uv:
        sh(f"{sys.executable} -m pip install -q uv==0.9.5", timeout=600)
        uv = f"{sys.executable} -m uv"
    if not (venv / "bin" / "python").exists():
        sh(f"{uv} venv -q -p 3.12 {venv}", timeout=900)
        sh(f"{uv} pip install -q --python {venv}/bin/python -e {src}", timeout=1800)
    head = sh(f"git -C {src} rev-parse HEAD")
    if head != commit:
        raise RuntimeError(f"role-set is at {head[:12]}, not {commit[:12]}")
    return {"commit": head, "src": str(src), "role_set": str(venv / "bin" / "role-set"),
            "python": sh(f"{venv}/bin/python -c \"import sys; print(sys.version.split()[0])\"")}


def critic_control(rs: dict, model: dict, machine_file: Path, seed: int, earlier: Path | None) -> dict:
    out = WORK / f"critic-control-{model['served_as']}-s{seed}.json"
    cmd = [rs["role_set"], "critic-control", "--items", str(CONTENT / "known-answer-v1.jsonl"), "--planted", "10", "--seed", "0",
           "--critic-endpoint", f"http://127.0.0.1:{CRITIC_PORT}/v1", "--critic-model", model["served_as"],
           "--critic-gguf-sha256", model["gguf_sha256"], "--llama-server-sha256", server_sha256(),
           "--machine", str(machine_file), "--writers", f"{rs['src']}/writers.json", "--cards", str(CONTENT / "card-values.json"),
           "--critic-seed", str(seed), "--out", str(out)]
    if earlier is not None:
        cmd += ["--agreement-with", str(earlier)]
    t0 = time.time()
    run = run_proc(cmd, timeout=4 * 3600, cwd=rs["src"])
    got = {"seed": seed, "exit": run.returncode, "seconds": round(time.time() - t0), "stdout": run.stdout[-1500:],
           "stderr": run.stderr[-1500:], "out": str(out)}
    if out.exists():
        got["result"] = json.loads(out.read_text(encoding="utf-8"))
    return got


def critic_window(built: dict, specs: list, result: dict) -> None:
    """PIN_ROLE=critic (R&D, 2026-10-17): each critic reconverted and checked against its pin, its offload read, Muse's
    pilot-prompt tokenizer check and replay fingerprint, then role-set's critic-control at the card values. Any check
    that fails stops that model before critic-control; the next model still runs."""

    rs = result["role_set"] = role_set_env(os.environ["PIN_ROLESET_COMMIT"])
    result["server_pin"] = server_pin_entry(server_sha256(), Path(rs["src"]) / "writers.json")
    if isinstance(result.get("llama_server"), dict):
        result["llama_server"].update(result["server_pin"])
    machine_file = WORK / "machine.json"
    machine_file.write_text(json.dumps(result["machine"]), encoding="utf-8")
    seeds = {"muse-glimmer-30b": [0, 1], "nemotron-3.5-lightning-30b-a3b": [0]}
    result["models"] = []
    for spec in specs:
        rec: dict = {"served_as": spec["served_as"]}
        result["models"].append(rec)
        try:
            m = fetch_model({**spec, "brief_check": spec["served_as"] == "muse-glimmer-30b"}, built)
            rec.update({k: v for k, v in m.items() if k not in ("path", "brief_texts", "brief_hf", "tokenizer_hf")})
            if spec.get("expect_sha256") and m["gguf_sha256"] != spec["expect_sha256"]:
                rec["stopped"] = f"GGUF {m['gguf_sha256'][:12]} doesn't reproduce the pin {spec['expect_sha256'][:12]}"
                continue
            rec["reproduced"] = True
            proc, log_fh, server_log = start_server(built, m, CRITIC_CTX, CRITIC_PORT)
            try:
                wait_for_log(server_log)
                rec["offload_lines"] = _offload_lines(server_log)
                rec["server_log_tail"] = log_tail(server_log)
                rec["fit"] = all_on_gpu(rec["offload_lines"])
                if not rec["fit"]["ok"]:
                    rec["stopped"] = "not every layer is on the GPU"
                    continue
                rec["slots"] = slots(CRITIC_PORT, CRITIC_CTX)
                if not rec["slots"]["ok"]:
                    rec["stopped"] = f"per-slot context {rec['slots']['n_ctx_per_slot']} is under {CRITIC_CTX}"
                    continue
                if m.get("tokenizer_hf"):
                    rec["tokenizer"] = tokenizer_check(CRITIC_PORT, m)
                if m.get("brief_hf"):
                    rec["brief_check"] = tokenize_all(CRITIC_PORT, m["brief_texts"], m["brief_hf"])
                    rec["reference"] = reference(CRITIC_PORT, m)
                    if not rec["brief_check"]["match"]:
                        rec["stopped"] = "a pilot prompt tokenizes differently (R&D: any mismatch blocks)"
                        continue
                runs, earlier = [], None
                for s in seeds.get(spec["served_as"], [0]):
                    say(f"critic-control {spec['served_as']} seed {s}")
                    r = critic_control(rs, m, machine_file, s, earlier)
                    runs.append(r)
                    earlier = Path(r["out"]) if "result" in r else None
                rec["critic_control"] = runs
            finally:
                stop_server(proc, log_fh)
        except DeadlineReached:
            raise
        except Exception as e:  # noqa: BLE001 - recorded with its model; the next model still runs
            rec["error"] = f"{type(e).__name__}: {e}"
        finally:
            g = Path(WORK / f"{spec['served_as']}-Q8_0.gguf")
            if g.exists():
                g.unlink(missing_ok=True)
                resources(f"{spec['served_as']}: after unlinking Q8_0")
        say(f"{spec['served_as']}: {json.dumps({k: rec.get(k) for k in ('reproduced', 'fit', 'brief_check', 'stopped', 'error')})[:600]}")
    result["model_errors"] = sum(1 for r in result["models"] if r.get("stopped") or r.get("error"))


def build_quantize(src: Path) -> dict:
    """llama-quantize from the pinned tree as its own CPU-only target, so the pinned llama-server is untouched."""

    build = src / "build-quantize"
    sh(f"cmake -S {src} -B {build} -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release", timeout=900)
    sh(f"cmake --build {build} --target llama-quantize -j {os.cpu_count() or 4}", timeout=1800)
    path = build / "bin" / "llama-quantize"
    return {"path": str(path), "sha256": sha256(path), "commit": sh(f"git -C {src} rev-parse HEAD"),
            "cmake": sh("cmake --version").splitlines()[0], "cc": sh("cc --version").splitlines()[0]}


def q6k_window(built: dict, specs: list, result: dict) -> None:
    """PIN_ROLE=q6k: each model's Q8_0 rebuilt and checked against its pin (Gemma's is made and recorded), quantized to
    Q6_K with the pinned llama-quantize, hashed, and served at 32768 with its fit and slots read. A labelled arm."""

    result["models"] = []
    for spec in specs:
        rec: dict = {"served_as": spec["served_as"], "quant": "Q6_K"}
        result["models"].append(rec)
        try:
            m = fetch_model(spec, built, keep_snapshot=True)
            rec["q8_0"] = {"gguf_sha256": m["gguf_sha256"], "gguf_bytes": m["gguf_bytes"], "revision": m["revision"],
                           "converter": m.get("converter"), "pinned": spec.get("expect_sha256")}
            if spec.get("expect_sha256") and m["gguf_sha256"] != spec["expect_sha256"]:
                rec["stopped"] = f"Q8_0 {m['gguf_sha256'][:12]} doesn't reproduce the pin {spec['expect_sha256'][:12]}"
                continue
            m6 = to_q6k(built, spec, m, result)
            rec.update({k: v for k, v in m6.items() if k not in ("path", "tokenizer_hf", "converter_py")})
            if spec.get("expect_q6k_sha256"):
                rec["q6k_reproduced"] = m6["gguf_sha256"] == spec["expect_q6k_sha256"]
                if not rec["q6k_reproduced"]:
                    rec["stopped"] = f"Q6_K {m6['gguf_sha256'][:12]} doesn't reproduce {spec['expect_q6k_sha256'][:12]}"
                    continue
            m6["smoke"] = smoke(built, m6)
            rec["smoke"] = m6["smoke"]
            if not (rec["smoke"].get("fit") or {}).get("ok") or not (rec["smoke"].get("slots") or {}).get("ok"):
                rec["stopped"] = "the Q6_K doesn't fit the card at 32768 with every layer on the GPU and one full slot"
        except DeadlineReached:
            raise
        except Exception as e:  # noqa: BLE001 - recorded with its model; the next model still runs
            rec["error"] = f"{type(e).__name__}: {e}"
        finally:
            for f in (WORK / f"{spec['served_as']}-Q8_0.gguf", WORK / f"{spec['served_as']}-Q6_K.gguf"):
                if f.exists():
                    f.unlink(missing_ok=True)
                    resources(f"{spec['served_as']}: after unlinking {f.name}")
        say(f"{spec['served_as']} Q6_K: {json.dumps({k: rec.get(k) for k in ('gguf_sha256', 'stopped', 'error')})}")
    result["model_errors"] = sum(1 for r in result["models"] if r.get("stopped") or r.get("error"))


PILOT_PORT = 8093
PILOT_PARTITION = "trial:j1-wave1"
PILOT_WAVES = 6  # write, critique, rewrite, critique, rewrite, critique: the attempt budget (3) ends here, on a critique
# The w0 screen (R&D, 2026-10-17: the pilot as designed costs 147-175 units): PIN_PILOT_WAVES=2 runs the first write and
# its critique, on the frozen slot lists in PIN_PILOT_SLOT_IDS and PIN_PILOT_TWIN_IDS.
PILOT_WAVES = int(os.environ.get("PIN_PILOT_WAVES") or PILOT_WAVES)


_STUB_WRITER = None


def _repo_root() -> Path:
    """This checkout's root when run as a file; /content/role-set on the VM, where `colab exec -f` runs the script in
    the kernel and `__file__` isn't defined (stub run 12s2, 2026-10-17: NameError before any wave)."""

    f = globals().get("__file__")
    return Path(f).resolve().parents[2] if f else CONTENT / "role-set"


class _SlowStub:
    """A stub writer or critic that takes PIN_STUB_CALL_S per call, so a stub rung has waves long enough to cut at a
    forced deadline mid-wave (R&D on #91). Stub mode only; everything else passes through."""

    def __init__(self, inner, seconds: float):
        self.inner, self.seconds = inner, seconds

    def __call__(self, *a, **k):
        time.sleep(self.seconds)
        return self.inner(*a, **k)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def _slow_stub(inner):
    s = float(os.environ.get("PIN_STUB_CALL_S") or 0)
    return _SlowStub(inner, s) if s > 0 else inner


def _start_stub(name: str, port: int, *, ollama_model: str | None = None, ollama_digest: str | None = None,
                ollama_version: str = "0.40.4"):
    """StubLlama on PILOT_PORT, answering with role-set's MockWriter and MockCritic."""

    global _STUB_WRITER
    for p in (CONTENT / "role-set" / "src", _repo_root() / "src", CONTENT / "role-set" / "tests", _repo_root() / "tests"):
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import stub_llama as stub_mod  # noqa: PLC0415
    from role_set.mock import MockCritic, MockWriter  # noqa: PLC0415

    if _STUB_WRITER is None:
        _STUB_WRITER = MockWriter()
    w = _STUB_WRITER
    c = MockCritic(answers=w.answers)
    if ollama_model is not None and "ollama_model" in stub_mod.StubLlama.__init__.__code__.co_varnames:
        return stub_mod.StubLlama(
            name,
            name,
            writer=_slow_stub(w),
            critic=_slow_stub(c),
            port=port,
            ollama_model=ollama_model,
            ollama_digest=ollama_digest,
            ollama_version=ollama_version,
        )
    if ollama_model is not None:
        raise ImportError("StubLlama has no ollama_model support")
    if "port" in stub_mod.StubLlama.__init__.__code__.co_varnames:
        return stub_mod.StubLlama(name, name, writer=_slow_stub(w), critic=_slow_stub(c), port=port)
    # Fallback when /content/role-set is checked out at an earlier commit before StubLlama gained `port`.
    orig_srv = stub_mod.ThreadingHTTPServer

    class _BoundServer(orig_srv):
        allow_reuse_address = True

        def __init__(self, addr, handler):
            super().__init__((addr[0], port), handler)

    stub_mod.ThreadingHTTPServer = _BoundServer
    try:
        s = stub_mod.StubLlama(name, "__critic__", critic=c)
    finally:
        stub_mod.ThreadingHTTPServer = orig_srv
    state = {"apply": False}

    def _route(msgs, seed=0):
        sys_p = msgs[0].get("content") or "" if msgs else ""
        usr_p = msgs[-1].get("content") or "" if msgs else ""
        if "## The trait lesson\n" in usr_p:
            state["apply"] = "# 4." in usr_p or "## 4." in usr_p or "Lesson T04" in usr_p
        elif "Lesson T04," in usr_p:
            state["apply"] = True
        elif "Lesson T02," in usr_p:
            state["apply"] = False
        if "training items" in sys_p or "training item" in sys_p or "everyday material" in sys_p:
            return w(msgs, seed=seed, slot_id="T04-AP" if state["apply"] else None)
        return c(msgs, seed=seed)

    s.writer = _route
    return s


class Served:
    """One model's GGUF on disk; each start re-checks its sha256 against the pin, its offload and its slots."""

    def __init__(self, built: dict, model: dict, pinned: str):
        self.built, self.model, self.pinned = built, model, pinned
        self.proc = self.log_fh = self.stub = None
        self.starts: list = []

    def __enter__(self):
        if stub_mode():
            t0 = time.time()
            self.stub = _start_stub(self.model["served_as"], PILOT_PORT)
            fit = {"offloaded": 1, "layers": 1, "ok": True, "stub": True}
            sl = {"total_slots": 1, "n_ctx_per_slot": CRITIC_CTX, "ok": True, "stub": True}
            self.starts.append({"seconds": round(time.time() - t0), "fit": fit, "slots": sl, "stub": True})
            return self
        got = sha256(Path(self.model["path"]))
        if got != self.pinned:
            raise RuntimeError(f"{self.model['served_as']}: GGUF {got[:12]} isn't the pin {self.pinned[:12]}")
        t0 = time.time()
        self.proc, self.log_fh, log = start_server(self.built, self.model, CRITIC_CTX, PILOT_PORT)
        self.log = log
        wait_for_log(log)
        fit, sl = all_on_gpu(_offload_lines(log)), slots(PILOT_PORT, CRITIC_CTX)
        self.starts.append({"seconds": round(time.time() - t0), "fit": fit, "slots": sl})
        if not fit["ok"] or not sl["ok"]:
            stop_server(self.proc, self.log_fh)
            raise RuntimeError(f"{self.model['served_as']}: fit {fit} slots {sl}; the arm stops")
        return self

    def __exit__(self, *exc):
        if self.stub is not None:
            self.stub.close()
            self.stub = None
            return
        stop_server(self.proc, self.log_fh)


def install_rnd(commit: str) -> dict:
    """rnd's logging proxy (stdlib only), from a clone at a pinned commit on sys.path, imported here so a missing module
    fails the window in its first minute. A pip install isn't used: rnd's package list at 9667a32 leaves out
    mcptoolshop_rnd.localbuilders (window 10c failed on that after both conversions)."""

    src = CONTENT / "rnd"
    if not (src / ".git").exists():
        sh(f"git clone -q https://github.com/mcp-tool-shop-org/rnd {src} && git -C {src} checkout -q {commit}", timeout=600)
    head = sh(f"git -C {src} rev-parse HEAD")
    if head != commit:
        raise RuntimeError(f"rnd is at {head[:12]}, not {commit[:12]}")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from mcptoolshop_rnd.localbuilders.proxy import LoggingProxyServer, ProxyState  # noqa: F401, PLC0415

    return {"commit": head, "src": str(src), "proxy_import": "ok"}


def j1_card(model: dict, cards_path: Path, out_dir: Path) -> tuple[Path, str, dict]:
    """The model's card-values entry as a one-candidate file for the proxy, values unchanged. reattach_thinking is
    J2's multi-turn transform and doesn't apply in J1 (R&D, 2026-10-17), so it is the one thing removed."""

    cid = model.get("card") or model["served_as"]
    entry = next(c for c in json.loads(cards_path.read_text(encoding="utf-8"))["candidates"] if c["candidate"] == cid)
    j1 = {**entry, "transforms": [x for x in entry.get("transforms") or [] if x != "reattach_thinking"]}
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{cid}.json"
    path.write_text(json.dumps({"candidates": [j1]}, ensure_ascii=False, indent=1), encoding="utf-8")
    return path, cid, {"candidate": cid, "removed": sorted(set(entry.get("transforms") or []) - set(j1["transforms"])),
                       "sha256": sha256(path)}


def generations_summary(path: Path) -> dict:
    """What the dataminer saw in one wave: calls, truncations, thinking and output tokens, call wall time."""

    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.exists() else []
    res = [r for r in rows if r.get("kind") == "result"]
    return {"calls": len(res), "refused": sum(1 for r in rows if r.get("kind") != "result"),
            "truncated": sum(1 for r in res if r.get("truncated")),
            "thinking_tokens": sum(r.get("thinking_tokens") or 0 for r in res),
            "output_tokens": sum(r.get("output_tokens") or 0 for r in res),
            "wall_s": round(sum(r.get("wall_time_s") or 0 for r in res), 1),
            "sha256": sha256(path) if path.exists() else None}


def write_wave(rs: dict, common: list, wave: str, out: Path, *, writer: dict | None = None, critic: dict | None = None,
               drafts: Path | None = None, extra: list | None = None, proxy: dict | None = None,
               reuse_calls: Path | None = None) -> dict:
    """One role-set wave. With `proxy` ({config, candidate}), rnd's logging proxy sits between role-set and llama-server
    and writes this wave's generation records beside the wave directory."""

    gen = out.parent / f"generations-{out.name}-{(writer or critic or {}).get('served_as', 'model')}.jsonl"
    px = None
    endpoint = f"http://127.0.0.1:{PILOT_PORT}/v1"
    if proxy is not None:
        from mcptoolshop_rnd.localbuilders.proxy import LoggingProxyServer, ProxyState  # noqa: PLC0415

        state = ProxyState(upstream_url=f"http://127.0.0.1:{PILOT_PORT}", config=proxy["config"], candidate=proxy["candidate"],
                           # J1 has no per-task caps. The wall is also the proxy's upstream timeout, so it stays a
                           # finite 24 h (10**9 s overflows a C timeval).
                           generations_path=gen, max_calls=10**6, wall_limit_s=86400)
        px = LoggingProxyServer(state).start()
        endpoint = px.openai_base_url
    check_deadline(f"the {wave} wave at {out}")
    cmd = [rs["role_set"], "write", *common, "--wave", wave, "--out", str(out)]
    if wave in ("critique", "inline"):
        cmd += ["--critique-scope", critique_scope()]
    # Every finished call is journaled as it returns and streamed to the rig log, so a cut-off wave resumes at the call
    # (R&D, 2026-10-17); a resume replays the cut wave's journal with --reuse-calls.
    arm, label = out.parent.name, f"{out.parent.parent.name}-{out.name}"
    journal = out.parent / f"calls-{out.name}-{(writer or critic or {}).get('served_as', 'model')}.jsonl"
    cmd += ["--call-journal", str(journal)]
    if reuse_calls is not None:
        cmd += ["--reuse-calls", str(reuse_calls)]
    stop = threading.Event()
    tail = threading.Thread(target=stream_calls, args=(journal, arm, label, stop), daemon=True)
    tail.start()
    if writer is not None:
        cmd += ["--endpoint", endpoint, "--model", writer["served_as"], "--writer-gguf-sha256", writer["gguf_sha256"]]
    if critic is not None:
        cmd += ["--critic-endpoint", endpoint, "--critic-model", critic["served_as"], "--critic-gguf-sha256", critic["gguf_sha256"]]
    if drafts is not None:
        cmd += ["--drafts", str(drafts)]
    cmd += extra or []
    t0 = time.time()
    left = deadline_left()
    try:
        try:
            run = run_proc(cmd, timeout=6 * 3600 if left is None else min(6 * 3600, max(left, 1)), cwd=rs["src"],
                           term_first=left is not None)
        except subprocess.TimeoutExpired:
            if left is None or left >= 6 * 3600:
                raise
            global CUT_AT
            CUT_AT = {"wave": wave, "out": str(out), "arm": arm, "label": label, "journal": str(journal),
                      "seconds": round(time.time() - t0), "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            say(f"CUT AT THE DEADLINE {arm} {label} after {CUT_AT['seconds']} s; the finished calls are in {journal.name}")
            raise DeadlineReached(f"the {wave} wave at {out} was cut at the window's deadline") from None
    finally:
        stop.set()
        tail.join(timeout=60)
        if px is not None:
            px.close()
    got = {"wave": wave, "out": str(out), "exit": run.returncode, "seconds": round(time.time() - t0), "stderr": run.stderr[-1500:]}
    if proxy is not None:
        got["generations"] = {"path": str(gen), **generations_summary(gen)}
    try:
        got["result"] = json.loads(run.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        got["stdout"] = run.stdout[-1500:]
    # A replay is exact at concurrency 1, role-set's default, which this window never changes; journaled calls left
    # unused mean the resumed wave asked for something the cut one didn't, so its drafts aren't the cut run's.
    unused = ((got.get("result") or {}).get("calls") or {}).get("unused_from_reuse") or 0
    if reuse_calls is not None and unused:
        got["replay_warning"] = f"{unused} journaled calls went unused: the resumed wave diverged from the cut one"
        say(f"WARNING {arm} {label}: {got['replay_warning']}")
    return got


def stream_calls(journal: Path, arm: str, label: str, stop: threading.Event) -> None:
    """Each new line of a wave's call journal, printed as `CALL_CHECKPOINT <arm> <label> <base64 gzip line>` as soon as
    it is written; `checkpoints.py extract` turns them back into the journal. A torn last line waits for its newline."""

    import base64  # noqa: PLC0415
    import gzip  # noqa: PLC0415

    pos = 0
    while True:
        last = stop.is_set()
        if journal.exists():
            with journal.open("rb") as f:
                f.seek(pos)
                data = f.read()
            end = data.rfind(b"\n") + 1
            for line in data[:end].splitlines():
                print(f"CALL_CHECKPOINT {arm} {label} {base64.b64encode(gzip.compress(line)).decode()}", flush=True)
            pos += end
        if last:
            return
        stop.wait(5)


def run_dir(out: Path) -> Path:
    """Where role-set wrote a wave: the --out directory itself, or its mock/ or rehearsal/ subdirectory."""

    for d in (out, out / "mock", out / "rehearsal"):
        if (d / "wave-drafts.jsonl").exists():
            return d
    return out


def checkpoint(arm: str, wave_name: str, drafts: Path) -> None:
    """The wave's drafts, printed as one line, so they reach the rig's log as soon as the wave ends (R&D, 2026-10-17:
    a dropped VM loses minutes, not the window). `checkpoints.py extract` turns the lines back into files."""

    import base64  # noqa: PLC0415
    import gzip  # noqa: PLC0415

    if drafts.exists():
        blob = base64.b64encode(gzip.compress(drafts.read_bytes())).decode()
        print(f"WAVE_CHECKPOINT {arm} {wave_name} {sha256(drafts)} {blob}", flush=True)


RESUME_TAR = CONTENT / "resume.tar"


def unpack_resume(tar: Path | None = None) -> Path | None:
    """A resume's drafts arrive as one resume.tar (R&D, from Argon's review: uploads land flat in /content, so drafts and
    call journals with the same <arm>.jsonl name would overwrite each other). It unpacks to /content/resume; set
    PIN_RESUME_DIR to /content/resume/<the wave's directory>."""

    tar = CONTENT / "resume.tar" if tar is None else tar
    if not tar.exists():
        return None
    import tarfile  # noqa: PLC0415

    dest = tar.parent / tar.name.removesuffix(".tar")  # /content/resume, or /content/prior for prior.tar
    dest.mkdir(exist_ok=True)
    with tarfile.open(tar) as tf:
        tf.extractall(dest, filter="data")
    return dest


def is_stub_drafts(data: bytes) -> bool:
    """Whether a wave-drafts.jsonl header came from a PIN_STUB=1 or --mock run."""

    first = data.splitlines()[0] if data else b""
    if not first:
        return False
    h = json.loads(first)
    if h.get("stub") or (h.get("batch_pins") or {}).get("mock"):
        return True
    if ((h.get("writer_host") or {}).get("machine") or {}).get("stub"):
        return True
    return any(((w.get("pins") or {}).get("machine") or {}).get("stub") for w in (h.get("waves") or []))


def resume_from() -> tuple[int, dict]:
    """PIN_RESUME_WAVE=<i> and PIN_RESUME_DIR=<dir of <arm>.jsonl>: start a batch after wave i, from those drafts. Each
    drafts file needs its seal, <arm>.jsonl.sha256, beside it (checkpoints.py extract writes both)."""

    i = os.environ.get("PIN_RESUME_WAVE")
    if not i:
        return 0, {}
    d = Path(os.environ["PIN_RESUME_DIR"])
    if not d.exists() and str(d).startswith("/content/"):
        d = CONTENT / str(d).removeprefix("/content/")
    found = {f.stem: f for f in d.glob("*.jsonl")}
    missing = [n for n, f in found.items() if not f.with_name(f.name + ".sha256").exists()]
    if not found or missing:
        raise RuntimeError(f"PIN_RESUME_DIR {d}: {'no drafts' if not found else 'no seal for ' + ', '.join(missing)}")
    if not stub_mode():
        stub_arms = [n for n, f in sorted(found.items()) if is_stub_drafts(f.read_bytes())]
        if stub_arms:
            raise RuntimeError(f"REFUSED: PIN_RESUME_DIR {d} has STUB drafts ({', '.join(stub_arms)}); mock drafts can never seed a real resume")
    return int(i) + 1, found


def resume_priors() -> dict:
    """PIN_RESUME_PRIOR_DIR=<dir of <arm>.jsonl>: for an arm whose resumed drafts lost the writer's card pins (an old
    header through a critique before the fix), the write wave's sealed drafts that critique read. role-set checks the
    chain (waves.chain_card_pins); this only finds the files. They arrive as prior.tar, so resume.tar stays unchanged."""

    d = os.environ.get("PIN_RESUME_PRIOR_DIR")
    if not d:
        return {}
    p = Path(d)
    if not p.exists() and str(p).startswith("/content/"):
        p = CONTENT / str(p).removeprefix("/content/")
    found = {f.stem: f for f in p.glob("*.jsonl")}
    missing = [n for n, f in found.items() if not f.with_name(f.name + ".sha256").exists()]
    if not found or missing:
        raise RuntimeError(f"PIN_RESUME_PRIOR_DIR {p}: {'no drafts' if not found else 'no seal for ' + ', '.join(missing)}")
    return found


def resume_calls() -> dict:
    """PIN_RESUME_CALLS_DIR=<dir of <arm>.jsonl call journals>: the wave a dropped window was in when it stopped. The
    first wave this window runs replays each arm's journal and asks the model only for the calls it lacks."""

    d = os.environ.get("PIN_RESUME_CALLS_DIR")
    if not d:
        return {}
    p = Path(d)
    if not p.exists() and str(p).startswith("/content/"):
        p = CONTENT / str(p).removeprefix("/content/")
    return {f.stem: f for f in p.glob("*.jsonl")}


FUNNEL_DROP_SHARE = 0.3  # 6 of the pilot's 20 slots; 3 of the screen's 8 (R&D, 2026-10-17)


def funnel_stop(wave: dict, slots: int = 20) -> str | None:
    """R&D's funnel (2026-10-17), thresholds set before any result: after w0 an arm stops at 30% or more of its slots
    dropped (6 of 20, 3 of 8), a failed warm-up or a halted cell."""

    if wave.get("exit") and not wave.get("result"):
        return f"w0 exited {wave['exit']} with no result"
    r = wave.get("result") or {}
    limit = max(1, math.ceil(FUNNEL_DROP_SHARE * slots - 1e-9))
    if (r.get("dropped") or 0) >= limit:
        return f"{r['dropped']} of {slots} slots dropped at w0 (threshold {limit})"
    if r.get("halted"):
        return f"halted at w0: {r['halted']}"
    if r.get("lesson_halts"):
        return f"cell halt at w0: {r['lesson_halts']}"
    if r.get("warmup_failed") or r.get("warmups_failed"):
        return "warm-up failed"
    return None


def pending(wave: dict) -> bool:
    states = ((wave.get("result") or {}).get("states")) or {}
    return bool(states.get("awaiting_critique") or states.get("needs_rewrite"))


def drafts_pending(drafts: Path) -> int:
    """How many drafts in a sealed drafts file still wait for a critique or a rewrite."""

    n = 0
    for line in drafts.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            n += d.get("type") == "draft" and d.get("state") in ("awaiting_critique", "needs_rewrite")
    return n


REBUILD_CODE = r"""
import json, sys
from pathlib import Path
from role_set import waves
drafts, out = Path(sys.argv[1]), Path(sys.argv[2])
h, d = waves.read(drafts)
waves.write_batch(d, out)
if not waves.CARD_KEYS & set(waves.header_writer_pins(h)):
    side = {"drafts_sha256": waves.file_sha256(drafts)}
    if len(sys.argv) < 4:
        side["refused"] = "no prior drafts for this arm"
    else:
        try:
            pins, sha = waves.chain_card_pins(h, Path(sys.argv[3]))
            side.update(via="chain", prior_sha256=sha, pins=pins)
        except waves.WaveError as e:
            side["refused"] = str(e)
    (out / "card-pins-chain.json").write_text(json.dumps(side, indent=1, sort_keys=True) + "\n", encoding="utf-8")
"""


def rebuild_batch(rs: dict, drafts: Path, out: Path, prior: Path | None = None) -> Path:
    """A resumed arm with nothing pending: its batch files, rebuilt from its drafts the way the wave that finished it
    writes them (waves.write_batch), so its twins have an accepted.jsonl to train against (R&D, 2026-10-17). Runs in
    role-set's own Python; the drafts are read through waves.read, so a file that doesn't match its seal is refused.
    When the drafts' header lost the writer's card pins, card-pins-chain.json beside the batch files records them
    through waves.chain_card_pins from `prior`, or why the chain refused; no wave needs it, so the run goes on (R&D,
    2026-10-18). The batch files themselves stay byte for byte what the window wrote."""

    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(drafts, out / "wave-drafts.jsonl")
    shutil.copyfile(drafts.with_name(drafts.name + ".sha256"), out / "wave-drafts.jsonl.sha256")
    python = str(Path(rs["role_set"]).with_name("python")) if rs.get("role_set") else sys.executable
    args = [str(out / "wave-drafts.jsonl"), str(out), *([str(prior)] if prior is not None else [])]
    r = run_proc([python, "-c", REBUILD_CODE, *args], timeout=600, cwd=rs.get("src"))
    if r.returncode != 0:
        raise RuntimeError(f"rebuilding {out}'s batch files failed: {(r.stderr or r.stdout)[-1500:]}")
    return out


class Pool:
    """The window's models, each set up the first time a wave needs it, and only once (R&D, 2026-10-17, from Argon's
    reply): a writer just before its own write wave, a critic just before its first critique. A VM reclaimed partway
    then costs at most the setup in hand, and the arms already written have their calls in job.log. A model that is
    both a writer and a critic (Muse and Lightning critiquing each other) is fetched once."""

    def __init__(self, built: dict, result: dict, policies: dict, cards: Path, cards_out: Path):
        self.built, self.result, self.policies, self.cards, self.cards_out = built, result, policies, cards, cards_out
        self.by_name = {m["served_as"]: m for m in MODELS}
        self.models: dict = {}
        self.srv: dict = {}
        self.proxies: dict = {}

    def get(self, name: str) -> dict:
        if name in self.models:
            if self.models[name].get("dropped"):
                # Its GGUF went after what the window planned as its last use; serving it now would start llama-server
                # on a deleted file (R&D on #89).
                raise RuntimeError(f"{name}: its GGUF was deleted after its last planned use, and a later step asked for it")
            return self.models[name]
        spec = self.by_name[name]
        policy = self.policies.get(name, "q8")
        if stub_mode():
            quant = "Q6_K" if policy == "q6k" else "Q8_0"
            served = f"{spec['served_as']}-q6k" if policy == "q6k" else spec["served_as"]
            sha = (spec.get("expect_q6k_sha256") or spec["expect_sha256"]) if policy == "q6k" else spec["expect_sha256"]
            m = {"served_as": served, "card": spec.get("card") or spec["served_as"], "repo": spec["repo"],
                 "how": "stub", "gguf_sha256": sha, "quant": quant, "stub": True}
            self.result.setdefault("models", []).append(dict(m))
            self.result.setdefault("quant", {})[name] = {"policy": policy, "ran": quant, "stub": True}
        else:
            m = fetch_model(spec, self.built, keep_snapshot=policy in ("q6k", "q8_or_q6k"))
            self.result.setdefault("models", []).append({k: v for k, v in m.items() if k not in ("path", "tokenizer_hf", "converter_py")})
            if spec.get("expect_sha256") and m["gguf_sha256"] != spec["expect_sha256"]:
                raise RuntimeError(f"{name}: GGUF {m['gguf_sha256'][:12]} doesn't reproduce the pin {spec['expect_sha256'][:12]}")
            m, self.result.setdefault("quant", {})[name] = choose_quant(self.built, spec, m, policy, self.result)
        path, cid, info = j1_card(m, self.cards, self.cards_out)
        self.proxies[name] = {"config": str(path), "candidate": cid}
        self.result.setdefault("proxy_configs", {})[m["served_as"]] = info
        self.models[name] = m
        self.srv[name] = Served(self.built, m, m["gguf_sha256"])
        return m

    def serve(self, name: str) -> "Served":
        self.get(name)
        return self.srv[name]

    def proxy(self, name: str) -> dict | None:
        return self.proxies.get(name)

    def drop(self, name: str) -> None:
        """Delete a model's GGUF once no later wave or twin batch in this window needs it, and log resources()."""

        m = self.models.get(name)
        if not m or m.get("dropped") or m.get("stub"):
            return
        p = Path(m["path"]) if m.get("path") else None
        if p and p != Path("/dev/null") and p.is_file():
            p.unlink(missing_ok=True)
        m["dropped"] = True
        resources(f"{name}: after deleting GGUF")

    def drop_unused(self, needed: set[str]) -> None:
        for name in list(self.models):
            if name not in needed:
                self.drop(name)


def critics_for(writer_names: list, families: dict | None = None, *, waves: int | None = None) -> dict:
    """Each arm's critic: PIN_PILOT_CRITIC for all, with PIN_PILOT_CRITIC_FOR=<writer>=<critic>,... per arm (W3:
    Lightning and Muse critique each other's arm, the design's own critic rule). role-set refuses a same-family pair.
    PIN_PILOT_CRITIC=none is allowed only when PIN_PILOT_WAVES=1 (write-only)."""

    default = os.environ.get("PIN_PILOT_CRITIC", "").strip()
    over = dict(x.split("=", 1) for x in os.environ.get("PIN_PILOT_CRITIC_FOR", "").split(",") if "=" in x)
    eff_waves = int(os.environ.get("PIN_PILOT_WAVES") or PILOT_WAVES) if waves is None else int(waves)
    if default.lower() == "none":
        if over:
            raise RuntimeError("REFUSED: PIN_PILOT_CRITIC=none takes no PIN_PILOT_CRITIC_FOR")
        if eff_waves != 1:
            raise RuntimeError(f"REFUSED: PIN_PILOT_CRITIC=none is allowed only when PIN_PILOT_WAVES=1 (got {eff_waves})")
        return {}
    if any(c.strip().lower() == "none" for c in over.values()):
        raise RuntimeError("REFUSED: 'none' is only valid as PIN_PILOT_CRITIC=none with PIN_PILOT_WAVES=1")
    out = {n: over.get(n, default) for n in writer_names}
    missing = [n for n, c in out.items() if not c]
    if missing:
        raise RuntimeError(f"no critic for {', '.join(missing)}: set PIN_PILOT_CRITIC or PIN_PILOT_CRITIC_FOR")
    if any(n == c for n, c in out.items()):
        raise RuntimeError(f"an arm can't critique itself: {out}")
    # The family rule role-set enforces, checked here too, so a same-family pair fails before any model loads.
    same = [f"{n}<-{c}" for n, c in out.items() if families and families.get(n) and families.get(n) == families.get(c)]
    if same:
        raise RuntimeError(f"a critic reads another family's drafts; same family: {', '.join(same)}")
    return out


def _needed_models(arms: dict, critic_for: dict, i: int, left_in_wave: list[str], kind: str,
                   *, twins_after: bool, later_arms: list[str], finished_batch: bool = False) -> set[str]:
    needed: set[str] = set()
    if not finished_batch:
        if kind == "critique":
            needed.update(critic_for[n] for n in left_in_wave if n in critic_for)
        else:
            needed.update(left_in_wave)
        active = [n for n, a in arms.items() if not a["done"]]
        if any(j == 0 or j % 2 == 0 for j in range(i + 1, PILOT_WAVES)):
            needed.update(active)
        if any(j > 0 and j % 2 == 1 for j in range(i + 1, PILOT_WAVES)):
            needed.update(critic_for[n] for n in active if n in critic_for)
    twin_arms = list(later_arms)
    if twins_after:
        for n, a in arms.items():
            if a.get("funnel_stop") or any(w.get("exit") for w in a["waves"]):
                continue
            if finished_batch or a["done"]:
                acc = a.get("last") and Path(a["last"]) / "accepted.jsonl"
                if not (acc and acc.exists() and acc.read_text(encoding="utf-8").strip()):
                    continue
            twin_arms.append(n)
    if PILOT_WAVES >= 1:
        needed.update(twin_arms)
    if PILOT_WAVES >= 2:
        needed.update(critic_for[n] for n in twin_arms if n in critic_for)
    return needed


def pilot_batch(rs, common, plan: Path, base: Path, arms_in: list, critic_for: dict, pool: Pool, extra=None,
                *, twins_after: bool = False, later_arms: list[str] = ()) -> dict:
    """One batch (the pilot, or its twins) for every writer arm, waves interleaved around the critics: all write
    waves, then each critic once for the arms it reads, then all rewrites, and so on. Every model is set up the first
    time it is needed (Pool). Returns each arm's waves and its last directory."""

    arms = {name: {"waves": [], "drafts": None, "done": False} for name in arms_in}
    start, resumed = resume_from() if extra is None else (0, {})  # the twins never resume from the pilot's drafts
    calls = resume_calls() if extra is None else {}
    priors = resume_priors() if extra is None and start > 0 else {}
    for n, f in resumed.items():
        if n in arms:
            arms[n]["drafts"] = f
            arms[n]["resumed_after_wave"] = start - 1
            if n in priors:
                arms[n]["prior"] = priors[n]  # for the first wave after the resume only
            if not drafts_pending(f):
                # Its batch finished in the cut window: no wave to run, and its writer is never set up for this batch.
                # Its accepted rows are rebuilt so its twins can train against them.
                arms[n]["done"], arms[n]["resumed_finished"] = True, True
                arms[n]["last"] = rebuild_batch(rs, f, base / n / f"w{start - 1}-resumed", priors.get(n))
    if start > 0:
        for n, a in arms.items():  # listed in PIN_PILOT_WRITERS but no drafts to resume: it can't run wave `start`
            if n not in resumed:
                a["done"], a["not_resumed"] = True, f"no drafts in PIN_RESUME_DIR for wave {start}"
    for i in range(start, PILOT_WAVES):
        kind = "write" if i == 0 else ("critique" if i % 2 else "rewrite")
        todo = [n for n, a in arms.items() if not a["done"]]
        if not todo:
            break
        if kind == "critique":
            assert critic_for and all(critic_for.get(n) for n in todo), f"pilot_batch reached a critique wave with no critic for {todo}"
            done_crit: set[str] = set()
            for c in dict.fromkeys(critic_for[n] for n in todo):  # each critic served once, in the arms' order
                with pool.serve(c):
                    for n in [n for n in todo if critic_for[n] == c]:
                        w = write_wave(rs, common + ["--plan", str(plan)], "critique", base / n / f"w{i}", critic=pool.get(c),
                                       drafts=arms[n]["drafts"], extra=extra, proxy=pool.proxy(c),
                                       reuse_calls=calls.get(n) if i == start else None)
                        arms[n]["waves"].append(w)
                        arms[n]["last"] = run_dir(base / n / f"w{i}")
                        arms[n]["drafts"] = arms[n]["last"] / "wave-drafts.jsonl"
                        checkpoint(n, f"{base.name}-w{i}", arms[n]["drafts"])
                        arms[n]["done"] = w["exit"] != 0 or not pending(w)
                        done_crit.add(n)
                left = [n for n in todo if n not in done_crit]
                pool.drop_unused(_needed_models(arms, critic_for, i, left, kind, twins_after=twins_after, later_arms=list(later_arms)))
        else:
            done_write: set[str] = set()
            for n in todo:
                with pool.serve(n):
                    chain = ["--prior-drafts", str(arms[n]["prior"])] if kind == "rewrite" and i == start and arms[n].get("prior") else []
                    w = write_wave(rs, common + ["--plan", str(plan)], kind, base / n / f"w{i}", writer=pool.get(n),
                                   drafts=arms[n]["drafts"], extra=[*(extra or []), *chain] or None, proxy=pool.proxy(n),
                                   reuse_calls=calls.get(n) if i == start else None)
                arms[n]["waves"].append(w)
                arms[n]["last"] = run_dir(base / n / f"w{i}")
                arms[n]["drafts"] = arms[n]["last"] / "wave-drafts.jsonl"
                checkpoint(n, f"{base.name}-w{i}", arms[n]["drafts"])
                # Nothing left to critique (every slot dropped, R&D on 13b) ends the arm, as the critique branch does.
                arms[n]["done"] = w["exit"] != 0 or not pending(w)
                # ---- (9) the funnel, on the pilot's first write wave only
                if i == 0 and extra is None:
                    why = funnel_stop(w, len(json.loads(plan.read_text(encoding="utf-8"))["slots"]))
                    if why:
                        arms[n]["done"], arms[n]["funnel_stop"] = True, why
                done_write.add(n)
                left = [x for x in todo if x not in done_write]
                pool.drop_unused(_needed_models(arms, critic_for, i, left, kind, twins_after=twins_after, later_arms=list(later_arms)))
    pool.drop_unused(_needed_models(arms, critic_for, PILOT_WAVES, [], "done", twins_after=twins_after,
                                    later_arms=list(later_arms), finished_batch=True))
    return arms


def drop_snapshot(m: dict) -> None:
    if m.get("snapshot"):
        shutil.rmtree(m["snapshot"], ignore_errors=True)
        m.pop("snapshot", None)
        resources(f"{m.get('served_as', 'model')}: after dropping snapshot")


def to_q6k(built: dict, spec: dict, m: dict, result: dict) -> dict:
    """The writer's Q6_K from a BF16 conversion of the snapshot (R&D, 2026-10-17: never requantized from the Q8_0; the
    quantize README warns that requantizing can severely reduce quality). Order: unlink the Q8_0, BF16 convert, drop the
    snapshot, BF16 -> Q6_K, drop the BF16. Served as writers.json's '<name>-q6k'."""

    if not m.get("snapshot"):
        raise RuntimeError(f"{spec['served_as']}: no snapshot kept for the BF16 conversion")
    if "llama_quantize" not in result:
        result["llama_quantize"] = build_quantize(Path(built["src"]))
    if m.get("path"):
        Path(m["path"]).unlink(missing_ok=True)
        resources(f"{spec['served_as']}: after unlinking Q8_0")
    bf16 = WORK / f"{spec['served_as']}-BF16.gguf"
    say(f"converting {spec['served_as']} to BF16 for Q6_K")
    sh(f"{m['converter_py']} {built['src']}/convert_hf_to_gguf.py {m['snapshot']} --outtype bf16 --outfile {bf16}", timeout=3600)
    resources(f"{spec['served_as']}: after BF16 conversion")
    drop_snapshot(m)
    q6 = WORK / f"{spec['served_as']}-Q6_K.gguf"
    sh(f"{result['llama_quantize']['path']} {bf16} {q6} Q6_K", timeout=3600)
    bf16_sha = sha256(bf16)
    bf16.unlink(missing_ok=True)
    resources(f"{spec['served_as']}: after Q6_K")
    got = sha256(q6)
    if spec.get("expect_q6k_sha256") and got != spec["expect_q6k_sha256"]:
        raise RuntimeError(f"{spec['served_as']}: Q6_K {got[:12]} doesn't reproduce {spec['expect_q6k_sha256'][:12]}")
    return {**m, "served_as": f"{spec['served_as']}-q6k", "card": m.get("card") or spec["served_as"], "path": str(q6),
            "gguf_file": q6.name, "gguf_sha256": got, "gguf_bytes": q6.stat().st_size, "quant": "Q6_K",
            "bf16_sha256": bf16_sha, "label": f"Q6_K (from BF16 {bf16_sha[:12]}, the same snapshot as pinned Q8_0 {m['gguf_sha256'][:12]})"}


def choose_quant(built: dict, spec: dict, m: dict, policy: str, result: dict) -> tuple[dict, dict]:
    """q8: the Q8_0. q6k: the pinned Q6_K. q8_or_q6k: the Q8_0 if it passes the gates (offload N/N, 1 slot x 32768),
    otherwise the pinned Q6_K (R&D, 2026-10-17). Returns the model to serve and the record of the choice."""

    if policy == "q8":
        drop_snapshot(m)
        return m, {"policy": policy, "ran": "Q8_0"}
    if policy == "q6k":
        return to_q6k(built, spec, m, result), {"policy": policy, "ran": "Q6_K"}
    try:
        with Served(built, m, m["gguf_sha256"]) as srv:
            first = srv.starts[-1]
        drop_snapshot(m)
        return m, {"policy": policy, "ran": "Q8_0", "q8_gates": first}
    except RuntimeError as e:
        return to_q6k(built, spec, m, result), {"policy": policy, "ran": "Q6_K", "why": str(e)[:400]}


def twins_enabled() -> bool:
    """PIN_PILOT_TWINS=0 leaves the twins for a later window, for the arms that go on (R&D, 2026-10-17: the screen is
    w0 and its critique for every arm; rewrites and twins only for the arms that pass)."""

    return os.environ.get("PIN_PILOT_TWINS", "1") != "0"


def buffers_enabled() -> bool:
    """PIN_PILOT_BUFFERS=1: plan T10 with --buffers (40 base + 8 training buffers + 10 apply + 2 apply buffers)."""

    raw = os.environ.get("PIN_PILOT_BUFFERS", "0").strip()
    if raw not in ("", "0", "1"):
        raise RuntimeError(f"REFUSED: PIN_PILOT_BUFFERS must be '0' or '1', got {raw!r}")
    return raw == "1"


def critique_scope() -> str:
    """PIN_CRITIQUE_SCOPE=all|verdict (default 'all'): passed as --critique-scope to role-set write critique waves."""

    scope = (os.environ.get("PIN_CRITIQUE_SCOPE") or "all").strip().lower()
    if scope not in ("all", "verdict"):
        raise RuntimeError(f"REFUSED: PIN_CRITIQUE_SCOPE must be 'all' or 'verdict', got {scope!r}")
    return scope


def check_pins(names: list, listed: dict, policies: dict) -> None:
    """Every model the window will set up has a pin in MODELS, equal to writers.json's, before anything downloads (R&D,
    from Argon's review: three writers had none, so a mismatch showed only at w0, after 15-27 minutes of setup). A Q6_K
    policy also needs writers.json's '<name>-q6k' entry."""

    by_name = {m["served_as"]: m for m in MODELS}
    listed_pins = {e["served_as"]: e.get("gguf_sha256") for e in listed["writers"] + listed["critics"]}
    bad = []
    for n in names:
        mine, theirs = (by_name.get(n) or {}).get("expect_sha256"), listed_pins.get(n)
        if not mine or mine != theirs:
            bad.append(f"{n}: MODELS {str(mine)[:12]} vs writers.json {str(theirs)[:12]}")
        if policies.get(n) in ("q6k", "q8_or_q6k"):
            mine6, theirs6 = (by_name.get(n) or {}).get("expect_q6k_sha256"), listed_pins.get(f"{n}-q6k")
            if not mine6 or mine6 != theirs6:
                bad.append(f"{n}-q6k: MODELS {str(mine6)[:12]} vs writers.json {str(theirs6)[:12]}")
    if bad:
        raise RuntimeError("pins don't agree, nothing downloaded: " + "; ".join(bad))


def twins_skip_reason(a: dict) -> str | None:
    """Why an arm gets no twins, or None when it does. The same arms _needed_models keeps a writer for, so a twin batch
    never asks for a writer the pool has deleted (Argon, v2 re-review)."""

    if a.get("funnel_stop"):
        return f"stopped by the funnel: {a['funnel_stop']}"
    if not twins_enabled():
        return "PIN_PILOT_TWINS=0: twins wait for the arms that go on"
    if any(w.get("exit") for w in a.get("waves") or []):
        return "a pilot wave exited non-zero"
    accepted = a.get("last") and Path(a["last"]) / "accepted.jsonl"
    if not (accepted and accepted.exists() and accepted.read_text(encoding="utf-8").strip()):
        return "the pilot accepted no rows to train the twins against"
    return None


def screen_plan(plan: Path, var: str) -> list[str] | None:
    """The w0 screen's frozen slot list (R&D, 2026-10-17): the same slot ids for every arm, set in the window config.
    The plan keeps only those slots, in its own order; an id the plan lacks fails the window before any model loads.
    PIN_PILOT_SLOT_IDS=ALL keeps every planned slot (requires PIN_PILOT_BUFFERS=1)."""

    ids = [x.strip() for x in os.environ.get(var, "").split(",") if x.strip()]
    if not ids:
        return None
    if ids == ["ALL"] or "ALL" in ids:
        if ids != ["ALL"] or var != "PIN_PILOT_SLOT_IDS" or not buffers_enabled():
            raise RuntimeError(f"REFUSED: {var}=ALL requires PIN_PILOT_BUFFERS=1")
        data = json.loads(plan.read_text(encoding="utf-8"))
        return [s["id"] for s in data["slots"]]
    data = json.loads(plan.read_text(encoding="utf-8"))
    have = {s["id"] for s in data["slots"]}
    missing = [i for i in ids if i not in have]
    if missing:
        raise RuntimeError(f"{var}: {', '.join(missing)} not in {plan.name}")
    data["slots"] = [s for s in data["slots"] if s["id"] in ids]
    plan.write_text(json.dumps(data, indent=1), encoding="utf-8", newline="\n")  # the same bytes on Windows and the VM
    return ids


def plan_pilot_lessons(rs: dict, lessons: list[str], domains: str, out: Path, work_dir: Path | None = None) -> dict:
    """Plan each lesson alone with `role-set plan --lessons <L> [--buffers] --seed <seed> --domains <naics> --out <tmp>` and merge into `out`."""

    buffers = buffers_enabled()
    if buffers:
        if twins_enabled():
            raise RuntimeError("REFUSED: PIN_PILOT_BUFFERS=1 requires PIN_PILOT_TWINS=0")
        if not lessons or any(x != "T10" for x in lessons):
            raise RuntimeError(f"REFUSED: PIN_PILOT_BUFFERS=1 is allowed only for T10, got {lessons!r}")
    slot_ids = [x.strip() for x in os.environ.get("PIN_PILOT_SLOT_IDS", "").split(",") if x.strip()]
    if not slot_ids:
        raise RuntimeError("REFUSED: PIN_PILOT_LESSONS requires PIN_PILOT_SLOT_IDS")
    if twins_enabled():
        raise RuntimeError("REFUSED: PIN_PILOT_LESSONS requires PIN_PILOT_TWINS=0")
    seed = int(os.environ.get("PIN_PILOT_SEED") or "0")
    tmp_dir = work_dir or out.parent
    tmp_dir.mkdir(parents=True, exist_ok=True)
    per_lesson_sha256: dict[str, str] = {}
    plans: list[dict] = []
    excl_shas: dict[str, str | None] = {}
    buf_flag = " --buffers" if buffers else ""
    for lesson in lessons:
        tmp = tmp_dir / f"lesson-{lesson}-plan.json"
        sh(f"{rs['role_set']} plan --lessons {lesson}{buf_flag} --seed {seed} --domains {domains} --out {tmp}", cwd=rs.get("src"))
        per_lesson_sha256[lesson] = sha256(tmp)
        pdata = json.loads(tmp.read_text(encoding="utf-8"))
        plans.append(pdata)
        excl_shas[lesson] = pdata.get("exclusion_state_sha256")
    if len(set(excl_shas.values())) > 1:
        raise RuntimeError(f"REFUSED: lesson plans have mismatched exclusion_state_sha256: {excl_shas}")
    merged = dict(plans[0])
    merged["slots"] = [s for p in plans for s in p.get("slots", [])]
    merged["figures"] = [f for p in plans for f in (p.get("figures") or [])]
    out.write_text(json.dumps(merged, indent=1), encoding="utf-8", newline="\n")
    return {
        "lessons": list(lessons),
        "buffers": buffers,
        "seed": seed,
        "per_lesson_sha256": per_lesson_sha256,
        "exclusion_state_sha256": merged.get("exclusion_state_sha256"),
    }


# Stage 1 frozen training files in training order (aspire-si docs/runs/stage1-training-data/FROZEN.md,
# "## Training files", branch verifier-distill-prereg-draft). Never sort by path: the order is the training order.
# Each sha256 is over LF-normalized content (CRLF read as LF, matching receipts.sha256_file).
FROZEN_TRAINING_FILES: tuple[tuple[str, str], ...] = (
    ("docs/runs/stage1-training-data/pilot-t04.jsonl", "a82c7302b40ae323a853a34cd0ceb7b9adda4414536cd2549db08a4224c0e7e6"),
    ("docs/runs/stage1-training-data/batch2-t01.jsonl", "f29eea91326bd3fb6da01b82cf94b2aed67bb549a2facd93db69e21d83c66df8"),
    ("docs/runs/stage1-training-data/batch2-t14.jsonl", "5486a1dc483fcae51c4497e411223fb6d661f5b61a1b2d905472c4d4a226c96d"),
    ("docs/runs/stage1-training-data/batch2-t16.jsonl", "6dc77b59d7e7ba71792d6afe09ede43605301a89f71c1ccbe6c8fc669b348cd3"),
    ("docs/runs/stage1-training-data/batch3-t05.jsonl", "fe7325218f9df85a1040f0c833b801806a33f3da7ede45a412c9e78caeaacf06"),
    ("docs/runs/stage1-training-data/batch3-t13.jsonl", "4f45c013d698022e6e292c1c5410ab42c2efe4ab6fd01501bee18d6e3257cd5a"),
    ("docs/runs/stage1-training-data/batch3-t17.jsonl", "3411b69b9234974aad22b82127524ba48e49214bedf670593da2d4daa6e14479"),
    ("docs/runs/stage1-training-data/batch4-t02.jsonl", "3724dddcd31cea00813a1b384757bba080d9f348ec2aeebc8bcb3e9789ccf3af"),
    ("docs/runs/stage1-training-data/batch4-t03.jsonl", "0f3577654ef4624e3959ec4cfc6d444148aad7b3199a3bb28b46d6ab98297f80"),
    ("docs/runs/stage1-training-data/batch4-t06.jsonl", "95e9f6f21785fa1220e78e98543580f494536fc9b59df15ef401944f2e512cf1"),
    ("docs/runs/stage1-training-data/batch5-t07.jsonl", "20e18c67d80859f656ae3721a504b6c742cba587991fc2641085953c238a74a5"),
    ("docs/runs/stage1-training-data/batch5-t08.jsonl", "01077ee791a5c8cadf5a9ef249d9c9344e0e0e0e39a0ed3a848e4c86fd5ea172"),
    ("docs/runs/stage1-training-data/batch5-t09.jsonl", "a76b5de018d859f342d0cbd067e0054321d6a3550b9f62b8084e7b94eb9cc7a6"),
    ("docs/runs/stage1-training-data/batch6-t10.jsonl", "f6c09e47caf62dc215ddd64e18fbc20538e36e96f2c596d58e964ff62ac3b97e"),
    ("docs/runs/stage1-training-data/batch6-t11.jsonl", "b9e01765264b6f4da75f6a5d2e5322f7feabd356eb8e77fba4b20a64dececdfd"),
    ("docs/runs/stage1-training-data/batch6-t12.jsonl", "a2db220b9b84ddb16e6af3850d4fb43104cee28bd8f2992393b11ecd760c12ee"),
    ("docs/runs/stage1-training-data/batch7-t15.jsonl", "9455e6eafe4d53d497186f842a068d4a57066fc401b722c5d27df8d14800fe06"),
    ("docs/runs/stage1-training-data/batch7-t18.jsonl", "512c38c99078b41b2acf2c2816776025d9e52028ad0e86461c43e5fc8f2e5e5f"),
    ("docs/runs/stage1-training-data/batch8-l1.jsonl", "a0abcc2bd67c9ef92476a29e584a229567ed8c6224508b4453f133f0a1829f11"),
    ("docs/runs/stage1-training-data/batch8-l2.jsonl", "b1256d296d3f62eb1c7b84596de097f715e8db0b01496a6beb2494d3c8a7b1c0"),
    ("docs/runs/stage1-training-data/batch8-l3.jsonl", "cf557a06bf17253269966811b1abf220106b24b3ed5bc2efc1f9ba9c90c9066a"),
)


def sha256_lf(path: Path) -> str:
    """LF-normalized sha256 (CRLF read as LF), matching aspire-si's receipts.sha256_file and FROZEN.md."""

    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def plan_has_apply_slots(plan: Path) -> bool:
    """Whether the screened plan has any slot with split == 'apply'."""

    if not plan.exists():
        return False
    data = json.loads(plan.read_text(encoding="utf-8"))
    return any(s.get("split") == "apply" for s in data.get("slots", []))


def training_inputs(result: dict) -> list[Path]:
    """The frozen Stage 1 training set and its .models.jsonl sidecars, unpacked read-only and sha-checked before any
    server starts (Argon round 1 F1, round 2 F1). Returns the training .jsonl paths in FROZEN.md's training order."""

    import tarfile  # noqa: PLC0415

    raw_shas = os.environ.get("PIN_TRAINING_SHAS", "").strip()
    shas = dict(x.split("=", 1) for x in raw_shas.split(",") if "=" in x)
    if not shas:
        raise RuntimeError("REFUSED: plan has apply slots; PIN_TRAINING_SHAS is required")
    raw_tar = os.environ.get("PIN_TRAINING_INPUTS", "").strip()
    tar = Path(raw_tar) if raw_tar else (CONTENT / "training-inputs.tar")
    if not tar.is_file():
        raise RuntimeError(f"REFUSED: plan has apply slots; PIN_TRAINING_INPUTS tar not found: {tar}")
    root = WORK / "training-in"
    if root.exists():
        for p in sorted(root.rglob("*")):
            os.chmod(p, 0o700)
        os.chmod(root, 0o700)
        shutil.rmtree(root)
    root.mkdir(parents=True)
    with tarfile.open(tar) as tf:
        tf.extractall(root, filter="data")
    unpacked = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    unpinned = [n for n in unpacked if n not in shas]
    if unpinned:
        raise RuntimeError(f"REFUSED: no sha256 in PIN_TRAINING_SHAS for {', '.join(unpinned)}")
    missing = [n for n in shas if not (root / n).is_file()]
    if missing:
        raise RuntimeError(f"REFUSED: training input missing from tar: {', '.join(missing)}")
    data_files = [n for n in shas if n.endswith(".jsonl") and not n.endswith(".models.jsonl")]
    if not data_files:
        raise RuntimeError("REFUSED: PIN_TRAINING_SHAS names no training .jsonl files")
    ordered_files: list[str] = []
    expected_frozen_sha: dict[str, str] = {}
    missing_frozen: list[str] = []
    claimed: set[str] = set()
    for rel_path, frozen_sha in FROZEN_TRAINING_FILES:
        fname = Path(rel_path).name
        match = next((n for n in data_files if n not in claimed and n in (rel_path, fname)), None)
        if match is None:
            missing_frozen.append(rel_path)
        else:
            claimed.add(match)
            ordered_files.append(match)
            expected_frozen_sha[match] = frozen_sha
    if missing_frozen:
        raise RuntimeError(f"REFUSED: missing frozen training file: {', '.join(missing_frozen)}")
    extra_data = [n for n in data_files if n not in claimed]
    if extra_data:
        raise RuntimeError(f"REFUSED: extra training .jsonl file not in FROZEN_TRAINING_FILES: {', '.join(extra_data)}")
    missing_sidecars = [
        f"{n[:-len('.jsonl')]}.models.jsonl"
        for n in ordered_files
        if f"{n[:-len('.jsonl')]}.models.jsonl" not in shas or not (root / f"{n[:-len('.jsonl')]}.models.jsonl").is_file()
    ]
    if missing_sidecars:
        raise RuntimeError(f"REFUSED: missing training sidecar: {', '.join(missing_sidecars)}")
    got = {n: sha256_lf(root / n) for n in shas}
    bad = [f"{n} is {got[n][:12]}, not {shas[n][:12]}" for n in shas if got[n] != shas[n]]
    if bad:
        raise RuntimeError("REFUSED: training input sha256 mismatch: " + "; ".join(bad))
    bad_frozen = [
        f"{n} is {got[n][:12]}, not FROZEN_TRAINING_FILES {expected_frozen_sha[n][:12]}"
        for n in ordered_files
        if got[n] != expected_frozen_sha[n]
    ]
    if bad_frozen:
        raise RuntimeError("REFUSED: frozen training sha256 mismatch: " + "; ".join(bad_frozen))
    for p in sorted(root.rglob("*"), reverse=True):
        os.chmod(p, 0o555 if p.is_dir() else 0o444)
    os.chmod(root, 0o555)
    result["training_inputs"] = {"tar_sha256": sha256(tar), "order": ordered_files, "files": got}
    return [root / n for n in ordered_files]


def rescore_inputs(result: dict) -> tuple[Path, dict]:
    """The rescore's inputs, read-only (R&D, 2026-10-18): one tar the rig uploads (PIN_RESCORE_INPUTS), unpacked into a
    directory of its own and made read-only, and every file named in PIN_RESCORE_SHAS ("<relative path>=<sha256>,...")
    checked against its sha256 before use. A file the run or drafts list names without a sha refuses the window, and so
    does any mismatch. The rescore never writes beside them."""

    tar = Path(os.environ.get("PIN_RESCORE_INPUTS") or (CONTENT / "rescore-inputs.tar"))
    shas = dict(x.split("=", 1) for x in os.environ.get("PIN_RESCORE_SHAS", "").split(",") if "=" in x)
    named = [x for x in os.environ.get("PIN_RESCORE_RUNS", "").split(",") if x]
    for spec in [x for x in os.environ.get("PIN_RESCORE_DRAFTS", "").split(";") if x]:
        named += spec.split("=", 1)[1].split(",")
    if not named:
        raise RuntimeError("REFUSED: PIN_RESCORE_RUNS and PIN_RESCORE_DRAFTS name no input")
    unpinned = [n for n in named if n not in shas]
    if unpinned:
        raise RuntimeError(f"REFUSED: no sha256 in PIN_RESCORE_SHAS for {', '.join(unpinned)}")
    root = WORK / "rescore-in"
    if root.exists():
        sh(f"chmod -R u+w {root} && rm -rf {root}")
    root.mkdir(parents=True)
    sh(f"tar -xf {tar} -C {root}")
    got = {n: sha256(root / n) for n in shas}
    bad = [f"{n} is {got[n][:12]}, not {shas[n][:12]}" for n in shas if got[n] != shas[n]]
    if bad:
        raise RuntimeError("REFUSED: rescore input sha256 mismatch: " + "; ".join(bad))
    sh(f"chmod -R a-w {root}")
    result["rescore_inputs"] = {"tar_sha256": sha256(tar), "files": got}
    return root, shas


def rescore_window(built: dict, result: dict) -> None:
    """PIN_ROLE=rescore (R&D, 2026-10-18): the second critic (PIN_RESCORE_CRITIC) set up and served exactly as in the
    screen's critique waves (the Pool: converted in-window, byte-matched to its pin before serving, its card's proxy
    config from card-values.json), then one `role-set rescore` against it on the read-only inputs, with PIN_RESCORE_SEED.
    The rescore's process sits under the window's deadline like a wave: past it, SIGTERM, then SIGKILL, DeadlineReached,
    and the finally returns what exists. sample-manifest.json is written before scoring, so a cut window still returns
    the draw. Known limit: the rescore's critic calls aren't journaled, so a cut means scoring again (7 items).
    rescore-out.tar holds the manifest, rescore.json, rescore-report.json, the rescore's log and the critic's server
    log; pin-result.json carries critic_card and the input shas."""

    # No token is sent to a rescore window; it records the same five-place check as every window (R&D on #95).
    take_token()
    check_token_clean(result)
    rs = result["role_set"] = role_set_env(os.environ["PIN_ROLESET_COMMIT"])
    result["rnd"] = install_rnd(os.environ["PIN_RND_COMMIT"])
    critic = os.environ["PIN_RESCORE_CRITIC"]
    seed = os.environ.get("PIN_RESCORE_SEED", "0")
    if critic not in {m["served_as"] for m in MODELS}:
        raise RuntimeError(f"{critic} isn't in pin_window.MODELS")
    root, _ = rescore_inputs(result)
    pack = CONTENT / "pack"
    if not (pack / "pack.json").exists():
        pack.mkdir(parents=True, exist_ok=True)
        sh(f"tar -xf {CONTENT / 'pack.tar'} -C {pack}")
    cards_file = CONTENT / "card-values.json"
    out = WORK / "rescore"
    out.mkdir(parents=True, exist_ok=True)
    log = out / "rescore.log"
    srv = None
    try:
        pool = Pool(built, result, {}, cards_file, WORK / "cards-j1")
        m = pool.get(critic)  # converted or stubbed; a GGUF that isn't its pin raises here, before any serve
        result["critic"] = {"served_as": m["served_as"], "gguf_sha256": m["gguf_sha256"]}
        check_deadline("the rescore")
        srv = pool.serve(critic)
        with srv:
            proxy = pool.proxy(critic)
            px, endpoint = None, f"http://127.0.0.1:{PILOT_PORT}/v1"
            if proxy is not None:
                from mcptoolshop_rnd.localbuilders.proxy import LoggingProxyServer, ProxyState  # noqa: PLC0415

                state = ProxyState(upstream_url=f"http://127.0.0.1:{PILOT_PORT}", config=proxy["config"],
                                   candidate=proxy["candidate"], generations_path=out / "generations.jsonl",
                                   max_calls=10**6, wall_limit_s=86400)
                px = LoggingProxyServer(state).start()
                endpoint = px.openai_base_url
            cmd = [rs["role_set"], "rescore", "--pack", str(pack), "--writers", f"{rs['src']}/writers.json",
                   "--cards", str(cards_file), "--critic-endpoint", endpoint, "--critic-model", m["served_as"],
                   "--critic-gguf-sha256", m["gguf_sha256"], "--seed", str(seed), "--out", str(out)]
            for r in [x for x in os.environ.get("PIN_RESCORE_RUNS", "").split(",") if x]:
                cmd += ["--runs", str(root / r)]
            for spec in [x for x in os.environ.get("PIN_RESCORE_DRAFTS", "").split(";") if x]:
                arm, files = spec.split("=", 1)
                cmd += ["--drafts", f"{arm}=" + ",".join(str(root / f) for f in files.split(","))]
            left = deadline_left()
            t0 = time.time()
            try:
                run = run_proc(cmd, timeout=4 * 3600 if left is None else min(4 * 3600, max(left, 1)), cwd=rs["src"],
                               term_first=left is not None)
            except subprocess.TimeoutExpired:
                if left is None or left >= 4 * 3600:
                    raise
                global CUT_AT
                CUT_AT = {"wave": "rescore", "out": str(out), "seconds": round(time.time() - t0),
                          "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                say(f"CUT AT THE DEADLINE rescore after {CUT_AT['seconds']} s; the manifest was written before scoring")
                raise DeadlineReached("the rescore was cut at the window's deadline") from None
            finally:
                if px is not None:
                    px.close()
            log.write_text(run.stdout + "\n--- stderr ---\n" + run.stderr, encoding="utf-8")
            result["rescore"] = {"exit": run.returncode, "seconds": round(time.time() - t0), "table": run.stdout[-3000:]}
            if run.returncode != 0:
                raise RuntimeError(f"role-set rescore exited {run.returncode}: {(run.stderr or run.stdout)[-1500:]}")
        report = json.loads((out / "rescore-report.json").read_text(encoding="utf-8"))
        result["critic_card"] = report.get("critic_card")
        if not result["critic_card"] and not stub_mode():
            raise RuntimeError("the rescore report has no critic_card: no number from it may be read (R&D)")
        result["rescore"]["flagged"] = report.get("flagged")
        result["model_errors"] = 0
    finally:
        if srv is not None and getattr(srv, "log", None) and Path(srv.log).exists():
            shutil.copy(srv.log, out / "critic-server.log")
        try:
            sh(f"tar -C {WORK} -cf {WORK}/rescore-out.tar rescore")
        except Exception as e:  # noqa: BLE001 - recorded; the result still prints
            result["tar_error"] = str(e)[:300]


def critic_eval_inputs(result: dict) -> tuple[Path, Path, dict]:
    """Read-only, sha-checked inputs for PIN_ROLE=critic-eval (PIN_CRITIC_EVAL_INPUTS, PIN_CRITIC_EVAL_SHAS).
    Inputs may be .tar or .tar.gz (GNU tar's -xf detects compression)."""

    default_tar = CONTENT / "critic-eval-inputs.tar.gz" if (CONTENT / "critic-eval-inputs.tar.gz").exists() else CONTENT / "critic-eval-inputs.tar"
    tar = Path(os.environ.get("PIN_CRITIC_EVAL_INPUTS") or default_tar)
    shas = dict(x.split("=", 1) for x in os.environ.get("PIN_CRITIC_EVAL_SHAS", "").split(",") if "=" in x)
    if not shas:
        raise RuntimeError("REFUSED: PIN_CRITIC_EVAL_SHAS names no input")
    verdict_from_rel = (os.environ.get("PIN_CRITIC_EVAL_VERDICT_FROM") or "").strip() or None
    if verdict_from_rel and verdict_from_rel not in shas:
        raise RuntimeError(f"REFUSED: no sha256 in PIN_CRITIC_EVAL_SHAS for {verdict_from_rel}")
    items_rel = os.environ.get("PIN_CRITIC_EVAL_ITEMS")
    if items_rel:
        if items_rel not in shas:
            raise RuntimeError(f"REFUSED: no sha256 in PIN_CRITIC_EVAL_SHAS for {items_rel}")
    else:
        jsonls = [k for k in shas if k.endswith(".jsonl") and k != verdict_from_rel]
        if not jsonls:
            raise RuntimeError("REFUSED: PIN_CRITIC_EVAL_SHAS names no items .jsonl file")
        items_rel = jsonls[0]
    root = WORK / "critic-eval-in"
    if root.exists():
        sh(f"chmod -R u+w {root} && rm -rf {root}")
    root.mkdir(parents=True)
    sh(f"tar -xf {tar} -C {root}")
    missing = [n for n in shas if not (root / n).is_file()]
    if missing:
        raise RuntimeError("REFUSED: critic-eval input missing from tar: " + ", ".join(missing))
    got = {n: sha256(root / n) for n in shas}
    bad = [f"{n} is {got[n][:12]}, not {shas[n][:12]}" for n in shas if got[n] != shas[n]]
    if bad:
        raise RuntimeError("REFUSED: critic-eval input sha256 mismatch: " + "; ".join(bad))
    items_path = root / items_rel
    seal_p = items_path.with_name(items_path.name + ".sha256")
    if not seal_p.exists():
        seal_p.write_text(f"{shas[items_rel]}  {items_path.name}\n", encoding="utf-8")
    sh(f"chmod -R a-w {root}")
    inputs_meta: dict = {"tar_sha256": sha256(tar), "files": got}
    if verdict_from_rel:
        inputs_meta["verdict_from"] = str(root / verdict_from_rel)
        inputs_meta["verdict_from_sha256"] = got[verdict_from_rel]
    result["critic_eval_inputs"] = inputs_meta
    return root, items_path, shas


def critic_eval_resume(critic: str, seeds: list[int], result: dict | None = None) -> dict[int, Path]:
    """PIN_CRITIC_EVAL_RESUME: a tar of an earlier window's journals; maps each seed to its journal path.
    Accepts only critic-eval/s{seed}/calls-journal.jsonl and critic-eval-s{seed}/calls/{critic}.jsonl."""

    raw = os.environ.get("PIN_CRITIC_EVAL_RESUME")
    if not raw:
        return {}
    tar = Path(raw)
    if not tar.exists() and str(tar).startswith("/content/"):
        tar = CONTENT / str(tar).removeprefix("/content/")
    if not tar.exists():
        raise RuntimeError(f"REFUSED: PIN_CRITIC_EVAL_RESUME {tar} does not exist")
    tar_sha = sha256(tar)
    dest = WORK / "critic-eval-resume"
    if dest.exists():
        sh(f"rm -rf {dest}")
    dest.mkdir(parents=True)
    sh(f"tar -xf {tar} -C {dest}")
    found: dict[int, Path] = {}
    for s in seeds:
        for c in (
            dest / "critic-eval" / f"s{s}" / "calls-journal.jsonl",
            dest / f"critic-eval-s{s}" / "calls" / f"{critic}.jsonl",
        ):
            if c.is_file():
                found[s] = c
                break
    if result is not None:
        result["critic_eval_resume"] = {
            "tar_sha256": tar_sha,
            "journals": {str(s): str(p) for s, p in found.items()},
        }
    return found


def critic_eval_window(built: dict, result: dict) -> None:
    """PIN_ROLE=critic-eval: score every item of a sealed set across PIN_CRITIC_SEEDS, streaming each seed's call
    journal as CALL_CHECKPOINT lines and tarring critic-eval-out.tar in finally."""

    global _TOKEN, CUT_AT
    held_token = take_token()
    try:
        check_token_clean(result)
        if not os.environ.get("PIN_GGUF_REPO"):
            held_token = None
        rs = result["role_set"] = role_set_env(os.environ["PIN_ROLESET_COMMIT"])
        result["rnd"] = install_rnd(os.environ["PIN_RND_COMMIT"])
        critic = (
            os.environ.get("PIN_CRITIC_EVAL_CRITIC")
            or os.environ.get("PIN_CRITIC_MODEL")
            or os.environ["PIN_RESCORE_CRITIC"]
        )
        raw_seeds = os.environ.get("PIN_CRITIC_SEEDS") or os.environ.get("PIN_CRITIC_EVAL_SEEDS") or "0"
        seeds = [int(x.strip()) for x in raw_seeds.split(",") if x.strip()]
        flaw_prompt = (
            os.environ.get("PIN_CRITIC_FLAW_PROMPT")
            or os.environ.get("PIN_CRITIC_EVAL_FLAW_PROMPT")
            or "old"
        )
        if flaw_prompt not in ("old", "new", "trackb"):
            raise RuntimeError(f"REFUSED: PIN_CRITIC_FLAW_PROMPT must be 'old', 'new' or 'trackb', got {flaw_prompt!r}")
        raw_flaw_votes = (
            os.environ.get("PIN_CRITIC_FLAW_VOTES")
            or os.environ.get("PIN_CRITIC_EVAL_FLAW_VOTES")
            or "1"
        ).strip()
        try:
            flaw_votes = int(raw_flaw_votes)
        except ValueError as e:
            raise RuntimeError(f"REFUSED: PIN_CRITIC_FLAW_VOTES must be 1 or 3, got {raw_flaw_votes!r}") from e
        if flaw_votes not in (1, 3):
            raise RuntimeError(f"REFUSED: PIN_CRITIC_FLAW_VOTES must be 1 or 3, got {flaw_votes!r}")
        verdict_from_rel = (os.environ.get("PIN_CRITIC_EVAL_VERDICT_FROM") or "").strip() or None
        by_name = {m["served_as"]: m for m in MODELS}
        if critic not in by_name:
            raise RuntimeError(f"{critic} isn't in pin_window.MODELS")
        in_root, items_path, _ = critic_eval_inputs(result)
        verdict_from_path = (in_root / verdict_from_rel) if verdict_from_rel else None
        resumed = critic_eval_resume(critic, seeds, result)
        if resumed:
            result["critic_eval_resumed"] = {str(s): str(p) for s, p in resumed.items()}
        pack = CONTENT / "pack"
        if not (pack / "pack.json").exists():
            pack.mkdir(parents=True, exist_ok=True)
            sh(f"tar -xf {CONTENT / 'pack.tar'} -C {pack}")
        cards_file = CONTENT / "card-values.json"
        if os.environ.get("PIN_GGUF_REPO"):
            _TOKEN, held_token = held_token, None
        prefetch_hosted([critic], by_name, result)
        check_token_clean(result)
    finally:
        held_token = None
        _TOKEN = None

    out = WORK / "critic-eval"
    out.mkdir(parents=True, exist_ok=True)
    srv = None
    try:
        pool = Pool(built, result, {}, cards_file, WORK / "cards-j1")
        m = pool.get(critic)
        result["critic"] = {"served_as": m["served_as"], "gguf_sha256": m["gguf_sha256"]}
        check_deadline("the critic-eval")
        srv = pool.serve(critic)
        with srv:
            proxy = pool.proxy(critic)
            ce_meta: dict = {"seeds": seeds, "flaw_prompt": flaw_prompt, "flaw_votes": flaw_votes, "runs": []}
            if verdict_from_path is not None:
                ce_meta["verdict_from"] = str(verdict_from_path)
            result["critic_eval"] = ce_meta
            for seed in seeds:
                check_deadline(f"critic-eval seed {seed}")
                seed_out = out / f"s{seed}"
                seed_out.mkdir(parents=True, exist_ok=True)
                journal = seed_out / "calls-journal.jsonl"
                label = f"critic-eval-s{seed}"
                px, endpoint = None, f"http://127.0.0.1:{PILOT_PORT}/v1"
                if proxy is not None:
                    from mcptoolshop_rnd.localbuilders.proxy import LoggingProxyServer, ProxyState  # noqa: PLC0415

                    state = ProxyState(
                        upstream_url=f"http://127.0.0.1:{PILOT_PORT}",
                        config=proxy["config"],
                        candidate=proxy["candidate"],
                        generations_path=seed_out / "generations.jsonl",
                        max_calls=10**6,
                        wall_limit_s=86400,
                    )
                    px = LoggingProxyServer(state).start()
                    endpoint = px.openai_base_url
                cmd = [
                    rs["role_set"], "critic-eval",
                    "--items", str(items_path),
                    "--pack", str(pack),
                    "--writers", f"{rs['src']}/writers.json",
                    "--cards", str(cards_file),
                    "--critic-endpoint", endpoint,
                    "--critic-model", m["served_as"],
                    "--critic-gguf-sha256", m["gguf_sha256"],
                    "--critic-seed", str(seed),
                    "--flaw-prompt", flaw_prompt,
                    "--flaw-votes", str(flaw_votes),
                    "--call-journal", str(journal),
                    "--out", str(seed_out),
                ]
                if verdict_from_path is not None:
                    cmd += ["--verdict-from", str(verdict_from_path)]
                if seed in resumed:
                    cmd += ["--reuse-calls", str(resumed[seed])]
                stop = threading.Event()
                tail = threading.Thread(target=stream_calls, args=(journal, m["served_as"], label, stop), daemon=True)
                tail.start()
                left = deadline_left()
                t0 = time.time()
                try:
                    try:
                        run = run_proc(
                            cmd,
                            timeout=6 * 3600 if left is None else min(6 * 3600, max(left, 1)),
                            cwd=rs["src"],
                            term_first=left is not None,
                        )
                    except subprocess.TimeoutExpired:
                        if left is None or left >= 6 * 3600:
                            raise
                        CUT_AT = {
                            "wave": "critic-eval",
                            "seed": seed,
                            "out": str(seed_out),
                            "arm": m["served_as"],
                            "label": label,
                            "journal": str(journal),
                            "seconds": round(time.time() - t0),
                            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        }
                        say(f"CUT AT THE DEADLINE {m['served_as']} {label} after {CUT_AT['seconds']} s; finished calls are in {journal.name}")
                        raise DeadlineReached(f"the critic-eval seed {seed} was cut at the window's deadline") from None
                finally:
                    stop.set()
                    tail.join(timeout=60)
                    if px is not None:
                        px.close()
                (seed_out / "critic-eval.log").write_text(run.stdout + "\n--- stderr ---\n" + run.stderr, encoding="utf-8")
                if run.returncode != 0:
                    raise RuntimeError(f"role-set critic-eval (seed {seed}) exited {run.returncode}: {(run.stderr or run.stdout)[-1500:]}")
                report = json.loads((seed_out / "critic-eval-report.json").read_text(encoding="utf-8"))
                result["critic_card"] = report.get("critic_card")
                if not result["critic_card"] and not stub_mode():
                    raise RuntimeError("the critic-eval report has no critic_card: no number from it may be read (R&D)")
                run_entry = {
                    "seed": seed,
                    "flaw_prompt": report.get("flaw_prompt") or flaw_prompt,
                    "flaw_votes": report.get("flaw_votes") or flaw_votes,
                    "exit": run.returncode,
                    "seconds": round(time.time() - t0),
                    "items_scored": report.get("items_scored"),
                    "skipped_count": report.get("skipped_count"),
                    "calls": report.get("calls"),
                }
                if report.get("verdict_from_sha256"):
                    run_entry["verdict_from_sha256"] = report["verdict_from_sha256"]
                    run_entry["verdict_reused_count"] = report.get("verdict_reused_count")
                result["critic_eval"]["runs"].append(run_entry)
        result["model_errors"] = 0
    finally:
        if srv is not None and getattr(srv, "log", None) and Path(srv.log).exists():
            shutil.copy(srv.log, out / "critic-server.log")
        try:
            sh(f"tar -C {WORK} -cf {WORK}/critic-eval-out.tar critic-eval")
        except Exception as e:  # noqa: BLE001 - recorded; the result still prints
            result["tar_error"] = str(e)[:300]


# ---- PIN_ROLE=offrig-bench (R&D, 2026-10-23): cross-rig baseline (offrig bench30 at batch 1 through Ollama 0.40.4) ------
OLLAMA_VERSION = "0.40.4"
OLLAMA_SHA256 = "15f31e9576d15a481e099b4145dba6e9f8c1f295927d6b8895300ae30ed3740a"  # ollama-linux-amd64.tar.zst (v0.40.4)
OLLAMA_URL = f"https://github.com/ollama/ollama/releases/download/v{OLLAMA_VERSION}/ollama-linux-amd64.tar.zst"
OLLAMA_PORT = int(os.environ.get("PIN_OLLAMA_PORT") or 11434)
# Colab's image (release-colab-external-images_20261020-060116_RC00) has no system `zstd` binary (windows 16t and 16u),
# so `.tar.zst` archives are decompressed in Python with a pinned `zstandard` wheel (`--require-hashes`).
ZSTANDARD_VERSION = "0.25.0"
ZSTANDARD_WHEEL_SHA256S: dict[str, str] = {
    "cp310": "4b6d83057e713ff235a12e73916b6d356e3084fd3d14ced499d84240f3eecee0",
    "cp311": "9300d02ea7c6506f00e627e287e0492a5eb0371ec1670ae852fefffa6164b072",
    "cp312": "5a56ba0db2d244117ed744dfa8f6f5b366e14148e00de44723413b2f3938a902",
    "cp313": "8e735494da3db08694d26480f1493ad2cf86e99bdd53e8e9771b2752a5c0246a",
    "cp314": "e09bb6252b6476d8d56100e8147b803befa9a12cea144bbe629dd508800d1ad0",
}
ZSTANDARD_WHEEL_SHA256 = ZSTANDARD_WHEEL_SHA256S["cp313"]  # Colab's VM Python (3.13, manylinux2014_x86_64)
STUB_OLLAMA_SHA256 = "e55a61f612f162db35b49cc1275d6d8dd942edb4e07572e572dd28e9b868b7e2"  # tests/fixtures/stub-ollama.tar.zst
STUB_OLLAMA_TAR_ZST_B64 = (
    "KLUv/WAAJ3UEACJGFBiAJ24Ab//acH6e42dgqcWOQyEdnnOBAhnmqDF71UE7tJKO1plYES3XcmzG"
    "xbgNwoDICgJBis7FKQGBn4FaH3c8wc/5b///u+f/DxWQUiwXpxUAvA96SIACAMEavCjgrxWMAVUD"
    "GQN7CJhngJyNI1BBAih0ILCCALXEG9INQDpuhZvaADhWLmMT8AI="
)
OFFRIG_MODEL = "gemma4:31b"
OFFRIG_MODEL_DIGEST = "6316f0629137b426c9d9b853ffc4c8209589f30ee39aebede6285096c0ff47e7"
OFFRIG_GOLD_SHA256 = "0c88ec0f915c911a99d3fcf2d2e91104c0c65b0f279656f27afa2244099ce547"
OFFRIG_NUM_CTX = 8192
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


def _require_sha256(name: str, raw: str | None) -> str:
    val = (raw or "").strip().lower().removeprefix("sha256:")
    if not HEX64_RE.fullmatch(val):
        raise RuntimeError(f"REFUSED: {name} must be a 64-char hex sha256, got {raw!r}")
    return val


def offrig_bench_inputs(result: dict) -> tuple[Path, Path, dict]:
    """Read-only, sha-checked uploaded offrig Linux binary and gold jsonl for PIN_ROLE=offrig-bench."""

    offrig_sha = _require_sha256("PIN_OFFRIG_SHA256", os.environ.get("PIN_OFFRIG_SHA256"))
    raw_bin = (os.environ.get("PIN_OFFRIG_BIN") or "").strip()
    bin_src = Path(raw_bin) if raw_bin else (CONTENT / "offrig")
    if not bin_src.exists() and str(bin_src).startswith("/content/"):
        bin_src = CONTENT / str(bin_src).removeprefix("/content/")
    parts = sorted(CONTENT.glob("offrig.gz.part*"))
    if not bin_src.is_file() and parts:
        sh(f"cat {' '.join(str(p) for p in parts)} | gunzip -c > {bin_src}")
    if not bin_src.is_file():
        raise RuntimeError(f"REFUSED: offrig binary not found at {bin_src}")
    raw_gold = (os.environ.get("PIN_OFFRIG_GOLD") or "").strip()
    gold_src = Path(raw_gold) if raw_gold else (CONTENT / "bench30.jsonl")
    if not gold_src.exists() and str(gold_src).startswith("/content/"):
        gold_src = CONTENT / str(gold_src).removeprefix("/content/")
    raw_gold_sha = os.environ.get("PIN_OFFRIG_GOLD_SHA256")
    gold_sha = _require_sha256(
        "PIN_OFFRIG_GOLD_SHA256",
        OFFRIG_GOLD_SHA256 if (raw_gold_sha is None and gold_src.name == "bench30.jsonl") else raw_gold_sha,
    )
    if not gold_src.is_file():
        raise RuntimeError(f"REFUSED: offrig gold jsonl not found at {gold_src}")
    got_bin_sha = sha256(bin_src).lower()
    if got_bin_sha != offrig_sha:
        raise RuntimeError(f"REFUSED: offrig binary sha256 mismatch: {got_bin_sha[:12]} != PIN_OFFRIG_SHA256 {offrig_sha[:12]}")
    got_gold_sha = sha256(gold_src).lower()
    if got_gold_sha != gold_sha:
        raise RuntimeError(f"REFUSED: offrig gold {gold_src.name} sha256 mismatch: {got_gold_sha[:12]} != PIN_OFFRIG_GOLD_SHA256 {gold_sha[:12]}")
    root = WORK / "offrig-bench-in"
    if root.exists():
        for p in sorted(root.rglob("*")):
            os.chmod(p, 0o700)
        os.chmod(root, 0o700)
        shutil.rmtree(root)
    root.mkdir(parents=True)
    bin_dst = root / "offrig"
    gold_dst = root / gold_src.name
    shutil.copyfile(bin_src, bin_dst)
    shutil.copyfile(gold_src, gold_dst)
    os.chmod(bin_dst, 0o555)
    os.chmod(gold_dst, 0o444)
    os.chmod(root, 0o555)
    meta = {"offrig_sha256": got_bin_sha, "gold": gold_dst.name, "gold_sha256": got_gold_sha}
    result["offrig_bench_inputs"] = meta
    return bin_dst, gold_dst, meta


def zstandard_wheel_sha256(py_tag: str | None = None) -> str:
    """Return the pinned manylinux x86_64 wheel sha256 of zstandard==ZSTANDARD_VERSION for the VM's Python."""

    tag = py_tag or f"cp{sys.version_info.major}{sys.version_info.minor}"
    if tag == "cp313":
        return _require_sha256("ZSTANDARD_WHEEL_SHA256", ZSTANDARD_WHEEL_SHA256)
    sha = ZSTANDARD_WHEEL_SHA256S.get(tag)
    if not sha:
        raise RuntimeError(f"REFUSED: no pinned zstandard=={ZSTANDARD_VERSION} manylinux x86_64 wheel sha256 for Python {tag}")
    return _require_sha256(f"ZSTANDARD_WHEEL_SHA256S[{tag!r}]", sha)


def ensure_zstandard() -> dict[str, str]:
    """Install and verify `zstandard` at `ZSTANDARD_VERSION` pinned by version and manylinux x86_64 wheel sha256 (`pip --require-hashes`)."""

    import importlib.metadata as _im  # noqa: PLC0415

    wheel_sha = zstandard_wheel_sha256()
    req = WORK / "zstandard-requirements.txt"
    req.parent.mkdir(parents=True, exist_ok=True)
    req.write_text(f"zstandard=={ZSTANDARD_VERSION} --hash=sha256:{wheel_sha}\n", encoding="utf-8", newline="\n")
    try:
        have_ver = _im.version("zstandard")
    except _im.PackageNotFoundError:
        have_ver = None
    if not stub_mode() or have_ver != ZSTANDARD_VERSION:
        sh(
            f"{sys.executable} -m pip install -q --require-hashes --only-binary=:all: --no-deps -r {req}",
            timeout=300,
        )
    import zstandard  # noqa: F401, PLC0415

    got_ver = _im.version("zstandard")
    if got_ver != ZSTANDARD_VERSION:
        raise RuntimeError(f"REFUSED: zstandard version is {got_ver!r}, expected {ZSTANDARD_VERSION!r}")
    return {"version": got_ver, "wheel_sha256": wheel_sha}


def _safe_tar_filter(member, dest_path: str):
    """Reject absolute paths, '..' segments, and links outside dest_path before delegating to tarfile.data_filter."""

    import tarfile as _tarfile  # noqa: PLC0415

    raw_name = member.name
    norm_slashes = raw_name.replace("\\", "/")
    if raw_name.startswith(("/", "\\")) or os.path.isabs(raw_name) or re.match(r"^[A-Za-z]:", raw_name):
        raise RuntimeError(f"REFUSED: absolute path in tar archive: {raw_name!r}")
    parts = [p for p in norm_slashes.split("/") if p]
    if ".." in parts:
        raise RuntimeError(f"REFUSED: '..' segment in tar archive member: {raw_name!r}")
    dest_real = os.path.realpath(dest_path)
    target_real = os.path.realpath(os.path.join(dest_real, *parts))
    if os.path.commonpath([target_real, dest_real]) != dest_real:
        raise RuntimeError(f"REFUSED: tar member escapes destination: {raw_name!r}")
    if member.issym() or member.islnk():
        link = member.linkname
        if link.startswith(("/", "\\")) or os.path.isabs(link) or re.match(r"^[A-Za-z]:", link):
            raise RuntimeError(f"REFUSED: absolute link target in tar member {raw_name!r}: {link!r}")
        link_base = os.path.dirname(target_real) if member.issym() else dest_real
        link_real = os.path.realpath(os.path.join(link_base, link.replace("\\", "/")))
        if os.path.commonpath([link_real, dest_real]) != dest_real:
            raise RuntimeError(f"REFUSED: link outside destination in tar member {raw_name!r} -> {link!r}")
    try:
        return _tarfile.data_filter(member, dest_path)
    except _tarfile.FilterError as e:
        raise RuntimeError(f"REFUSED: unsafe tar member {raw_name!r}: {e}") from e


def _stub_ollama_archive() -> Path:
    """The tiny .tar.zst fixture unpacked in stub mode so a stub window exercises the same extraction code as a paid window."""

    for candidate in (
        _repo_root() / "tests" / "fixtures" / "stub-ollama.tar.zst",
        CONTENT / "role-set" / "tests" / "fixtures" / "stub-ollama.tar.zst",
    ):
        if candidate.is_file():
            return candidate
    import base64 as _b64  # noqa: PLC0415

    out = WORK / "stub-ollama.tar.zst"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(_b64.b64decode(STUB_OLLAMA_TAR_ZST_B64))
    return out


def unpack_ollama_archive(archive: Path, dest: Path) -> dict[str, str] | None:
    """Decompress and extract an Ollama archive into `dest` in Python using `_safe_tar_filter` (streaming `.tar.zst`
    through the pinned `zstandard` wheel without any system `zstd` binary)."""

    import tarfile as _tarfile  # noqa: PLC0415

    dest.mkdir(parents=True, exist_ok=True)
    if archive.name.endswith((".tar.zst", ".zst")):
        zstd_meta = ensure_zstandard()
        import zstandard  # noqa: PLC0415

        dctx = zstandard.ZstdDecompressor()
        with archive.open("rb") as fh, dctx.stream_reader(fh) as reader, _tarfile.open(fileobj=reader, mode="r|") as tf:
            tf.extractall(dest, filter=_safe_tar_filter)
        return zstd_meta
    with _tarfile.open(archive, mode="r:*") as tf:
        tf.extractall(dest, filter=_safe_tar_filter)
    return None


def install_ollama(result: dict) -> Path:
    """Install the pinned Ollama version (0.40.4) verified by sha256 checksum before unpacking or running."""

    ver = (os.environ.get("PIN_OLLAMA_VERSION") or OLLAMA_VERSION).strip()
    if ver != OLLAMA_VERSION:
        raise RuntimeError(f"REFUSED: PIN_OLLAMA_VERSION must be {OLLAMA_VERSION!r}, got {ver!r}")
    raw_sha = os.environ.get("PIN_OLLAMA_SHA256")
    if raw_sha is not None:
        _require_sha256("PIN_OLLAMA_SHA256", raw_sha)
    raw_arch = (os.environ.get("PIN_OLLAMA_ARCHIVE") or "").strip()
    url = None
    using_stub_fixture = False
    if raw_arch:
        archive = Path(raw_arch)
        if not archive.exists() and str(archive).startswith("/content/"):
            archive = CONTENT / str(archive).removeprefix("/content/")
        if not archive.is_file():
            raise RuntimeError(f"REFUSED: PIN_OLLAMA_ARCHIVE not found at {archive}")
    elif (CONTENT / "ollama-linux-amd64.tar.zst").is_file():
        archive = CONTENT / "ollama-linux-amd64.tar.zst"
    elif (CONTENT / "ollama-linux-amd64.tgz").is_file():
        archive = CONTENT / "ollama-linux-amd64.tgz"
    elif stub_mode():
        archive = _stub_ollama_archive()
        using_stub_fixture = True
    else:
        url = (os.environ.get("PIN_OLLAMA_URL") or OLLAMA_URL).strip()
        check_deadline("installing ollama")
        archive = WORK / (Path(url).name or "ollama-linux-amd64.tar.zst")
        say(f"downloading ollama {OLLAMA_VERSION} from {url}")
        sh(f"curl -fsSL '{url}' -o {archive}", timeout=900)
    default_sha = STUB_OLLAMA_SHA256 if using_stub_fixture else OLLAMA_SHA256
    expect_sha = _require_sha256("PIN_OLLAMA_SHA256", default_sha if raw_sha is None else raw_sha)
    got_sha = sha256(archive).lower()
    if got_sha != expect_sha:
        if url is not None:
            archive.unlink(missing_ok=True)
        raise RuntimeError(f"REFUSED: ollama archive sha256 mismatch: {got_sha[:12]} != PIN_OLLAMA_SHA256 {expect_sha[:12]}")
    dest = WORK / "ollama-install"
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)
    zstd_meta: dict[str, str] | None = None
    if archive.name.endswith((".tar.zst", ".zst", ".tgz", ".tar.gz", ".tar")):
        zstd_meta = unpack_ollama_archive(archive, dest)
        ollama_bin = dest / "bin" / "ollama" if (dest / "bin" / "ollama").exists() else dest / "ollama"
    else:
        (dest / "bin").mkdir(parents=True, exist_ok=True)
        ollama_bin = dest / "bin" / "ollama"
        shutil.copyfile(archive, ollama_bin)
    if not ollama_bin.is_file():
        raise RuntimeError(f"ollama binary not found under {dest} after unpacking {archive.name}")
    ollama_bin.chmod(0o755)
    if url is not None:
        archive.unlink(missing_ok=True)
    if stub_mode():
        bin_text = ollama_bin.read_text(encoding="utf-8", errors="replace")
        if OLLAMA_VERSION not in bin_text:
            raise RuntimeError(f"REFUSED: stub ollama binary is {bin_text.strip()!r}, expected {OLLAMA_VERSION}")
        ver_out = f"ollama version is {OLLAMA_VERSION}"
    else:
        ver_out = sh(f"{ollama_bin} --version", timeout=30)
        if OLLAMA_VERSION not in ver_out:
            raise RuntimeError(f"REFUSED: ollama --version is {ver_out!r}, expected {OLLAMA_VERSION}")
    ollama_rec: dict = {
        "version": OLLAMA_VERSION,
        "sha256": got_sha,
        "bin": str(ollama_bin),
        "version_line": ver_out,
    }
    if zstd_meta is not None:
        ollama_rec["zstandard"] = zstd_meta
        ollama_rec["zstandard_version"] = zstd_meta["version"]
        ollama_rec["zstandard_wheel_sha256"] = zstd_meta["wheel_sha256"]
    if stub_mode():
        ollama_rec["stub"] = True
    result["ollama"] = ollama_rec
    return ollama_bin


def _start_stub_ollama(model: str, digest: str, port: int, version: str = OLLAMA_VERSION):
    """Fake Ollama server on `port` for PIN_STUB=1: uses StubLlama when tests/stub_llama.py is present, or an inline
    loopback HTTP server with the same /api/* routes when pin_window.py runs standalone on a CPU VM."""

    try:
        return _start_stub(model, port, ollama_model=model, ollama_digest=digest, ollama_version=version)
    except ImportError:
        pass
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # noqa: PLC0415

    class _Reuse(ThreadingHTTPServer):
        allow_reuse_address = True

    class _InlineStubOllama:
        def __init__(self):
            self.ollama_model = model
            self.ollama_digest = digest.removeprefix("sha256:")
            self.ollama_version = version
            self.ollama_loaded = False
            self.ollama_size = 20281829620
            self.ollama_size_vram = 20281829620
            self.ollama_context_length = 4096
            self.ollama_requests: list[tuple[str, dict]] = []
            self.call_fn = _slow_stub(lambda *_a, **_k: None)
            stub = self

            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *a):
                    pass

                def _send_json(self, status: int, payload: dict) -> None:
                    data = json.dumps(payload).encode("utf-8")
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)

                def do_GET(self):  # noqa: N802
                    if self.path == "/api/version":
                        self._send_json(200, {"version": stub.ollama_version})
                    elif self.path == "/api/tags":
                        self._send_json(
                            200,
                            {"models": [{"name": stub.ollama_model, "model": stub.ollama_model, "size": stub.ollama_size, "digest": stub.ollama_digest}]},
                        )
                    elif self.path == "/api/ps":
                        ms = (
                            [{"name": stub.ollama_model, "model": stub.ollama_model, "size": stub.ollama_size, "digest": stub.ollama_digest, "size_vram": stub.ollama_size_vram, "context_length": stub.ollama_context_length}]
                            if stub.ollama_loaded
                            else []
                        )
                        self._send_json(200, {"models": ms})
                    else:
                        self.send_response(404)
                        self.end_headers()

                def do_POST(self):  # noqa: N802
                    length = int(self.headers.get("Content-Length") or 0)
                    body = json.loads(self.rfile.read(length)) if length else {}
                    stub.ollama_requests.append((self.path, body))
                    if self.path == "/api/show":
                        self._send_json(200, {"capabilities": ["completion", "thinking"], "model_info": {"gemma4.context_length": 262144}})
                    elif self.path == "/api/generate":
                        opts = body.get("options") or {}
                        if opts.get("num_ctx"):
                            stub.ollama_context_length = int(opts["num_ctx"])
                        stub.ollama_loaded = body.get("keep_alive") not in (0, "0", "0s")
                        self._send_json(200, {"model": body.get("model") or stub.ollama_model, "response": "", "done": True, "done_reason": "load" if stub.ollama_loaded else "unload"})
                    elif self.path == "/api/chat":
                        stub.ollama_loaded = True
                        opts = body.get("options") or {}
                        if opts.get("num_ctx"):
                            stub.ollama_context_length = int(opts["num_ctx"])
                        if opts.get("num_predict") == 1:
                            self._send_json(200, {"model": body.get("model") or stub.ollama_model, "message": {"role": "assistant", "content": "ok"}, "done": True, "done_reason": "length", "prompt_eval_count": 30, "eval_count": 1, "total_duration": 5})
                            return
                        stub.call_fn(body.get("messages") or [], seed=opts.get("seed") or 0)
                        content = json.dumps({"reasoning": "stub check", "verdict": "unsupported", "evidence_quote": "", "evidence_source": ""})
                        self._send_json(200, {"model": body.get("model") or stub.ollama_model, "message": {"role": "assistant", "content": content, "thinking": "stub thinking"}, "done": True, "done_reason": "stop", "prompt_eval_count": 30, "eval_count": 3, "total_duration": 5})
                    else:
                        self.send_response(404)
                        self.end_headers()

            self.server = _Reuse(("127.0.0.1", port), Handler)
            self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}"
            threading.Thread(target=self.server.serve_forever, daemon=True).start()

        def close(self) -> None:
            self.server.shutdown()
            self.server.server_close()

    return _InlineStubOllama()


class OllamaServed:
    """One `ollama serve` instance with A1's environment (OLLAMA_NUM_PARALLEL=1, OLLAMA_KV_CACHE_TYPE=f16;
    OLLAMA_FLASH_ATTENTION left unset), or StubLlama in stub mode."""

    def __init__(self, ollama_bin: Path | None, model: str, digest: str, port: int):
        self.ollama_bin, self.model, self.digest, self.port = ollama_bin, model, digest, port
        self.proc = self.log_fh = self.stub = self.log = None

    def __enter__(self):
        if stub_mode():
            self.stub = _start_stub_ollama(self.model, self.digest, self.port, OLLAMA_VERSION)
            return self
        assert self.ollama_bin is not None
        lib_dirs = [p for p in [self.ollama_bin.parent.parent / "lib" / "ollama", *_cuda_lib_dirs()] if p.is_dir()]
        ld_path = ":".join(str(p) for p in lib_dirs) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        models_dir = WORK / "ollama-models"
        models_dir.mkdir(parents=True, exist_ok=True)
        env = {
            **{k: v for k, v in os.environ.items() if k != "OLLAMA_FLASH_ATTENTION"},
            "OLLAMA_HOST": f"127.0.0.1:{self.port}",
            "OLLAMA_NUM_PARALLEL": "1",
            "OLLAMA_KV_CACHE_TYPE": "f16",
            "OLLAMA_MODELS": str(models_dir),
            "LD_LIBRARY_PATH": ld_path,
        }
        server_log = WORK / "ollama-serve.log"
        self.log = server_log
        self.log_fh = open(server_log, "w", encoding="utf-8")  # noqa: SIM115 - closed by stop_server
        self.proc = subprocess.Popen(
            [str(self.ollama_bin), "serve"],
            stdout=self.log_fh,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            start_new_session=True,
        )
        ver_data = None
        for _ in range(60):
            if self.proc.poll() is not None:
                self.log_fh.close()
                raise RuntimeError(
                    f"ollama serve exited ({self.proc.returncode}): "
                    + server_log.read_text(encoding="utf-8", errors="replace")[-800:]
                )
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/api/version", timeout=5) as r:  # noqa: S310 - loopback
                    if r.status == 200:
                        ver_data = json.loads(r.read())
                        break
            except (OSError, ValueError):
                time.sleep(1)
        if ver_data is None:
            stop_server(self.proc, self.log_fh)
            raise RuntimeError("ollama serve never came up: " + server_log.read_text(encoding="utf-8", errors="replace")[-800:])
        if ver_data.get("version") != OLLAMA_VERSION:
            stop_server(self.proc, self.log_fh)
            raise RuntimeError(f"REFUSED: ollama /api/version is {ver_data.get('version')!r}, not {OLLAMA_VERSION!r}")
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/api/ps", timeout=10) as r:  # noqa: S310 - loopback
            ps0 = json.loads(r.read())
        if ps0.get("models"):
            stop_server(self.proc, self.log_fh)
            raise RuntimeError(f"REFUSED: ollama card not empty at start: {ps0.get('models')}")
        return self

    def __exit__(self, *exc):
        if self.stub is not None:
            self.stub.close()
            self.stub = None
            return
        if self.proc is not None and self.log_fh is not None:
            stop_server(self.proc, self.log_fh)


def pull_ollama_model(ollama_bin: Path | None, model: str, digest: str, port: int, result: dict) -> dict:
    """Pull `model` through Ollama and verify that its digest in /api/tags equals `digest`."""

    check_deadline(f"pulling {model}")
    if not stub_mode():
        assert ollama_bin is not None
        left = deadline_left()
        env = {**os.environ, "OLLAMA_HOST": f"127.0.0.1:{port}"}
        try:
            r = run_proc(
                [str(ollama_bin), "pull", model],
                env=env,
                timeout=1800 if left is None else min(1800, max(left, 1)),
                term_first=left is not None,
            )
        except subprocess.TimeoutExpired:
            if left is None or left >= 1800:
                raise
            raise DeadlineReached(f"ollama pull {model} was cut at the window's deadline") from None
        if r.returncode != 0:
            raise RuntimeError(f"ollama pull {model} exited {r.returncode}: {(r.stderr or r.stdout)[-1500:]}")
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/tags", timeout=30) as resp:  # noqa: S310 - loopback
        tags = json.loads(resp.read())
    candidates = {model, f"{model}:latest"}
    entry = next(
        (m for m in (tags.get("models") or []) if m.get("name") in candidates or m.get("model") in candidates),
        None,
    )
    if entry is None:
        raise RuntimeError(f"REFUSED: model {model!r} not in /api/tags after pull")
    got_digest = (entry.get("digest") or "").strip().lower().removeprefix("sha256:")
    if got_digest != digest:
        raise RuntimeError(f"REFUSED: model {model} digest {got_digest[:12]} != PIN_OFFRIG_MODEL_DIGEST {digest[:12]}")
    info = {"model": model, "digest": got_digest, "size": entry.get("size")}
    result["offrig_model"] = info
    return info


def warmup_ollama_model(
    model: str, digest: str, port: int, base: Path, result: dict, num_ctx: int = OFFRIG_NUM_CTX
) -> dict:
    """Warm-up load `model` via /api/generate at `num_ctx` before offrig starts (so a cold load never hits offrig's
    request timeout or reloads at a different context), poll /api/ps until loaded, write ps.json, and verify digest,
    context_length, and full GPU offload (size_vram == size)."""

    check_deadline(f"warming up {model}")
    left = deadline_left()
    req_timeout = 600 if left is None else min(600, max(int(left), 1))
    body = json.dumps(
        {"model": model, "prompt": "", "keep_alive": "60m", "stream": False, "options": {"num_ctx": num_ctx}}
    ).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=req_timeout) as resp:  # noqa: S310 - loopback
        resp.read()
    ps: dict = {"models": []}
    poll_sleep = 0.05 if stub_mode() else 2.0
    for _ in range(120):
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/ps", timeout=15) as resp:  # noqa: S310 - loopback
            ps = json.loads(resp.read())
        if len(ps.get("models") or []) == 1:
            break
        time.sleep(poll_sleep)
    ps_text = json.dumps(ps, indent=1) + "\n"
    (base / "ps.json").write_text(ps_text, encoding="utf-8")
    (WORK / "ps.json").write_text(ps_text, encoding="utf-8")
    result["ps"] = ps
    models = ps.get("models") or []
    if len(models) != 1:
        raise RuntimeError(f"REFUSED: {model} did not load in /api/ps after warm-up: {ps}")
    m_ps = models[0]
    got_digest = (m_ps.get("digest") or "").strip().lower().removeprefix("sha256:")
    if got_digest != digest:
        raise RuntimeError(f"REFUSED: /api/ps model digest {got_digest[:12]} != PIN_OFFRIG_MODEL_DIGEST {digest[:12]}")
    ctx_len = int(m_ps.get("context_length") or 0)
    if ctx_len != num_ctx:
        raise RuntimeError(f"REFUSED: {model} context_length in /api/ps is {ctx_len} != {num_ctx}")
    size = int(m_ps.get("size") or 0)
    size_vram = int(m_ps.get("size_vram") or 0)
    if size <= 0 or size_vram < size:
        raise RuntimeError(f"REFUSED: {model} offloaded in /api/ps: size_vram {size_vram} < size {size}")
    return ps


def _query_smi_csv() -> str:
    """One `nvidia-smi` CSV query line (`timestamp,memory.used,utilization.gpu,temperature.gpu,power.draw`)."""

    if stub_mode():
        return f"{time.strftime('%Y/%m/%d %H:%M:%S.000', time.gmtime())}, 0 MiB, 0 %, 0, 0.00 W\n"
    return subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,memory.used,utilization.gpu,temperature.gpu,power.draw",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    ).stdout


SMI_INTERVAL_S = 2.0


def sample_smi_every_2s(smi_csv: Path, stop: threading.Event, stats: dict) -> None:
    """Sample `nvidia-smi` every 2 s into `smi_csv` and track peak VRAM in MiB."""

    while not stop.is_set():
        try:
            out = _query_smi_csv()
            if out and out.strip():
                with smi_csv.open("a", encoding="utf-8") as f:
                    f.write(out if out.endswith("\n") else out + "\n")
                for line in out.splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 2 and parts[1]:
                        mib = int(float(parts[1].split()[0]))
                        stats["peak_vram_mib"] = max(stats["peak_vram_mib"], mib)
                        stats["samples"] += 1
        except (subprocess.SubprocessError, OSError, ValueError):
            pass
        stop.wait(SMI_INTERVAL_S)


def offrig_bench_window(result: dict) -> None:
    """PIN_ROLE=offrig-bench (R&D, 2026-10-23): install pinned Ollama 0.40.4 by checksum, pull model by digest,
    warm-up load and verify /api/ps, sample nvidia-smi every 2 s, and run
    `offrig verify calibrate <gold> --model <m> --think on --structured on --split tune --jobs 1`."""

    global CUT_AT
    # No HF token is needed for Ollama's registry; drop any token and verify the five-place residue check.
    take_token()
    check_token_clean(result)
    ver = (os.environ.get("PIN_OLLAMA_VERSION") or OLLAMA_VERSION).strip()
    if ver != OLLAMA_VERSION:
        raise RuntimeError(f"REFUSED: PIN_OLLAMA_VERSION must be {OLLAMA_VERSION!r}, got {ver!r}")
    raw_ollama_sha = os.environ.get("PIN_OLLAMA_SHA256")
    if raw_ollama_sha is not None:
        _require_sha256("PIN_OLLAMA_SHA256", raw_ollama_sha)
    model = (os.environ.get("PIN_OFFRIG_MODEL") or OFFRIG_MODEL).strip()
    if not model:
        raise RuntimeError("REFUSED: PIN_OFFRIG_MODEL is empty")
    raw_digest = os.environ.get("PIN_OFFRIG_MODEL_DIGEST")
    digest = _require_sha256(
        "PIN_OFFRIG_MODEL_DIGEST",
        OFFRIG_MODEL_DIGEST if (raw_digest is None and model == OFFRIG_MODEL) else raw_digest,
    )
    offrig_bin, gold_path, inputs = offrig_bench_inputs(result)
    base = WORK / "offrig-bench"
    cal_out = base / "out"
    proj_dir = base / "project"
    smi_csv = base / "smi.csv"
    base.mkdir(parents=True, exist_ok=True)
    proj_dir.mkdir(parents=True, exist_ok=True)
    ollama_bin = install_ollama(result)
    port = int(os.environ.get("PIN_OLLAMA_PORT") or OLLAMA_PORT)
    srv = OllamaServed(ollama_bin, model, digest, port)
    try:
        with srv:
            pull_ollama_model(ollama_bin, model, digest, port, result)
            ps_warm = warmup_ollama_model(model, digest, port, base, result)
            check_deadline("offrig verify calibrate")
            cmd = [
                str(offrig_bin),
                "verify",
                "calibrate",
                str(gold_path),
                "--model",
                model,
                "--think",
                "on",
                "--structured",
                "on",
                "--split",
                "tune",
                "--jobs",
                "1",
                "--project",
                str(proj_dir),
                "--out",
                str(cal_out),
            ]
            if port != 11434:
                cmd += ["--url", f"http://127.0.0.1:{port}"]
            stop = threading.Event()
            smi_stats = {"samples": 0, "peak_vram_mib": 0}
            smi_thread = threading.Thread(target=sample_smi_every_2s, args=(smi_csv, stop, smi_stats), daemon=True)
            smi_thread.start()
            left = deadline_left()
            t0 = time.time()
            ps_end: dict | None = None
            try:
                try:
                    run = run_proc(
                        cmd,
                        timeout=4 * 3600 if left is None else min(4 * 3600, max(left, 1)),
                        cwd=str(base),
                        term_first=left is not None,
                    )
                except subprocess.TimeoutExpired:
                    if left is None or left >= 4 * 3600:
                        raise
                    CUT_AT = {
                        "wave": "offrig-bench",
                        "out": str(cal_out),
                        "model": model,
                        "seconds": round(time.time() - t0),
                        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }
                    say(f"CUT AT THE DEADLINE offrig-bench {model} after {CUT_AT['seconds']} s")
                    raise DeadlineReached("the offrig-bench calibrate run was cut at the window's deadline") from None
            finally:
                stop.set()
                smi_thread.join(timeout=15)
                result["smi"] = {
                    "interval_s": SMI_INTERVAL_S,
                    "samples": smi_stats["samples"],
                    "peak_vram_mib": smi_stats["peak_vram_mib"],
                    "csv": str(smi_csv),
                }
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/ps", timeout=10) as resp:  # noqa: S310 - loopback
                        ps_end = json.loads(resp.read())
                    (base / "ps-end.json").write_text(json.dumps(ps_end, indent=1) + "\n", encoding="utf-8")
                    result["ps_end"] = ps_end
                    unload_req = urllib.request.Request(
                        f"http://127.0.0.1:{port}/api/generate",
                        data=json.dumps({"model": model, "keep_alive": 0}).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                    )
                    with urllib.request.urlopen(unload_req, timeout=10) as resp:  # noqa: S310 - loopback
                        resp.read()
                except (OSError, ValueError):
                    pass
            (base / "offrig.log").write_text(run.stdout + "\n--- stderr ---\n" + run.stderr, encoding="utf-8")
            if run.returncode != 0:
                raise RuntimeError(f"offrig verify calibrate exited {run.returncode}: {(run.stderr or run.stdout)[-1500:]}")
            warm_m = (ps_warm.get("models") or [{}])[0]
            end_models = (ps_end or {}).get("models") or []
            if len(end_models) != 1:
                msg = f"REFUSED: {model} did not stay loaded in /api/ps after offrig run: {ps_end}"
                result["error"] = msg
                raise RuntimeError(msg)
            end_m = end_models[0]
            warm_ctx = int(warm_m.get("context_length") or 0)
            end_ctx = int(end_m.get("context_length") or 0)
            if end_ctx != warm_ctx:
                msg = f"REFUSED: /api/ps context_length changed during offrig run: warm-up {warm_ctx} != post-run {end_ctx}"
                result["error"] = msg
                raise RuntimeError(msg)
            warm_vram = int(warm_m.get("size_vram") or 0)
            end_vram = int(end_m.get("size_vram") or 0)
            if end_vram != warm_vram:
                msg = f"REFUSED: /api/ps size_vram changed during offrig run: warm-up {warm_vram} != post-run {end_vram}"
                result["error"] = msg
                raise RuntimeError(msg)
            verdicts_file = cal_out / "verdicts.jsonl"
            verdicts_count = (
                sum(1 for ln in verdicts_file.read_text(encoding="utf-8").splitlines() if ln.strip())
                if verdicts_file.is_file()
                else 0
            )
            metrics_file = cal_out / "metrics.json"
            manifest_file = cal_out / "manifest.json"
            metrics = json.loads(metrics_file.read_text(encoding="utf-8")) if metrics_file.is_file() else None
            if stub_mode():
                for jf in (metrics_file, manifest_file):
                    if jf.is_file():
                        jd = json.loads(jf.read_text(encoding="utf-8"))
                        jd["stub"] = True
                        jf.write_text(json.dumps(jd, indent=1) + "\n", encoding="utf-8")
                if isinstance(metrics, dict):
                    metrics["stub"] = True
            out_tail = run.stdout
            if stub_mode() and not out_tail.startswith("# STUB"):
                out_tail = "# STUB RUN (PIN_STUB=1)\n" + out_tail
            result["offrig_bench"] = {
                "model": model,
                "model_digest": digest,
                "gold": gold_path.name,
                "gold_sha256": inputs["gold_sha256"],
                "offrig_sha256": inputs["offrig_sha256"],
                "exit": run.returncode,
                "seconds": round(time.time() - t0),
                "verdicts_count": verdicts_count,
                "metrics": metrics,
                "stdout_tail": out_tail[-2500:],
                **({"stub": True} if stub_mode() else {}),
            }
            result["model_errors"] = 0
    finally:
        if srv.log is not None and Path(srv.log).exists() and Path(srv.log) != base / "ollama-serve.log":
            shutil.copy(srv.log, base / "ollama-serve.log")
        if (base / "ps.json").exists() and not (WORK / "ps.json").exists():
            shutil.copy(base / "ps.json", WORK / "ps.json")
        try:
            sh(f"tar -C {WORK} -cf {WORK}/offrig-bench-out.tar offrig-bench")
        except Exception as e:  # noqa: BLE001 - recorded; the result still prints
            result["tar_error"] = str(e)[:300]


def resolve_pilot_pool(rs: dict, plan: Path, twins_plan: Path | None = None) -> tuple[Path | None, str | None]:
    """Verify the pool's sha256 before any llama-server starts (Argon F2). When PIN_POOL is unset, defaults to
    Path(rs['src']) / 'roles' / 'verifier' / 'pool.jsonl' if it exists, falling back to CONTENT / 'pool.jsonl'.
    PIN_POOL_SHA256 stays required and checked."""

    pool_env = (os.environ.get("PIN_POOL") or "").strip()
    if pool_env:
        pool_path = Path(pool_env)
    else:
        committed = Path(rs["src"]) / "roles" / "verifier" / "pool.jsonl"
        pool_path = committed if committed.is_file() else (CONTENT / "pool.jsonl")
    expect_sha = (os.environ.get("PIN_POOL_SHA256") or "").strip().lower()
    plans_to_check = [plan]
    if twins_plan is not None and twins_enabled() and twins_plan.is_file():
        plans_to_check.append(twins_plan)
    has_real = any(
        bool(s.get("real_source"))
        for p in plans_to_check
        if p.is_file()
        for s in (json.loads(p.read_text(encoding="utf-8")).get("slots") or [])
    )
    if has_real:
        if not expect_sha:
            raise RuntimeError("REFUSED: PIN_POOL_SHA256 is required when the plan has real_source slots")
        if not pool_path.is_file():
            raise RuntimeError(f"REFUSED: pool file {pool_path} is missing")
    if expect_sha:
        if not pool_path.is_file():
            raise RuntimeError(f"REFUSED: pool file {pool_path} is missing")
        got_sha = sha256(pool_path).lower()
        if got_sha != expect_sha:
            raise RuntimeError(f"REFUSED: pool {pool_path.name} sha256 {got_sha[:12]} != PIN_POOL_SHA256 {expect_sha[:12]}")
        return pool_path, got_sha
    if pool_path.is_file():
        return pool_path, sha256(pool_path).lower()
    return None, None


def pilot_window(built: dict, result: dict) -> None:
    """PIN_ROLE=pilot (R&D, 2026-10-17): the T02 pilot (or the screen's slot list), then its T04 apply twins (the arm's
    accepted rows as --training), for each writer in PIN_PILOT_WRITERS, in that order, each read by its critic
    (PIN_PILOT_CRITIC, or PIN_PILOT_CRITIC_FOR per arm). One A100, one model loaded at a time, swapped by restart; each
    model is set up when first needed; every start re-checks the GGUF's sha256, the offload and the slots."""

    global PILOT_WAVES
    if "PIN_PILOT_WAVES" in os.environ:
        PILOT_WAVES = int(os.environ["PIN_PILOT_WAVES"])
    rs = result["role_set"] = role_set_env(os.environ["PIN_ROLESET_COMMIT"])
    result["rnd"] = install_rnd(os.environ["PIN_RND_COMMIT"])
    writer_names = [x.strip() for x in os.environ["PIN_PILOT_WRITERS"].split(",") if x.strip()]
    by_name = {m["served_as"]: m for m in MODELS}
    result["disk_free_gb_before"] = round(shutil.disk_usage(_disk_target()).free / 1e9, 1)
    machine_file = WORK / "machine.json"
    machine_file.write_text(json.dumps(result["machine"]), encoding="utf-8")
    pack = CONTENT / "pack"
    if not (pack / "pack.json").exists():
        pack.mkdir(parents=True, exist_ok=True)
        sh(f"tar -xf {CONTENT / 'pack.tar'} -C {pack}")
    result["pack_verify"] = sh(f"{rs['role_set']} pack verify --pack {pack}", cwd=rs["src"])[-400:]
    base = WORK / "pilot"
    base.mkdir(exist_ok=True)
    result["resume_unpacked"] = str(unpack_resume() or "")
    result["prior_unpacked"] = str(unpack_resume(CONTENT / "prior.tar") or "")
    plan, twins_plan = base / "pilot-plan.json", base / "twins-plan.json"
    domains = f"{rs['src']}/roles/domains/naics-2022.json"
    result["critique_scope"] = critique_scope()
    lessons = [x.strip() for x in os.environ.get("PIN_PILOT_LESSONS", "").split(",") if x.strip()]
    if lessons:
        lessons_meta = plan_pilot_lessons(rs, lessons, domains, plan, base)
        screened_pilot = screen_plan(plan, "PIN_PILOT_SLOT_IDS")
        kept_count = len(screened_pilot) if screened_pilot is not None else len(json.loads(plan.read_text(encoding="utf-8"))["slots"])
        result["screen"] = {"pilot": screened_pilot, "kept": kept_count, "twins": None, "waves": PILOT_WAVES}
        result["plans"] = {
            "lessons": lessons_meta["lessons"],
            "buffers": lessons_meta["buffers"],
            "seed": lessons_meta["seed"],
            "per_lesson_sha256": lessons_meta["per_lesson_sha256"],
            "pilot_sha256": sha256(plan),
            "twins_sha256": None,
            "exclusion_state_sha256": lessons_meta["exclusion_state_sha256"],
        }
    else:
        if buffers_enabled():
            raise RuntimeError("REFUSED: PIN_PILOT_BUFFERS=1 requires PIN_PILOT_LESSONS=T10")
        pilot_seed = int(os.environ.get("PIN_PILOT_SEED") or "0")
        sh(f"{rs['role_set']} plan --pilot --seed {pilot_seed} --domains {domains} --out {plan}", cwd=rs["src"])
        sh(f"{rs['role_set']} plan --pilot-twins --seed 7 --domains {domains} --out {twins_plan}", cwd=rs["src"])
        screened_pilot = screen_plan(plan, "PIN_PILOT_SLOT_IDS")
        screened_twins = screen_plan(twins_plan, "PIN_PILOT_TWIN_IDS")
        kept_count = len(screened_pilot) if screened_pilot is not None else len(json.loads(plan.read_text(encoding="utf-8"))["slots"])
        result["screen"] = {"pilot": screened_pilot, "kept": kept_count, "twins": screened_twins,
                            "waves": PILOT_WAVES}
        result["plans"] = {"pilot_sha256": sha256(plan), "twins_sha256": sha256(twins_plan),
                           # Made with --domains and no --exclusion-state: both plans and every write load role-set's empty
                           # state, so the shas agree by construction (write refuses a mismatch).
                           "exclusion_state_sha256": json.loads(plan.read_text(encoding="utf-8")).get("exclusion_state_sha256")}

    pool_file, pool_sha = resolve_pilot_pool(rs, plan, twins_plan)
    result["pool_sha256"] = pool_sha
    training_files = training_inputs(result) if plan_has_apply_slots(plan) else []

    policies = dict(x.split("=", 1) for x in os.environ.get("PIN_PILOT_QUANT", "").split(",") if "=" in x)
    writers_path = Path(rs["src"]) / "writers.json"
    result["server_pin"] = server_pin_entry(server_sha256(), writers_path)
    if isinstance(result.get("llama_server"), dict):
        result["llama_server"].update(result["server_pin"])
    listed = json.loads(writers_path.read_text(encoding="utf-8"))
    families = {e["served_as"]: e.get("family") for e in listed["writers"] + listed["critics"]}
    critic_for = critics_for(writer_names, families, waves=PILOT_WAVES)
    result["critic_for"] = critic_for if critic_for else "none (write only)"
    check_pins(list(dict.fromkeys([*writer_names, *critic_for.values()])), listed, policies)
    for name in [*writer_names, *critic_for.values()]:
        if name not in by_name:
            raise RuntimeError(f"{name} isn't in pin_window.MODELS")
    # Critics run at Q8_0 (R&D, 2026-10-17); a writer's quant policy applies to it as a writer only.
    for c in set(critic_for.values()) - set(writer_names):
        policies.pop(c, None)
    cards_file = CONTENT / "card-values.json"
    # Pre-built GGUFs (PIN_GGUF_REPO): all of them before any wave, so the token is gone before a wave runs. Q8_0 only;
    # a Q6_K policy still converts here.
    prefetch_hosted([n for n in dict.fromkeys([*writer_names, *critic_for.values()]) if policies.get(n, "q8") == "q8"],
                    by_name, result)
    pool = Pool(built, result, policies, cards_file, WORK / "cards-j1")
    common = ["--pack", str(pack), "--writers", f"{rs['src']}/writers.json", "--cards", str(cards_file),
              "--llama-server-sha256", server_sha256(), "--machine", str(machine_file), "--partition", PILOT_PARTITION]
    if pool_file is not None:
        common += ["--pool", str(pool_file)]
    if training_files:
        common += ["--training", *(str(f) for f in training_files)]

    # Whatever happens in the waves, the wave directories come back: the tar is made in finally (window 10c lost its
    # result to a late failure, with nothing downloaded).
    result["arms"] = {}
    try:
        t0 = time.time()
        pilot = pilot_batch(rs, common, plan, base / "pilot", writer_names, critic_for, pool, twins_after=twins_enabled())
        result["pilot_seconds"] = round(time.time() - t0)
        twins_writers = {}
        for n, a in pilot.items():
            result["arms"][n] = {"pilot": {"waves": a["waves"], "last": str(a.get("last"))}}
            if a.get("funnel_stop"):
                result["arms"][n]["funnel_stop"] = a["funnel_stop"]
            why = twins_skip_reason(a)
            if why:
                result["arms"][n]["twins"] = {"skipped": why}
            else:
                twins_writers[n] = True
        t1 = time.time()
        twins_list = list(twins_writers)
        for idx, n in enumerate(twins_list):  # each arm's twins train against its own accepted rows, so one arm at a time here
            accepted = Path(pilot[n]["last"]) / "accepted.jsonl"
            tw = pilot_batch(rs, common, twins_plan, base / "twins", [n], critic_for, pool,
                             extra=["--training", str(accepted)], later_arms=twins_list[idx + 1:])
            result["arms"][n]["twins"] = {"waves": tw[n]["waves"], "last": str(tw[n].get("last"))}
        result["twins_seconds"] = round(time.time() - t1)
        for n in writer_names:
            if stub_mode():
                result["arms"][n]["stub"] = True
            for part in ("pilot", "twins"):
                if stub_mode():
                    result["arms"][n][part]["stub"] = True
                last = (result["arms"][n].get(part) or {}).get("last")
                if last and last != "None" and (Path(last) / "batch-report.json").exists():
                    r = run_proc([rs["role_set"], "readout", "--run", last], timeout=600, cwd=rs["src"])
                    result["arms"][n][part]["readout_exit"] = r.returncode
                    out_text = r.stdout
                    if stub_mode():
                        if not out_text.startswith("# STUB"):
                            out_text = "# STUB RUN (PIN_STUB=1)\n" + out_text
                        md_p, js_p = Path(last) / "readout.md", Path(last) / "readout.json"
                        if md_p.exists() and "STUB" not in md_p.read_text(encoding="utf-8"):
                            md_p.write_text("# STUB RUN (PIN_STUB=1)\n" + md_p.read_text(encoding="utf-8"), encoding="utf-8")
                        if js_p.exists():
                            jd = json.loads(js_p.read_text(encoding="utf-8"))
                            jd["stub"] = True
                            js_p.write_text(json.dumps(jd, indent=1), encoding="utf-8")
                    result["arms"][n][part]["readout_tail"] = out_text[-2500:]
                    if r.returncode != 0:
                        result["arms"][n][part]["readout_stderr"] = r.stderr[-2500:]
                elif stub_mode():
                    result["arms"][n][part].setdefault("readout_tail", "# STUB RUN (PIN_STUB=1)")
        result["serves"] = {n: s.starts for n, s in pool.srv.items()}
        result["model_errors"] = sum(1 for n in writer_names for part in ("pilot", "twins")
                                     for w in (result["arms"][n].get(part) or {}).get("waves", []) if w.get("exit"))
    finally:
        try:
            sh(f"tar -C {WORK} -cf {WORK}/pilot-out.tar pilot")
        except Exception as e:  # noqa: BLE001 - recorded; the result still prints
            result["tar_error"] = str(e)[:300]


def runtime() -> dict:
    """PIN_ROLE=smoke: no build. The uploaded window-1 binary, checked by sha256, with the CUDA 13.4 runtime wheels it
    loads; their exact versions are recorded so the library pins can be reproduced."""

    want_sha = server_sha256()
    pin = server_pin_entry(want_sha)
    override = "PIN_SERVER_SHA256" in os.environ
    sh(f"{sys.executable} -m pip install -q 'nvidia-cuda-runtime=={NVCC_VERSION}' 'nvidia-cublas=={CUBLAS_VERSION}'", timeout=1200)
    server, uploaded = WORK / "llama-server", CONTENT / "llama-server"
    parts = sorted(CONTENT.glob("llama-server.gz.part*"))
    if parts and not server.exists():
        # The contents API drops a 166 MB upload (window 2, 2026-10-17), so the binary arrives gzipped in 40 MB parts;
        # the sha256 check below is what proves it whole.
        sh(f"cat {' '.join(str(p) for p in parts)} | gunzip -c > {server}")
    elif uploaded.exists() and not server.exists():
        shutil.move(str(uploaded), server)
    if not server.exists():
        raise RuntimeError(f"upload window 1's llama-server to {uploaded} first")
    server.chmod(0o755)
    got = sha256(server)
    if got != want_sha:
        raise RuntimeError(f"llama-server {got[:12]} isn't pinned {want_sha[:12]}")
    src = WORK / "llama.cpp"
    if not (src / ".git").exists():
        sh(f"git clone -q https://github.com/ggml-org/llama.cpp {src} && git -C {src} checkout -q {LLAMA_CPP_COMMIT}", timeout=900)
    wheels = [x for x in sh(f"{sys.executable} -m pip freeze").splitlines() if x.lower().startswith("nvidia-")]
    for want in (f"nvidia-cublas=={CUBLAS_VERSION}", f"nvidia-cuda-runtime=={NVCC_VERSION}"):
        if want not in wheels:
            raise RuntimeError(f"{want} isn't what pip installed: {[w for w in wheels if 'cublas' in w or 'runtime' in w]}")
    return {"commit": pin["commit"], "sha256": got, "archs": pin["archs"], "sha256_override": override, "path": str(server),
            "src": str(src), "cuda_wheels": wheels, "cuda_toolkit": CUDA_TOOLKIT}


def take_job_lock(work: Path):
    """One job per VM, per window (R&D, 2026-10-17; fixed after Browser Argon's v2 re-review).

    The window re-runs an exec that never reached the kernel, so a late first exec can still arrive. The job takes an
    exclusive flock on <work>/job.lock for its own life and releases it when it ends (main's finally). A second job:
    - finds this window's result already written (pin-result.json carrying this window's PIN_JOB_TOKEN): it stops;
    - finds the lock held: it waits for the first job to end, then stops if that job wrote this window's result, or runs
      if it didn't (the first died without one).
    A pin-result.json from another run, with another token, never makes a real job skip itself. The lock used to live in a
    module global held for the kernel's life; `colab exec -f` re-runs the whole file in the kernel's globals, which
    rebound it to None and released the lock (Argon). Returns the open lock file, or None when this job must stop."""

    import fcntl  # noqa: PLC0415 - Linux, where the window runs

    work.mkdir(parents=True, exist_ok=True)

    def done_here() -> bool:
        token = os.environ.get("PIN_JOB_TOKEN")
        try:
            return bool(token) and json.loads((work / "pin-result.json").read_text(encoding="utf-8")).get("job_token") == token
        except (OSError, ValueError):
            return False

    def note(text: str) -> None:
        with (work / "job.log").open("a", encoding="utf-8") as f:
            f.write(f"{text} {time.strftime('%H:%M:%S')}\n")

    if done_here():
        note("JOB ALREADY DONE: this window's result is written; this exec stops")
        return None
    fh = (work / "job.lock").open("w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        note("JOB ALREADY RUNNING: this exec waits for it")
        fcntl.flock(fh, fcntl.LOCK_EX)  # until the first job ends
        if done_here():
            fh.close()
            note("JOB ALREADY DONE: the first job finished this window; this exec stops")
            return None
        note("JOB ALREADY RUNNING ended with no result for this window: this exec runs the job")
    return fh


def main() -> int:
    global CONTENT, WORK, JOB_LOG, BRIEFS, RESUME_TAR
    if "PIN_CONTENT" in os.environ:
        CONTENT = Path(os.environ["PIN_CONTENT"])
        BRIEFS, RESUME_TAR = CONTENT / "brief-strings.json", CONTENT / "resume.tar"
    if "PIN_WORK" in os.environ or "PIN_CONTENT" in os.environ:
        WORK = Path(os.environ.get("PIN_WORK") or str(CONTENT / "pin"))
        JOB_LOG = WORK / "job.log"
    WORK.mkdir(parents=True, exist_ok=True)
    lock = take_job_lock(WORK)
    if lock is None:
        return 0
    out, err = sys.stdout, sys.stderr
    sys.stdout = Tee(out, JOB_LOG)
    sys.stderr = Tee(err, JOB_LOG)
    try:
        return _main()
    finally:
        sys.stdout, sys.stderr = out, err
        lock.close()  # the job's own life: a later exec in this kernel waits on it, then reads this window's result


def _main() -> int:
    result: dict = {"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "job_token": os.environ.get("PIN_JOB_TOKEN")}
    # The private repo's token: read once, here, before anything else runs (R&D, from Argon). Only a prebuild, or a
    # pilot/critic-eval window with PIN_GGUF_REPO, keeps the value; for any other window it's dropped at once (R&D on 346ea2d).
    role0 = os.environ.get("PIN_ROLE", "all")
    if os.environ.get("PIN_STUB_CALL_S") and not stub_mode():
        # A test knob that slows the stub's calls (R&D on #91): it can never slow a real run.
        result["ok"], result["error"] = False, "REFUSED: PIN_STUB_CALL_S is set without PIN_STUB=1; nothing ran"
        (WORK / "pin-result.json").write_text(json.dumps(result, indent=1))
        print("PIN_RESULT " + json.dumps(result), flush=True)
        return 1
    needs_token = role0 == "prebuild" or (role0 in ("pilot", "critic-eval") and bool(os.environ.get("PIN_GGUF_REPO")))
    if needs_token and not token_file().parent.is_dir():
        result["ok"], result["error"] = False, "REFUSED: no /dev/shm on this VM, where the token file goes; nothing downloaded"
        (WORK / "pin-result.json").write_text(json.dumps(result, indent=1))
        print("PIN_RESULT " + json.dumps(result), flush=True)
        return 1
    read_token_file(needed=needs_token)
    if stub_mode():
        result["stub"] = True
        say("STUB MODE (PIN_STUB=1): mock writer and critic on StubLlama; NOT A REAL RUN")
    try:
        want_server_sha = server_sha256()
        matched_server_pin = server_pin_entry(want_server_sha)
        server_sha_overridden = "PIN_SERVER_SHA256" in os.environ
        result["server_sha256"] = want_server_sha
        result["server_sha256_override"] = server_sha_overridden
        result["server_pin"] = matched_server_pin
        result["machine"] = machine_line(cpu_ok=os.environ.get("PIN_ROLE") in ("convert", "prebuild"))
        say(f"machine: {result['machine']}")
        # Split windows run side by side (Mike, 2026-10-16): PIN_ROLE=build builds the server and pins and smokes
        # the maker GGUFs; PIN_ROLE=convert only converts and hashes (no CUDA build, so no smoke; the build VM's
        # binary serves it in the next window). Unset: everything, on one VM.
        role = os.environ.get("PIN_ROLE", "all")
        result["role"] = role
        if role in ("smoke", "critic", "q6k", "pilot", "rescore", "critic-eval"):
            if stub_mode() and role in ("pilot", "rescore", "critic-eval"):
                result["llama_server"] = built = {"commit": matched_server_pin["commit"], "sha256": want_server_sha,
                                                  "archs": matched_server_pin["archs"], "sha256_override": server_sha_overridden,
                                                  "path": "stub", "stub": True}
            else:
                result["llama_server"] = built = runtime()
                # The libraries first, so they're pinned even if a model fails later in the window.
                result["llama_server"]["libs"] = pin_libs(built["path"])
        elif role in ("convert", "prebuild"):
            src = WORK / "llama.cpp"
            sh(f"git clone -q https://github.com/ggml-org/llama.cpp {src} && git -C {src} checkout -q {LLAMA_CPP_COMMIT}", timeout=900)
            built = {"commit": LLAMA_CPP_COMMIT, "src": str(src)}
        elif role == "offrig-bench":
            built = {}
        else:
            result["llama_server"] = built = build()
        specs = MODELS if role in ("all", "smoke", "critic", "q6k") else [m for m in MODELS if (role == "build") == (m["how"] == "maker_gguf")]
        if role == "prebuild" and len([x for x in os.environ.get("PIN_ONLY", "").split(",") if x.strip()]) != 1:
            raise RuntimeError("PIN_ROLE=prebuild converts one model per window: set PIN_ONLY to exactly one")
        if role == "prebuild":
            unpinned = [m["served_as"] for m in specs if m["served_as"] in os.environ["PIN_ONLY"].split(",") and not m.get("expect_sha256")]
            if unpinned:  # byte-match or don't host means nothing without a pin to match (R&D)
                raise RuntimeError(f"REFUSED: {', '.join(unpinned)} has no expect_sha256; a prebuild needs a pin to match")
        # PIN_ONLY=<served_as,...> runs just those models (window 4: Muse alone, Lightning already pinned three times).
        only = [x.strip() for x in os.environ.get("PIN_ONLY", "").split(",") if x.strip()]
        if only:
            specs = [m for m in specs if m["served_as"] in only]
            if not specs:
                raise RuntimeError(f"PIN_ONLY names no model: {only}")
            result["only"] = only
        if role == "critic":
            critic_window(built, specs, result)
            specs = []
        elif role == "q6k":
            q6k_window(built, specs, result)
            specs = []
        elif role == "pilot":
            pilot_window(built, result)
            specs = []
        elif role == "rescore":
            rescore_window(built, result)
            specs = []
        elif role == "critic-eval":
            critic_eval_window(built, result)
            specs = []
        elif role == "offrig-bench":
            offrig_bench_window(result)
            specs = []
        else:
            result["models"] = []
        for spec in specs:
            if role == "smoke":
                # Each model is independent here: a failure is recorded with its model and the next one still runs.
                try:
                    m = fetch_model(spec, built)
                except DeadlineReached:
                    raise
                except Exception as e:  # noqa: BLE001 - recorded, and the window's result is not ok
                    result["models"].append({"served_as": spec["served_as"], "error": f"{type(e).__name__}: {e}"})
                    result["model_errors"] = result.get("model_errors", 0) + 1
                    continue
            else:
                m = fetch_model(spec, built)
            if spec.get("expect_sha256"):
                m["reproduced"] = m["gguf_sha256"] == spec["expect_sha256"]
                say(f"{m['served_as']}: reproduces its pinned GGUF: {m['reproduced']}")
            if spec.get("expect_sha256") and not m["reproduced"]:
                # A pin that doesn't reproduce never serves: it goes to R&D first (andon).
                m["smoke"] = {"ok": False, "error": f"GGUF {m['gguf_sha256'][:12]} doesn't reproduce the pin {spec['expect_sha256'][:12]}; not served"}
                result["model_errors"] = result.get("model_errors", 0) + 1
            elif role == "prebuild":
                upload_hosted(spec, m, result)
            elif role != "convert":
                try:
                    m["smoke"] = smoke(built, m)
                except DeadlineReached:
                    raise
                except Exception as e:  # noqa: BLE001 - a smoke window records a failed serve and goes on
                    if role != "smoke":
                        raise
                    m["smoke"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                    result["model_errors"] = result.get("model_errors", 0) + 1
                say(f"{m['served_as']}: {m['smoke']}")
            result["models"].append({k: v for k, v in m.items() if k != "path"})
            Path(m["path"]).unlink(missing_ok=True)  # the hash is the record; the file is rebuilt per window
            resources(f"{m['served_as']}: after unlinking {Path(m['path']).name}")
        if role not in ("convert", "prebuild", "smoke", "offrig-bench") and not (stub_mode() and role in ("pilot", "rescore", "critic-eval")):
            # The server binary is the one artifact fetched back: every later window and pod runs this exact file.
            if Path(built["path"]) != WORK / "llama-server":
                shutil.copy(built["path"], WORK / "llama-server")
            result["llama_server"]["libs"] = pin_libs(built["path"])
        result["ok"] = not result.get("model_errors")
    except Exception as e:  # noqa: BLE001 - the window reports and stops; nothing falls back silently
        result["ok"], result["error"] = False, f"{type(e).__name__}: {e}"
        if isinstance(e, DeadlineReached):
            result["cut_at_deadline"] = CUT_AT or {"before": str(e)}
    finally:
        result["resources"] = RESOURCES
        result["ended"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        # The file first: the rig takes the PIN_RESULT line in job.log as the sign that pin-result.json is there to fetch.
        (WORK / "pin-result.json").write_text(json.dumps(result, indent=1))
        print("PIN_RESULT " + json.dumps(result), flush=True)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":  # pragma: no cover - runs on the VM
    sys.exit(main())
