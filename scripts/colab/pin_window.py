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
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

LLAMA_CPP_COMMIT = "620d2225cc436ab0b001e3200e590e3b286ef9ab"  # ggml-org/llama.cpp release v0.7.0, 2026-10-15
# SASS for A100 (80), H100/H200 (90) and B200 (100): one binary for Colab and RunPod, no PTX JIT.
CUDA_ARCHS = "80;90;100"
NVCC_VERSION = "13.4.92"  # nvcc, nvvm, crt and the runtime, one release
# cuBLAS follows its own numbering: the CUDA 13.4.2 toolkit (runtime 13.4.92) ships cuBLAS 13.8.0.4. Pinned exactly; a
# `~=13.4` spec pulled the later standalone 13.8.1.7 in window 2 (2026-10-17).
CUBLAS_VERSION = "13.8.0.4"
CUDA_TOOLKIT = "13.4.2 (redistrib 2026-09-16)"
NVCC_PIP = f"nvidia-cuda-nvcc=={NVCC_VERSION}"
WORK = Path("/content/pin")
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
     "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", "tokenizer_check": True},
    # Window 5 pinned it on the A100 with an unproven fit; window 6 (H100) reconverts it, so the hash checks determinism.
    {"served_as": "qwen3.6-35b-a3b", "how": "convert", "repo": "Qwen/Qwen3.6-35B-A3B",
     "revision": "995ad96eacd98c81ed38be0c5b274b04031597b0", "tokenizer_check": True,
     "expect_sha256": "b7c0c8eb16a54af29e87845d187969f99ecbbf79cd47cc198be8890d109dc164"},
    {"served_as": "granite-4.2-30b", "how": "convert", "repo": "ibm-granite/granite-4.2-30b",
     "revision": "9e668ce1c538387ef24d3644e9b0606647762636", "tokenizer_check": True},
    # Gemma 4 was saved with transformers 5.5.0.dev0; its tokenizer_config's list-shaped extra_special_tokens breaks
    # llama.cpp's pinned 4.57.6 (window 5), so it converts in its own env, as Muse does.
    {"served_as": "gemma-4-31b-it", "how": "convert", "repo": "google/gemma-4-31B-it",
     "revision": "842da3794eaa0b77d5f08bae87a17459d91ff475", "env": {"transformers": "5.20.1"}, "tokenizer_check": True},
    # Pinned in window 5; its 32k KV cache doesn't fit the A100, so it serves on the H100 (R&D, 2026-10-17).
    {"served_as": "deepseek-r1-distill-qwen-32b", "how": "convert", "repo": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
     "revision": "711ad2ea6aa40cfca18895e8aca02ab92df1a746", "tokenizer_check": True,
     "expect_sha256": "b791775c3bb0d1ab3cd7a73c6760c4ab915680f3587be8a7d29937ca2b9ad24e"},
    # NVIDIA's own 24GB quantization-aware GGUF: fetched and hashed, not converted.
    {"served_as": "nemotron-3.5-bolt-74b-a9b", "how": "maker_gguf", "repo": "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF",
     "revision": "e0fecb69cb3e5e4c5e8a056fa2aec98c78e91246", "file": "NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf",
     "card": "nemotron-3.5-bolt-74b-a9b-24gb"},
]
# Window 1's llama-server (built 2026-10-17 at LLAMA_CPP_COMMIT, nvcc 13.4.92): every later window runs this file.
SERVER_SHA256 = "b4659b3fc74037deb96b2148c455753121fa21dde174d7124547e2bf9ffd37c3"


def sh(cmd, cwd=None, timeout=3600, env=None) -> str:
    out = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env)
    if out.returncode != 0:
        raise RuntimeError(f"{cmd[:80]}... exit {out.returncode}: {(out.stderr or out.stdout)[-1500:]}")
    return out.stdout.strip()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def machine_line() -> dict:
    gpu = sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader")
    name, mem, driver = [x.strip() for x in gpu.split(",")]
    return {"card": name, "memory": mem, "driver": driver, "compat": False,
            "driver_cuda": sh("nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | cut -d' ' -f3"),
            "colab_release_tag": os.environ.get("COLAB_RELEASE_TAG"), "cores": os.cpu_count(),
            "disk_free_gb": round(shutil.disk_usage("/content").free / 2**30, 1),
            # The CUDA runtime the build targets; role-set's load_machine requires it (R&D on #77).
            "runtime": "13.4"}


def build() -> dict:
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
    sh(f"cmake -S {src} -B {src}/build -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES='{CUDA_ARCHS}' "
       f"-DCUDAToolkit_ROOT={cu} -DCMAKE_CUDA_COMPILER={nvcc} -DCUDA_cuda_driver_LIBRARY={libcuda} "
       f"-DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release", env=env, timeout=900)
    say("building llama-server and llama-quantize")
    sh(f"cmake --build {src}/build --target llama-server llama-quantize -j {os.cpu_count()}", env=env, timeout=3600)
    server = src / "build" / "bin" / "llama-server"
    return {"commit": LLAMA_CPP_COMMIT, "nvcc": version.splitlines()[-1], "cuda_archs": CUDA_ARCHS,
            "sha256": sha256(server), "path": str(server), "quantize": str(src / "build" / "bin" / "llama-quantize"),
            "src": str(src)}


def hf_revision(repo: str) -> str:
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}", timeout=60) as r:  # noqa: S310 - fixed host
        return json.loads(r.read())["sha"]


def fetch_model(spec: dict, built: dict) -> dict:
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download  # noqa: PLC0415

    rev = spec.get("revision") or hf_revision(spec["repo"])
    out = {"served_as": spec["served_as"], "repo": spec["repo"], "revision": rev, "card": spec.get("card")}
    if spec["how"] == "maker_gguf":
        listed = HfApi().list_repo_files(spec["repo"], revision=rev)
        # An exact `file` when the repo's names overlap (Bolt's mmproj file contains the model file's name).
        files = [f for f in listed if f == spec["file"]] if spec.get("file") else \
            [f for f in listed if f.endswith(".gguf") and spec["pattern"] in f]
        if len(files) != 1:
            raise RuntimeError(f"{spec['repo']}: expected one GGUF for {spec.get('file') or spec['pattern']}, found {files}")
        say(f"downloading {files[0]}")
        path = Path(hf_hub_download(spec["repo"], files[0], revision=rev, local_dir=WORK / spec["served_as"]))
        out.update({"gguf_source": "maker", "gguf_file": files[0]})
    else:
        say(f"downloading {spec['repo']} (BF16) for conversion")
        snap = Path(snapshot_download(spec["repo"], revision=rev, local_dir=WORK / (spec["served_as"] + "-bf16"),
                                      allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.py", "*.jinja"]))
        py = converter_python(spec, built)
        path = WORK / f"{spec['served_as']}-Q8_0.gguf"
        say("converting to Q8_0")
        sh(f"{py} {built['src']}/convert_hf_to_gguf.py {snap} --outtype q8_0 --outfile {path}", timeout=3600)
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
        shutil.rmtree(snap, ignore_errors=True)
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
                             "-c", str(ctx), "--jinja", "--reasoning-format", "deepseek"], stdout=log_fh, stderr=subprocess.STDOUT,
                            text=True, env=env)
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
        tok = tokenizer_check(port, model) if model.get("tokenizer_hf") else None
        # Amendment E1's thinking split: reasoning comes back apart, and no thinking tag is left in the content.
        content = msg.get("content") or ""
        tags = [tg for tg in ("<think>", "</think>", "<|channel>thought", "<channel|>") if tg in content]
        return {"ok": bool((msg.get("content") or "").strip()), "tokens_per_s": round(toks / secs, 1) if secs else None,
                "completion_tokens": toks, "finish_reason": reply["choices"][0].get("finish_reason"), "tokenizer": tok,
                "reasoning_content": bool(msg.get("reasoning_content")), "content_head": (msg.get("content") or "")[:120],
                "logprob_steps": len(steps), "top5_ok": bool(steps) and all(len(s.get("top_logprobs") or []) == 5 for s in steps),
                "ctx": ctx, "peak_vram_mib": peak["mib"], "total_vram_mib": peak.get("total_mib"), "max_tokens": max_tokens,
                "offload_lines": _offload_lines(server_log),
                "chat_template_kwargs": kwargs,
                "thinking_split": {"reasoning_nonempty": bool((msg.get("reasoning_content") or "").strip()), "tags_in_content": tags,
                                   "ok": bool((msg.get("reasoning_content") or "").strip()) and not tags}}
    finally:
        stop.set()
        proc.terminate()
        proc.wait(timeout=60)
        log_fh.close()


SMOKE_MAX_TOKENS = 2048


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


BRIEFS = Path("/content/brief-strings.json")  # uploaded by the window script: 20 writer briefs, 20 critic prompts
CRITIC_CTX = 32768
CRITIC_PORT = 8092


def start_server(built: dict, model: dict, ctx: int, port: int):
    """llama-server for one model, its log in a file; waits until healthy and fails at once if the server exits."""

    env = {**os.environ, "LD_LIBRARY_PATH": ":".join(str(p) for p in _cuda_lib_dirs()) + ":" + os.environ.get("LD_LIBRARY_PATH", "")}
    server_log = WORK / f"{model['served_as']}-critic-server.log"
    log_fh = open(server_log, "w", encoding="utf-8")  # noqa: SIM115 - closed by stop_server
    proc = subprocess.Popen([built["path"], "-m", model["path"], "--port", str(port), "--host", "127.0.0.1", "-ngl", "999",
                             "-c", str(ctx), "--jinja", "--reasoning-format", "deepseek"], stdout=log_fh, stderr=subprocess.STDOUT,
                            text=True, env=env)
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
    proc.terminate()
    log_fh.close()
    raise RuntimeError("llama-server never came up: " + server_log.read_text(encoding="utf-8", errors="replace")[-800:])


def stop_server(proc, log_fh) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=60)
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

    src, venv = Path("/content/role-set"), Path("/content/rs-venv")
    if not (src / ".git").exists():
        sh(f"git clone -q https://github.com/mcp-tool-shop-org/role-set {src} && git -C {src} checkout -q {commit}", timeout=600)
    sh(f"{sys.executable} -m pip install -q uv==0.9.5", timeout=600)
    if not (venv / "bin" / "python").exists():
        sh(f"{sys.executable} -m uv venv -q -p 3.12 {venv}", timeout=900)
        sh(f"{sys.executable} -m uv pip install -q --python {venv}/bin/python -e {src}", timeout=1800)
    head = sh(f"git -C {src} rev-parse HEAD")
    if head != commit:
        raise RuntimeError(f"role-set is at {head[:12]}, not {commit[:12]}")
    return {"commit": head, "src": str(src), "role_set": str(venv / "bin" / "role-set"),
            "python": sh(f"{venv}/bin/python -c \"import sys; print(sys.version.split()[0])\"")}


def critic_control(rs: dict, model: dict, machine_file: Path, seed: int, earlier: Path | None) -> dict:
    out = WORK / f"critic-control-{model['served_as']}-s{seed}.json"
    cmd = [rs["role_set"], "critic-control", "--items", "/content/known-answer-v1.jsonl", "--planted", "10", "--seed", "0",
           "--critic-endpoint", f"http://127.0.0.1:{CRITIC_PORT}/v1", "--critic-model", model["served_as"],
           "--critic-gguf-sha256", model["gguf_sha256"], "--llama-server-sha256", SERVER_SHA256,
           "--machine", str(machine_file), "--writers", f"{rs['src']}/writers.json", "--cards", "/content/card-values.json",
           "--critic-seed", str(seed), "--out", str(out)]
    if earlier is not None:
        cmd += ["--agreement-with", str(earlier)]
    t0 = time.time()
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=4 * 3600, cwd=rs["src"])
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
                rec["offload_lines"] = _offload_lines(server_log)
                rec["fit"] = all_on_gpu(rec["offload_lines"])
                if not rec["fit"]["ok"]:
                    rec["stopped"] = "not every layer is on the GPU"
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
        except Exception as e:  # noqa: BLE001 - recorded with its model; the next model still runs
            rec["error"] = f"{type(e).__name__}: {e}"
        finally:
            Path(WORK / f"{spec['served_as']}-Q8_0.gguf").unlink(missing_ok=True)
        say(f"{spec['served_as']}: {json.dumps({k: rec.get(k) for k in ('reproduced', 'fit', 'brief_check', 'stopped', 'error')})[:600]}")
    result["model_errors"] = sum(1 for r in result["models"] if r.get("stopped") or r.get("error"))


def runtime() -> dict:
    """PIN_ROLE=smoke: no build. The uploaded window-1 binary, checked by sha256, with the CUDA 13.4 runtime wheels it
    loads; their exact versions are recorded so the library pins can be reproduced."""

    sh(f"{sys.executable} -m pip install -q 'nvidia-cuda-runtime=={NVCC_VERSION}' 'nvidia-cublas=={CUBLAS_VERSION}'", timeout=1200)
    server, uploaded = WORK / "llama-server", Path("/content/llama-server")
    parts = sorted(Path("/content").glob("llama-server.gz.part*"))
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
    if got != SERVER_SHA256:
        raise RuntimeError(f"llama-server {got[:12]} isn't window 1's {SERVER_SHA256[:12]}")
    src = WORK / "llama.cpp"
    if not (src / ".git").exists():
        sh(f"git clone -q https://github.com/ggml-org/llama.cpp {src} && git -C {src} checkout -q {LLAMA_CPP_COMMIT}", timeout=900)
    wheels = [x for x in sh(f"{sys.executable} -m pip freeze").splitlines() if x.lower().startswith("nvidia-")]
    for want in (f"nvidia-cublas=={CUBLAS_VERSION}", f"nvidia-cuda-runtime=={NVCC_VERSION}"):
        if want not in wheels:
            raise RuntimeError(f"{want} isn't what pip installed: {[w for w in wheels if 'cublas' in w or 'runtime' in w]}")
    return {"commit": LLAMA_CPP_COMMIT, "sha256": got, "path": str(server), "src": str(src), "cuda_wheels": wheels,
            "cuda_toolkit": CUDA_TOOLKIT}


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    result: dict = {"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        result["machine"] = machine_line()
        say(f"machine: {result['machine']}")
        # Split windows run side by side (Mike, 2026-10-16): PIN_ROLE=build builds the server and pins and smokes
        # the maker GGUFs; PIN_ROLE=convert only converts and hashes (no CUDA build, so no smoke; the build VM's
        # binary serves it in the next window). Unset: everything, on one VM.
        role = os.environ.get("PIN_ROLE", "all")
        result["role"] = role
        if role in ("smoke", "critic"):
            result["llama_server"] = built = runtime()
            # The libraries first, so they're pinned even if a model fails later in the window.
            result["llama_server"]["libs"] = pin_libs(built["path"])
        elif role == "convert":
            src = WORK / "llama.cpp"
            sh(f"git clone -q https://github.com/ggml-org/llama.cpp {src} && git -C {src} checkout -q {LLAMA_CPP_COMMIT}", timeout=900)
            built = {"commit": LLAMA_CPP_COMMIT, "src": str(src)}
        else:
            result["llama_server"] = built = build()
        specs = MODELS if role in ("all", "smoke", "critic") else [m for m in MODELS if (role == "build") == (m["how"] == "maker_gguf")]
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
        else:
            result["models"] = []
        for spec in specs:
            if role == "smoke":
                # Each model is independent here: a failure is recorded with its model and the next one still runs.
                try:
                    m = fetch_model(spec, built)
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
            elif role != "convert":
                try:
                    m["smoke"] = smoke(built, m)
                except Exception as e:  # noqa: BLE001 - a smoke window records a failed serve and goes on
                    if role != "smoke":
                        raise
                    m["smoke"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                    result["model_errors"] = result.get("model_errors", 0) + 1
                say(f"{m['served_as']}: {m['smoke']}")
            result["models"].append({k: v for k, v in m.items() if k != "path"})
            Path(m["path"]).unlink(missing_ok=True)  # the hash is the record; the file is rebuilt per window
        if role not in ("convert", "smoke"):
            # The server binary is the one artifact fetched back: every later window and pod runs this exact file.
            if Path(built["path"]) != WORK / "llama-server":
                shutil.copy(built["path"], WORK / "llama-server")
            result["llama_server"]["libs"] = pin_libs(built["path"])
        result["ok"] = not result.get("model_errors")
    except Exception as e:  # noqa: BLE001 - the window reports and stops; nothing falls back silently
        result["ok"], result["error"] = False, f"{type(e).__name__}: {e}"
    finally:
        result["ended"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        print("PIN_RESULT " + json.dumps(result), flush=True)
        (WORK / "pin-result.json").write_text(json.dumps(result, indent=1))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":  # pragma: no cover - runs on the VM
    sys.exit(main())
