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
import time
import urllib.request
from pathlib import Path

LLAMA_CPP_COMMIT = "620d2225cc436ab0b001e3200e590e3b286ef9ab"  # ggml-org/llama.cpp release v0.7.0, 2026-10-15
# SASS for A100 (80), H100/H200 (90) and B200 (100): one binary for Colab and RunPod, no PTX JIT.
CUDA_ARCHS = "80;90;100"
NVCC_PIP = "nvidia-cuda-nvcc~=13.4.0"
WORK = Path("/content/pin")
MODELS = [
    # name, how, repo, file or None (convert), revision (None = the repo's current main, recorded)
    {"served_as": "muse-glimmer-30b", "how": "maker_gguf", "repo": "meta-models/Muse-Glimmer-30B-GGUF", "pattern": "Q8_0"},
    {"served_as": "nemotron-3.5-lightning-30b-a3b", "how": "convert", "repo": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"},
]


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
    sh(f"{sys.executable} -m pip install -q '{NVCC_PIP}' 'nvidia-cuda-runtime~=13.4.0' 'nvidia-cublas~=13.4' cmake ninja", timeout=1200)
    import nvidia.cuda_nvcc as n  # noqa: PLC0415 - installed just above

    nvcc_root = Path(list(n.__path__)[0])
    nvcc = next(nvcc_root.rglob("bin/nvcc"))
    version = sh(f"{nvcc} --version | tail -2")
    say(f"nvcc: {version.splitlines()[-1]}")
    src = WORK / "llama.cpp"
    sh(f"git clone -q https://github.com/ggml-org/llama.cpp {src} && git -C {src} checkout -q {LLAMA_CPP_COMMIT}", timeout=900)
    env = {**os.environ, "CUDACXX": str(nvcc), "PATH": f"{nvcc.parent}:{os.environ['PATH']}"}
    sh(f"cmake -S {src} -B {src}/build -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES='{CUDA_ARCHS}' "
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
    out = {"served_as": spec["served_as"], "repo": spec["repo"], "revision": rev}
    if spec["how"] == "maker_gguf":
        files = [f for f in HfApi().list_repo_files(spec["repo"], revision=rev) if f.endswith(".gguf") and spec["pattern"] in f]
        if len(files) != 1:
            raise RuntimeError(f"{spec['repo']}: expected one {spec['pattern']} GGUF, found {files}")
        say(f"downloading {files[0]}")
        path = Path(hf_hub_download(spec["repo"], files[0], revision=rev, local_dir=WORK / spec["served_as"]))
        out.update({"gguf_source": "maker", "gguf_file": files[0]})
    else:
        say(f"downloading {spec['repo']} (BF16) for conversion")
        snap = Path(snapshot_download(spec["repo"], revision=rev, local_dir=WORK / (spec["served_as"] + "-bf16"),
                                      allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.py", "*.jinja"]))
        sh(f"{sys.executable} -m pip install -q -r {built['src']}/requirements/requirements-convert_hf_to_gguf.txt", timeout=1200)
        path = WORK / f"{spec['served_as']}-Q8_0.gguf"
        say("converting to Q8_0")
        sh(f"{sys.executable} {built['src']}/convert_hf_to_gguf.py {snap} --outtype q8_0 --outfile {path}", timeout=3600)
        shutil.rmtree(snap, ignore_errors=True)
        # The converter's environment, so the Q8_0 hash can be reproduced (R&D on #77).
        freeze = sh(f"{sys.executable} -m pip freeze")
        gguf_file = sh(f"{sys.executable} -c \"import gguf; print(gguf.__file__)\"")
        out.update({"gguf_source": f"converted at llama.cpp {LLAMA_CPP_COMMIT[:12]} from {spec['repo']}@{rev[:12]}",
                    "gguf_file": path.name, "converter": {"pip_freeze_sha256": hashlib.sha256(freeze.encode()).hexdigest(),
                                                           "pip_freeze": freeze.splitlines(), "gguf_module": gguf_file}})
    say(f"hashing {path.name}")
    out.update({"gguf_sha256": sha256(path), "gguf_bytes": path.stat().st_size, "path": str(path)})
    return out


def smoke(built: dict, model: dict) -> dict:
    """One served completion: the server starts, answers, and we see whether reasoning comes back apart."""

    port = 8091
    env = {**os.environ, "LD_LIBRARY_PATH": ":".join(str(p) for p in _cuda_lib_dirs()) + ":" + os.environ.get("LD_LIBRARY_PATH", "")}
    proc = subprocess.Popen([built["path"], "-m", model["path"], "--port", str(port), "--host", "127.0.0.1", "-ngl", "999",
                             "-c", "8192", "--reasoning-format", "deepseek"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, env=env)
    try:
        for _ in range(180):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:  # noqa: S310 - loopback
                    if r.status == 200:
                        break
            except Exception:  # noqa: BLE001 - starting up
                time.sleep(5)
        else:
            raise RuntimeError("llama-server never came up: " + (proc.stderr.read()[-800:] if proc.stderr else ""))
        body = json.dumps({"model": model["served_as"], "temperature": 0, "seed": 0, "max_tokens": 256,
                           "logprobs": True, "top_logprobs": 5,
                           "messages": [{"role": "user", "content": "In two sentences, say what a changelog is for."}]}).encode()
        t0 = time.time()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:  # noqa: S310 - loopback
            reply = json.loads(r.read())
        secs = time.time() - t0
        msg = reply["choices"][0]["message"]
        toks = (reply.get("usage") or {}).get("completion_tokens") or 0
        # The dataminer record (rnd abmine's format): the first check that llama-server's logprobs come back usable.
        steps = ((reply["choices"][0].get("logprobs") or {}).get("content")) or []
        record = {"type": "result", "arm": model["served_as"], "prompt": "pin-smoke-0",
                  "tokens": [s.get("token") for s in steps],
                  "top": [[[t.get("token"), t.get("logprob")] for t in (s.get("top_logprobs") or [])] for s in steps]}
        with open(WORK / "generations.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return {"ok": bool((msg.get("content") or "").strip()), "tokens_per_s": round(toks / secs, 1) if secs else None,
                "reasoning_content": bool(msg.get("reasoning_content")), "content_head": (msg.get("content") or "")[:120],
                "logprob_steps": len(steps), "top5_ok": bool(steps) and all(len(s.get("top_logprobs") or []) == 5 for s in steps)}
    finally:
        proc.terminate()
        proc.wait(timeout=60)


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
        if role == "convert":
            src = WORK / "llama.cpp"
            sh(f"git clone -q https://github.com/ggml-org/llama.cpp {src} && git -C {src} checkout -q {LLAMA_CPP_COMMIT}", timeout=900)
            built = {"commit": LLAMA_CPP_COMMIT, "src": str(src)}
        else:
            result["llama_server"] = built = build()
        specs = [m for m in MODELS if role == "all" or (role == "build") == (m["how"] == "maker_gguf")]
        result["models"] = []
        for spec in specs:
            m = fetch_model(spec, built)
            if role != "convert":
                m["smoke"] = smoke(built, m)
                say(f"{m['served_as']}: {m['smoke']}")
            result["models"].append({k: v for k, v in m.items() if k != "path"})
            Path(m["path"]).unlink(missing_ok=True)  # the hash is the record; the file is rebuilt per window
        if role != "convert":
            # The server binary is the one artifact fetched back: every later window and pod runs this exact file.
            shutil.copy(built["path"], WORK / "llama-server")
            result["llama_server"]["libs"] = pin_libs(built["path"])
        result["ok"] = True
    except Exception as e:  # noqa: BLE001 - the window reports and stops; nothing falls back silently
        result["ok"], result["error"] = False, f"{type(e).__name__}: {e}"
    finally:
        result["ended"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        print("PIN_RESULT " + json.dumps(result), flush=True)
        (WORK / "pin-result.json").write_text(json.dumps(result, indent=1))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":  # pragma: no cover - runs on the VM
    sys.exit(main())
