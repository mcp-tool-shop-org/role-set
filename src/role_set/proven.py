"""The proven-model list (`writers.json`) and the pins a wave must match (BUILD-SPEC B6).

A use is allowed unless a clause in the maker's licence or terms restricts it (the maintainer's rule, 2026-10-17:
silence is not a restriction). Each entry records the search: the terms page, the day it was read, who read it, its
record in rnd's licence registry, and the training verdict. An entry without that record, or whose verdict is
`barred`, is refused when the list loads. A listed model is still refused until its weights
are pinned: the repo revision, the GGUF file and that file's sha256. The wave script hashes the GGUF and the
llama-server binary before launch and passes the hashes in; any drift from the list refuses the run before a byte
is sent. A critic of the writer's own family is refused.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from role_set.guards import CLOUD_MARKERS, GuardError

TERMS_FIELDS = ("family", "licence", "terms_url", "terms_date", "checked_by", "registry_id", "training")
# The registry's verdicts for training on the model's outputs. `conditions` needs the conditions written out;
# `barred` is refused, with the clause that bars it.
TRAINING_VERDICTS = ("allowed", "conditions", "barred")
PIN_FIELDS = ("repo", "revision", "gguf_file", "gguf_sha256")
SERVER_FIELDS = ("commit", "sha256")
MACHINE_FIELDS = ("driver", "compat", "runtime", "card")
PARTITION_RE = re.compile(r"production|trial:[a-z0-9][a-z0-9-]*")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
# The id of the entry's record in rnd's licence registry (rnd licences/registry.json), where the page text is
# snapshotted and any restricting clause quoted.
REGISTRY_ID_RE = re.compile(r"[a-z0-9][a-z0-9.-]*")


@dataclass(frozen=True)
class ProvenList:
    writers: dict[str, dict]
    critics: dict[str, dict]
    server: dict
    sha256: str
    server_builds: tuple[dict, ...] = ()


def _text(entry: dict, field: str) -> str:
    value = entry.get(field)
    return value.strip() if isinstance(value, str) else ""


def require_terms(entry: dict, where: str) -> dict:
    """The terms search recorded and well formed, a verdict that allows training, and no cloud tag."""

    name = _text(entry, "served_as")
    if not name:
        raise GuardError(f"REFUSED: a {where} entry has no served_as")
    if any(m in name.lower() for m in CLOUD_MARKERS):
        raise GuardError(f"REFUSED: cloud model tags are never used ({name})")
    missing = [f for f in TERMS_FIELDS if not _text(entry, f)]
    if missing:
        raise GuardError(f"REFUSED: {name} has no {', '.join(missing)}; a model is listed with the record of its terms search")
    if not _text(entry, "terms_url").startswith("https://"):
        raise GuardError(f"REFUSED: {name}'s terms_url must be the maker's live https page")
    if not DATE_RE.fullmatch(_text(entry, "terms_date")):
        raise GuardError(f"REFUSED: {name}'s terms_date must be YYYY-MM-DD, the day the terms were read")
    verdict = _text(entry, "training")
    if verdict not in TRAINING_VERDICTS:
        raise GuardError(f"REFUSED: {name}'s training verdict must be one of {', '.join(TRAINING_VERDICTS)}")
    if verdict == "barred":
        raise GuardError(f"REFUSED: {name}'s terms bar training on its outputs ({_text(entry, 'restriction') or 'see its registry record'})")
    if verdict == "conditions" and not _text(entry, "conditions"):
        raise GuardError(f"REFUSED: {name}'s training verdict is conditions, but no conditions are written out")
    if not REGISTRY_ID_RE.fullmatch(_text(entry, "registry_id")):
        raise GuardError(f"REFUSED: {name}'s registry_id must be an rnd licence registry id, like openmdw-1.1")
    return entry


def load(path: Path) -> ProvenList:
    """The list, with every entry's terms record checked. A duplicate name is refused."""

    raw = path.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    out: dict[str, dict[str, dict]] = {"writers": {}, "critics": {}}
    for where in ("writers", "critics"):
        for entry in data.get(where, []):
            require_terms(entry, where[:-1])
            name = entry["served_as"].strip()
            if name in out[where]:
                raise GuardError(f"REFUSED: {name} is listed twice among the {where}")
            out[where][name] = entry
    return ProvenList(writers=out["writers"], critics=out["critics"], server=data.get("llama_server") or {},
                      sha256=hashlib.sha256(raw).hexdigest(),
                      server_builds=tuple(data.get("llama_server_builds") or ()))


def require_pinned(entry: dict, measured_sha256: str | None) -> dict:
    """The weights the wave script measured are the weights the list pins."""

    name = entry["served_as"]
    missing = [f for f in PIN_FIELDS if not _text(entry, f)]
    if missing:
        raise GuardError(f"REFUSED: {name} is not pinned yet (no {', '.join(missing)}); pin it after the download, before a wave")
    pinned = _text(entry, "gguf_sha256").lower()
    if not SHA256_RE.fullmatch(pinned):
        raise GuardError(f"REFUSED: {name}'s gguf_sha256 is not a sha256")
    got = (measured_sha256 or "").strip().lower()
    if not got:
        raise GuardError(f"REFUSED: {name} needs the GGUF's measured sha256 from the wave script")
    if got != pinned:
        raise GuardError(f"REFUSED: {name}'s GGUF {got[:12]} differs from the pinned {pinned[:12]}; re-check and re-pin")
    return {"family": entry["family"], "repo": entry["repo"], "revision": entry["revision"], "gguf_file": entry["gguf_file"],
            "gguf_sha256": pinned, "licence": entry["licence"], "terms_url": entry["terms_url"], "registry_id": entry["registry_id"],
            "training": entry["training"]}


def require_server(server: dict | ProvenList, measured_sha256: str | None,
                   other_builds: tuple[dict, ...] | list[dict] = ()) -> dict:
    """The llama-server binary the wave launched is the one the list pins (built with CUDA 13.4)."""

    if isinstance(server, ProvenList):
        default_server = server.server
        builds = tuple(other_builds) if other_builds else server.server_builds
    else:
        default_server = server
        builds = tuple(other_builds)
    missing = [f for f in SERVER_FIELDS if not _text(default_server, f)]
    if missing:
        raise GuardError(f"REFUSED: llama_server is not pinned yet (no {', '.join(missing)})")
    for idx, entry in enumerate(builds):
        b_missing = [f for f in SERVER_FIELDS if not _text(entry, f)]
        if b_missing:
            raise GuardError(f"REFUSED: llama_server_builds[{idx}] is not pinned yet (no {', '.join(b_missing)})")
    pinned = _text(default_server, "sha256").lower()
    got = (measured_sha256 or "").strip().lower()
    if not got:
        raise GuardError("REFUSED: the llama-server binary's measured sha256 is required from the wave script")
    candidates = [default_server, *builds]
    matched = next((entry for entry in candidates if _text(entry, "sha256").lower() == got), None)
    if matched is None:
        raise GuardError(f"REFUSED: llama-server {got[:12]} differs from the pinned {pinned[:12]}; rebuild or re-pin")
    out = {"commit": _text(matched, "commit"), "sha256": _text(matched, "sha256").lower()}
    for opt in ("cuda", "archs"):
        val = _text(matched, opt)
        if val:
            out[opt] = val
    return out


def require_other_family(writer: dict, critic: dict) -> None:
    if _text(writer, "family").lower() == _text(critic, "family").lower():
        raise GuardError(f"REFUSED: the critic {critic['served_as']} is the writer's own family ({critic['family']}); "
                         "a critic reads another family's drafts")


def require_partition(partition: str | None) -> str:
    p = (partition or "").strip()
    if not PARTITION_RE.fullmatch(p):
        raise GuardError(f"REFUSED: --partition must be production or trial:<name>, got {partition!r}")
    return p


def load_machine(path: Path | None) -> dict:
    """The CUDA machine line the wave script recorded: driver, compat loaded or not, runtime, card, and where."""

    if path is None:
        raise GuardError("REFUSED: --machine <machine line JSON> is required (driver, compat, runtime, card, and the Colab tag or pod id)")
    try:
        line = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise GuardError(f"REFUSED: cannot read the machine line ({type(e).__name__})") from e
    if not isinstance(line, dict):
        raise GuardError("REFUSED: the machine line must be a JSON object")
    missing = [f for f in MACHINE_FIELDS if line.get(f) in (None, "")]
    if missing:
        raise GuardError(f"REFUSED: the machine line has no {', '.join(missing)}")
    if not isinstance(line["compat"], bool):
        raise GuardError("REFUSED: the machine line's compat must be true or false")
    if not (line.get("colab_release_tag") or line.get("pod_id") or line.get("rig")):
        raise GuardError("REFUSED: the machine line must name where it ran: colab_release_tag, pod_id or rig")
    return line


def needs_grant(machine: dict) -> bool:
    """Fail closed: every step needs the Publisher's grant and a fresh heartbeat unless its machine line names a
    rented host (a Colab VM or a RunPod pod). A line that names the rig needs them, whatever its card string says."""

    if machine.get("rig"):
        return True
    return not (machine.get("colab_release_tag") or machine.get("pod_id"))
