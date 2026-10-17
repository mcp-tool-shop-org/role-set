import json
from pathlib import Path

import pytest

from role_set import proven
from role_set.guards import GuardError

SHA_W = "a" * 64
SHA_C = "c" * 64
SHA_S = "5" * 64


def entry(name: str, family: str, sha: str | None = None, **extra) -> dict:
    return {"served_as": name, "family": family, "repo": f"maker/{name}", "revision": "r1" if sha else None,
            "gguf_file": f"{name}-Q8_0.gguf" if sha else None, "gguf_sha256": sha, "quant": "Q8_0", "licence": "OpenMDW-1.1",
            "terms_url": f"https://huggingface.co/maker/{name}", "terms_date": "2026-10-16", "checked_by": "R&D, 2026-10-16",
            "registry_id": "openmdw-1.1", "training": "allowed", **extra}


def proven_file(tmp_path: Path, *, writers=None, critics=None, server=None) -> Path:
    path = tmp_path / "writers.json"
    path.write_text(json.dumps({
        "llama_server": server if server is not None else {"commit": "b6000", "sha256": SHA_S},
        "writers": writers if writers is not None else [entry("big-writer", "nemotron", SHA_W)],
        "critics": critics if critics is not None else [entry("other-critic", "muse", SHA_C), entry("same-critic", "nemotron", SHA_C),
                                                         entry("compare-critic", "granite", SHA_C, role="critic-comparison")],
    }), encoding="utf-8")
    return path


def test_the_list_loads_with_every_proof_and_hashes_itself(tmp_path: Path):
    listed = proven.load(proven_file(tmp_path))
    assert sorted(listed.writers) == ["big-writer"] and sorted(listed.critics) == ["compare-critic", "other-critic", "same-critic"]
    assert len(listed.sha256) == 64 and listed.server["commit"] == "b6000"


@pytest.mark.parametrize("field", proven.TERMS_FIELDS)
def test_an_entry_without_any_terms_field_is_refused(tmp_path: Path, field: str):
    bad = entry("big-writer", "nemotron", SHA_W)
    bad[field] = ""
    with pytest.raises(GuardError, match=f"no {field}"):
        proven.load(proven_file(tmp_path, writers=[bad]))


def test_terms_shape_name_cloud_and_duplicates_are_refused(tmp_path: Path):
    with pytest.raises(GuardError, match="live https"):
        proven.load(proven_file(tmp_path, writers=[entry("w", "x", terms_url="http://example.org/card")]))
    with pytest.raises(GuardError, match="YYYY-MM-DD"):
        proven.load(proven_file(tmp_path, writers=[entry("w", "x", terms_date="last week")]))
    with pytest.raises(GuardError, match="registry id"):
        proven.load(proven_file(tmp_path, writers=[entry("w", "x", registry_id="OpenMDW 1.1")]))
    with pytest.raises(GuardError, match="no served_as"):
        proven.load(proven_file(tmp_path, writers=[entry("", "x")]))
    with pytest.raises(GuardError, match="cloud"):
        proven.load(proven_file(tmp_path, writers=[entry("kimi-k3:cloud", "kimi")]))
    with pytest.raises(GuardError, match="listed twice"):
        proven.load(proven_file(tmp_path, critics=[entry("c", "muse"), entry("c", "muse")]))


def test_an_unpinned_model_and_a_drifted_gguf_are_refused():
    unpinned = entry("big-writer", "nemotron", None)
    with pytest.raises(GuardError, match="not pinned yet"):
        proven.require_pinned(unpinned, SHA_W)
    with pytest.raises(GuardError, match="not a sha256"):
        proven.require_pinned(entry("w", "x", "nothex"), "nothex")
    with pytest.raises(GuardError, match="measured sha256"):
        proven.require_pinned(entry("w", "x", SHA_W), None)
    with pytest.raises(GuardError, match="differs from the pinned"):
        proven.require_pinned(entry("w", "x", SHA_W), "b" * 64)
    pins = proven.require_pinned(entry("w", "x", SHA_W), SHA_W.upper())
    assert pins["gguf_sha256"] == SHA_W and pins["family"] == "x" and pins["terms_url"].startswith("https://")
    assert pins["registry_id"] == "openmdw-1.1" and pins["training"] == "allowed"


def test_only_a_barring_clause_refuses_a_model(tmp_path: Path):
    """The 2026-10-17 rule: a use is allowed unless a clause restricts it. Silence lists the model (no quote that grants
    the use is asked for); `barred` refuses it; `conditions` lists it with the conditions written out."""

    listed = proven.load(proven_file(tmp_path, writers=[entry("silent", "x", SHA_W)]))
    assert "silent" in listed.writers and "terms_note" not in listed.writers["silent"]
    with pytest.raises(GuardError, match="bar training"):
        proven.load(proven_file(tmp_path, writers=[entry("w", "x", training="barred", restriction="no distilling Outputs")]))
    with pytest.raises(GuardError, match="no conditions"):
        proven.load(proven_file(tmp_path, writers=[entry("w", "x", training="conditions")]))
    assert "w" in proven.load(proven_file(tmp_path, writers=[entry("w", "x", training="conditions", conditions="credit the maker")])).writers
    with pytest.raises(GuardError, match="one of allowed"):
        proven.load(proven_file(tmp_path, writers=[entry("w", "x", training="proven")]))


def test_the_llama_server_binary_is_pinned():
    with pytest.raises(GuardError, match="not pinned yet"):
        proven.require_server({"commit": None, "sha256": None}, SHA_S)
    with pytest.raises(GuardError, match="measured sha256"):
        proven.require_server({"commit": "b1", "sha256": SHA_S}, "")
    with pytest.raises(GuardError, match="rebuild or re-pin"):
        proven.require_server({"commit": "b1", "sha256": SHA_S}, "6" * 64)
    assert proven.require_server({"commit": "b1", "sha256": SHA_S}, SHA_S) == {"commit": "b1", "sha256": SHA_S}


def test_a_critic_of_the_writers_family_is_refused():
    with pytest.raises(GuardError, match="own family"):
        proven.require_other_family(entry("w", "Nemotron"), entry("c", "nemotron"))
    proven.require_other_family(entry("w", "nemotron"), entry("c", "muse"))


def test_partitions():
    assert proven.require_partition("production") == "production"
    assert proven.require_partition(" trial:j1-wave1 ") == "trial:j1-wave1"
    for bad in (None, "", "prod", "trial:", "trial:J1", "production2"):
        with pytest.raises(GuardError, match="--partition"):
            proven.require_partition(bad)


def test_the_machine_line(tmp_path: Path):
    with pytest.raises(GuardError, match="--machine"):
        proven.load_machine(None)
    with pytest.raises(GuardError, match="cannot read"):
        proven.load_machine(tmp_path / "missing.json")
    path = tmp_path / "m.json"
    for body, match in (("[1]", "JSON object"), ('{"driver": "580"}', "no compat, runtime, card"),
                        ('{"driver": "580", "compat": "yes", "runtime": "13.4", "card": "A100"}', "true or false"),
                        ('{"driver": "580", "compat": true, "runtime": "13.4", "card": "A100"}', "where it ran")):
        path.write_text(body, encoding="utf-8")
        with pytest.raises(GuardError, match=match):
            proven.load_machine(path)
    path.write_text(json.dumps({"driver": "580.82", "compat": True, "runtime": "13.4", "card": "NVIDIA A100-SXM4-40GB",
                                "colab_release_tag": "release-colab_20261015-060052_RC00"}), encoding="utf-8")
    line = proven.load_machine(path)
    assert line["card"].endswith("40GB") and not proven.needs_grant(line)
    # Fail closed (R&D on #70): only a rented host goes without the grant; the rig needs it whatever its card says.
    assert not proven.needs_grant({"card": "NVIDIA H200 NVL", "pod_id": "abc123"})
    assert proven.needs_grant({"card": "NVIDIA A100-SXM4-40GB", "rig": "robot"})
    assert proven.needs_grant({"card": "NVIDIA A100-SXM4-40GB", "rig": "robot", "pod_id": "abc123"})
    assert proven.needs_grant({"card": "NVIDIA GeForce RTX 5090"})
    assert proven.needs_grant({"rig": "robot"})  # a rig line with no card string at all


def test_the_committed_list_records_its_terms_and_its_pins():
    listed = proven.load(Path(__file__).resolve().parents[1] / "writers.json")
    names = set(listed.writers) | set(listed.critics)
    # The 2026-10-11 pilots' Ollama tags stay off: the pilot's arms are llama-server GGUFs at pinned revisions.
    assert not names & {"qwen3-235b-instruct-2507", "qwen3:8b", "gemma4:31b", "granite4.1:30b"}
    assert {"muse-glimmer-30b", "nemotron-3.5-lightning-30b-a3b"} <= set(listed.critics)
    # The pilot's eight arms (R&D, 2026-10-17): Apache-2.0 and MIT models are eligible under the terms rule.
    assert {"qwen3.8-27b", "qwen3.6-35b-a3b", "granite-4.2-30b", "gemma-4-31b-it", "deepseek-r1-distill-qwen-32b",
            "nemotron-3.5-bolt-74b-a9b", "muse-glimmer-30b", "nemotron-3.5-lightning-30b-a3b"} <= set(listed.writers)
    assert {e["registry_id"] for e in [*listed.writers.values(), *listed.critics.values()]} <= {
        "openmdw-1.1", "deepseek-r1-0528-card", "muse-glimmer-30b-card", "apache-2.0", "deepseek-r1-card"}
    assert all(e["training"] == "allowed" for e in [*listed.writers.values(), *listed.critics.values()])
    families = {e["family"] for e in listed.critics.values()}
    # Every writer has a critic of another family on the list.
    for w in listed.writers.values():
        assert families - {w["family"]}, w["served_as"]
    # The server is pinned (window 1); a pinned arm loads only its own GGUF; an unconverted one is refused. Window 9's
    # requantized Q6_K entries are retired (R&D, 2026-10-17): kept in the file as history, never loaded.
    raw = json.loads((Path(__file__).resolve().parents[1] / "writers.json").read_text(encoding="utf-8"))
    assert {e["served_as"] for e in raw["retired"]} == {"qwen3.6-35b-a3b-q6k", "gemma-4-31b-it-q6k", "deepseek-r1-distill-qwen-32b-q6k"}
    assert not {e["served_as"] for e in raw["retired"]} & set(listed.writers)
    assert listed.server["cuda"] == "13.4" and listed.server["commit"].startswith("620d2225")
    assert proven.require_server(listed.server, listed.server["sha256"])["sha256"] == listed.server["sha256"]
    with pytest.raises(GuardError, match="rebuild or re-pin"):
        proven.require_server(listed.server, SHA_S)
    lightning = listed.critics["nemotron-3.5-lightning-30b-a3b"]
    assert proven.require_pinned(lightning, lightning["gguf_sha256"])["revision"].startswith("a9904d24")
    with pytest.raises(GuardError, match="not pinned yet"):
        proven.require_pinned(listed.writers["nemotron-3-ultra-550b-a55b"], SHA_W)  # a pod writer, not converted yet
