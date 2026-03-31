from __future__ import annotations

from pathlib import Path


REPO_NAME = "flashinfer-ai/mlsys26-contest"


def infer_parent(def_name: str) -> str:
    if def_name.startswith("dsa_"):
        return "dsa_paged"
    if def_name.startswith("gdn_"):
        return "gdn"
    if def_name.startswith("moe_"):
        return "moe"
    raise ValueError(f"Unsupported definition: {def_name}")


def resolve_dataset_root(local_path: str | None = None) -> Path:
    if local_path:
        repo_path = Path(local_path).expanduser().resolve()
        if not repo_path.exists():
            raise FileNotFoundError(f"Local dataset not found: {repo_path}")
        return repo_path

    from huggingface_hub import snapshot_download

    # snapshot_download reuses the local Hugging Face cache and only fetches
    # changed files when needed.
    return Path(snapshot_download(REPO_NAME, repo_type="dataset"))
