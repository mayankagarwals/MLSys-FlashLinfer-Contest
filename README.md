# FlashInfer Competition Starter

Starter workspace for the FlashInfer competition. It contains problem definitions, workloads, reference code, and task-specific solution areas for local development.

## Structure

- `definitions/`: task specs grouped by problem family.
- `workloads/`: input workloads used for testing and evaluation.
- `gdn_decode/` and `gdn_prefill/`: main working directories with scripts and `solution/` code.
- `reference_solutions/`: baseline implementations.
- `docker/`: container image definition.
- `.devcontainer/`: VS Code Dev Container configuration.
- `other/`: small examples and experiments.

## Open with Docker

This repo is configured for VS Code Dev Containers.

1. Install Docker with NVIDIA GPU support.
2. Open this folder in VS Code.
3. Run `Dev Containers: Reopen in Container`.

The container is configured in `.devcontainer/devcontainer.json` and builds from `docker/Dockerfile`.

## Open with `uv`

For a local Python workflow instead of Docker:

```bash
uv venv
uv pip install -r requirements.txt
```
