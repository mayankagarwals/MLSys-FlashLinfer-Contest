---
description: How to access the remote B200 GPU server for building and testing CUDA kernels
---

# B200 Remote GPU Access

## SSH Connection
The remote server requires specific SSH options to avoid agent forwarding issues:

```bash
export SSH_AUTH_SOCK=""
SSH_OPTS="-o StrictHostKeyChecking=no -o IdentitiesOnly=yes -i $HOME/.ssh/id_ed25519_9701account -F /dev/null"
ssh $SSH_OPTS yue@95.133.252.11 'COMMAND'
```

**All three are required**: `SSH_AUTH_SOCK=""`, `-o IdentitiesOnly=yes`, `-F /dev/null`.

## SCP Files to Remote
```bash
export SSH_AUTH_SOCK=""
scp -o StrictHostKeyChecking=no -o IdentitiesOnly=yes -i $HOME/.ssh/id_ed25519_9701account -F /dev/null \
  LOCAL_FILE yue@95.133.252.11:/home/yue/MLSys-FlashLinfer-Contest/solution/cuda/
```

## Build & Test Workflow Example (path might changed, you need to read the repo for latest)
Note: there's a venv we can use on the machine under /home/yue
1. Edit kernel locally in `solution/cuda/`
2. SCP changed files to remote
3. **Clear build cache** (required after code changes):
   ```bash
   rm -rf /home/yue/.cache/flashinfer_bench/cache/python/fib_python_gdn_prefill_cuda_tcgen05*
   ```
4. Run test from `/home/yue/MLSys-FlashLinfer-Contest/scripts/`:
   ```bash
   cd /home/yue/MLSys-FlashLinfer-Contest/scripts && uv run python3 -c 'TEST_CODE'
   ```
Use the run_local_fast variants to make the test run fast.

## Key Paths on Remote
- Project: `/home/yue/MLSys-FlashLinfer-Contest/`
- Dataset (100 workloads): `/home/yue/mlsys26-contest/`
- Build cache: `/home/yue/.cache/flashinfer_bench/cache/python/`
- Scripts: `/home/yue/MLSys-FlashLinfer-Contest/scripts/`

## Common Test Pattern
```bash
export SSH_AUTH_SOCK="" && \
scp -o StrictHostKeyChecking=no -o IdentitiesOnly=yes -i $HOME/.ssh/id_ed25519_9701account -F /dev/null \
  /Users/yzhang16/MLSys-FlashLinfer-Contest/solution/cuda/FILE.cu \
  yue@95.133.252.11:/home/yue/MLSys-FlashLinfer-Contest/solution/cuda/ && \
ssh -o StrictHostKeyChecking=no -o IdentitiesOnly=yes -i $HOME/.ssh/id_ed25519_9701account -F /dev/null \
  yue@95.133.252.11 "killall -9 python3 2>/dev/null; \
  rm -rf /home/yue/.cache/flashinfer_bench/cache/python/fib_python_gdn_prefill_cuda_tcgen05*; \
  cd /home/yue/MLSys-FlashLinfer-Contest/scripts && timeout 120 uv run python3 -c 'TEST_CODE'"
```

## GPU Info
- GPU: NVIDIA B200 (Blackwell, sm_100a)
- Driver: 590.48.01, CUDA 13.1
- 192 SMs, 183GB HBM
