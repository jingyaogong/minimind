# MiniMind native vLLM adapter

This out-of-tree plugin targets the MiniMind hybrid checkpoint format and
vLLM 0.17.0. It keeps the checkpoint and the repository's existing model
implementation unchanged.

## Start

From the repository root in PowerShell:

```powershell
.\run_vllm_native.ps1
```

The default checkpoint directory is
`D:\Tools\code\TinyLLM\out\grpo_768_hybrid_hf`. Override it when needed:

```powershell
.\run_vllm_native.ps1 -ModelPath 'D:\path\to\your\converted\checkpoint'
```

The script builds `minimind:vllm-native-0.17.0`, mounts the checkpoint read-only,
and starts the OpenAI-compatible API on `http://127.0.0.1:8000/v1`. If local
Python is on PATH, it also starts a Chinese web chat at
`http://127.0.0.1:8080`. The API and web chat bind to loopback only.

## Stop

Stop the model container with:

```powershell
docker stop minimind-vllm-native
```

The launch script prints the web chat helper's process ID; stop that specific
process with `Stop-Process -Id <PID>` when finished.
