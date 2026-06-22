# sparkinfer

**Blackwell-native MoE/LLM inference runtime.** The engineering arm of [SN74 on Gittensor](https://github.com/gittensor-ai-lab) — reproducible, hardware-level inference-speed gains for NVIDIA Blackwell consumer/edge GPUs: RTX Spark (`sm_121`), RTX 5090 & RTX PRO 6000 (`sm_120`), Jetson Thor (`sm_121`).

Unified monorepo for the kernels, MoE engine, runtime, and benchmarks. (The `agent` autotuner and the private `kernel-wiki` live in their own repos.)

## Proven

Qwen3-30B-A3B (Q4_K_M GGUF) runs end-to-end on an RTX PRO 6000 (sm_120), decode optimized **0.60 → 134 tok/s (≈220×)** across 6 source-verifiable passes — within **1.8×** of llama.cpp on the same model + GPU, output verified correct, **21.7 GB** resident (experts kept quantized).

## Layout & emission weights

| Path | Weight | What |
|---|--:|---|
| [`kernels/`](kernels) | **0.42** | CUDA kernels — flash-decode (hd128/256/512), decode GEMV, fused quantized MoE expert FFN, GEMM, RMSNorm, RoPE, GGUF dequant |
| [`runtime/`](runtime) | **0.26** | scheduler, paged KV cache, CUDA-graph decode, native GGUF loading, model forward |
| [`moe/`](moe) | **0.21** | sync-free MoE router + expert dispatch (on-device counts, CUDA-graph-ready) |
| [`bench/`](bench) | **0.11** | reproducible benchmarks + eval harness (source-required builds, frozen weights) |

`Weight` = intra-repo emission share for SN74 (**path-based**, sums to 1.0; see [`.gittensor/weights.json`](.gittensor/weights.json)). Performance paths (`kernels`/`runtime`/`moe`) are scored by **verified frontier-delta speedup** (XL/L/M/S/XS); `bench/` by code quality. Weights are **maturity-adaptive** — see the [org reward model](https://github.com/gittensor-ai-lab).

## Build

Requires **CUDA Toolkit 12.8+** (first toolkit with `sm_120` / `sm_121` codegen).

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=120   # or 121 for RTX Spark / Jetson Thor
cmake --build build -j
ctest --test-dir build
```

The top-level `CMakeLists.txt` is a superbuild (`kernels → moe → runtime`); each subsystem also builds standalone (the sibling `../kernels` references resolve within the monorepo). A direct `nvcc` build from the repo root works too — see [`bench/scripts`](bench/scripts).

## Targets

**Blackwell only, by design:** `sm_120` (RTX 5090, RTX PRO 6000) and `sm_121` (RTX Spark / GB10, Jetson Thor). **Not** `sm_100` (datacenter B200/GB200 — binary-incompatible).

## Contributing

Source-required and reproducible — no pre-built binaries. Contributions are rewarded on SN74 by the **verified marginal speedup** they add over the live frontier, correctness-gated against a frozen reference, validated on both basket models (Qwen + Gemma). Full model: the [org profile](https://github.com/gittensor-ai-lab).
