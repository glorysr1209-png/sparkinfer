# sparkinfer-kernels

Native C++/CUDA and CuTe DSL kernel library for edge MoE inference on **NVIDIA RTX Spark**, RTX 5090, and Jetson Thor.

Part of [gittensor-ai-lab](https://github.com/orgs/gittensor-ai-lab) — SN74.

---

## Why native CUDA + CuTe DSL, not Triton

| | Triton | Native CUDA + CuTe DSL |
|---|---|---|
| TMA (Tensor Memory Accelerator) | limited / indirect | full access |
| WGMMA (Blackwell warpgroup MMA) | not exposed | direct |
| Persistent kernel architecture | no | yes |
| Epilogue visitor composition | no | yes (CuTe) |
| Register layout control | no | full |
| sm_100 (RTX Spark) maturity | maturing | production-ready |
| Blackwell smem (228 KB/CTA) | partial | full |
| Performance ceiling | ~80% of peak | ~95%+ of peak |

Triton is useful for rapid prototyping. For the kernel-level optimization work that SN74 rewards — and for the specific challenges of RTX Spark's unified LPDDR5X memory — native CUDA + CuTe DSL is the only path to the performance ceiling.

---

## Stack

```
csrc/
├── cuda/              # native CUDA kernels — warp intrinsics, shared memory, CUDA primitives
│   ├── attention/     # flash decode: generic, GQA-8 (Qwen3.5), hd256-sw (Gemma4 local), hd512 (Gemma4 global)
│   ├── gemm/          # batched GEMM, GEMV for single-token decode
│   ├── moe/           # MoE router (256-expert top-8), token dispatch
│   ├── quant/         # quantization helpers
│   └── fused/         # CODA-style GEMM+epilogue (non-TMA path)
│
└── cute/              # CuTe DSL kernels — TMA, WGMMA, persistent, epilogue visitors
    ├── moe_swiglu/    # sync-free GroupGEMM + SwiGLU epilogue (no CPU sync, CUDA-graph safe)
    └── moe_gemm/      # sync-free GroupGEMM baseline

include/sparkinfer/kernels/
├── attention.h        # flash decode launch API
├── gemm.h             # GEMM / GEMV launch API
└── moe.h              # GroupGEMM + SwiGLU launch API
```

---

## Target kernels

### Attention
| Kernel | Model | Head dim | GQA | Window |
|---|---|---|---|---|
| `flash_decode.cu` | generic | 128 | any | full |
| `flash_decode_gqa8.cu` | Qwen3.5-35B-A3B | 128 | 8:1 | full |
| `flash_decode_local_hd256.cu` | Gemma 4 26B-A4B local layers | 256 | 2:1 | 1024 |
| `flash_decode_global_hd512.cu` | Gemma 4 26B-A4B global layers | **512** | 8:1 | full |

`head_dim=512` has no public implementation in FlashInfer, vLLM, or llama.cpp as of 2026-06.

### CuTe DSL — MoE
| Kernel | Operations fused | Sync-free |
|---|---|---|
| `moe_swiglu/group_gemm_swiglu.cu` | GroupGEMM + SwiGLU epilogue | yes — token counts on GPU |
| `moe_gemm/group_gemm.cu` | GroupGEMM | yes |

Sync-free design means token-per-expert counts live in GPU memory only. No CPU readback, no synchronization barrier mid-graph. Enables end-to-end CUDA graph capture across the full MoE forward pass — the primary latency win on RTX Spark.

---

## Build

```bash
cmake -B build \
  -DCMAKE_CUDA_ARCHITECTURES="100" \
  -DBUILD_TESTS=ON \
  -DBUILD_BENCHMARKS=ON
cmake --build build -j$(nproc)
```

CUTLASS (and CuTe DSL) is fetched automatically via CMake FetchContent.

---

## Namespace

```cpp
#include "sparkinfer/kernels/attention.h"
#include "sparkinfer/kernels/moe.h"

sparkinfer::kernels::launch_flash_decode_gqa8(...);
sparkinfer::kernels::launch_group_gemm_swiglu(...);
```
