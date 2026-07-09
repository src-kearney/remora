# Peak FLOPs Reference

Theoretical peak throughput figures used as denominators in remora's %-of-peak benchmarks with primary sources.

## RTX 4090 (Ada Lovelace, AD102, sm_89)

Primary source for all NVIDIA figures: [NVIDIA Ada GPU Architecture Whitepaper v2.02](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf), Appendix A, Table 2 ("GeForce RTX 4090 Full Specifications").

| Metric | Value |
|---|---|
| Boost clock | 2.52 GHz |
| CUDA cores | 16,384 |
| Tensor cores (4th gen) | 512 |
| Memory bandwidth | 1008 GB/s |
| FP32 (non-tensor, CUDA cores) | 82.6 TFLOPS |
| FP16 (non-tensor, CUDA cores) | 82.6 TFLOPS |
| FP16 Tensor, FP16 accumulate | 330.3 TFLOPS (dense) |
| FP16 Tensor, FP32 accumulate | 165.2 TFLOPS (dense) |

All tensor-core figures are dense. Structured 2:4 sparsity doubles each figure (e.g. 330.3 → 660.6 TFLOPS); remora does not (yet) use sparsity, so those numbers aren't tabulated here.

FP16 (non-tensor, CUDA cores) is not a separately published spec: Ada's CUDA cores have no packed-FP16 throughput path, so FP16 vector ops run at the same rate as FP32. This row is architectural, not a whitepaper figure.

## Which peak? Accumulate mode matters

FP16-input tensor-core GEMM has two real, both-published peaks, depending on whether the accumulator is FP16 or FP32: **330.3 TFLOPS with FP16 accumulate, 165.2 TFLOPS with FP32 accumulate**, half the FP16-accumulate rate.

The %-of-peak denominator must match the kernel's actual accumulate mode, or the reported percentage is wrong by 2×. **FP32 accumulate is the numerically standard choice**, requested via cuBLAS's `CUBLAS_COMPUTE_32F` compute type, so **165.2 TFLOPS is the default denominator** unless a kernel deliberately accumulates in FP16.

The 330.3-vs-165.2 split seen across secondary web sources is not a discrepancy or an error in either source. Both are real NVIDIA figures; they just describe different accumulate modes, usually without saying so.

## AMD (RDNA3, gfx1100 / 7900 XTX): placeholder

No AMD hardware yet. Values below are TODO, to be filled in from AMD's RDNA3 ISA / architecture documentation and confirmed empirically once the card is available.

| Metric | Value |
|---|---|
| Boost clock | TODO |
| Compute units | TODO |
| Memory bandwidth | TODO |
| FP32 (non-tensor) | TODO |
| FP16 (non-tensor) | TODO |
| FP16 Matrix (WMMA) | TODO: confirm accumulate-mode structure against RDNA3 ISA; may differ from Ada's clean 2× split |

## Code

`benchmarks/peaks.py` encodes these values in code. This document is the source of truth; if a figure here changes, update `peaks.py` to match.
