---
_layout: landing
---

# NVMath.Net

NVMath.Net brings NVIDIA's CUDA math libraries to .NET. It is a C# counterpart to
[nvmath-python](https://github.com/NVIDIA/nvmath-python): a small set of idiomatic, `async`-friendly
classes over **cuFFT, cuBLASLt, cuBLAS, cuSPARSE, cuDSS, cuTENSOR, cuSOLVER and cuRAND**, with the
same plan → execute workflow, explicit resource ownership and no native build step.

```csharp
using NVMathNet;
using NVMathNet.LinAlg;

using var matmul = new Matmul(m: 4, n: 4, k: 4);
await matmul.PlanAsync();
await matmul.ExecuteAsync<float>(a, b, c, d, alpha: 1.0, beta: 0.0);   // D = α·A·B + β·C on the GPU
```

## What is in the box

| Area | Highlights |
|---|---|
| **FFT** | C2C / R2C / C2R, N-D and batched transforms, leading- or trailing-axis layouts, output normalization |
| **Matrix multiplication** | cuBLASLt `Matmul` with epilogs, bias, batching, FP8 scaling, autotuning; GEMM, TRSM, SYMM, DGMM |
| **Sparse** | CSR / COO / BSR / CSC, SpMV, SpMM, SpSV, SpSM, SDDMM, SpGEMM, transpose; cuDSS direct solver |
| **Tensors** | Contractions, element-wise unary/binary/trinary operations, permutation, reduction |
| **Dense solvers** | LU, QR, least squares, Cholesky, SVD, eigenvalues — single, batched and complex variants |
| **Random numbers** | Pseudo- and quasi-random generators, uniform / normal / log-normal / Poisson, skip-ahead |
| **Infrastructure** | `DeviceBuffer<T>`, streams, events, multi-GPU helpers, pluggable device memory managers |

## Where to go next

- **New here?** Read [Getting started](articles/getting-started.md), then [Core concepts](articles/concepts.md).
- **Native libraries not found?** See [Installing the native libraries](articles/native-libraries.md).
- **Know nvmath-python?** See the [mapping](articles/nvmath-python.md).
- **Looking for a type or method?** Use the [API reference](reference/NVMathNet.yml) or the [module guides](api/index.md).

NVMath.Net targets .NET 8, 9 and 10 and needs an NVIDIA GPU (compute capability 7.0 or newer recommended)
with the matching CUDA libraries on the library path at run time.
