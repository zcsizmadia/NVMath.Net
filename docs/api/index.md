# Module reference

Hand-written guides to each module: what it wraps, the types you use, and the conventions that apply.
For the complete, generated listing of every public type and member see the [API reference](../reference/NVMathNet.yml).

| Namespace | Wraps | Guide |
|---|---|---|
| `NVMathNet` | CUDA runtime | [Core](Core.md) |
| `NVMathNet.Memory` | CUDA allocation | [Core](Core.md) |
| `NVMathNet.Fft` | cuFFT | [FFT](Fft.md) |
| `NVMathNet.LinAlg` | cuBLASLt, cuBLAS | [Linear algebra](Linalg.md) |
| `NVMathNet.Sparse` | cuSPARSE, cuDSS | [Sparse](Sparse.md) |
| `NVMathNet.Tensor` | cuTENSOR | [Tensor](Tensor.md) |
| `NVMathNet.Solver` | cuSOLVER (dense), cuBLAS batched | [Dense solvers](Solver.md) |
| `NVMathNet.Rand` | cuRAND | [Random numbers](Rand.md) |
| `NVMathNet.Interop` | raw bindings | [Interop](Interop.md) |

All matrices and tensors are **column-major** unless a guide says otherwise.
