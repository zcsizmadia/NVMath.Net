# Interop & Exceptions — `NVMathNet.Interop`

Raw P/Invoke bindings. Not intended for direct use by application code.

## Native Binding Classes

| Class | CUDA library |
|---|---|
| `CudaRuntime` | CUDA Runtime |
| `CuFftNative` | cuFFT |
| `CuBlasNative` | cuBLAS / cuBLASLt |
| `CuSparseNative` | cuSPARSE |
| `CuTensorNative` | cuTENSOR v2 |
| `CuSolverNative` | cuSOLVER |
| `CuRandNative` | cuRAND |
| `NativeLibraryLoader` | Cross-platform library loader |

## Exceptions

| Exception | Thrown by |
|---|---|
| `CudaException` | CUDA runtime errors |
| `CuFftException` | cuFFT errors |
| `CuBlasException` | cuBLAS / cuBLASLt errors |
| `CuSparseException` | cuSPARSE errors |
| `CuTensorException` | cuTENSOR errors |
| `CuSolverException` | cuSOLVER errors |
| `CuRandException` | cuRAND errors |

All exceptions expose an `int ErrorCode` property with the native status code.
