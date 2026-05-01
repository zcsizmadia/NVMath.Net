# NVMath.Net — Copilot Instructions

## What This Project Is

A .NET wrapper for NVIDIA CUDA math libraries (cuFFT, cuBLASLt, cuBLAS, cuSPARSE, cuTENSOR, cuSOLVER, cuRAND). Modelled after [nvmath-python](https://github.com/NVIDIA/nvmath-python).

- **Root namespace**: `NVMathNet`  
- **Assembly**: `NVMath.Net`  
- **NuGet ID**: `NVMath.Net`  
- **SDK**: .NET 10 (see `global.json`)  
- **Target frameworks**: `net8.0;net9.0;net10.0` (CI) / `net10.0` (local)

---

## Project Structure

```
src/NVMathNet/          Main library
  Interop/              Native bindings (lazy-loaded, no DllImport)
  Fft/                  cuFFT
  LinAlg/               cuBLASLt + cuBLAS
  Sparse/               cuSPARSE
  Tensor/               cuTENSOR
  Solver/               cuSOLVER
  Rand/                 cuRAND
tests/NVMathNet.Tests/  TUnit 1.41.0 tests (need CUDA GPU to run)
samples/                One sample project per feature
docs/api/               API docs per module
```

---

## Build & Test

```bash
dotnet build                          # local (net10.0)
CI=true dotnet build -c Release       # CI (all TFMs)
dotnet pack src/NVMathNet/ -c Release -o nupkgs
dotnet test                           # requires CUDA GPU
```

CI (`build.yml`) runs build + pack on `ubuntu-latest`; tests are commented out (no GPU).

---

## Architecture Rules

### Native Interop
- **No `[DllImport]`** — all native calls use `NativeLibraryLoader` (lazy delegate loading).
- Each library has its own `*Native.cs` in `Interop/` (e.g., `CuFftNative.cs`).

### Resource Ownership
- `_streamOwned` pattern: classes take an optional external `CudaStream`; they only dispose it if they created it.
- All GPU-resource classes implement `IDisposable`. Consumers must use `using`.
- Sparse matrices own their `DeviceBuffer` instances and dispose them.

### API Conventions
- All GPU operations are **`async`** (`Task`/`Task<T>`).
- Matrices are **column-major** (cuBLAS convention).
- `DeviceBuffer<T>` is generic; `AllowUnsafeBlocks` is enabled.
- XML doc comments on all public members are **required** — `CS1591` is a build error.

---

## Native Library Versions

| Library | Windows | Linux | Version |
|---|---|---|---|
| CUDA Runtime | `cudart64_13.dll` | `libcudart.so.13` | 13.x |
| cuBLAS | `cublas64_13.dll` | `libcublas.so.13` | 13.x |
| cuBLASLt | `cublasLt64_13.dll` | `libcublasLt.so.13` | 13.x |
| cuFFT | `cufft64_12.dll` | `libcufft.so.12` | 12.x |
| cuSPARSE | `cusparse64_12.dll` | `libcusparse.so.12` | 12.x |
| cuSOLVER | `cusolver64_12.dll` | `libcusolver.so.12` | 12.x |
| cuRAND | `curand64_10.dll` | `libcurand.so.10` | 10.x |
| cuTENSOR | `cutensor.dll` | `libcutensor.so.2` | 2.x |

---

## Adding a New Module Checklist

1. `src/NVMathNet/<Module>/` — wrapper implementation
2. `src/NVMathNet/Interop/<Lib>Native.cs` — native bindings
3. Public API: `IDisposable`, `_streamOwned`, `async` methods, XML docs
4. `tests/NVMathNet.Tests/<Module>Tests.cs` — TUnit tests
5. `samples/<Module>Sample/` — runnable sample
6. `docs/api/<Module>.md` — API documentation
