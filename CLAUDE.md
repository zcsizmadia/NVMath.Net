# NVMath.Net — AI Assistant Context

## Project Overview

**NVMath.Net** is a .NET wrapper library for NVIDIA CUDA math libraries, modelled after Python's [nvmath-python](https://github.com/NVIDIA/nvmath-python). It wraps: cuFFT, cuBLASLt, cuBLAS, cuSPARSE, cuTENSOR, cuSOLVER, and cuRAND.

- **NuGet package ID**: `NVMath.Net`
- **Root namespace**: `NVMathNet`
- **Assembly name**: `NVMath.Net`
- **Version**: see `Directory.Build.props` → `<Version>`
- **License**: MIT
- **Author**: Zoltan Csizmadia
- **GitHub**: https://github.com/zcsizmadia/NVMath.Net

---

## Repository Layout

```
src/NVMathNet/          # Main library
  Interop/              # P/Invoke bindings (lazy-loaded via NativeLibraryLoader)
  Fft/                  # cuFFT wrapper
  LinAlg/               # cuBLASLt + cuBLAS wrappers (Matmul, TriangularSolve, SymmetricMultiply)
  Sparse/               # cuSPARSE wrapper (CsrMatrix, SparseLinearAlgebra)
  Tensor/               # cuTENSOR wrapper (TensorContraction)
  Solver/               # cuSOLVER wrapper (DenseSolver)
  Rand/                 # cuRAND wrapper (CudaRandom)
  CudaDevice.cs         # Device enumeration helpers
  CudaEvent.cs          # CUDA event wrapper
  CudaStream.cs         # CUDA stream wrapper
  DeviceBuffer.cs       # GPU memory buffer (generic, IDisposable)
  PinnedBuffer.cs       # Pinned host memory buffer
  MultiGpuContext.cs    # Multi-GPU concurrency helper
tests/NVMathNet.Tests/  # TUnit test suite (requires CUDA GPU at runtime)
samples/                # One project per feature demonstrating usage
docs/api/               # API documentation per module
.github/workflows/      # CI: build.yml (build + pack; tests skipped — no GPU)
```

---

## Build System

### SDK & Frameworks

- `global.json` pins SDK **10.0.100** with `latestFeature` rollForward
- CI (`$CI=true`) targets `net8.0;net9.0;net10.0`
- Local builds target `net10.0` only
- Controlled via `ClassLibraryTargetFrameworks` in `Directory.Build.props`

### Common Commands

```bash
# Restore
dotnet restore

# Build (local — net10.0 only)
dotnet build

# Build (CI — all TFMs)
CI=true dotnet build -c Release --no-restore

# Pack NuGet
dotnet pack src/NVMathNet/ -c Release -o nupkgs

# Run tests (requires CUDA GPU)
dotnet test

# Run a specific sample
dotnet run --project samples/FftSample/
```

### Key `Directory.Build.props` Properties

| Property | CI value | Local value |
|---|---|---|
| `ClassLibraryTargetFrameworks` | `net8.0;net9.0;net10.0` | `net10.0` |
| `TestTargetFrameworks` | `net8.0;net9.0;net10.0` | `net10.0` |
| `LangVersion` | `latest` | `latest` |
| `Nullable` | `enable` | `enable` |
| `ImplicitUsings` | `enable` | `enable` |

Missing XML doc comments (`CS1591`) are treated as **errors** for library projects that set `<GenerateDocumentationFile>true</GenerateDocumentationFile>`.

---

## Native Library Dependencies

| Library | Windows DLL | Linux SO | Version |
|---|---|---|---|
| CUDA Runtime | `cudart64_13.dll` | `libcudart.so.13` | CUDA 13.x |
| cuBLAS | `cublas64_13.dll` | `libcublas.so.13` | 13.x |
| cuBLASLt | `cublasLt64_13.dll` | `libcublasLt.so.13` | 13.x |
| cuFFT | `cufft64_12.dll` | `libcufft.so.12` | 12.x |
| cuSPARSE | `cusparse64_12.dll` | `libcusparse.so.12` | 12.x |
| cuSOLVER | `cusolver64_12.dll` | `libcusolver.so.12` | 12.x |
| cuRAND | `curand64_10.dll` | `libcurand.so.10` | 10.x |
| cuTENSOR | `cutensor.dll` | `libcutensor.so.2` | 2.x |

Native libraries are resolved via **`NativeLibraryLoader`** (lazy-loaded delegates — no `[DllImport]`). The native DLL directories must be on `PATH` (Windows) or `LD_LIBRARY_PATH` (Linux).

---

## Architecture & Coding Conventions

### Interop Layer (`src/NVMathNet/Interop/`)

- No `[DllImport]` — all native calls go through `NativeLibraryLoader` which loads the DLL on first use.
- Each native library has its own `*Native.cs` file (e.g., `CuFftNative.cs`, `CuBlasNative.cs`).
- `Exceptions.cs` defines CUDA status exception types.

### Resource Ownership

- **`_streamOwned` pattern**: wrapper classes accept an external `CudaStream` or create one internally. They only dispose the stream if they created it (`_streamOwned = true`).
- **Sparse matrices** (`CsrMatrix`) own their `DeviceBuffer` allocations and dispose them in `Dispose()`.
- All public classes that hold GPU resources implement `IDisposable`; always use `using` in consuming code.

### API Style

- All GPU operations are `async` and return `Task` or `Task<T>`.
- Matrix layouts are **column-major** (matching cuBLAS conventions).
- `DeviceBuffer<T>` is generic over numeric types; unsafe blocks are allowed (`<AllowUnsafeBlocks>true`).

### Tests

- Test framework: **TUnit 1.41.0**
- Test files: `tests/NVMathNet.Tests/`
- Tests are skipped in CI because no GPU is available; they are commented out in `build.yml`.
- GPU hardware requirement: compute capability 7.0+.

---

## CI Workflow (`.github/workflows/build.yml`)

Two jobs:
1. **build** — `dotnet restore` + `dotnet build -c Release --no-restore` on `ubuntu-latest`
2. **pack** — `dotnet pack src/NVMathNet/ -c Release -o nupkgs` + uploads artifact `nuget-package`

Tests are intentionally **commented out** in CI (`# Tests require NVIDIA GPU`).

---

## Adding a New Module

1. Create a subdirectory under `src/NVMathNet/<ModuleName>/`.
2. Add native bindings to `src/NVMathNet/Interop/<LibName>Native.cs`.
3. Implement the public wrapper with `IDisposable`, `_streamOwned` pattern, and `async` methods.
4. Add XML doc comments on all public members (required — CS1591 is an error).
5. Add tests in `tests/NVMathNet.Tests/<ModuleName>Tests.cs`.
6. Add a sample in `samples/<ModuleName>Sample/`.
7. Document the API in `docs/api/<ModuleName>.md`.
