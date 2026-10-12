# Installing the native libraries

NVMath.Net does not ship NVIDIA binaries. Install the ones you need and make sure they can be found when
your program starts. A library is loaded **lazily on first use**, so you only need the libraries for the modules
you actually call.

## What each module needs

| Module | Native library | Windows file | Linux file | Version |
|---|---|---|---|---|
| Core (devices, buffers, streams) | CUDA runtime | `cudart64_13.dll` | `libcudart.so.13` | CUDA 13.x |
| Linear algebra | cuBLAS, cuBLASLt | `cublas64_13.dll`, `cublasLt64_13.dll` | `libcublas.so.13`, `libcublasLt.so.13` | 13.x |
| FFT | cuFFT | `cufft64_12.dll` | `libcufft.so.12` | 12.x |
| Sparse | cuSPARSE | `cusparse64_12.dll` | `libcusparse.so.12` | 12.x |
| Sparse direct solver | cuDSS | `cudss64_0.dll` | `libcudss.so.0` | 0.x |
| Dense solvers | cuSOLVER (+ cuBLAS) | `cusolver64_12.dll` | `libcusolver.so.12` | 12.x |
| Random numbers | cuRAND | `curand64_10.dll` | `libcurand.so.10` | 10.x |
| Tensors | cuTENSOR | `cutensor.dll` | `libcutensor.so.2` | 2.x |

cuBLAS, cuBLASLt, cuFFT, cuSPARSE, cuSOLVER and cuRAND come with the **CUDA Toolkit**. cuTENSOR and cuDSS are
separate downloads and are only needed for the tensor module and `SparseDirectSolver`.

## Installing

### CUDA Toolkit

Download the CUDA Toolkit 13.x installer from NVIDIA (or use your distribution's packages) and install it. The
GPU driver must support the toolkit version.

### cuTENSOR and cuDSS

Both are available from NVIDIA's developer downloads (pick the build for your CUDA major version), as pip wheels
(`cutensor-cu13`, and the corresponding cuDSS wheel) or through conda-forge. On Windows the archives install into
`C:\Program Files\NVIDIA cuTENSOR\vX.Y\` and `C:\Program Files\NVIDIA cuDSS\vX.Y\`, with the DLLs under `bin\12` and
`bin\13`.

## How the library finds them

Libraries are resolved by name in this order:

1. The operating system's normal search path: `PATH` on Windows, `LD_LIBRARY_PATH` / `ldconfig` on Linux.
2. **Windows:** the CUDA Toolkit folders under `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin` (newest first),
   and the standard install folders of cuTENSOR and cuDSS (`…\bin\<cuda-major>`, newest first). The `CUTENSOR_ROOT`
   and `CUDSS_ROOT` environment variables can point at a custom location.
3. **Linux:** `/usr/local/cuda/lib64` and `/usr/lib/x86_64-linux-gnu`.

In practice a default CUDA Toolkit, cuTENSOR and cuDSS installation on Windows works without touching `PATH`. If
you installed elsewhere, add the folder containing the DLLs (or `.so` files) to `PATH` / `LD_LIBRARY_PATH`.

## Checking your setup

```csharp
using NVMathNet;

Console.WriteLine($"Devices: {CudaDevice.GetDeviceCount()}");
Console.WriteLine($"Driver:  {CudaDevice.GetDriverVersion()}  Runtime: {CudaDevice.GetRuntimeVersion()}");
```

If a library cannot be found you get a `DllNotFoundException` naming the file. See [Troubleshooting](troubleshooting.md).
