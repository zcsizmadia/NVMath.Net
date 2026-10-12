# Troubleshooting

## `DllNotFoundException: Unable to load shared library 'cufft64_12.dll'` (or similar)

The named NVIDIA library could not be found. Install it (see [Installing the native libraries](native-libraries.md)) and make
sure its folder is on `PATH` (Windows) or `LD_LIBRARY_PATH` (Linux). Remember that the file name encodes the CUDA
generation: a CUDA 12 toolkit provides `cublas64_12.dll`, but this library expects the CUDA 13 names listed on that page.

## `EntryPointNotFoundException: Symbol '…' not found`

The library was found but is an incompatible version, for example an older cuTENSOR 1.x where 2.x is required, or a
cuDSS release that predates the function. Install the version listed in the requirements.

## `CudaException: no CUDA-capable device is detected` / driver errors

Check `nvidia-smi`. The driver must be recent enough for the CUDA runtime you installed, and the process must be able
to see a GPU (in containers, use the NVIDIA container runtime).

## `CudaException: an illegal memory access was encountered`

A native routine read or wrote outside a buffer. CUDA errors are **sticky**: after one illegal access every later CUDA
call in the process fails, often with the same message from an unrelated line such as a `Dispose`. Typical causes:

- A buffer smaller than the operation needs (check the documented size, for example `batchCount · n · n`).
- A matrix passed in row-major order where column-major is required.
- Using a `DeviceBuffer<T>` after it was disposed, or disposing it while queued work still uses it. Synchronise the
  stream before disposing.

## Results look transposed or scrambled

Matrices are **column-major**. A row-major array uploaded as-is represents the transpose. Either transpose on the host
or swap the roles of the operands.

## A cuSPARSE / cuDSS call fails with `InvalidValue`

- CSR column indices must be **sorted within each row** for cuDSS and most cuSPARSE routines.
- Sizes (`rows`, `cols`, `nnz`) must match the arrays. `SparseDirectSolver` needs a square matrix.
- Symmetric and positive-definite structures with a `Lower`/`Upper` view expect only that triangle to be stored.

## cuTENSOR returns `NotSupported`

Some operation/mode combinations are not supported by the library. For `ElementwiseBinary`, `C` must have the same
modes and extents as `D`; only `A` may be permuted. Check data types and alignment as well.

## Slow first call

Planning, library loading and, for some libraries, JIT compilation happen on first use. Reuse plan objects and warm up
before timing.

## Tests are skipped in CI

The GPU test suite needs real hardware and is not run by the hosted CI. Run it locally with `dotnet test`; the build
workflow only checks that the library, tests and samples compile.
