# Getting started

This page takes you from an empty project to a working GPU computation.

## Requirements

- An NVIDIA GPU. Compute capability 7.0 or newer is recommended.
- The NVIDIA driver and the CUDA libraries your code uses (see [Installing the native libraries](native-libraries.md)).
- .NET 8, 9 or 10.

NVMath.Net has **no native build step**: the NVIDIA libraries are loaded at run time, so the same
managed assembly runs on Windows and Linux.

## Install

```bash
dotnet add package NVMath.Net
```

or reference the project directly:

```xml
<ProjectReference Include="path/to/src/NVMathNet/NVMathNet.csproj" />
```

The library uses unsafe code internally; your own project does not need `AllowUnsafeBlocks`.

## Your first computation: a matrix product

```csharp
using NVMathNet;
using NVMathNet.LinAlg;

const int N = 4;
float[] hostA = Enumerable.Range(1, N * N).Select(i => (float)i).ToArray();
float[] hostB = new float[N * N];
for (int i = 0; i < N; i++) hostB[i + N * i] = 1f;          // identity (column-major)

using var a = new DeviceBuffer<float>(N * N);
using var b = new DeviceBuffer<float>(N * N);
using var c = new DeviceBuffer<float>(N * N);
using var d = new DeviceBuffer<float>(N * N);
a.CopyFrom(hostA);
b.CopyFrom(hostB);
c.Clear();

// D = alpha * A * B + beta * C
using var matmul = new Matmul(m: N, n: N, k: N, options: new MatmulOptions { Blocking = true });
await matmul.PlanAsync();                                      // choose an algorithm
await matmul.ExecuteAsync<float>(a, b, c, d, alpha: 1.0, beta: 0.0);

float[] result = d.ToArray();                                  // equals hostA
```

Three things to notice:

1. **Data lives in `DeviceBuffer<T>`**, GPU memory you allocate, fill and dispose explicitly.
2. **Operations follow a plan → execute lifecycle.** Construct the operation, call `Plan`/`PlanAsync` once,
   then `Execute`/`ExecuteAsync` as many times as you like. Planning is the expensive part.
3. **Matrices are column-major**, the cuBLAS convention. Element *(row, col)* of an `m × n` matrix is at `row + m * col`.

## One-shot helpers

Some modules have static helpers that create a plan, run it and clean up, which is convenient for
occasional use:

```csharp
using NVMathNet.Fft;

using var output = await Fft.FftAsync(input, new long[] { 512 });   // forward complex FFT
```

Plan-based classes are better when the same shape is reused.

## Choosing what to read next

| I want to… | Read |
|---|---|
| Understand buffers, streams and ownership | [Core concepts](concepts.md) |
| Transform signals | [FFT guide](../api/Fft.md) |
| Multiply matrices, apply epilogs, use FP8 | [Linear algebra guide](../api/Linalg.md) |
| Work with sparse matrices or solve sparse systems | [Sparse guide](../api/Sparse.md) |
| Contract, permute or reduce tensors | [Tensor guide](../api/Tensor.md) |
| Factorise or solve dense systems | [Solver guide](../api/Solver.md) |
| Generate random numbers | [Random numbers guide](../api/Rand.md) |
| Port nvmath-python code | [Coming from nvmath-python](nvmath-python.md) |
