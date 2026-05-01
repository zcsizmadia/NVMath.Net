// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the Apache 2.0 License.

using NVMathNet;
using NVMathNet.Sparse;

namespace NVMathNet.Net.Tests;

/// <summary>
/// Tests for the Sparse module (SpMV, SpMM). Requires a CUDA-capable GPU.
/// </summary>
public sealed class SparseTests
{
    // Build a simple diagonal CSR matrix of size N x N with value 'diagVal'
    private static CsrMatrix<float> MakeDiagCsr(int n, float diagVal)
    {
        float[] values = Enumerable.Repeat(diagVal, n).ToArray();
        int[]   colIdx = Enumerable.Range(0, n).ToArray();
        int[]   rowPtr = Enumerable.Range(0, n + 1).ToArray();

        var dValues  = new DeviceBuffer<float>(n);
        var dColIdx  = new DeviceBuffer<int>(n);
        var dRowPtr  = new DeviceBuffer<int>(n + 1);
        dValues.CopyFrom(values.AsSpan());
        dColIdx.CopyFrom(colIdx.AsSpan());
        dRowPtr.CopyFrom(rowPtr.AsSpan());

        return new CsrMatrix<float>(n, n, dRowPtr, dColIdx, dValues);
    }

    [Test]
    public async Task SpMV_DiagonalMatrix_ScalesVector()
    {
        const int N = 128;
        const float DiagVal = 3.0f;

        using var csr = MakeDiagCsr(N, DiagVal);

        float[] hX = new float[N];
        for (int i = 0; i < N; i++) hX[i] = i + 1.0f;

        using var dX = new DeviceBuffer<float>(N);
        using var dY = new DeviceBuffer<float>(N);
        dX.CopyFrom(hX.AsSpan());
        dY.Clear();

        await SparseLinearAlgebra.SpMVAsync(csr, dX, dY, alpha: 1.0f, beta: 0.0f);

        float[] result = dY.ToArray();
        for (int i = 0; i < N; i++)
        {
            float expected = DiagVal * hX[i];
            float err = Math.Abs(result[i] - expected);
            await Assert.That(err).IsLessThan(1e-4f);
        }
    }

    [Test]
    public async Task SpMM_DiagonalMatrix_ScalesMatrix()
    {
        const int N = 32;
        const int K = 4;
        const float DiagVal = 2.5f;

        using var csr = MakeDiagCsr(N, DiagVal);

        // Dense matrix B: N rows x K cols, B[i,j] = i+j+1 (col-major for cuSPARSE)
        float[] hB = new float[N * K];
        for (int i = 0; i < N; i++)
            for (int j = 0; j < K; j++)
                hB[j * N + i] = i + j + 1; // col-major

        using var dB = new DeviceBuffer<float>(N * K);
        using var dC = new DeviceBuffer<float>(N * K);
        dB.CopyFrom(hB.AsSpan());
        dC.Clear();

        await SparseLinearAlgebra.SpMMAsync(csr, dB, dC, K, alpha: 1.0f, beta: 0.0f);

        float[] result = dC.ToArray();
        for (int i = 0; i < N; i++)
            for (int j = 0; j < K; j++)
            {
                float expected = DiagVal * hB[j * N + i];
                float err = Math.Abs(result[j * N + i] - expected);
                await Assert.That(err).IsLessThan(1e-4f);
            }
    }
}
