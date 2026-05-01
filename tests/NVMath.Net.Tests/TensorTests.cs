// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the Apache 2.0 License.

using NVMathNet;
using NVMathNet.Tensor;

namespace NVMathNet.Net.Tests;

/// <summary>
/// Tests for the Tensor contraction module (cuTENSOR). Requires a CUDA-capable GPU.
/// </summary>
public sealed class TensorTests
{
    [Test]
    public async Task TensorContraction_MatrixMultiply()
    {
        // Represent matrix multiplication as a tensor contraction:
        // C[m,n] = A[m,k] * B[k,n]
        // Mode labels: m=0, n=1, k=2
        const int M = 4, N = 4, K = 4;

        float[] hA = new float[M * K];
        float[] hB = new float[K * N];
        for (int i = 0; i < M * K; i++) hA[i] = i + 1.0f;
        for (int i = 0; i < K * N; i++) hB[i] = (i % K) == (i / K) ? 1.0f : 0.0f; // identity

        using var dA = new DeviceBuffer<float>(M * K);
        using var dB = new DeviceBuffer<float>(K * N);
        using var dC = new DeviceBuffer<float>(M * N);
        using var dD = new DeviceBuffer<float>(M * N);
        dA.CopyFrom(hA.AsSpan());
        dB.CopyFrom(hB.AsSpan());
        dC.Clear();
        dD.Clear();

        long[] extA = [M, K];
        long[] extB = [K, N];
        long[] extC = [M, N];
        long[] extD = [M, N];
        int[] modeA = [0, 2];  // m, k
        int[] modeB = [2, 1];  // k, n
        int[] modeC = [0, 1];  // m, n
        int[] modeD = [0, 1];  // m, n

        var opts = new TensorContractionOptions { Blocking = true };

        using var contraction = new TensorContraction(
            extA, modeA, extB, modeB, extC, modeC, extD, modeD,
            TensorDataType.Float32, opts);
        await contraction.PlanAsync();
        await contraction.ExecuteAsync(
            dA.PointerAsInt, dB.PointerAsInt, dC.PointerAsInt, dD.PointerAsInt,
            alpha: 1.0, beta: 0.0);

        // A * I = A
        float[] result = dD.ToArray();
        for (int i = 0; i < M * N; i++)
        {
            float err = Math.Abs(result[i] - hA[i]);
            await Assert.That(err).IsLessThan(1e-3f);
        }
    }
}
