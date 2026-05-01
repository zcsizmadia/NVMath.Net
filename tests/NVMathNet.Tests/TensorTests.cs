// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the MIT License.

using NVMathNet.Tensor;

namespace NVMathNet.Tests;

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
        for (int i = 0; i < M * K; i++)
        {
            hA[i] = i + 1.0f;
        }

        for (int i = 0; i < K * N; i++)
        {
            hB[i] = (i % K) == (i / K) ? 1.0f : 0.0f; // identity
        }

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

    [Test]
    public async Task TensorContraction_NonTrivialProduct()
    {
        // A (3x2) * B (2x4) = D (3x4) with non-trivial data
        // Mode labels: m=0, n=1, k=2
        const int M = 3, N = 4, K = 2;

        // A[r,c]: [[1,4],[2,5],[3,6]]  stored row-major in memory
        float[] hA = [1, 2, 3, 4, 5, 6]; // element (r,c) at r + M*c
        // B[r,c]: [[7,9,11,13],[8,10,12,14]]
        float[] hB = [7, 8, 9, 10, 11, 12, 13, 14]; // element (r,c) at r + K*c

        // CPU reference D[r,c] = Σ_k A[r,k]*B[k,c]
        float[] expected = new float[M * N];
        for (int r = 0; r < M; r++)
        {
            for (int c = 0; c < N; c++)
            {
                float sum = 0;
                for (int kk = 0; kk < K; kk++)
                {
                    sum += hA[r + M * kk] * hB[kk + K * c];
                }

                expected[r + M * c] = sum;
            }
        }

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

        float[] result = dD.ToArray();
        for (int i = 0; i < M * N; i++)
        {
            float err = Math.Abs(result[i] - expected[i]);
            await Assert.That(err).IsLessThan(1e-2f);
        }
    }

    [Test]
    public async Task TensorContraction_TypedExecuteAsync()
    {
        // Test the DeviceBuffer<T> generic overload
        const int S = 4;
        float[] hA = new float[S * S];
        float[] hB = new float[S * S];
        for (int i = 0; i < S * S; i++)
        {
            hA[i] = i + 1f;
            hB[i] = (S * S) - i;
        }

        // CPU reference
        float[] expected = new float[S * S];
        for (int r = 0; r < S; r++)
        {
            for (int c = 0; c < S; c++)
            {
                float sum = 0;
                for (int kk = 0; kk < S; kk++)
                {
                    sum += hA[r + S * kk] * hB[kk + S * c];
                }

                expected[r + S * c] = sum;
            }
        }

        using var dA = new DeviceBuffer<float>(S * S);
        using var dB = new DeviceBuffer<float>(S * S);
        using var dC = new DeviceBuffer<float>(S * S);
        using var dD = new DeviceBuffer<float>(S * S);
        dA.CopyFrom(hA.AsSpan());
        dB.CopyFrom(hB.AsSpan());
        dC.Clear();
        dD.Clear();

        long[] ext = [S, S];
        int[] modeA = [0, 2];
        int[] modeB = [2, 1];
        int[] modeC = [0, 1];

        var opts = new TensorContractionOptions { Blocking = true };
        using var tc = new TensorContraction(
            ext, modeA, ext, modeB, ext, modeC, ext, modeC,
            TensorDataType.Float32, opts);
        await tc.PlanAsync();
        await tc.ExecuteAsync<float>(dA, dB, dC, dD, alpha: 1.0, beta: 0.0);

        float[] result = dD.ToArray();
        for (int i = 0; i < S * S; i++)
        {
            float err = Math.Abs(result[i] - expected[i]);
            await Assert.That(err).IsLessThan(1e-1f);
        }
    }
}
