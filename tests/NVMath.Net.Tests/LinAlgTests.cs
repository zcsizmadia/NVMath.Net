// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the Apache 2.0 License.

using NVMathNet;
using NVMathNet.LinAlg;

namespace NVMathNet.Net.Tests;

/// <summary>
/// Tests for the LinAlg (cuBLASLt) matrix multiplication module.
/// Requires a CUDA-capable GPU.
/// </summary>
public sealed class LinAlgTests
{
    // Helper: create a row-major float matrix on device, initialized to a[i,j] = (i+1)*(j+1)
    private static DeviceBuffer<float> MakeMatrix(int rows, int cols)
    {
        float[] h = new float[rows * cols];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                h[r * cols + c] = (r + 1) * (c + 1);
        var buf = new DeviceBuffer<float>(rows * cols);
        buf.CopyFrom(h.AsSpan());
        return buf;
    }

    [Test]
    public async Task Matmul_Float_Identity()
    {
        // C = A * I  =>  C == A
        const int M = 4, N = 4, K = 4;

        // A[i,j] = i+j+1  (row-major)
        float[] hA = new float[M * K];
        for (int i = 0; i < M; i++)
            for (int j = 0; j < K; j++)
                hA[i * K + j] = i + j + 1;

        // Identity
        float[] hI = new float[K * N];
        for (int i = 0; i < K; i++) hI[i * N + i] = 1.0f;

        using var dA = new DeviceBuffer<float>(M * K);
        using var dI = new DeviceBuffer<float>(K * N);
        using var dC = new DeviceBuffer<float>(M * N);
        using var dD = new DeviceBuffer<float>(M * N);
        dA.CopyFrom(hA.AsSpan());
        dI.CopyFrom(hI.AsSpan());
        dC.Clear(); // beta=0 so C doesn't matter

        using var matmul = new Matmul(M, N, K, options: new MatmulOptions { Blocking = true });
        await matmul.PlanAsync();
        await matmul.ExecuteAsync<float>(dA, dI, dC, dD, alpha: 1f, beta: 0f);

        float[] result = dD.ToArray();
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
            {
                float expected = hA[i * K + j]; // A * I = A
                float err = Math.Abs(result[i * N + j] - expected);
                await Assert.That(err).IsLessThan(1e-4f);
            }
    }

    [Test]
    public async Task Matmul_Float_2x3_Times_3x2()
    {
        // A (2x3) * B (3x2) = C (2x2)
        // A = [[1,2,3],[4,5,6]], B = [[7,8],[9,10],[11,12]]
        // Expected C = [[58,64],[139,154]]
        float[] hA = [1, 2, 3, 4, 5, 6];
        float[] hB = [7, 8, 9, 10, 11, 12];

        using var dA = new DeviceBuffer<float>(6);
        using var dB = new DeviceBuffer<float>(6);
        using var dC = new DeviceBuffer<float>(4);
        using var dD = new DeviceBuffer<float>(4);
        dA.CopyFrom(hA.AsSpan());
        dB.CopyFrom(hB.AsSpan());
        dC.Clear();

        using var matmul = new Matmul(2, 2, 3, options: new MatmulOptions { Blocking = true });
        await matmul.PlanAsync();
        await matmul.ExecuteAsync<float>(dA, dB, dC, dD, alpha: 1f, beta: 0f);

        float[] result = dD.ToArray();
        float[] expected = [58, 64, 139, 154];
        for (int i = 0; i < 4; i++)
        {
            float err = Math.Abs(result[i] - expected[i]);
            await Assert.That(err).IsLessThan(1e-3f);
        }
    }
}
