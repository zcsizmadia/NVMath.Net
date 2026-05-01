// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the MIT License.

using NVMathNet.LinAlg;

namespace NVMathNet.Tests;

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
        {
            for (int c = 0; c < cols; c++)
            {
                h[r * cols + c] = (r + 1) * (c + 1);
            }
        }

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
        {
            for (int j = 0; j < K; j++)
            {
                hA[i * K + j] = i + j + 1;
            }
        }

        // Identity
        float[] hI = new float[K * N];
        for (int i = 0; i < K; i++)
        {
            hI[i * N + i] = 1.0f;
        }

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
        {
            for (int j = 0; j < N; j++)
            {
                float expected = hA[i * K + j]; // A * I = A
                float err = Math.Abs(result[i * N + j] - expected);
                await Assert.That(err).IsLessThan(1e-4f);
            }
        }
    }

    [Test]
    public async Task Matmul_Float_2x3_Times_3x2()
    {
        // A (2x3) * B (3x2) = D (2x2)
        // A = [[1,2,3],[4,5,6]], B = [[7,8],[9,10],[11,12]]
        // Expected D = [[58,64],[139,154]]
        // cuBLASLt uses column-major storage:
        float[] hA = [1, 4, 2, 5, 3, 6];        // A col-major
        float[] hB = [7, 9, 11, 8, 10, 12];      // B col-major

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
        // Expected in col-major: [58, 139, 64, 154]
        float[] expected = [58, 139, 64, 154];
        for (int i = 0; i < 4; i++)
        {
            float err = Math.Abs(result[i] - expected[i]);
            await Assert.That(err).IsLessThan(1e-3f);
        }
    }

    [Test]
    public async Task Matmul_Float_Rectangular_8x16_Times_16x4()
    {
        // Non-square rectangular test: A (8x16) * B (16x4) = D (8x4)
        // A[r,c] = r*16+c+1, B = Hilbert-like: B[r,c] = 1/(r+c+1)
        const int M = 8, N = 4, K = 16;

        float[] hA = new float[M * K]; // col-major
        float[] hB = new float[K * N]; // col-major
        for (int r = 0; r < M; r++)
        {
            for (int c = 0; c < K; c++)
            {
                hA[r + M * c] = r * K + c + 1f;
            }
        }
        for (int r = 0; r < K; r++)
        {
            for (int c = 0; c < N; c++)
            {
                hB[r + K * c] = 1f / (r + c + 1);
            }
        }

        // CPU reference
        float[] expected = new float[M * N];
        for (int r = 0; r < M; r++)
        {
            for (int c = 0; c < N; c++)
            {
                float sum = 0;
                for (int kk = 0; kk < K; kk++)
                {
                    sum += (r * K + kk + 1f) * (1f / (kk + c + 1));
                }

                expected[r + M * c] = sum;
            }
        }

        using var dA = new DeviceBuffer<float>(M * K);
        using var dB = new DeviceBuffer<float>(K * N);
        using var dD = new DeviceBuffer<float>(M * N);
        dA.CopyFrom(hA.AsSpan());
        dB.CopyFrom(hB.AsSpan());
        dD.Clear();

        using var matmul = new Matmul(M, N, K, options: new MatmulOptions { Blocking = true });
        await matmul.PlanAsync();
        await matmul.ExecuteAsync<float>(dA, dB, dD, dD, alpha: 1f, beta: 0f);

        float[] result = dD.ToArray();
        for (int i = 0; i < M * N; i++)
        {
            float relErr = Math.Abs(result[i] - expected[i]) / (Math.Abs(expected[i]) + 1e-10f);
            await Assert.That(relErr).IsLessThan(1e-3f);
        }
    }

    [Test]
    public async Task Matmul_AlphaBeta_Accumulation()
    {
        // Test D = 2*A*B + 0.5*C with non-trivial data
        const int S = 3;
        // A = [[1,4,7],[2,5,8],[3,6,9]] (col-major), B = [[9,6,3],[8,5,2],[7,4,1]]
        float[] hA = [1, 2, 3, 4, 5, 6, 7, 8, 9];
        float[] hB = [9, 8, 7, 6, 5, 4, 3, 2, 1];
        float[] hC = [1, 1, 1, 1, 1, 1, 1, 1, 1];

        // CPU reference: D = 2*A*B + 0.5*C
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

                expected[r + S * c] = 2f * sum + 0.5f * hC[r + S * c];
            }
        }

        using var dA = new DeviceBuffer<float>(S * S);
        using var dB = new DeviceBuffer<float>(S * S);
        using var dC = new DeviceBuffer<float>(S * S);
        using var dD = new DeviceBuffer<float>(S * S);
        dA.CopyFrom(hA.AsSpan());
        dB.CopyFrom(hB.AsSpan());
        dC.CopyFrom(hC.AsSpan());
        dD.Clear();

        using var matmul = new Matmul(S, S, S, options: new MatmulOptions { Blocking = true });
        await matmul.PlanAsync();
        await matmul.ExecuteAsync<float>(dA, dB, dC, dD, alpha: 2f, beta: 0.5f);

        float[] result = dD.ToArray();
        for (int i = 0; i < S * S; i++)
        {
            float err = Math.Abs(result[i] - expected[i]);
            await Assert.That(err).IsLessThan(1e-2f);
        }
    }

    [Test]
    public async Task Gemm_Sgemm_4x4()
    {
        // Test the static Gemm.SgemmAsync helper
        const int S = 4;
        // A = upper-triangular, B = [1..16] col-major
        float[] hA = new float[S * S];
        float[] hB = new float[S * S];
        for (int r = 0; r < S; r++)
        {
            for (int c = 0; c < S; c++)
            {
                hA[r + S * c] = c >= r ? (r + 1f) : 0f; // upper-tri col-major
                hB[r + S * c] = r + S * c + 1f;
            }
        }

        float[] hC = new float[S * S]; // zeros

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
        dA.CopyFrom(hA.AsSpan());
        dB.CopyFrom(hB.AsSpan());
        dC.CopyFrom(hC.AsSpan());

        await Gemm.SgemmAsync(dA, dB, dC, S, S, S);

        float[] result = dC.ToArray();
        for (int i = 0; i < S * S; i++)
        {
            float err = Math.Abs(result[i] - expected[i]);
            await Assert.That(err).IsLessThan(1e-2f);
        }
    }
}
