// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the MIT License.

using System.Numerics;

using NVMathNet.Solver;

namespace NVMathNet.Tests;

/// <summary>
/// Tests for the complex SVD and the complex batched solvers. Requires a CUDA-capable GPU.
/// </summary>
public sealed class ComplexSolverBatchedTests
{
    private static Complex[] RandomLike(int rows, int cols, int seed, bool diagonalBoost)
    {
        var m = new Complex[rows * cols];
        for (int c = 0; c < cols; c++)
        {
            for (int r = 0; r < rows; r++)
            {
                m[r + rows * c] = new Complex(
                    Math.Sin(seed + r * 1.3 + c * 0.7) + (diagonalBoost && r == c ? 4 : 0),
                    Math.Cos(seed + r * 0.9 - c * 1.1));
            }
        }

        return m;
    }

    [Test]
    public async Task Svd_ReconstructsMatrix_AndFactorsAreUnitary()
    {
        const int M = 5, N = 3;
        var a = RandomLike(M, N, 1, false);
        using var dA = new DeviceBuffer<Complex>(M * N);
        using var dS = new DeviceBuffer<double>(N);
        using var dU = new DeviceBuffer<Complex>(M * M);
        using var dVh = new DeviceBuffer<Complex>(N * N);
        dA.CopyFrom(a.AsSpan());

        await DenseSolver.SvdAsync(dA, dS, dU, dVh, M, N);

        double[] s = dS.ToArray();
        var u = dU.ToArray();
        var vh = dVh.ToArray();
        for (int i = 0; i + 1 < N; i++)
        {
            await Assert.That(s[i] >= s[i + 1] && s[i + 1] >= 0).IsTrue();
        }

        for (int r = 0; r < M; r++)
        {
            for (int c = 0; c < N; c++)
            {
                Complex sum = 0;
                for (int k = 0; k < N; k++)
                {
                    sum += u[r + M * k] * s[k] * vh[k + N * c];
                }

                await Assert.That((sum - a[r + M * c]).Magnitude).IsLessThan(1e-9);
            }
        }

        // Vh Vh^H = I
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                Complex dot = 0;
                for (int k = 0; k < N; k++)
                {
                    dot += vh[i + N * k] * Complex.Conjugate(vh[j + N * k]);
                }

                await Assert.That((dot - (i == j ? 1.0 : 0.0)).Magnitude).IsLessThan(1e-9);
            }
        }
    }

    [Test]
    public async Task Svd_WideMatrix_Throws()
    {
        using var dA = new DeviceBuffer<Complex>(6);
        using var dS = new DeviceBuffer<double>(2);
        using var dU = new DeviceBuffer<Complex>(4);
        using var dVh = new DeviceBuffer<Complex>(9);
        await Assert.That(async () => await DenseSolver.SvdAsync(dA, dS, dU, dVh, 2, 3)).Throws<ArgumentException>();
    }

    [Test]
    public async Task CholeskyBatched_FactorsHermitianMatrices_AndFlagsIndefinite()
    {
        // Three 2x2 column-major matrices: two Hermitian positive definite, one indefinite.
        Complex[] h =
        [
            new(4, 0), new(1, -1), new(1, 1), new(3, 0),
            new(9, 0), new(3, 0), new(3, 0), new(5, 0),
            new(1, 0), new(3, -1), new(3, 1), new(1, 0),
        ];
        using var dA = new DeviceBuffer<Complex>(h.Length);
        dA.CopyFrom(h.AsSpan());

        int[] infos = await DenseSolver.CholeskyBatchedAsync(dA, n: 2, batchCount: 3);
        var r = dA.ToArray();

        await Assert.That(infos[0]).IsEqualTo(0);
        await Assert.That(infos[1]).IsEqualTo(0);
        await Assert.That(infos[2]).IsGreaterThan(0);
        // Lower factor of matrix 0: L00 = 2, L10 = (1-i)/2, L11 = sqrt(3 - 0.5)
        await Assert.That((r[0] - 2.0).Magnitude).IsLessThan(1e-9);
        await Assert.That((r[1] - new Complex(0.5, -0.5)).Magnitude).IsLessThan(1e-9);
        await Assert.That(Math.Abs(r[3].Real - Math.Sqrt(2.5))).IsLessThan(1e-9);
        // Matrix 1: L = [[3, 0], [1, 2]]
        await Assert.That((r[4] - 3.0).Magnitude).IsLessThan(1e-9);
        await Assert.That((r[5] - 1.0).Magnitude).IsLessThan(1e-9);
        await Assert.That((r[7] - 2.0).Magnitude).IsLessThan(1e-9);
    }

    [Test]
    public async Task LuBatched_FactorAndSolve_IncludingConjugateTranspose()
    {
        const int N = 4, Batch = 3;
        var a = new Complex[N * N * Batch];
        var b = new Complex[N * Batch];
        for (int m = 0; m < Batch; m++)
        {
            Array.Copy(RandomLike(N, N, m + 5, true), 0, a, m * N * N, N * N);
            for (int r = 0; r < N; r++)
            {
                b[m * N + r] = new Complex(m + r + 1, 0.5 * r - m);
            }
        }

        foreach (bool conj in new[] { false, true })
        {
            using var dA = new DeviceBuffer<Complex>(a.Length);
            using var dB = new DeviceBuffer<Complex>(b.Length);
            using var dPiv = new DeviceBuffer<int>(N * Batch);
            dA.CopyFrom(a.AsSpan());
            dB.CopyFrom(b.AsSpan());

            int[] infos = await DenseSolver.LuFactorBatchedAsync(dA, dPiv, N, Batch);
            await Assert.That(infos.All(i => i == 0)).IsTrue();
            await DenseSolver.LuSolveBatchedAsync(dA, dPiv, dB, N, 1, Batch, transpose: conj, conjugate: conj);

            var x = dB.ToArray();
            for (int m = 0; m < Batch; m++)
            {
                for (int r = 0; r < N; r++)
                {
                    Complex sum = 0;
                    for (int c = 0; c < N; c++)
                    {
                        sum += conj
                            ? Complex.Conjugate(a[m * N * N + c + N * r]) * x[m * N + c]
                            : a[m * N * N + r + N * c] * x[m * N + c];
                    }

                    await Assert.That((sum - b[m * N + r]).Magnitude).IsLessThan(1e-9);
                }
            }
        }
    }

    [Test]
    public async Task SvdBatched_ReconstructsEachMatrix_AndValuesOnlyMatches()
    {
        const int M = 3, N = 2, Batch = 2, K = 2;
        var a = new Complex[M * N * Batch];
        for (int b = 0; b < Batch; b++)
        {
            Array.Copy(RandomLike(M, N, 10 + b, false), 0, a, b * M * N, M * N);
        }

        using var dA = new DeviceBuffer<Complex>(a.Length);
        using var dA2 = new DeviceBuffer<Complex>(a.Length);
        using var dS = new DeviceBuffer<double>(K * Batch);
        using var dS2 = new DeviceBuffer<double>(K * Batch);
        using var dU = new DeviceBuffer<Complex>(M * M * Batch);
        using var dV = new DeviceBuffer<Complex>(N * N * Batch);
        dA.CopyFrom(a.AsSpan());
        dA2.CopyFrom(a.AsSpan());

        int[] infos = await DenseSolver.SvdBatchedAsync(dA, dS, dU, dV, M, N, Batch);
        await Assert.That(infos.All(i => i == 0)).IsTrue();

        double[] s = dS.ToArray();
        var u = dU.ToArray();
        var v = dV.ToArray();
        for (int b = 0; b < Batch; b++)
        {
            for (int r = 0; r < M; r++)
            {
                for (int c = 0; c < N; c++)
                {
                    Complex sum = 0;
                    for (int k = 0; k < K; k++)
                    {
                        // A = U diag(S) V^H, V returned as an n x n column-major matrix.
                        sum += u[b * M * M + r + M * k] * s[b * K + k] * Complex.Conjugate(v[b * N * N + c + N * k]);
                    }

                    await Assert.That((sum - a[b * M * N + r + M * c]).Magnitude).IsLessThan(1e-9);
                }
            }
        }

        await DenseSolver.SvdBatchedAsync(dA2, dS2, null, null, M, N, Batch, computeVectors: false);
        double[] s2 = dS2.ToArray();
        for (int i = 0; i < s.Length; i++)
        {
            await Assert.That(Math.Abs(s[i] - s2[i])).IsLessThan(1e-9);
        }
    }
}
