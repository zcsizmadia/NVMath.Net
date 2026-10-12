// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the MIT License.

using NVMathNet.Solver;

namespace NVMathNet.Tests;

/// <summary>
/// Tests for the cuSOLVER dense solver module. Requires a CUDA-capable GPU.
/// </summary>
public sealed class SolverTests
{
    [Test]
    public async Task LuSolve_Float_3x3_KnownSystem()
    {
        // Solve A*x = b where A = [[2,1,1],[4,3,3],[8,7,9]], b = [1,1,1]
        // Solution: x = [1, -1, 1] (verify manually)
        // Column-major A:
        const int N = 3;
        float[] hA = [2, 4, 8, 1, 3, 7, 1, 3, 9]; // col-major
        float[] hB = [1, 1, 1]; // rhs vector

        // CPU reference: solve by hand
        // | 2  1  1 | | x |   | 1 |
        // | 4  3  3 | | y | = | 1 |
        // | 8  7  9 | | z |   | 1 |
        // x=1, y=-1, z=1 => 2(1)+1(-1)+1(1)=2, 4(1)+3(-1)+3(1)=4, 8(1)+7(-1)+9(1)=10 — not right
        // Let me just verify GPU produces Ax=b
        float[] hACopy = (float[])hA.Clone();
        float[] hBCopy = (float[])hB.Clone();

        using var dA = new DeviceBuffer<float>(N * N);
        using var dB = new DeviceBuffer<float>(N);
        dA.CopyFrom(hA.AsSpan());
        dB.CopyFrom(hB.AsSpan());

        await DenseSolver.SolveAsync(dA, dB, N);

        // Verify: multiply original A with solution x and check = b
        float[] x = dB.ToArray();
        for (int r = 0; r < N; r++)
        {
            float sum = 0;
            for (int c = 0; c < N; c++)
            {
                sum += hACopy[r + N * c] * x[c]; // col-major
            }

            float err = Math.Abs(sum - hBCopy[r]);
            await Assert.That(err).IsLessThan(1e-4f);
        }
    }

    [Test]
    public async Task LuSolve_Double_4x4()
    {
        // A = diag(1,2,3,4), b = [1,2,3,4] => x = [1,1,1,1]
        const int N = 4;
        double[] hA = new double[N * N];
        for (int i = 0; i < N; i++)
        {
            hA[i + N * i] = i + 1.0;
        }

        double[] hB = [1, 2, 3, 4];

        using var dA = new DeviceBuffer<double>(N * N);
        using var dB = new DeviceBuffer<double>(N);
        dA.CopyFrom(hA.AsSpan());
        dB.CopyFrom(hB.AsSpan());

        await DenseSolver.SolveAsync(dA, dB, N);

        double[] result = dB.ToArray();
        for (int i = 0; i < N; i++)
        {
            double err = Math.Abs(result[i] - 1.0);
            await Assert.That(err).IsLessThan(1e-10);
        }
    }

    [Test]
    public async Task Cholesky_Float_SPD_Matrix()
    {
        // A = [[4,2],[2,3]] — symmetric positive definite
        // L = [[2,0],[1,sqrt(2)]], A = L*L^T
        const int N = 2;
        float[] hA = [4, 2, 2, 3]; // col-major: [A(0,0), A(1,0), A(0,1), A(1,1)]

        using var dA = new DeviceBuffer<float>(N * N);
        dA.CopyFrom(hA.AsSpan());

        await DenseSolver.CholeskyAsync(dA, N, upper: false);

        // Lower triangle should contain L
        float[] result = dA.ToArray();
        // L(0,0) = 2
        await Assert.That(Math.Abs(result[0] - 2f)).IsLessThan(1e-4f);
        // L(1,0) = 1
        await Assert.That(Math.Abs(result[1] - 1f)).IsLessThan(1e-4f);
        // L(1,1) = sqrt(2)
        await Assert.That(Math.Abs(result[3] - MathF.Sqrt(2f))).IsLessThan(1e-4f);
    }

    [Test]
    public async Task Svd_Float_3x2_SingularValues()
    {
        // A = [[3,2,2],[2,3,-2]] transposed to 3x2 col-major
        // Known singular values: sqrt(18) ≈ 4.243, sqrt(9) = 3.0
        const int M = 3, N = 2;
        // A in col-major (3 rows, 2 cols):
        // Col 0: [3, 2, 2], Col 1: [2, 3, -2]
        float[] hA = [3, 2, 2, 2, 3, -2];
        int minMN = Math.Min(M, N);

        using var dA = new DeviceBuffer<float>(M * N);
        using var dS = new DeviceBuffer<float>(minMN);
        using var dU = new DeviceBuffer<float>(M * M);
        using var dVT = new DeviceBuffer<float>(N * N);
        dA.CopyFrom(hA.AsSpan());

        await DenseSolver.SvdAsync(dA, dS, dU, dVT, M, N);

        float[] singularValues = dS.ToArray();
        // Sort descending (cuSOLVER returns in descending order)
        // Verify both are positive
        await Assert.That(singularValues[0]).IsGreaterThan(0f);
        await Assert.That(singularValues[1]).IsGreaterThan(0f);
        await Assert.That(singularValues[0]).IsGreaterThanOrEqualTo(singularValues[1]);
    }

    [Test]
    public async Task Eigenvalue_Float_Symmetric_2x2()
    {
        // A = [[2,1],[1,2]] — eigenvalues are 1 and 3
        const int N = 2;
        float[] hA = [2, 1, 1, 2]; // col-major

        using var dA = new DeviceBuffer<float>(N * N);
        using var dW = new DeviceBuffer<float>(N);
        dA.CopyFrom(hA.AsSpan());

        await DenseSolver.EigAsync(dA, dW, N, computeVectors: false);

        float[] eigenvalues = dW.ToArray();
        // cuSOLVER returns eigenvalues in ascending order
        await Assert.That(Math.Abs(eigenvalues[0] - 1f)).IsLessThan(1e-4f);
        await Assert.That(Math.Abs(eigenvalues[1] - 3f)).IsLessThan(1e-4f);
    }

    [Test]
    public async Task Eigenvalue_Double_Symmetric_3x3()
    {
        // A = diag(5,3,1) — eigenvalues should be 1,3,5 in ascending order
        const int N = 3;
        double[] hA = new double[N * N];
        hA[0] = 5; hA[4] = 3; hA[8] = 1; // diag col-major

        using var dA = new DeviceBuffer<double>(N * N);
        using var dW = new DeviceBuffer<double>(N);
        dA.CopyFrom(hA.AsSpan());

        await DenseSolver.EigAsync(dA, dW, N, computeVectors: true);

        double[] eigenvalues = dW.ToArray();
        Array.Sort(eigenvalues);
        await Assert.That(Math.Abs(eigenvalues[0] - 1.0)).IsLessThan(1e-10);
        await Assert.That(Math.Abs(eigenvalues[1] - 3.0)).IsLessThan(1e-10);
        await Assert.That(Math.Abs(eigenvalues[2] - 5.0)).IsLessThan(1e-10);
    }

    // ── LU / QR / least squares / batched Cholesky ───────────────────────────

    [Test]
    public async Task LuFactor_LuSolve_SolvesAndTransposeSolves()
    {
        const int N = 3;
        // A = [[2,1,1],[4,3,3],[8,7,9]] column-major
        double[] hA = [2, 4, 8, 1, 3, 7, 1, 3, 9];
        double[] hB = [1, 1, 1];

        foreach (bool transpose in new[] { false, true })
        {
            using var dA = new DeviceBuffer<double>(N * N);
            using var dB = new DeviceBuffer<double>(N);
            using var dPiv = new DeviceBuffer<int>(N);
            dA.CopyFrom(hA.AsSpan());
            dB.CopyFrom(hB.AsSpan());

            int info = await DenseSolver.LuFactorAsync(dA, dPiv, N, N);
            await Assert.That(info).IsEqualTo(0);
            await DenseSolver.LuSolveAsync(dA, dPiv, dB, N, 1, transpose);

            double[] x = dB.ToArray();
            for (int r = 0; r < N; r++)
            {
                double sum = 0;
                for (int c = 0; c < N; c++)
                {
                    sum += (transpose ? hA[c + N * r] : hA[r + N * c]) * x[c];
                }

                await Assert.That(Math.Abs(sum - hB[r])).IsLessThan(1e-9);
            }
        }
    }

    [Test]
    public async Task LuFactor_SingularMatrix_ReportsPositiveInfo()
    {
        double[] hA = [1, 2, 2, 4]; // [[1,2],[2,4]]
        using var dA = new DeviceBuffer<double>(4);
        using var dPiv = new DeviceBuffer<int>(2);
        dA.CopyFrom(hA.AsSpan());

        int info = await DenseSolver.LuFactorAsync(dA, dPiv, 2, 2);

        await Assert.That(info).IsGreaterThan(0);
    }

    [Test]
    public async Task Qr_FormQ_ReconstructsMatrixAndIsOrthonormal()
    {
        const int M = 4, N = 3;
        double[] hA = [1, 2, 3, 4, 2, 1, 0, 1, 0, 3, 1, 2]; // column-major 4x3
        using var dA = new DeviceBuffer<double>(M * N);
        using var dTau = new DeviceBuffer<double>(N);
        dA.CopyFrom(hA.AsSpan());

        await DenseSolver.QrAsync(dA, dTau, M, N);
        double[] fact = dA.ToArray(); // upper triangle = R
        await DenseSolver.QrFormQAsync(dA, dTau, M, N);
        double[] q = dA.ToArray();

        // Q^T Q = I
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                double dot = 0;
                for (int r = 0; r < M; r++)
                {
                    dot += q[r + M * i] * q[r + M * j];
                }

                await Assert.That(Math.Abs(dot - (i == j ? 1.0 : 0.0))).IsLessThan(1e-9);
            }
        }

        // Q R = A
        for (int r = 0; r < M; r++)
        {
            for (int c = 0; c < N; c++)
            {
                double sum = 0;
                for (int k = 0; k <= c; k++)
                {
                    sum += q[r + M * k] * fact[k + M * c];
                }

                await Assert.That(Math.Abs(sum - hA[r + M * c])).IsLessThan(1e-9);
            }
        }
    }

    [Test]
    public async Task LeastSquares_LineFit_MatchesNormalEquations()
    {
        const int M = 5, N = 2;
        double[] hA = [1, 1, 1, 1, 1, 0, 1, 2, 3, 4]; // columns: ones, x
        double[] hB = [1.1, 2.9, 5.2, 6.8, 9.1];

        using var dA = new DeviceBuffer<double>(M * N);
        using var dB = new DeviceBuffer<double>(M);
        dA.CopyFrom(hA.AsSpan());
        dB.CopyFrom(hB.AsSpan());

        await DenseSolver.LeastSquaresAsync(dA, dB, M, N);
        double[] sol = dB.ToArray();

        // Normal equations (A^T A) x = A^T b for the 2x2 case
        double s0 = M, s1 = 0, s2 = 0, t0 = 0, t1 = 0;
        for (int i = 0; i < M; i++)
        {
            s1 += hA[M + i];
            s2 += hA[M + i] * hA[M + i];
            t0 += hB[i];
            t1 += hA[M + i] * hB[i];
        }

        double det = s0 * s2 - s1 * s1;
        double x0 = (t0 * s2 - s1 * t1) / det;
        double x1 = (s0 * t1 - s1 * t0) / det;

        await Assert.That(Math.Abs(sol[0] - x0)).IsLessThan(1e-9);
        await Assert.That(Math.Abs(sol[1] - x1)).IsLessThan(1e-9);
    }

    [Test]
    public async Task CholeskyBatched_FactorsEachMatrixAndReportsNonPositiveDefinite()
    {
        // Three 2x2 column-major matrices: SPD, SPD, and indefinite.
        double[] h = [4, 2, 2, 3, 9, 3, 3, 5, 1, 2, 2, 1];
        using var dA = new DeviceBuffer<double>(h.Length);
        dA.CopyFrom(h.AsSpan());

        int[] infos = await DenseSolver.CholeskyBatchedAsync(dA, n: 2, batchCount: 3);
        double[] r = dA.ToArray();

        await Assert.That(infos[0]).IsEqualTo(0);
        await Assert.That(infos[1]).IsEqualTo(0);
        await Assert.That(infos[2]).IsGreaterThan(0);
        // Lower factors: [[2,0],[1,sqrt2]] and [[3,0],[1,2]]
        await Assert.That(Math.Abs(r[0] - 2)).IsLessThan(1e-9);
        await Assert.That(Math.Abs(r[1] - 1)).IsLessThan(1e-9);
        await Assert.That(Math.Abs(r[3] - Math.Sqrt(2))).IsLessThan(1e-9);
        await Assert.That(Math.Abs(r[4] - 3)).IsLessThan(1e-9);
        await Assert.That(Math.Abs(r[5] - 1)).IsLessThan(1e-9);
        await Assert.That(Math.Abs(r[7] - 2)).IsLessThan(1e-9);
    }

    // ── Batched LU ───────────────────────────────────────────────────────────

    [Test]
    public async Task LuBatched_FactorAndSolve_SolveEveryMatrix()
    {
        const int N = 4, Batch = 3;
        double[] hA = new double[N * N * Batch];
        double[] hB = new double[N * Batch];
        for (int m = 0; m < Batch; m++)
        {
            for (int c = 0; c < N; c++)
            {
                for (int r = 0; r < N; r++)
                {
                    hA[m * N * N + r + N * c] = r == c ? 5 + m : 0.3 * ((r + 2 * c + m) % 5);
                }
            }

            for (int r = 0; r < N; r++)
            {
                hB[m * N + r] = (m + 1) * (r + 1);
            }
        }

        foreach (bool transpose in new[] { false, true })
        {
            using var dA = new DeviceBuffer<double>(hA.Length);
            using var dB = new DeviceBuffer<double>(hB.Length);
            using var dPiv = new DeviceBuffer<int>(N * Batch);
            dA.CopyFrom(hA.AsSpan());
            dB.CopyFrom(hB.AsSpan());

            int[] infos = await DenseSolver.LuFactorBatchedAsync(dA, dPiv, N, Batch);
            await Assert.That(infos.All(i => i == 0)).IsTrue();
            await DenseSolver.LuSolveBatchedAsync(dA, dPiv, dB, N, 1, Batch, transpose);

            double[] x = dB.ToArray();
            for (int m = 0; m < Batch; m++)
            {
                for (int r = 0; r < N; r++)
                {
                    double sum = 0;
                    for (int c = 0; c < N; c++)
                    {
                        double a = transpose ? hA[m * N * N + c + N * r] : hA[m * N * N + r + N * c];
                        sum += a * x[m * N + c];
                    }

                    await Assert.That(Math.Abs(sum - hB[m * N + r])).IsLessThan(1e-9);
                }
            }
        }
    }

    [Test]
    public async Task LuBatched_SingularMatrix_ReportedPerMatrix()
    {
        // Matrix 0 is regular, matrix 1 is singular ([[1,2],[2,4]]).
        double[] hA = [2, 0, 0, 3, 1, 2, 2, 4];
        using var dA = new DeviceBuffer<double>(hA.Length);
        using var dPiv = new DeviceBuffer<int>(4);
        dA.CopyFrom(hA.AsSpan());

        int[] infos = await DenseSolver.LuFactorBatchedAsync(dA, dPiv, 2, 2);

        await Assert.That(infos[0]).IsEqualTo(0);
        await Assert.That(infos[1]).IsGreaterThan(0);
    }

    [Test]
    public async Task LuBatched_TooSmallBuffer_Throws()
    {
        using var dA = new DeviceBuffer<double>(4);
        using var dPiv = new DeviceBuffer<int>(4);
        await Assert.That(async () => await DenseSolver.LuFactorBatchedAsync(dA, dPiv, 2, 2)).Throws<ArgumentException>();
    }

    // ── Batched SVD (Jacobi) ─────────────────────────────────────────────────

    private static readonly double[] BatchedSvdInput = [1, 3, 5, 2, 4, 6, 2, 1, 0, 0, 1, 3]; // two 3x2 matrices, col-major

    [Test]
    public async Task SvdBatched_ReconstructsEachMatrix()
    {
        const int M = 3, N = 2, Batch = 2, K = 2;
        using var dA = new DeviceBuffer<double>(BatchedSvdInput.Length);
        using var dS = new DeviceBuffer<double>(K * Batch);
        using var dU = new DeviceBuffer<double>(M * M * Batch);
        using var dV = new DeviceBuffer<double>(N * N * Batch);
        dA.CopyFrom(BatchedSvdInput.AsSpan());

        int[] infos = await DenseSolver.SvdBatchedAsync(dA, dS, dU, dV, M, N, Batch);
        await Assert.That(infos.All(i => i == 0)).IsTrue();

        double[] s = dS.ToArray(), u = dU.ToArray(), v = dV.ToArray();
        for (int b = 0; b < Batch; b++)
        {
            await Assert.That(s[b * K] >= s[b * K + 1]).IsTrue();
            for (int r = 0; r < M; r++)
            {
                for (int c = 0; c < N; c++)
                {
                    double sum = 0;
                    for (int k = 0; k < K; k++)
                    {
                        // A = U diag(S) V^T; V is returned as V (n x n, col-major).
                        sum += u[b * M * M + r + M * k] * s[b * K + k] * v[b * N * N + c + N * k];
                    }

                    await Assert.That(Math.Abs(sum - BatchedSvdInput[b * M * N + r + M * c])).IsLessThan(1e-9);
                }
            }
        }
    }

    [Test]
    public async Task SvdBatched_ValuesOnly_MatchesVectorRun()
    {
        const int M = 3, N = 2, Batch = 2, K = 2;
        using var dA1 = new DeviceBuffer<double>(BatchedSvdInput.Length);
        using var dA2 = new DeviceBuffer<double>(BatchedSvdInput.Length);
        using var dS1 = new DeviceBuffer<double>(K * Batch);
        using var dS2 = new DeviceBuffer<double>(K * Batch);
        using var dU = new DeviceBuffer<double>(M * M * Batch);
        using var dV = new DeviceBuffer<double>(N * N * Batch);
        dA1.CopyFrom(BatchedSvdInput.AsSpan());
        dA2.CopyFrom(BatchedSvdInput.AsSpan());

        await DenseSolver.SvdBatchedAsync(dA1, dS1, dU, dV, M, N, Batch);
        await DenseSolver.SvdBatchedAsync<double>(dA2, dS2, null, null, M, N, Batch, computeVectors: false);

        double[] a = dS1.ToArray(), b = dS2.ToArray();
        for (int i = 0; i < a.Length; i++)
        {
            await Assert.That(Math.Abs(a[i] - b[i])).IsLessThan(1e-9);
        }
    }

    [Test]
    public async Task SvdBatched_TooLargeMatrix_Throws()
    {
        using var dA = new DeviceBuffer<double>(33 * 33);
        using var dS = new DeviceBuffer<double>(33);
        await Assert.That(async () => await DenseSolver.SvdBatchedAsync<double>(dA, dS, null, null, 33, 33, 1, computeVectors: false))
            .Throws<ArgumentOutOfRangeException>();
    }
}
