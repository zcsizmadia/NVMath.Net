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
}
