// Copyright (c) 2026 Zoltan Csizmadia. All rights reserved.
// Licensed under the MIT License.

using NVMathNet.Sparse;

namespace NVMathNet.Tests;

/// <summary>
/// Tests for the cuDSS-backed sparse direct solver. Requires a CUDA-capable GPU and the cuDSS runtime.
/// </summary>
public sealed class SparseDirectSolverTests
{
    // Tridiagonal matrix: diagonal `diag`, off-diagonals -1. Optionally only the lower triangle.
    private static (CsrMatrix<double> Matrix, double[] Dense) MakeTridiagonal(int n, double diag, bool lowerOnly)
    {
        var rowPtr = new List<int> { 0 };
        var cols = new List<int>();
        var vals = new List<double>();
        double[] dense = new double[n * n];
        for (int r = 0; r < n; r++)
        {
            for (int c = Math.Max(0, r - 1); c <= Math.Min(n - 1, r + 1); c++)
            {
                double v = r == c ? diag : -1.0;
                dense[r + n * c] = v; // column-major full matrix for the host reference
                if (lowerOnly && c > r)
                {
                    continue;
                }

                cols.Add(c);
                vals.Add(v);
            }

            rowPtr.Add(cols.Count);
        }

        var dRow = new DeviceBuffer<int>(rowPtr.Count);
        var dCol = new DeviceBuffer<int>(cols.Count);
        var dVal = new DeviceBuffer<double>(vals.Count);
        dRow.CopyFrom(rowPtr.ToArray().AsSpan());
        dCol.CopyFrom(cols.ToArray().AsSpan());
        dVal.CopyFrom(vals.ToArray().AsSpan());
        return (new CsrMatrix<double>(n, n, dRow, dCol, dVal), dense);
    }

    // b = A * x for a column-major dense A (n x n) and one or more columns of x.
    private static double[] Multiply(double[] a, int n, double[] x, int nrhs)
    {
        double[] b = new double[n * nrhs];
        for (int k = 0; k < nrhs; k++)
        {
            for (int r = 0; r < n; r++)
            {
                double sum = 0;
                for (int c = 0; c < n; c++)
                {
                    sum += a[r + n * c] * x[c + n * k];
                }

                b[r + n * k] = sum;
            }
        }

        return b;
    }

    [Test]
    public async Task Solve_GeneralMatrix_RecoversKnownSolution()
    {
        const int N = 64;
        var (matrix, dense) = MakeTridiagonal(N, 4.0, lowerOnly: false);
        using var _ = matrix;
        double[] xTrue = [.. Enumerable.Range(0, N).Select(i => Math.Sin(i * 0.3) + 1.0)];
        double[] hB = Multiply(dense, N, xTrue, 1);

        using var dB = new DeviceBuffer<double>(N);
        using var dX = new DeviceBuffer<double>(N);
        dB.CopyFrom(hB.AsSpan());
        dX.Clear();

        using var solver = new SparseDirectSolver<double>(matrix);
        solver.Analyze();
        solver.Factorize();
        await solver.SolveAsync(dB, dX);

        double[] x = dX.ToArray();
        for (int i = 0; i < N; i++)
        {
            await Assert.That(Math.Abs(x[i] - xTrue[i])).IsLessThan(1e-9);
        }

        await Assert.That(solver.FactorNonZeros).IsGreaterThan(0L);
    }

    [Test]
    public async Task Solve_SpdMatrix_LowerTriangleOnly()
    {
        const int N = 100;
        var (matrix, dense) = MakeTridiagonal(N, 2.5, lowerOnly: true);
        using var _ = matrix;
        double[] xTrue = [.. Enumerable.Repeat(1.0, N)];
        double[] hB = Multiply(dense, N, xTrue, 1);

        using var dB = new DeviceBuffer<double>(N);
        using var dX = new DeviceBuffer<double>(N);
        dB.CopyFrom(hB.AsSpan());

        var options = new DirectSolverOptions
        {
            Structure = SparseMatrixStructure.SymmetricPositiveDefinite,
            View = SparseMatrixView.Lower,
        };
        using var solver = new SparseDirectSolver<double>(matrix, options);
        solver.Solve(dB, dX); // runs analysis and factorisation implicitly

        double[] x = dX.ToArray();
        for (int i = 0; i < N; i++)
        {
            await Assert.That(Math.Abs(x[i] - 1.0)).IsLessThan(1e-9);
        }
    }

    [Test]
    public async Task Solve_MultipleRightHandSides()
    {
        const int N = 32, Nrhs = 3;
        var (matrix, dense) = MakeTridiagonal(N, 3.0, lowerOnly: false);
        using var _ = matrix;
        double[] xTrue = new double[N * Nrhs];
        for (int i = 0; i < xTrue.Length; i++)
        {
            xTrue[i] = (i % 7) - 3 + 0.5 * (i / N);
        }

        double[] hB = Multiply(dense, N, xTrue, Nrhs);
        using var dB = new DeviceBuffer<double>(N * Nrhs);
        using var dX = new DeviceBuffer<double>(N * Nrhs);
        dB.CopyFrom(hB.AsSpan());

        using var solver = new SparseDirectSolver<double>(matrix);
        solver.Solve(dB, dX, Nrhs);

        double[] x = dX.ToArray();
        for (int i = 0; i < x.Length; i++)
        {
            await Assert.That(Math.Abs(x[i] - xTrue[i])).IsLessThan(1e-9);
        }
    }

    [Test]
    public async Task Refactorize_AfterValuesChange_ReusesAnalysis()
    {
        const int N = 48;
        var (matrix, dense) = MakeTridiagonal(N, 4.0, lowerOnly: false);
        using var _ = matrix;
        double[] xTrue = [.. Enumerable.Range(0, N).Select(i => 1.0 + i * 0.1)];
        double[] hB = Multiply(dense, N, xTrue, 1);

        using var dB = new DeviceBuffer<double>(N);
        using var dX = new DeviceBuffer<double>(N);
        dB.CopyFrom(hB.AsSpan());

        using var solver = new SparseDirectSolver<double>(matrix);
        solver.Factorize();
        solver.Solve(dB, dX);

        // Scale A by 2 in place: the solution of (2A) x = b is x_true / 2.
        double[] vals = matrix.Values.ToArray();
        matrix.Values.CopyFrom([.. vals.Select(v => v * 2.0)]);
        solver.Refactorize();
        solver.Solve(dB, dX);

        double[] x = dX.ToArray();
        for (int i = 0; i < N; i++)
        {
            await Assert.That(Math.Abs(x[i] - xTrue[i] / 2.0)).IsLessThan(1e-9);
        }
    }

    [Test]
    public async Task Solve_SinglePrecision()
    {
        const int N = 40;
        var rowPtr = Enumerable.Range(0, N + 1).ToArray();
        var cols = Enumerable.Range(0, N).ToArray();
        float[] diag = [.. Enumerable.Range(1, N).Select(i => (float)i)];
        var dRow = new DeviceBuffer<int>(N + 1);
        var dCol = new DeviceBuffer<int>(N);
        var dVal = new DeviceBuffer<float>(N);
        dRow.CopyFrom(rowPtr.AsSpan());
        dCol.CopyFrom(cols.AsSpan());
        dVal.CopyFrom(diag.AsSpan());
        using var matrix = new CsrMatrix<float>(N, N, dRow, dCol, dVal);

        using var dB = new DeviceBuffer<float>(N);
        using var dX = new DeviceBuffer<float>(N);
        dB.CopyFrom(diag.AsSpan()); // diag(i) * x = diag(i)  =>  x = 1

        using var solver = new SparseDirectSolver<float>(matrix);
        solver.Solve(dB, dX);

        float[] x = dX.ToArray();
        for (int i = 0; i < N; i++)
        {
            await Assert.That(Math.Abs(x[i] - 1f)).IsLessThan(1e-4f);
        }
    }

    [Test]
    public async Task NonSquareMatrix_Throws()
    {
        var dRow = new DeviceBuffer<int>(3);
        var dCol = new DeviceBuffer<int>(2);
        var dVal = new DeviceBuffer<double>(2);
        dRow.CopyFrom(new[] { 0, 1, 2 }.AsSpan());
        dCol.CopyFrom(new[] { 0, 2 }.AsSpan());
        dVal.CopyFrom(new[] { 1.0, 1.0 }.AsSpan());
        using var matrix = new CsrMatrix<double>(2, 3, dRow, dCol, dVal);

        await Assert.That(() => new SparseDirectSolver<double>(matrix)).Throws<ArgumentException>();
    }

    [Test]
    public async Task Factorize_SingularMatrix_IsReported()
    {
        // diag(1, 0, 1) is singular.
        var dRow = new DeviceBuffer<int>(4);
        var dCol = new DeviceBuffer<int>(3);
        var dVal = new DeviceBuffer<double>(3);
        dRow.CopyFrom(new[] { 0, 1, 2, 3 }.AsSpan());
        dCol.CopyFrom(new[] { 0, 1, 2 }.AsSpan());
        dVal.CopyFrom(new[] { 1.0, 0.0, 1.0 }.AsSpan());
        using var matrix = new CsrMatrix<double>(3, 3, dRow, dCol, dVal);

        using var solver = new SparseDirectSolver<double>(matrix);
        bool reported = false;
        try
        {
            solver.Factorize();
        }
        catch (InvalidOperationException)
        {
            reported = true;
        }

        // Either the factorisation fails outright or cuDSS perturbs the pivot and says so.
        await Assert.That(reported || solver.PerturbedPivots > 0).IsTrue();
    }
}
