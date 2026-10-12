// Copyright (c) 2026 Zoltan Csizmadia. All rights reserved.
// Licensed under the MIT License.

using NVMathNet.Rand;
using NVMathNet.Solver;
using NVMathNet.Sparse;

namespace NVMathNet.Tests;

/// <summary>
/// Regression tests for issues found in the pre-1.0 review of the sparse, dense-solver and cuRAND wrappers.
/// Requires a CUDA-capable GPU.
/// </summary>
public sealed class SparseSolverRandReviewTests
{
    private static CsrMatrix<double> Diagonal(int n, double value)
    {
        var rowPtr = new DeviceBuffer<int>(n + 1);
        var colIdx = new DeviceBuffer<int>(n);
        var vals = new DeviceBuffer<double>(n);
        rowPtr.CopyFrom([.. Enumerable.Range(0, n + 1)]);
        colIdx.CopyFrom([.. Enumerable.Range(0, n)]);
        vals.CopyFrom([.. Enumerable.Repeat(value, n)]);
        return new CsrMatrix<double>(n, n, rowPtr, colIdx, vals);
    }

    private static CsrMatrix<float> Empty(int rows, int cols)
    {
        var rowPtr = new DeviceBuffer<int>(rows + 1);
        rowPtr.Clear();
        var colIdx = new DeviceBuffer<int>(1);
        var vals = new DeviceBuffer<float>(1);
        colIdx.Clear();
        vals.Clear();
        return new CsrMatrix<float>(rows, cols, rowPtr, colIdx, vals, nnz: 0);
    }

    [Test]
    public async Task EmptyMatrix_TransposeAndToCsc_Work()
    {
        using var empty = Empty(3, 5);

        using var t = empty.Transpose();
        await Assert.That(t.Rows).IsEqualTo(5L);
        await Assert.That(t.Cols).IsEqualTo(3L);
        await Assert.That(t.NNZ).IsEqualTo(0L);

        using var csc = empty.ToCsc();
        await Assert.That(csc.NNZ).IsEqualTo(0L);
    }

    [Test]
    public async Task Constructors_RejectUndersizedBuffers()
    {
        using var rowPtr = new DeviceBuffer<int>(2); // needs rows + 1 = 4
        using var colIdx = new DeviceBuffer<int>(3);
        using var vals = new DeviceBuffer<float>(3);
        await Assert.That(() => new CsrMatrix<float>(3, 3, rowPtr, colIdx, vals)).Throws<ArgumentException>();

        using var rowPtr2 = new DeviceBuffer<int>(4);
        using var colIdx2 = new DeviceBuffer<int>(2); // fewer than the 3 values
        await Assert.That(() => new CsrMatrix<float>(3, 3, rowPtr2, colIdx2, vals)).Throws<ArgumentException>();
    }

    [Test]
    public async Task SpMV_UndersizedOperand_Throws()
    {
        using var matrix = Diagonal(8, 2.0);
        using var x = new DeviceBuffer<double>(4); // needs 8
        using var y = new DeviceBuffer<double>(8);
        await Assert.That(async () => await SparseLinearAlgebra.SpMVAsync(matrix, x, y)).Throws<ArgumentException>();
    }

    [Test]
    public async Task SpMM_NonPositiveColumns_Throws()
    {
        using var matrix = Diagonal(4, 2.0);
        using var b = new DeviceBuffer<double>(4);
        using var c = new DeviceBuffer<double>(4);
        await Assert.That(async () => await SparseLinearAlgebra.SpMMAsync(matrix, b, c, 0)).Throws<ArgumentOutOfRangeException>();
    }

    [Test]
    public async Task DirectSolver_SetValues_RefactorizesOnNextSolve()
    {
        const int N = 4;
        using var matrix = Diagonal(N, 2.0);
        using var dB = new DeviceBuffer<double>(N);
        using var dX = new DeviceBuffer<double>(N);
        dB.CopyFrom([2.0, 4.0, 6.0, 8.0]);

        using var solver = new SparseDirectSolver<double>(matrix);
        solver.Solve(dB, dX);
        double[] first = dX.ToArray();
        for (int i = 0; i < N; i++)
        {
            await Assert.That(Math.Abs(first[i] - (i + 1))).IsLessThan(1e-9);
        }

        using var doubled = new DeviceBuffer<double>(N);
        doubled.CopyFrom([.. Enumerable.Repeat(4.0, N)]);
        solver.SetValues(doubled);
        solver.Solve(dB, dX); // no explicit Refactorize: stale factors would give the old answer

        double[] second = dX.ToArray();
        for (int i = 0; i < N; i++)
        {
            await Assert.That(Math.Abs(second[i] - (i + 1) / 2.0)).IsLessThan(1e-9);
        }
    }

    [Test]
    public async Task DenseSolve_UndersizedOrNonPositive_Throws()
    {
        using var a = new DeviceBuffer<float>(4);
        using var b = new DeviceBuffer<float>(3);
        await Assert.That(async () => await DenseSolver.SolveAsync(a, b, 3)).Throws<ArgumentException>();
        await Assert.That(async () => await DenseSolver.SolveAsync(a, b, 0)).Throws<ArgumentOutOfRangeException>();
    }

    [Test]
    public async Task RealSvd_WideMatrix_Throws()
    {
        using var a = new DeviceBuffer<float>(6);
        using var s = new DeviceBuffer<float>(2);
        using var u = new DeviceBuffer<float>(4);
        using var vt = new DeviceBuffer<float>(9);
        await Assert.That(async () => await DenseSolver.SvdAsync(a, s, u, vt, 2, 3)).Throws<ArgumentException>();
    }

    [Test]
    [Arguments(double.NaN)]
    [Arguments(0.0)]
    [Arguments(500000.0)]
    public async Task FillPoisson_InvalidLambda_Throws(double lambda)
    {
        using var rng = new CudaRandom();
        using var buffer = new DeviceBuffer<uint>(16);
        await Assert.That(() => rng.FillPoisson(buffer, lambda)).Throws<ArgumentOutOfRangeException>();
    }
}
