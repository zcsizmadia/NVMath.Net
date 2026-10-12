// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the MIT License.

using System.Numerics;

using NVMathNet.Sparse;

namespace NVMathNet.Tests;

/// <summary>
/// Tests for sparse operations on complex (double) matrices, including cuDSS.
/// Requires a CUDA-capable GPU (and the cuDSS runtime for the direct-solver tests).
/// </summary>
public sealed class ComplexSparseTests
{
    private static CsrMatrix<Complex> MakeCsr(int rows, int cols, int[] rowPtr, int[] colIdx, Complex[] vals)
    {
        var dRow = new DeviceBuffer<int>(rowPtr.Length);
        var dCol = new DeviceBuffer<int>(colIdx.Length);
        var dVal = new DeviceBuffer<Complex>(vals.Length);
        dRow.CopyFrom(rowPtr.AsSpan());
        dCol.CopyFrom(colIdx.AsSpan());
        dVal.CopyFrom(vals.AsSpan());
        return new CsrMatrix<Complex>(rows, cols, dRow, dCol, dVal);
    }

    // A = [[1+i, 0, 2], [0, 3, -i]]
    private static CsrMatrix<Complex> MakeA() => MakeCsr(
        2, 3, [0, 2, 4], [0, 2, 1, 2],
        [new(1, 1), new(2, 0), new(3, 0), new(0, -1)]);

    [Test]
    public async Task SpMV_WithComplexAlphaAndBeta()
    {
        using var a = MakeA();
        Complex[] hx = [new(1, 0), new(0, 1), new(2, -1)];
        Complex[] hy = [new(1, 1), new(-1, 2)];
        using var x = new DeviceBuffer<Complex>(3);
        using var y = new DeviceBuffer<Complex>(2);
        x.CopyFrom(hx.AsSpan());
        y.CopyFrom(hy.AsSpan());
        var alpha = new Complex(0, 2);
        var beta = new Complex(1, -1);

        await SparseLinearAlgebra.SpMVAsync(a, x, y, alpha, beta);

        // y = alpha * A x + beta * y
        var ax0 = new Complex(1, 1) * hx[0] + 2 * hx[2];
        var ax1 = 3 * hx[1] + new Complex(0, -1) * hx[2];
        var r = y.ToArray();
        await Assert.That((r[0] - (alpha * ax0 + beta * hy[0])).Magnitude).IsLessThan(1e-12);
        await Assert.That((r[1] - (alpha * ax1 + beta * hy[1])).Magnitude).IsLessThan(1e-12);
    }

    [Test]
    public async Task SpMM_MatchesDense()
    {
        using var a = MakeA();
        Complex[] hb = [new(1, 0), new(0, 1), new(2, -1), new(0, 0), new(1, 1), new(1, 0)]; // 3x2 column-major
        using var b = new DeviceBuffer<Complex>(6);
        using var c = new DeviceBuffer<Complex>(4);
        b.CopyFrom(hb.AsSpan());
        c.Clear();

        await SparseLinearAlgebra.SpMMAsync(a, b, c, n: 2, Complex.One, Complex.Zero);

        var r = c.ToArray(); // 2x2 column-major
        for (int col = 0; col < 2; col++)
        {
            Complex[] x = [hb[3 * col], hb[3 * col + 1], hb[3 * col + 2]];
            var y0 = new Complex(1, 1) * x[0] + 2 * x[2];
            var y1 = 3 * x[1] + new Complex(0, -1) * x[2];
            await Assert.That((r[2 * col] - y0).Magnitude).IsLessThan(1e-12);
            await Assert.That((r[2 * col + 1] - y1).Magnitude).IsLessThan(1e-12);
        }
    }

    [Test]
    public async Task SpSV_LowerTriangular_Solves()
    {
        // A = [[2, 0], [1+i, 3]]; solve A y = x for y = [1-i, 2i]
        using var a = MakeCsr(2, 2, [0, 1, 3], [0, 0, 1], [new(2, 0), new(1, 1), new(3, 0)]);
        Complex[] yTrue = [new(1, -1), new(0, 2)];
        Complex[] hx = [2 * yTrue[0], new Complex(1, 1) * yTrue[0] + 3 * yTrue[1]];
        using var x = new DeviceBuffer<Complex>(2);
        using var y = new DeviceBuffer<Complex>(2);
        x.CopyFrom(hx.AsSpan());
        y.Clear();

        await SparseLinearAlgebra.SpSVAsync(a, x, y, SparseTriangle.Lower);

        var r = y.ToArray();
        await Assert.That((r[0] - yTrue[0]).Magnitude).IsLessThan(1e-12);
        await Assert.That((r[1] - yTrue[1]).Magnitude).IsLessThan(1e-12);
    }

    [Test]
    public async Task SpGemm_AtimesTranspose_MatchesDense()
    {
        // C = A * A^T (plain transpose, no conjugation)
        using var a = MakeA();
        using var aT = a.Transpose();
        using var c = await SparseLinearAlgebra.SpGemmAsync(a, aT);

        var rp = c.RowOffsets.ToArray();
        var ci = c.ColIndices.ToArray();
        var cv = c.Values.ToArray();
        var dense = new Complex[2, 2];
        for (int r = 0; r < 2; r++)
        {
            for (int k = rp[r]; k < rp[r + 1]; k++)
            {
                dense[r, ci[k]] = cv[k];
            }
        }

        var i1 = Complex.ImaginaryOne;
        await Assert.That((dense[0, 0] - ((1 + i1) * (1 + i1) + 4)).Magnitude).IsLessThan(1e-12);
        await Assert.That((dense[0, 1] - 2 * -i1).Magnitude).IsLessThan(1e-12);
        await Assert.That((dense[1, 0] - -i1 * 2).Magnitude).IsLessThan(1e-12);
        await Assert.That((dense[1, 1] - (9 + -i1 * -i1)).Magnitude).IsLessThan(1e-12);
    }

    // ── cuDSS with complex matrices ──────────────────────────────────────────

    private static CsrMatrix<Complex> Tridiagonal(
        int n, Complex diag, Complex lower, Complex upper, bool lowerOnly, out Complex[] dense)
    {
        var rowPtr = new List<int> { 0 };
        var cols = new List<int>();
        var vals = new List<Complex>();
        dense = new Complex[n * n];
        for (int r = 0; r < n; r++)
        {
            for (int c = Math.Max(0, r - 1); c <= Math.Min(n - 1, r + 1); c++)
            {
                var v = r == c ? diag : c < r ? lower : upper;
                dense[r + n * c] = v;
                if (lowerOnly && c > r)
                {
                    continue;
                }

                cols.Add(c);
                vals.Add(v);
            }

            rowPtr.Add(cols.Count);
        }

        return MakeCsr(n, n, [.. rowPtr], [.. cols], [.. vals]);
    }

    private static Complex[] Multiply(Complex[] a, int n, Complex[] x)
    {
        var b = new Complex[n];
        for (int r = 0; r < n; r++)
        {
            for (int c = 0; c < n; c++)
            {
                b[r] += a[r + n * c] * x[c];
            }
        }

        return b;
    }

    [Test]
    public async Task DirectSolver_GeneralComplexMatrix_RecoversKnownSolution()
    {
        const int N = 40;
        using var matrix = Tridiagonal(N, new(4, 1), new(-1, 0.5), new(-0.5, -1), false, out var dense);
        var xTrue = Enumerable.Range(0, N).Select(i => new Complex(Math.Sin(i), Math.Cos(i * 0.5))).ToArray();
        using var dB = new DeviceBuffer<Complex>(N);
        using var dX = new DeviceBuffer<Complex>(N);
        dB.CopyFrom(Multiply(dense, N, xTrue).AsSpan());

        using var solver = new SparseDirectSolver<Complex>(matrix);
        solver.Solve(dB, dX);

        var x = dX.ToArray();
        for (int i = 0; i < N; i++)
        {
            await Assert.That((x[i] - xTrue[i]).Magnitude).IsLessThan(1e-9);
        }
    }

    [Test]
    public async Task DirectSolver_HermitianPositiveDefinite_LowerTriangleOnly()
    {
        // Hermitian tridiagonal: diagonal 4, lower entries (-1+0.5i), upper entries their conjugates.
        const int N = 50;
        var lower = new Complex(-1, 0.5);
        using var matrix = Tridiagonal(N, 4, lower, Complex.Conjugate(lower), true, out var dense);
        var xTrue = Enumerable.Range(0, N).Select(i => new Complex(1 + i * 0.1, -i * 0.05)).ToArray();
        using var dB = new DeviceBuffer<Complex>(N);
        using var dX = new DeviceBuffer<Complex>(N);
        dB.CopyFrom(Multiply(dense, N, xTrue).AsSpan());

        var options = new DirectSolverOptions
        {
            Structure = SparseMatrixStructure.HermitianPositiveDefinite,
            View = SparseMatrixView.Lower,
        };
        using var solver = new SparseDirectSolver<Complex>(matrix, options);
        solver.Solve(dB, dX);

        var x = dX.ToArray();
        for (int i = 0; i < N; i++)
        {
            await Assert.That((x[i] - xTrue[i]).Magnitude).IsLessThan(1e-9);
        }
    }
}
