// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the MIT License.

using System.Numerics;

using NVMathNet.Solver;

namespace NVMathNet.Tests;

/// <summary>
/// The complex overloads must be callable the natural way, without naming every optional argument.
/// These tests fail to <i>compile</i> if overload resolution becomes ambiguous (as it once was for
/// <c>LuSolveAsync</c> with an extra optional <c>conjugate</c> parameter).
/// </summary>
public sealed class ComplexOverloadTests
{
    [Test]
    public async Task LuSolveAsync_WithoutConjugateArgument_ResolvesAndSolves()
    {
        // A = diag(2, 4) + i·I  => (2+i) x0 = b0, (4+i) x1 = b1
        Complex[] a = [new(2, 1), 0, 0, new(4, 1)];
        Complex[] xTrue = [new(1, -1), new(0.5, 2)];
        Complex[] b = [a[0] * xTrue[0], a[3] * xTrue[1]];
        using var dA = new DeviceBuffer<Complex>(4);
        using var dB = new DeviceBuffer<Complex>(2);
        using var dPiv = new DeviceBuffer<int>(2);
        dA.CopyFrom(a);
        dB.CopyFrom(b);

        await DenseSolver.LuFactorAsync(dA, dPiv, 2, 2);
        await DenseSolver.LuSolveAsync(dA, dPiv, dB, 2);              // no nrhs / transpose / conjugate

        var x = dB.ToArray();
        await Assert.That((x[0] - xTrue[0]).Magnitude).IsLessThan(1e-12);
        await Assert.That((x[1] - xTrue[1]).Magnitude).IsLessThan(1e-12);
    }

    [Test]
    public async Task LuSolveAsync_WithTransposeOnly_UsesThePlainTranspose()
    {
        // Non-symmetric A: A^T x = b must differ from A x = b.
        Complex[] a = [new(2, 0), new(1, 1), new(0, 0), new(3, 0)];   // column-major [[2,0],[1+i,3]]
        Complex[] xTrue = [new(1, 0), new(0, 1)];
        // (A^T) = [[2, 1+i],[0, 3]]
        Complex[] b = [2 * xTrue[0] + new Complex(1, 1) * xTrue[1], 3 * xTrue[1]];
        using var dA = new DeviceBuffer<Complex>(4);
        using var dB = new DeviceBuffer<Complex>(2);
        using var dPiv = new DeviceBuffer<int>(2);
        dA.CopyFrom(a);
        dB.CopyFrom(b);

        await DenseSolver.LuFactorAsync(dA, dPiv, 2, 2);
        await DenseSolver.LuSolveAsync(dA, dPiv, dB, 2, 1, true);      // plain transpose

        var x = dB.ToArray();
        await Assert.That((x[0] - xTrue[0]).Magnitude).IsLessThan(1e-12);
        await Assert.That((x[1] - xTrue[1]).Magnitude).IsLessThan(1e-12);
    }

    [Test]
    public async Task LuSolveBatchedAsync_WithoutConjugateArgument_ResolvesAndSolves()
    {
        // Two 1x1 "matrices": (3+i) and (1-2i).
        Complex[] a = [new(3, 1), new(1, -2)];
        Complex[] xTrue = [new(2, 1), new(-1, 0.5)];
        Complex[] b = [a[0] * xTrue[0], a[1] * xTrue[1]];
        using var dA = new DeviceBuffer<Complex>(2);
        using var dB = new DeviceBuffer<Complex>(2);
        using var dPiv = new DeviceBuffer<int>(2);
        dA.CopyFrom(a);
        dB.CopyFrom(b);

        await DenseSolver.LuFactorBatchedAsync(dA, dPiv, 1, 2);
        await DenseSolver.LuSolveBatchedAsync(dA, dPiv, dB, 1, 1, 2);   // batched, defaults for the rest

        var x = dB.ToArray();
        await Assert.That((x[0] - xTrue[0]).Magnitude).IsLessThan(1e-12);
        await Assert.That((x[1] - xTrue[1]).Magnitude).IsLessThan(1e-12);
    }
}
