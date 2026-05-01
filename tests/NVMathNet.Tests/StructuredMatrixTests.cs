// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the MIT License.

using NVMathNet.LinAlg;

namespace NVMathNet.Tests;

/// <summary>
/// Tests for structured matrix operations: TRSM (triangular solve) and SYMM (symmetric multiply).
/// </summary>
public sealed class StructuredMatrixTests
{
    [Test]
    public async Task TriangularSolve_Float_LowerIdentity()
    {
        // A = lower-triangular identity, B = [1..4]
        // Solve A*X = B => X = B
        const int N = 2;
        float[] hA = [1, 0, 0, 1]; // 2×2 identity (col-major, trivially lower-triangular)
        float[] hB = [1, 2];

        using var dA = new DeviceBuffer<float>(N * N);
        using var dB = new DeviceBuffer<float>(N);
        dA.CopyFrom(hA.AsSpan());
        dB.CopyFrom(hB.AsSpan());

        await TriangularSolve.StrsmAsync(dA, dB, N, 1, leftSide: true, upper: false);

        float[] result = dB.ToArray();
        await Assert.That(Math.Abs(result[0] - 1f)).IsLessThan(1e-4f);
        await Assert.That(Math.Abs(result[1] - 2f)).IsLessThan(1e-4f);
    }

    [Test]
    public async Task TriangularSolve_Float_LowerTriangular()
    {
        // A = [[2,0],[3,4]], solve A*x = b where b = [4, 17]
        // Forward sub: x0 = 4/2 = 2, x1 = (17 - 3*2)/4 = 11/4 = 2.75
        const int N = 2;
        float[] hA = [2, 3, 0, 4]; // col-major: A(0,0)=2, A(1,0)=3, A(0,1)=0, A(1,1)=4
        float[] hB = [4, 17];

        using var dA = new DeviceBuffer<float>(N * N);
        using var dB = new DeviceBuffer<float>(N);
        dA.CopyFrom(hA.AsSpan());
        dB.CopyFrom(hB.AsSpan());

        await TriangularSolve.StrsmAsync(dA, dB, N, 1, leftSide: true, upper: false);

        float[] result = dB.ToArray();
        await Assert.That(Math.Abs(result[0] - 2f)).IsLessThan(1e-4f);
        await Assert.That(Math.Abs(result[1] - 2.75f)).IsLessThan(1e-4f);
    }

    [Test]
    public async Task TriangularSolve_Double_UpperTriangular()
    {
        // A = [[3,1],[0,2]], solve A*x = b where b = [5, 4]
        // Back sub: x1 = 4/2 = 2, x0 = (5 - 1*2)/3 = 1
        const int N = 2;
        double[] hA = [3, 0, 1, 2]; // col-major: A(0,0)=3, A(1,0)=0, A(0,1)=1, A(1,1)=2
        double[] hB = [5, 4];

        using var dA = new DeviceBuffer<double>(N * N);
        using var dB = new DeviceBuffer<double>(N);
        dA.CopyFrom(hA.AsSpan());
        dB.CopyFrom(hB.AsSpan());

        await TriangularSolve.DtrsmAsync(dA, dB, N, 1, leftSide: true, upper: true);

        double[] result = dB.ToArray();
        await Assert.That(Math.Abs(result[0] - 1.0)).IsLessThan(1e-10);
        await Assert.That(Math.Abs(result[1] - 2.0)).IsLessThan(1e-10);
    }

    [Test]
    public async Task SymmetricMultiply_Float_2x2()
    {
        // A = [[2,1],[1,3]] (symmetric), B = [[1,2],[3,4]] (2×2)
        // C = A*B = [[2+3, 4+4],[1+9, 2+12]] = [[5, 8],[10, 14]]
        const int M = 2, N = 2;
        // A col-major: [A(0,0)=2, A(1,0)=1, A(0,1)=1, A(1,1)=3]
        float[] hA = [2, 1, 1, 3];
        // B col-major: [B(0,0)=1, B(1,0)=3, B(0,1)=2, B(1,1)=4]
        float[] hB = [1, 3, 2, 4];
        float[] expected = [5, 10, 8, 14]; // C col-major

        using var dA = new DeviceBuffer<float>(M * M);
        using var dB = new DeviceBuffer<float>(M * N);
        using var dC = new DeviceBuffer<float>(M * N);
        dA.CopyFrom(hA.AsSpan());
        dB.CopyFrom(hB.AsSpan());
        dC.Clear();

        await SymmetricMultiply.SsymmAsync(dA, dB, dC, M, N, leftSide: true, upper: false);

        float[] result = dC.ToArray();
        for (int i = 0; i < M * N; i++)
        {
            float err = Math.Abs(result[i] - expected[i]);
            await Assert.That(err).IsLessThan(1e-3f);
        }
    }

    [Test]
    public async Task SymmetricMultiply_Double_AlphaBeta()
    {
        // C = 2*A*B + 3*C_init
        // A = [[1,0],[0,1]] (identity, symmetric), B = [[2],[3]], C_init = [[1],[1]]
        // C = 2*B + 3*C_init = [[4+3],[6+3]] = [[7],[9]]
        const int M = 2, N = 1;
        double[] hA = [1, 0, 0, 1];
        double[] hB = [2, 3];
        double[] hC = [1, 1];

        using var dA = new DeviceBuffer<double>(M * M);
        using var dB = new DeviceBuffer<double>(M * N);
        using var dC = new DeviceBuffer<double>(M * N);
        dA.CopyFrom(hA.AsSpan());
        dB.CopyFrom(hB.AsSpan());
        dC.CopyFrom(hC.AsSpan());

        await SymmetricMultiply.DsymmAsync(dA, dB, dC, M, N, alpha: 2.0, beta: 3.0, leftSide: true);

        double[] result = dC.ToArray();
        await Assert.That(Math.Abs(result[0] - 7.0)).IsLessThan(1e-10);
        await Assert.That(Math.Abs(result[1] - 9.0)).IsLessThan(1e-10);
    }

    // ── DiagonalMultiply tests ────────────────────────────────────────────────

    [Test]
    public async Task DiagonalMultiply_Float_LeftSide()
    {
        // C = diag(x) * A
        // x = [2, 3], A = [[1, 2],[3, 4]] (col-major: [1,3,2,4])
        // C = [[2*1, 2*2],[3*3, 3*4]] = [[2, 4],[9, 12]] (col-major: [2,9,4,12])
        const int M = 2, N = 2;
        float[] hA = [1, 3, 2, 4]; // col-major
        float[] hX = [2, 3];
        float[] expected = [2, 9, 4, 12]; // col-major

        using var dA = new DeviceBuffer<float>(M * N);
        using var dX = new DeviceBuffer<float>(M);
        using var dC = new DeviceBuffer<float>(M * N);
        dA.CopyFrom(hA.AsSpan());
        dX.CopyFrom(hX.AsSpan());
        dC.Clear();

        await DiagonalMultiply.SdgmmAsync(dA, dX, dC, M, N, leftSide: true);

        float[] result = dC.ToArray();
        for (int i = 0; i < M * N; i++)
        {
            float err = Math.Abs(result[i] - expected[i]);
            await Assert.That(err).IsLessThan(1e-4f);
        }
    }

    [Test]
    public async Task DiagonalMultiply_Float_RightSide()
    {
        // C = A * diag(x)
        // x = [2, 3], A = [[1, 2],[3, 4]] (col-major: [1,3,2,4])
        // C = [[1*2, 2*3],[3*2, 4*3]] = [[2, 6],[6, 12]] (col-major: [2,6,6,12])
        const int M = 2, N = 2;
        float[] hA = [1, 3, 2, 4];
        float[] hX = [2, 3];
        float[] expected = [2, 6, 6, 12];

        using var dA = new DeviceBuffer<float>(M * N);
        using var dX = new DeviceBuffer<float>(N);
        using var dC = new DeviceBuffer<float>(M * N);
        dA.CopyFrom(hA.AsSpan());
        dX.CopyFrom(hX.AsSpan());
        dC.Clear();

        await DiagonalMultiply.SdgmmAsync(dA, dX, dC, M, N, leftSide: false);

        float[] result = dC.ToArray();
        for (int i = 0; i < M * N; i++)
        {
            float err = Math.Abs(result[i] - expected[i]);
            await Assert.That(err).IsLessThan(1e-4f);
        }
    }

    [Test]
    public async Task DiagonalMultiply_Double_LeftSide()
    {
        // C = diag(x) * A, 3x2
        // x = [1, 2, 3], A = [[1,4],[2,5],[3,6]] (col-major: [1,2,3,4,5,6])
        // C = [[1*1, 1*4],[2*2, 2*5],[3*3, 3*6]] = [[1,4],[4,10],[9,18]]
        // col-major: [1,4,9,4,10,18]
        const int M = 3, N = 2;
        double[] hA = [1, 2, 3, 4, 5, 6];
        double[] hX = [1, 2, 3];
        double[] expected = [1, 4, 9, 4, 10, 18];

        using var dA = new DeviceBuffer<double>(M * N);
        using var dX = new DeviceBuffer<double>(M);
        using var dC = new DeviceBuffer<double>(M * N);
        dA.CopyFrom(hA.AsSpan());
        dX.CopyFrom(hX.AsSpan());
        dC.Clear();

        await DiagonalMultiply.DdgmmAsync(dA, dX, dC, M, N, leftSide: true);

        double[] result = dC.ToArray();
        for (int i = 0; i < M * N; i++)
        {
            double err = Math.Abs(result[i] - expected[i]);
            await Assert.That(err).IsLessThan(1e-10);
        }
    }
}
