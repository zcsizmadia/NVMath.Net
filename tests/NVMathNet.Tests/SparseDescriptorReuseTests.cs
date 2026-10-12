// Copyright (c) 2026 Zoltan Csizmadia. All rights reserved.
// Licensed under the MIT License.

using NVMathNet.Sparse;

namespace NVMathNet.Tests;

/// <summary>
/// A matrix must stay usable for products after it was used in a triangular solve. Triangular solves attach
/// fill-mode attributes to a cuSPARSE descriptor, and a descriptor that carries them is rejected by SpMV and SpGEMM
/// (CUSPARSE_STATUS_NOT_SUPPORTED), so the solves work on a temporary descriptor.
/// </summary>
public sealed class SparseDescriptorReuseTests
{
    // A = [[2,0,0],[1,3,0],[4,5,6]]
    private static CsrMatrix<float> MakeCsr()
    {
        var dRow = new DeviceBuffer<int>(4);
        var dCol = new DeviceBuffer<int>(6);
        var dVal = new DeviceBuffer<float>(6);
        dRow.CopyFrom(new[] { 0, 1, 3, 6 });
        dCol.CopyFrom(new[] { 0, 0, 1, 0, 1, 2 });
        dVal.CopyFrom(new float[] { 2, 1, 3, 4, 5, 6 });
        return new CsrMatrix<float>(3, 3, dRow, dCol, dVal);
    }

    private static CooMatrix<float> MakeCoo()
    {
        var dRow = new DeviceBuffer<int>(6);
        var dCol = new DeviceBuffer<int>(6);
        var dVal = new DeviceBuffer<float>(6);
        dRow.CopyFrom(new[] { 0, 1, 1, 2, 2, 2 });
        dCol.CopyFrom(new[] { 0, 0, 1, 0, 1, 2 });
        dVal.CopyFrom(new float[] { 2, 1, 3, 4, 5, 6 });
        return new CooMatrix<float>(3, 3, dRow, dCol, dVal);
    }

    [Test]
    public async Task Csr_SolveThenMultiplyAndSpGemm_SameMatrix()
    {
        using var a = MakeCsr();
        using var x = new DeviceBuffer<float>(3);
        using var y = new DeviceBuffer<float>(3);
        x.CopyFrom(new float[] { 2, 7, 32 });
        y.Clear();

        await SparseLinearAlgebra.SpSVAsync(a, x, y, SparseTriangle.Lower);
        await Assert.That(Math.Abs(y.ToArray()[2] - 3f)).IsLessThan(1e-4f);

        // The same matrix is still accepted by SpMV ...
        using var z = new DeviceBuffer<float>(3);
        z.Clear();
        x.CopyFrom(new float[] { 1, 2, 3 });
        await SparseLinearAlgebra.SpMVAsync(a, x, z);
        await Assert.That(Math.Abs(z.ToArray()[2] - 32f)).IsLessThan(1e-4f);      // A·[1,2,3] = [2,7,32]

        // ... and by SpGEMM, including as both operands.
        using var c = await SparseLinearAlgebra.SpGemmAsync(a, a);
        await Assert.That(c.NNZ).IsEqualTo(6L);
    }

    [Test]
    public async Task Csr_SpSmThenSpMm_SameMatrix()
    {
        using var a = MakeCsr();
        using var b = new DeviceBuffer<float>(6);
        using var c = new DeviceBuffer<float>(6);
        b.CopyFrom(new float[] { 2, 7, 32, 4, 14, 64 });
        c.Clear();

        await SparseLinearAlgebra.SpSMAsync(a, b, c, n: 2, SparseTriangle.Lower);
        await Assert.That(Math.Abs(c.ToArray()[5] - 6f)).IsLessThan(1e-4f);

        using var d = new DeviceBuffer<float>(6);
        d.Clear();
        c.CopyFrom(new float[] { 1, 2, 3, 2, 4, 6 });
        await SparseLinearAlgebra.SpMMAsync(a, c, d, n: 2);                         // A · X, same matrix again
        await Assert.That(Math.Abs(d.ToArray()[2] - 32f)).IsLessThan(1e-3f);
    }

    [Test]
    public async Task Coo_SolveThenMultiply_SameMatrix()
    {
        using var a = MakeCoo();
        using var x = new DeviceBuffer<float>(3);
        using var y = new DeviceBuffer<float>(3);
        x.CopyFrom(new float[] { 2, 7, 32 });
        y.Clear();

        await SparseLinearAlgebra.SpSVAsync(a, x, y, SparseTriangle.Lower);
        await Assert.That(Math.Abs(y.ToArray()[2] - 3f)).IsLessThan(1e-4f);

        x.CopyFrom(new float[] { 1, 2, 3 });
        y.Clear();
        await SparseLinearAlgebra.SpMVAsync(a, x, y);
        await Assert.That(Math.Abs(y.ToArray()[2] - 32f)).IsLessThan(1e-4f);
    }
}
