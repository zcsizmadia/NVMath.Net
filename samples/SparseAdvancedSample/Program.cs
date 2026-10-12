// SparseAdvancedSample — triangular solves, sparse-sparse products, format conversion and COO matrices.
// Dense operands are column-major. Requires a CUDA-capable GPU at runtime.
// The sample verifies its own results and returns a non-zero exit code on failure.

using NVMathNet;
using NVMathNet.Sparse;

Console.WriteLine("=== NVMath.Net Advanced Sparse Sample ===");
int failures = 0;

void Check(string what, double error, double tolerance = 1e-4)
{
    bool ok = error < tolerance;
    failures += ok ? 0 : 1;
    Console.WriteLine($"  [{(ok ? "PASS" : "FAIL")}] {what} (max error {error:E2})");
}

static CsrMatrix<float> MakeCsr(int rows, int cols, int[] rowOffsets, int[] colIndices, float[] values)
{
    var dRow = new DeviceBuffer<int>(rowOffsets.Length);
    var dCol = new DeviceBuffer<int>(colIndices.Length);
    var dVal = new DeviceBuffer<float>(values.Length);
    dRow.CopyFrom(rowOffsets);
    dCol.CopyFrom(colIndices);
    dVal.CopyFrom(values);
    return new CsrMatrix<float>(rows, cols, dRow, dCol, dVal);   // the matrix owns the buffers
}

// A = [[2, 0, 0],
//      [1, 3, 0],
//      [4, 5, 6]]   (lower triangular, 6 non-zeros)
using var lower = MakeCsr(3, 3, [0, 1, 3, 6], [0, 0, 1, 0, 1, 2], [2, 1, 3, 4, 5, 6]);

// ── 1. Sparse triangular solve: A y = x ──────────────────────────────────────
{
    Console.WriteLine("\n--- SpSV: solve A y = x with A lower triangular ---");
    using var x = new DeviceBuffer<float>(3);
    using var y = new DeviceBuffer<float>(3);
    x.CopyFrom([2f, 7f, 32f]);                                  // = A · [1, 2, 3]
    y.Clear();

    await SparseLinearAlgebra.SpSVAsync(lower, x, y, SparseTriangle.Lower);
    var r = y.ToArray();
    Console.WriteLine($"  y = [{string.Join(", ", r)}]  (expected 1, 2, 3)");
    Check("solution", Math.Max(Math.Abs(r[0] - 1), Math.Max(Math.Abs(r[1] - 2), Math.Abs(r[2] - 3))));
}

// ── 2. Sparse · sparse = sparse (SpGEMM) ─────────────────────────────────────
{
    Console.WriteLine("\n--- SpGEMM: C = A·A ---");
    using var c = await SparseLinearAlgebra.SpGemmAsync(lower, lower);

    // Expand the CSR result to a dense 3x3 for comparison with the product computed on the host.
    var rp = c.RowOffsets.ToArray();
    var ci = c.ColIndices.ToArray();
    var cv = c.Values.ToArray();
    var dense = new float[3, 3];
    for (int r = 0; r < 3; r++)
    {
        for (int k = rp[r]; k < rp[r + 1]; k++)
        {
            dense[r, ci[k]] = cv[k];
        }
    }

    float[,] expected = { { 4, 0, 0 }, { 5, 9, 0 }, { 37, 45, 36 } };
    double worst = 0;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            worst = Math.Max(worst, Math.Abs(dense[i, j] - expected[i, j]));
        }
    }

    Console.WriteLine($"  result has {c.NNZ} non-zeros");
    Check("matches the dense product", worst);
}

// ── 3. Format conversion: CSR -> CSC, then multiply with the CSC matrix ──────
{
    Console.WriteLine("\n--- CSR -> CSC and Transpose ---");
    using var csc = lower.ToCsc();
    Console.WriteLine($"  CSC column offsets: [{string.Join(", ", csc.ColOffsets.ToArray())}]");

    using var x = new DeviceBuffer<float>(3);
    using var yCsr = new DeviceBuffer<float>(3);
    using var yCsc = new DeviceBuffer<float>(3);
    x.CopyFrom([1f, 2f, 3f]);
    yCsr.Clear();
    yCsc.Clear();

    await SparseLinearAlgebra.SpMVAsync(lower, x, yCsr);
    await SparseLinearAlgebra.SpMVAsync(csc, x, yCsc);          // same matrix, different storage format
    var a = yCsr.ToArray();
    var b = yCsc.ToArray();
    Check("SpMV agrees between CSR and CSC", Enumerable.Range(0, 3).Max(i => Math.Abs(a[i] - b[i])));

    using var transposed = lower.Transpose();                    // CSR of Aᵀ
    using var yT = new DeviceBuffer<float>(3);
    yT.Clear();
    await SparseLinearAlgebra.SpMVAsync(transposed, x, yT);
    var t = yT.ToArray();                                        // Aᵀ·[1,2,3] = [2+2+12, 6+15, 18]
    Check("transpose product", Math.Max(Math.Abs(t[0] - 16), Math.Max(Math.Abs(t[1] - 21), Math.Abs(t[2] - 18))));
}

// ── 4. A COO matrix: SpMV and a triangular solve ─────────────────────────────
{
    Console.WriteLine("\n--- COO matrix ---");
    var dRow = new DeviceBuffer<int>(6);
    var dCol = new DeviceBuffer<int>(6);
    var dVal = new DeviceBuffer<float>(6);
    dRow.CopyFrom([0, 1, 1, 2, 2, 2]);                           // entries sorted by row
    dCol.CopyFrom([0, 0, 1, 0, 1, 2]);
    dVal.CopyFrom([2f, 1f, 3f, 4f, 5f, 6f]);
    using var coo = new CooMatrix<float>(3, 3, dRow, dCol, dVal);

    using var x = new DeviceBuffer<float>(3);
    using var y = new DeviceBuffer<float>(3);
    x.CopyFrom([1f, 2f, 3f]);
    y.Clear();
    await SparseLinearAlgebra.SpMVAsync(coo, x, y);
    var r = y.ToArray();                                          // A·[1,2,3] = [2, 7, 32]
    Check("COO SpMV", Math.Max(Math.Abs(r[0] - 2), Math.Max(Math.Abs(r[1] - 7), Math.Abs(r[2] - 32))));

    using var rhs = new DeviceBuffer<float>(3);
    using var sol = new DeviceBuffer<float>(3);
    rhs.CopyFrom([2f, 7f, 32f]);
    sol.Clear();
    await SparseLinearAlgebra.SpSVAsync(coo, rhs, sol, SparseTriangle.Lower);
    var s = sol.ToArray();
    Check("COO triangular solve", Math.Max(Math.Abs(s[0] - 1), Math.Max(Math.Abs(s[1] - 2), Math.Abs(s[2] - 3))));
}

Console.WriteLine(failures == 0 ? "\nAll checks passed." : $"\n{failures} check(s) FAILED.");
return failures == 0 ? 0 : 1;
