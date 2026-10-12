// DirectSolverSample — demonstrates the cuDSS sparse direct solver.
// Requires a CUDA-capable GPU and the cuDSS runtime (cudss64_0.dll / libcudss.so.0) on the library path.

using NVMathNet;
using NVMathNet.Sparse;

Console.WriteLine("=== NVMath.Net Direct Solver Sample ===");

// Solve A x = b for the 5×5 tridiagonal matrix with 4 on the diagonal and -1 off the diagonal.
const int n = 5;
var rowOffsets = new List<int> { 0 };
var colIndices = new List<int>();
var values = new List<double>();
for (int r = 0; r < n; r++)
{
    for (int c = Math.Max(0, r - 1); c <= Math.Min(n - 1, r + 1); c++)
    {
        colIndices.Add(c);
        values.Add(r == c ? 4.0 : -1.0);
    }

    rowOffsets.Add(colIndices.Count);
}

var dRow = new DeviceBuffer<int>(rowOffsets.Count);
var dCol = new DeviceBuffer<int>(colIndices.Count);
var dVal = new DeviceBuffer<double>(values.Count);
dRow.CopyFrom(rowOffsets.ToArray());
dCol.CopyFrom(colIndices.ToArray());
dVal.CopyFrom(values.ToArray());
using var matrix = new CsrMatrix<double>(n, n, dRow, dCol, dVal);

// Right-hand side chosen so that the exact solution is x = [1, 1, 1, 1, 1].
using var b = new DeviceBuffer<double>(n);
using var x = new DeviceBuffer<double>(n);
b.CopyFrom(new double[] { 3, 2, 2, 2, 3 });

using var solver = new SparseDirectSolver<double>(matrix);
solver.Analyze();      // reordering + symbolic factorisation (depends only on the pattern)
solver.Factorize();    // numerical factorisation
solver.Solve(b, x);    // call again for further right-hand sides

Console.WriteLine($"x = [{string.Join(", ", x.ToArray().Select(v => v.ToString("F6")))}]");   // ≈ [1, 1, 1, 1, 1]
Console.WriteLine($"Non-zeros in the factors: {solver.FactorNonZeros}");
