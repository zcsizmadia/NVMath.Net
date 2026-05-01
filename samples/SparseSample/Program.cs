// SparseSample — demonstrates cuSPARSE SpMV and SpMM.
// Requires a CUDA-capable GPU at runtime.

using NVMathNet;
using NVMathNet.Sparse;

Console.WriteLine("=== NVMath.Net Sparse Sample ===");

// ── 1. SpMV: y = A * x where A is a 3×3 identity (CSR) ──────────────────────
//
//  A = [ 1 0 0 ]   rowOffsets = [0, 1, 2, 3]
//      [ 0 1 0 ]   colIndices = [0, 1, 2]
//      [ 0 0 1 ]   values     = [1, 1, 1]

int rows = 3, cols = 3, nnz = 3;

var hostRow = new int[] { 0, 1, 2, 3 };
var hostCol = new int[] { 0, 1, 2 };
var hostVal = new float[] { 1f, 1f, 1f };
var hostX   = new float[] { 4f, 5f, 6f };

await using var stream = new CudaStream();

using var rowOffsets = new DeviceBuffer<int>(4);
using var colIndices = new DeviceBuffer<int>(nnz);
using var values     = new DeviceBuffer<float>(nnz);
using var x          = new DeviceBuffer<float>(cols);
using var y          = new DeviceBuffer<float>(rows);

rowOffsets.CopyFrom(hostRow);
colIndices.CopyFrom(hostCol);
values.CopyFrom(hostVal);
x.CopyFrom(hostX);

var matrix = new CsrMatrix<float>(rows, cols, rowOffsets, colIndices, values);
await SparseLinearAlgebra.SpMVAsync<float>(matrix, x, y, alpha: 1f, beta: 0f, stream: stream);
await stream.SynchronizeAsync();

var result = y.ToArray();
Console.WriteLine($"SpMV result: [{string.Join(", ", result)}]");   // [4, 5, 6]

// ── 2. SpMM: C = A * B where B is a 3×2 dense matrix ────────────────────────
//
//  B = [ 1 2 ]    C = A * B = [ 1 2 ]
//      [ 3 4 ]                  [ 3 4 ]
//      [ 5 6 ]                  [ 5 6 ]

int denseN = 2;
var hostB = new float[] { 1f, 2f, 3f, 4f, 5f, 6f };

using var b = new DeviceBuffer<float>(rows * denseN);
using var c = new DeviceBuffer<float>(rows * denseN);
b.CopyFrom(hostB);

await SparseLinearAlgebra.SpMMAsync<float>(matrix, b, c, denseN,
    alpha: 1f, beta: 0f, stream: stream);
await stream.SynchronizeAsync();

var cResult = c.ToArray();
Console.WriteLine("SpMM result (row-major 3×2):");
for (int r = 0; r < rows; r++)
    Console.WriteLine($"  [{cResult[r * denseN],4:F0}, {cResult[r * denseN + 1],4:F0}]");

Console.WriteLine("All Sparse samples completed successfully.");
