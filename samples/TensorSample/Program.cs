// TensorSample — demonstrates cuTENSOR contraction as a matrix multiply.
// Contract A[m,k] * B[k,n] -> D[m,n]  (C = 0, beta = 0)
// Requires a CUDA-capable GPU at runtime.

using NVMathNet;
using NVMathNet.Tensor;

Console.WriteLine("=== NVMath.Net Tensor Sample ===");

// ── Dimensions ────────────────────────────────────────────────────────────────
const int Rows = 4, Cols = 4, Inner = 4;  // m, n, k

// Mode labels as ASCII chars: 'm'=109, 'n'=110, 'k'=107
var modeA = new int[] { 'm', 'k' };   // A[m,k]
var modeB = new int[] { 'k', 'n' };   // B[k,n]
var modeC = new int[] { 'm', 'n' };   // C[m,n]

var extA = new long[] { Rows, Inner };
var extB = new long[] { Inner, Cols };
var extC = new long[] { Rows, Cols };

// ── Host data ─────────────────────────────────────────────────────────────────
// A = identity, B = index+1  →  D should equal B
var hostA = new float[Rows * Inner];
for (int i = 0; i < Rows; i++) hostA[i * Inner + i] = 1f;

var hostB = new float[Inner * Cols];
for (int i = 0; i < Inner * Cols; i++) hostB[i] = i + 1f;

await using var stream = new CudaStream();
using var a = new DeviceBuffer<float>(Rows * Inner);
using var b = new DeviceBuffer<float>(Inner * Cols);
using var c = new DeviceBuffer<float>(Rows * Cols);   // output / zero
a.CopyFrom(hostA);
b.CopyFrom(hostB);
c.Clear();

// ── Plan and execute contraction ──────────────────────────────────────────────
using var tc = new TensorContraction(
    extA, modeA,
    extB, modeB,
    extC, modeC,
    extC, modeC,
    TensorDataType.Float32,
    stream: stream);

await tc.PlanAsync();
await tc.ExecuteAsync<float>(a, b, c, c, alpha: 1f, beta: 0f);
await tc.SynchronizeAsync();

// ── Print result ──────────────────────────────────────────────────────────────
var result = c.ToArray();
Console.WriteLine($"D = I × B  ({Rows}×{Cols}):");
for (int row = 0; row < Rows; row++)
{
    Console.Write("  [");
    for (int col = 0; col < Cols; col++)
        Console.Write($" {result[row * Cols + col],5:F1}");
    Console.WriteLine(" ]");
}

// Verify against expected (B)
double maxErr = 0;
for (int i = 0; i < Rows * Cols; i++)
    maxErr = Math.Max(maxErr, Math.Abs(result[i] - hostB[i]));
Console.WriteLine($"Max error vs expected: {maxErr:E3}");

Console.WriteLine("Tensor sample completed successfully.");
