// MatmulSample — demonstrates cuBLASLt matrix multiplication.
// Requires a CUDA-capable GPU at runtime.

using NVMathNet;
using NVMathNet.LinAlg;

Console.WriteLine("=== NVMath.Net Matmul Sample ===");

// ── 1. Basic 4×4 SGEMM: C = A * B ────────────────────────────────────────────
const int M = 4, N = 4, K = 4;

// Identity matrix A
var hostA = new float[M * K];
for (int i = 0; i < M; i++) hostA[i * K + i] = 1f;

// B = incremental values
var hostB = new float[K * N];
for (int i = 0; i < K * N; i++) hostB[i] = i + 1f;

await using var stream = new CudaStream();
using var a = new DeviceBuffer<float>(M * K);
using var b = new DeviceBuffer<float>(K * N);
using var c = new DeviceBuffer<float>(M * N);
a.CopyFrom(hostA);
b.CopyFrom(hostB);

using var matmul = new Matmul(M, N, K, stream: stream);
await matmul.PlanAsync();
await matmul.ExecuteAsync<float>(a, b, c, c, alpha: 1f, beta: 0f);
await stream.SynchronizeAsync();

var result = c.ToArray();
Console.WriteLine("C = I * B =");
for (int row = 0; row < M; row++)
{
    Console.Write("  [");
    for (int col = 0; col < N; col++)
        Console.Write($" {result[row * N + col],6:F1}");
    Console.WriteLine(" ]");
}

// ── 2. 2×2 SGEMM with alpha/beta scalars ─────────────────────────────────────
const int S = 2;
var hostA2 = new float[] { 2f, 0f, 0f, 3f };   // diagonal
var hostB2 = new float[] { 1f, 0f, 0f, 1f };   // identity
var hostC2 = new float[] { 1f, 1f, 1f, 1f };   // initial C

await using var stream2 = new CudaStream();
using var a2 = new DeviceBuffer<float>(S * S);
using var b2 = new DeviceBuffer<float>(S * S);
using var c2 = new DeviceBuffer<float>(S * S);
a2.CopyFrom(hostA2);
b2.CopyFrom(hostB2);
c2.CopyFrom(hostC2);

using var mm2 = new Matmul(S, S, S, stream: stream2);
await mm2.PlanAsync();
// Result = 2*diag(2,3) + 0.5*C = [4.5, 1, 1, 3.5] approx
await mm2.ExecuteAsync<float>(a2, b2, c2, c2, alpha: 2f, beta: 0.5f);
await stream2.SynchronizeAsync();

var r2 = c2.ToArray();
Console.WriteLine($"\n2×2 scaled GEMM: [{r2[0]:F1}, {r2[1]:F1}, {r2[2]:F1}, {r2[3]:F1}]");

Console.WriteLine("All Matmul samples completed successfully.");
