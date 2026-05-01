// MatmulSample — demonstrates cuBLASLt matrix multiplication.
// cuBLASLt uses column-major layout for all matrices.
// Requires a CUDA-capable GPU at runtime.

using NVMathNet;
using NVMathNet.LinAlg;

Console.WriteLine("=== NVMath.Net Matmul Sample ===");

// ── 1. Basic 4×4 SGEMM: C = A * B ────────────────────────────────────────────
const int M = 4, N = 4, K = 4;

// Identity matrix A (symmetric — same in row-major and column-major)
var hostA = new float[M * K];
for (int i = 0; i < M; i++)
{
    hostA[i + M * i] = 1f; // col-major: element (i,j) at index i + M*j
}

// B in column-major: B[i,j] = i*N + j + 1  (row-major interpretation)
// Stored col-major: element (r,c) at index r + K*c
var hostB = new float[K * N];
for (int r = 0; r < K; r++)
{
    for (int j = 0; j < N; j++)
    {
        hostB[r + K * j] = r * N + j + 1f;
    }
}

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
    {
        Console.Write($" {result[row + M * col],6:F1}"); // col-major read
    }

    Console.WriteLine(" ]");
}

// ── 2. 2×2 SGEMM with alpha/beta scalars ─────────────────────────────────────
const int S = 2;
// Column-major: diag(2,3) = [2, 0, 0, 3] (same for diagonal)
var hostA2 = new float[] { 2f, 0f, 0f, 3f };
// Column-major: identity = [1, 0, 0, 1] (same for identity)
var hostB2 = new float[] { 1f, 0f, 0f, 1f };
var hostC2 = new float[] { 1f, 1f, 1f, 1f };

await using var stream2 = new CudaStream();
using var a2 = new DeviceBuffer<float>(S * S);
using var b2 = new DeviceBuffer<float>(S * S);
using var c2 = new DeviceBuffer<float>(S * S);
a2.CopyFrom(hostA2);
b2.CopyFrom(hostB2);
c2.CopyFrom(hostC2);

using var mm2 = new Matmul(S, S, S, stream: stream2);
await mm2.PlanAsync();
// D = alpha*A*B + beta*C = 2*diag(2,3)*I + 0.5*ones
//   = [[4,0],[0,6]] + [[0.5,0.5],[0.5,0.5]] = [[4.5,0.5],[0.5,6.5]]
// Column-major result: [4.5, 0.5, 0.5, 6.5]
await mm2.ExecuteAsync<float>(a2, b2, c2, c2, alpha: 2f, beta: 0.5f);
await stream2.SynchronizeAsync();

var r2 = c2.ToArray();
// Read col-major: (0,0)=r2[0], (0,1)=r2[2], (1,0)=r2[1], (1,1)=r2[3]
Console.WriteLine($"\n2×2 scaled GEMM: [[{r2[0]:F1}, {r2[2]:F1}], [{r2[1]:F1}, {r2[3]:F1}]]");

Console.WriteLine("All Matmul samples completed successfully.");
