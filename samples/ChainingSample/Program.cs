// ChainingSample — demonstrates how to chain multiple GPU operations and
// synchronize efficiently using CudaStream and CudaEvent.
//
// Three patterns are shown:
//
//   Pattern 1 — Linear chain on a single stream
//     Queue several operations back-to-back on one stream.
//     The GPU executes them in order; the CPU awaits only once at the end.
//
//   Pattern 2 — Parallel pipelines on independent streams
//     Submit two independent operation sequences to separate streams.
//     The GPU can run both concurrently. The CPU awaits both with Task.WhenAll.
//
//   Pattern 3 — Event-based cross-stream dependency
//     Stream B must not begin its work until stream A's result is ready.
//     CudaEvent.Record + CudaStream.WaitEvent sets up a GPU-side ordering
//     constraint without blocking the CPU between submissions.
//
// Requires a CUDA-capable GPU at runtime.

using NVMathNet;
using NVMathNet.LinAlg;

Console.WriteLine("=== NVMath.Net Chaining Sample ===");
Console.WriteLine();

// ── Shared host data (4×4 matrices) ─────────────────────────────────────────
const int M = 4, N = 4, K = 4;

// A = identity
var hostA = new float[M * K];
for (int i = 0; i < M; i++) hostA[i * K + i] = 1f;

// B = row index + 1
var hostB = new float[K * N];
for (int i = 0; i < K * N; i++) hostB[i] = i + 1f;

// D = checkerboard values
var hostD = new float[K * N];
for (int i = 0; i < K * N; i++) hostD[i] = (i % 4) + 1f;

// ═════════════════════════════════════════════════════════════════════════════
// Pattern 1 — Linear chain on a single stream
//
// Three GEMMs queued on one stream with Blocking = false.
// The CPU enqueues all three before the GPU has finished even the first.
// A single SynchronizeAsync covers all three.
//
//   stream: ──[GEMM₁: C=A*B]──[GEMM₂: E=C*D]──[GEMM₃: G=E*A]──● sync
//
// ═════════════════════════════════════════════════════════════════════════════
Console.WriteLine("Pattern 1 — Linear chain: 3 GEMMs on one stream, one Synchronize");

await using var stream1 = new CudaStream();

// Blocking = false means ExecuteAsync enqueues GPU work and returns immediately
// without waiting for the kernel to finish. The caller is responsible for
// synchronizing the stream when the results are needed.
var nonBlocking = new MatmulOptions { Blocking = false };

using var devA  = new DeviceBuffer<float>(M * K);
using var devB  = new DeviceBuffer<float>(K * N);
using var devC1 = new DeviceBuffer<float>(M * N);   // GEMM₁ output
using var devD  = new DeviceBuffer<float>(K * N);
using var devE  = new DeviceBuffer<float>(M * N);   // GEMM₂ output
using var devG  = new DeviceBuffer<float>(M * N);   // GEMM₃ output

devA.CopyFrom(hostA);
devB.CopyFrom(hostB);
devD.CopyFrom(hostD);

// Three plans — all share the same stream so they execute in submission order.
using var mm1 = new Matmul(M, N, K, options: nonBlocking, stream: stream1);
using var mm2 = new Matmul(M, N, K, options: nonBlocking, stream: stream1);
using var mm3 = new Matmul(M, N, K, options: nonBlocking, stream: stream1);

// Plan selection is CPU-only work; does not enqueue GPU kernels.
await mm1.PlanAsync();
await mm2.PlanAsync();
await mm3.PlanAsync();

// Enqueue all three kernels. Because Blocking = false, none of these awaits
// waits for the GPU — they just submit work and return a completed Task.
// The GPU processes them in order on stream1.
_ = mm1.ExecuteAsync<float>(devA, devB, devC1, devC1, alpha: 1f, beta: 0f); // C = A * B
_ = mm2.ExecuteAsync<float>(devC1, devD, devE,  devE,  alpha: 1f, beta: 0f); // E = C * D
_ = mm3.ExecuteAsync<float>(devE,  devA, devG,  devG,  alpha: 1f, beta: 0f); // G = E * A

// Single barrier: wait for stream1 to drain all three kernels.
await stream1.SynchronizeAsync();

var result1 = devG.ToArray();
Console.WriteLine($"  G[0,0] = {result1[0]:F1}  (all three GEMMs complete)");
Console.WriteLine();

// ═════════════════════════════════════════════════════════════════════════════
// Pattern 2 — Parallel pipelines on independent streams
//
// Two chains run concurrently. The GPU schedules both on separate hardware
// queues and overlaps execution where resources allow.
//
//   streamA: ──[GEMM_A1]──[GEMM_A2]──● \
//                                        Task.WhenAll(syncA, syncB)
//   streamB: ──[GEMM_B1]──[GEMM_B2]──● /
//
// ═════════════════════════════════════════════════════════════════════════════
Console.WriteLine("Pattern 2 — Parallel pipelines: 2 independent chains, Task.WhenAll");

await using var streamA = new CudaStream();
await using var streamB = new CudaStream();

// Pipeline A buffers
using var pA_in1  = new DeviceBuffer<float>(M * K);
using var pA_in2  = new DeviceBuffer<float>(K * N);
using var pA_mid  = new DeviceBuffer<float>(M * N);
using var pA_out  = new DeviceBuffer<float>(M * N);

// Pipeline B buffers (same shapes, independent data)
using var pB_in1  = new DeviceBuffer<float>(M * K);
using var pB_in2  = new DeviceBuffer<float>(K * N);
using var pB_mid  = new DeviceBuffer<float>(M * N);
using var pB_out  = new DeviceBuffer<float>(M * N);

pA_in1.CopyFrom(hostA);
pA_in2.CopyFrom(hostB);
pB_in1.CopyFrom(hostA);
pB_in2.CopyFrom(hostD);

var nbA = new MatmulOptions { Blocking = false };
var nbB = new MatmulOptions { Blocking = false };

using var mmA1 = new Matmul(M, N, K, options: nbA, stream: streamA);
using var mmA2 = new Matmul(M, N, K, options: nbA, stream: streamA);
using var mmB1 = new Matmul(M, N, K, options: nbB, stream: streamB);
using var mmB2 = new Matmul(M, N, K, options: nbB, stream: streamB);

await Task.WhenAll(mmA1.PlanAsync(), mmA2.PlanAsync(),
                   mmB1.PlanAsync(), mmB2.PlanAsync());

// Enqueue both chains simultaneously — neither stream waits for the other.
_ = mmA1.ExecuteAsync<float>(pA_in1, pA_in2, pA_mid, pA_mid, alpha: 1f, beta: 0f);
_ = mmA2.ExecuteAsync<float>(pA_mid, pA_in1, pA_out, pA_out, alpha: 1f, beta: 0f);

_ = mmB1.ExecuteAsync<float>(pB_in1, pB_in2, pB_mid, pB_mid, alpha: 1f, beta: 0f);
_ = mmB2.ExecuteAsync<float>(pB_mid, pB_in1, pB_out, pB_out, alpha: 1f, beta: 0f);

// Await both pipelines concurrently from the CPU.
await Task.WhenAll(streamA.SynchronizeAsync(), streamB.SynchronizeAsync());

var rA = pA_out.ToArray();
var rB = pB_out.ToArray();
Console.WriteLine($"  Pipeline A out[0] = {rA[0]:F1}");
Console.WriteLine($"  Pipeline B out[0] = {rB[0]:F1}");
Console.WriteLine();

// ═════════════════════════════════════════════════════════════════════════════
// Pattern 3 — Event-based cross-stream dependency
//
// streamProducer produces data (GEMM). streamConsumer must not start until
// that data is ready. CudaEvent enforces the ordering on the GPU without
// the CPU blocking between the two stream submissions.
//
//   streamProducer: ──[GEMM: result=A*B]──record(evt)──────────────────────●
//                                                  ↓ (GPU-side signal)
//   streamConsumer:                        waitEvent(evt)──[GEMM: out=result*D]──● sync
//
// The CPU submits work to both streams before either has executed a single
// instruction on the GPU. Only one SynchronizeAsync call is needed (on the
// consumer, which is guaranteed to finish after the producer due to the event).
//
// ═════════════════════════════════════════════════════════════════════════════
Console.WriteLine("Pattern 3 — Event dependency: producer stream feeds consumer stream");

await using var streamProducer = new CudaStream();
await using var streamConsumer = new CudaStream();
using var completionEvent = new CudaEvent();   // timing disabled by default → low overhead

using var prod_A      = new DeviceBuffer<float>(M * K);
using var prod_B      = new DeviceBuffer<float>(K * N);
using var prod_result = new DeviceBuffer<float>(M * N);   // producer output / consumer input
using var cons_D      = new DeviceBuffer<float>(K * N);
using var cons_out    = new DeviceBuffer<float>(M * N);

prod_A.CopyFrom(hostA);
prod_B.CopyFrom(hostB);
cons_D.CopyFrom(hostD);

var nbProd = new MatmulOptions { Blocking = false };
var nbCons = new MatmulOptions { Blocking = false };

using var mmProd = new Matmul(M, N, K, options: nbProd, stream: streamProducer);
using var mmCons = new Matmul(M, N, K, options: nbCons, stream: streamConsumer);

await Task.WhenAll(mmProd.PlanAsync(), mmCons.PlanAsync());

// ── Submit producer work ───────────────────────────────────────────────────
// Enqueue GEMM on streamProducer.
_ = mmProd.ExecuteAsync<float>(prod_A, prod_B, prod_result, prod_result,
                                alpha: 1f, beta: 0f);

// Record the event *in* streamProducer's queue — it fires when the GEMM above
// (and everything before it on this stream) has finished on the GPU.
completionEvent.Record(streamProducer);

// ── Submit consumer work ───────────────────────────────────────────────────
// Tell streamConsumer to pause at this point until completionEvent fires.
// This is a GPU-side instruction; the CPU thread is NOT blocked here.
streamConsumer.WaitEvent(completionEvent);

// Enqueue GEMM on streamConsumer. The GPU won't execute this until the
// WaitEvent condition is satisfied, ensuring prod_result is fully written.
_ = mmCons.ExecuteAsync<float>(prod_result, cons_D, cons_out, cons_out,
                                alpha: 1f, beta: 0f);

// ── Single CPU sync point ──────────────────────────────────────────────────
// Awaiting streamConsumer is sufficient: because of the event dependency,
// streamConsumer cannot finish before streamProducer.
await streamConsumer.SynchronizeAsync();

var r3 = cons_out.ToArray();
Console.WriteLine($"  Consumer output[0] = {r3[0]:F1}  (guaranteed correct: producer finished first)");
Console.WriteLine();

Console.WriteLine("All chaining patterns completed successfully.");
