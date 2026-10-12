// MatmulAdvancedSample — cuBLASLt features beyond a plain product: bias epilog, batching, autotuning with the
// plan cache, and complex matrices with conjugate transpose.
// All matrices are column-major. Requires a CUDA-capable GPU at runtime.
// The sample verifies its own results and returns a non-zero exit code on failure.

using System.Numerics;

using NVMathNet;
using NVMathNet.LinAlg;

Console.WriteLine("=== NVMath.Net Advanced Matmul Sample ===");
int failures = 0;

void Check(string what, double error, double tolerance = 1e-4)
{
    bool ok = error < tolerance;
    failures += ok ? 0 : 1;
    Console.WriteLine($"  [{(ok ? "PASS" : "FAIL")}] {what} (max error {error:E2})");
}

// ── 1. Bias epilog: D = A·B + bias (one bias value per row of D) ─────────────
{
    Console.WriteLine("\n--- Bias epilog (4x3 = 4x2 · 2x3 + bias) ---");
    const int M = 4, N = 3, K = 2;
    float[] a = [1, 2, 3, 4, 5, 6, 7, 8];              // 4x2 column-major
    float[] b = [1, 0, 2, 1, 0, 3];                    // 2x3 column-major
    float[] bias = [10, 20, 30, 40];                   // length M

    using var dA = new DeviceBuffer<float>(M * K);
    using var dB = new DeviceBuffer<float>(K * N);
    using var dC = new DeviceBuffer<float>(M * N);
    using var dD = new DeviceBuffer<float>(M * N);
    using var dBias = new DeviceBuffer<float>(M);
    dA.CopyFrom(a);
    dB.CopyFrom(b);
    dBias.CopyFrom(bias);
    dC.Clear();

    using var matmul = new Matmul(M, N, K, options: new MatmulOptions { Epilog = MatmulEpilog.Bias, Blocking = true });
    matmul.SetBias(dBias.PointerAsInt);               // must be set before planning
    await matmul.PlanAsync();
    await matmul.ExecuteAsync<float>(dA, dB, dC, dD);

    var d = dD.ToArray();
    double worst = 0;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float expected = bias[i];
            for (int k = 0; k < K; k++)
            {
                expected += a[i + M * k] * b[k + K * j];
            }

            worst = Math.Max(worst, Math.Abs(d[i + M * j] - expected));
        }
    }

    Check("D equals A·B + bias", worst);
}

// ── 2. Strided-batch matmul: three independent 2x2 products in one call ──────
{
    Console.WriteLine("\n--- Batched matmul (3 x 2x2) ---");
    const int N = 2, Batch = 3;
    float[] a = new float[N * N * Batch];
    float[] b = new float[N * N * Batch];
    for (int i = 0; i < a.Length; i++)
    {
        a[i] = i + 1;
        b[i] = (i % 3) - 1;
    }

    using var dA = new DeviceBuffer<float>(a.Length);
    using var dB = new DeviceBuffer<float>(b.Length);
    using var dC = new DeviceBuffer<float>(a.Length);
    using var dD = new DeviceBuffer<float>(a.Length);
    dA.CopyFrom(a);
    dB.CopyFrom(b);
    dC.Clear();

    // Matrices of a batch are packed back-to-back.
    using var matmul = new Matmul(N, N, N, batchCount: Batch, options: new MatmulOptions { Blocking = true });
    await matmul.PlanAsync();
    await matmul.ExecuteAsync<float>(dA, dB, dC, dD);

    var d = dD.ToArray();
    double worst = 0;
    for (int m = 0; m < Batch; m++)
    {
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                float expected = 0;
                for (int k = 0; k < N; k++)
                {
                    expected += a[m * N * N + i + N * k] * b[m * N * N + k + N * j];
                }

                worst = Math.Max(worst, Math.Abs(d[m * N * N + i + N * j] - expected));
            }
        }
    }

    Check("every matrix of the batch is correct", worst);
}

// ── 3. Autotuning and the plan cache ─────────────────────────────────────────
{
    Console.WriteLine("\n--- Autotune and plan cache (256 x 256 x 256) ---");
    const int N = 256;
    using var a = new DeviceBuffer<float>(N * N);
    using var b = new DeviceBuffer<float>(N * N);
    using var c = new DeviceBuffer<float>(N * N);
    using var d = new DeviceBuffer<float>(N * N);
    a.Clear();
    b.Clear();
    c.Clear();

    var options = new MatmulOptions { Blocking = true };
    var preferences = new MatmulPlanPreferences { RequestedAlgoCount = 6 };   // ask for several candidates

    int tuned;
    using (var first = new Matmul(N, N, N, options: options))
    {
        await first.PlanAsync(preferences);
        float[] times = first.Autotune(a.PointerAsInt, b.PointerAsInt, c.PointerAsInt, d.PointerAsInt);
        tuned = first.SelectedAlgorithm;
        Console.WriteLine($"  {first.AlgorithmCount} candidate(s), timings (ms): [{string.Join(", ", times.Select(t => t.ToString("F3")))}]");
        Console.WriteLine($"  selected algorithm #{tuned}");
    }

    long hitsBefore = MatmulPlanCache.Hits;
    using var second = new Matmul(N, N, N, options: options);
    await second.PlanAsync(preferences);                 // identical configuration: served from the plan cache
    Console.WriteLine($"  second plan: cache hits {hitsBefore} -> {MatmulPlanCache.Hits}, selected algorithm #{second.SelectedAlgorithm}");
    Check("cache hit reuses the tuned algorithm",
        MatmulPlanCache.Hits > hitsBefore && second.SelectedAlgorithm == tuned ? 0 : 1, 0.5);
}

// ── 4. Complex matmul with conjugate transpose: D = alpha · Aᴴ·B ─────────────
{
    Console.WriteLine("\n--- Complex matmul with conjugate transpose (3x3) ---");
    const int N = 3;
    var a = new Complex[N * N];
    var b = new Complex[N * N];
    for (int i = 0; i < N * N; i++)
    {
        a[i] = new Complex(i + 1, 0.5 * i - 2);
        b[i] = new Complex(Math.Sin(i), Math.Cos(i * 1.5));
    }

    var options = new MatmulOptions
    {
        ComputeType = MatmulComputeType.Compute64F,
        TypeA = MatrixDataType.Complex128,
        TypeB = MatrixDataType.Complex128,
        TypeC = MatrixDataType.Complex128,
        TypeD = MatrixDataType.Complex128,
        ScaleType = MatrixDataType.Complex128,        // complex alpha / beta
        ConjugateTransposeA = true,                   // use Aᴴ
        Blocking = true,
    };

    using var dA = new DeviceBuffer<Complex>(N * N);
    using var dB = new DeviceBuffer<Complex>(N * N);
    using var dC = new DeviceBuffer<Complex>(N * N);
    using var dD = new DeviceBuffer<Complex>(N * N);
    dA.CopyFrom(a);
    dB.CopyFrom(b);
    dC.Clear();

    var alpha = new Complex(0, 1);
    using var matmul = new Matmul(N, N, N, options: options);
    await matmul.PlanAsync();
    await matmul.ExecuteAsync(dA, dB, dC, dD, alpha, Complex.Zero);

    var d = dD.ToArray();
    double worst = 0;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            Complex sum = 0;
            for (int k = 0; k < N; k++)
            {
                sum += Complex.Conjugate(a[k + N * i]) * b[k + N * j];   // (Aᴴ)[i,k] = conj(A[k,i])
            }

            worst = Math.Max(worst, (d[i + N * j] - alpha * sum).Magnitude);
        }
    }

    Check("D equals i · Aᴴ·B", worst, 1e-9);
}

Console.WriteLine(failures == 0 ? "\nAll checks passed." : $"\n{failures} check(s) FAILED.");
return failures == 0 ? 0 : 1;
