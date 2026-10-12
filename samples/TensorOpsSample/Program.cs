// TensorOpsSample — cuTENSOR operations: permutation, reduction, element-wise binary and a contraction.
//
// Layout note: tensors are packed COLUMN-major — the first mode varies fastest in memory. Modes are integer
// labels; the same label in two operands means the same dimension.
// Requires a CUDA-capable GPU and the cuTENSOR 2.x runtime. The sample verifies its own results and
// returns a non-zero exit code on failure (or 0 with a note when cuTENSOR is not installed).

using NVMathNet;
using NVMathNet.Interop;
using NVMathNet.Tensor;

Console.WriteLine("=== NVMath.Net Tensor Operations Sample ===");
int failures = 0;

void Check(string what, double error, double tolerance = 1e-4)
{
    bool ok = error < tolerance;
    failures += ok ? 0 : 1;
    Console.WriteLine($"  [{(ok ? "PASS" : "FAIL")}] {what} (max error {error:E2})");
}

try
{
    var options = new TensorContractionOptions { Blocking = true };

    // ── 1. Permutation: transpose a 2 x 3 tensor ─────────────────────────────
    {
        Console.WriteLine("\n--- Permutation (2x3 -> 3x2, B[j,i] = A[i,j]) ---");
        const int I = 2, J = 3;
        float[] a = new float[I * J];
        for (int i = 0; i < I; i++)
        {
            for (int j = 0; j < J; j++)
            {
                a[i + I * j] = i * 10 + j;
            }
        }

        using var dA = new DeviceBuffer<float>(a.Length);
        using var dB = new DeviceBuffer<float>(a.Length);
        dA.CopyFrom(a);

        // Mode labels: A has modes (0, 1); B lists the same labels in the new order (1, 0).
        using var permute = new TensorPermutation([I, J], [0, 1], [1, 0], options: options);
        await permute.ExecuteAsync(dA, dB);

        var b = dB.ToArray();                              // B is 3 x 2
        double worst = 0;
        for (int i = 0; i < I; i++)
        {
            for (int j = 0; j < J; j++)
            {
                worst = Math.Max(worst, Math.Abs(b[j + J * i] - a[i + I * j]));
            }
        }

        Check("transposed correctly", worst);
    }

    // ── 2. Reduction: row sums and a maximum ─────────────────────────────────
    {
        Console.WriteLine("\n--- Reduction (3x4 tensor: row sums and row maxima) ---");
        const int I = 3, J = 4;
        float[] a = new float[I * J];
        for (int i = 0; i < I; i++)
        {
            for (int j = 0; j < J; j++)
            {
                a[i + I * j] = (i + 1) * (j + 2) - 5;
            }
        }

        using var dA = new DeviceBuffer<float>(a.Length);
        using var dSum = new DeviceBuffer<float>(I);
        using var dMax = new DeviceBuffer<float>(I);
        dA.CopyFrom(a);
        dSum.Clear();
        dMax.Clear();

        // Output keeps mode 0; the modes missing from the output (mode 1) are reduced.
        using var sum = new TensorReduction([I, J], [0, 1], [I], [0], options: options);
        await sum.ExecuteAsync(dA, dSum);
        using var max = new TensorReduction([I, J], [0, 1], [I], [0], opReduce: CuTensorOperator.Max, options: options);
        await max.ExecuteAsync(dA, dMax);

        var sums = dSum.ToArray();
        var maxima = dMax.ToArray();
        double worst = 0;
        for (int i = 0; i < I; i++)
        {
            float expectedSum = 0, expectedMax = float.MinValue;
            for (int j = 0; j < J; j++)
            {
                expectedSum += a[i + I * j];
                expectedMax = Math.Max(expectedMax, a[i + I * j]);
            }

            worst = Math.Max(worst, Math.Max(Math.Abs(sums[i] - expectedSum), Math.Abs(maxima[i] - expectedMax)));
        }

        Console.WriteLine($"  row sums = [{string.Join(", ", sums)}], row maxima = [{string.Join(", ", maxima)}]");
        Check("sums and maxima", worst);
    }

    // ── 3. Element-wise binary: D = A^T + 2·C (only A may be permuted) ───────
    {
        Console.WriteLine("\n--- Element-wise binary (D[i,j] = A[j,i] + 2·C[i,j]) ---");
        const int I = 2, J = 3;
        float[] a = [.. Enumerable.Range(0, I * J).Select(v => (float)v)];          // A is J x I
        float[] c = [.. Enumerable.Range(0, I * J).Select(v => 100f + v)];          // C and D are I x J
        using var dA = new DeviceBuffer<float>(a.Length);
        using var dC = new DeviceBuffer<float>(c.Length);
        using var dD = new DeviceBuffer<float>(c.Length);
        dA.CopyFrom(a);
        dC.CopyFrom(c);

        using var add = new ElementwiseBinary(
            [J, I], [1, 0],            // A with modes (1, 0): transposed relative to D
            [I, J], [0, 1],            // C
            [I, J], [0, 1],            // D
            options: options);
        await add.ExecuteAsync(dA, dC, dD, alpha: 1.0, gamma: 2.0);

        var d = dD.ToArray();
        double worst = 0;
        for (int i = 0; i < I; i++)
        {
            for (int j = 0; j < J; j++)
            {
                worst = Math.Max(worst, Math.Abs(d[i + I * j] - (a[j + J * i] + 2 * c[i + I * j])));
            }
        }

        Check("D = Aᵀ + 2C", worst);
    }

    // ── 4. Contraction: a matrix product expressed with modes ────────────────
    {
        Console.WriteLine("\n--- Contraction (C[m,n] = Σk A[m,k]·B[k,n], modes m=0, n=1, k=2) ---");
        const int M = 3, N = 2, K = 4;
        float[] a = [.. Enumerable.Range(0, M * K).Select(v => (v % 5) * 0.5f)];
        float[] b = [.. Enumerable.Range(0, K * N).Select(v => (v % 3) - 1f)];
        using var dA = new DeviceBuffer<float>(a.Length);
        using var dB = new DeviceBuffer<float>(b.Length);
        using var dC = new DeviceBuffer<float>(M * N);
        using var dD = new DeviceBuffer<float>(M * N);
        dA.CopyFrom(a);
        dB.CopyFrom(b);
        dC.Clear();

        using var contraction = new TensorContraction(
            [M, K], [0, 2],            // A[m, k]
            [K, N], [2, 1],            // B[k, n]
            [M, N], [0, 1],            // C[m, n]
            [M, N], [0, 1],            // D[m, n]
            TensorDataType.Float32, options);
        await contraction.PlanAsync();
        await contraction.ExecuteAsync(dA.PointerAsInt, dB.PointerAsInt, dC.PointerAsInt, dD.PointerAsInt, alpha: 1.0, beta: 0.0);

        var d = dD.ToArray();
        double worst = 0;
        for (int m = 0; m < M; m++)
        {
            for (int n = 0; n < N; n++)
            {
                float expected = 0;
                for (int k = 0; k < K; k++)
                {
                    expected += a[m + M * k] * b[k + K * n];
                }

                worst = Math.Max(worst, Math.Abs(d[m + M * n] - expected));
            }
        }

        Check("matches a host matrix product", worst);
    }
}
catch (DllNotFoundException ex)
{
    Console.WriteLine($"\ncuTENSOR is not installed or not on the library path ({ex.Message.Split('\n')[0]}). Skipping.");
    return 0;
}

Console.WriteLine(failures == 0 ? "\nAll checks passed." : $"\n{failures} check(s) FAILED.");
return failures == 0 ? 0 : 1;
