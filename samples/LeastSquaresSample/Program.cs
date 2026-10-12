// LeastSquaresSample — QR factorisation, polynomial least-squares fitting and batched dense solvers.
// All matrices are column-major. Requires a CUDA-capable GPU at runtime.
// The sample verifies its own results and returns a non-zero exit code on failure.

using NVMathNet;
using NVMathNet.Solver;

Console.WriteLine("=== NVMath.Net Least Squares & Batched Solvers Sample ===");
int failures = 0;

void Check(string what, double error, double tolerance = 1e-8)
{
    bool ok = error < tolerance;
    failures += ok ? 0 : 1;
    Console.WriteLine($"  [{(ok ? "PASS" : "FAIL")}] {what} (max error {error:E2})");
}

// ── 1. Polynomial fit by least squares ───────────────────────────────────────
// Fit y = c0 + c1·x + c2·x² to 10 points sampled from y = 1 + 2x + 0.5x² with a small
// deterministic perturbation; the solver minimises ‖A·c − y‖₂ using QR.
{
    Console.WriteLine("\n--- Polynomial least squares (10 points, degree 2) ---");
    const int M = 10, N = 3;
    double[] x = [.. Enumerable.Range(0, M).Select(i => (double)i)];
    double[] y = [.. x.Select(v => 1 + 2 * v + 0.5 * v * v + 0.01 * Math.Sin(7 * v))];

    // Vandermonde matrix, column-major: columns are 1, x, x².
    double[] a = new double[M * N];
    for (int i = 0; i < M; i++)
    {
        a[i] = 1;
        a[i + M] = x[i];
        a[i + 2 * M] = x[i] * x[i];
    }

    using var dA = new DeviceBuffer<double>(M * N);
    using var dB = new DeviceBuffer<double>(M);
    dA.CopyFrom(a);
    dB.CopyFrom(y);

    await DenseSolver.LeastSquaresAsync(dA, dB, M, N);
    var coefficients = dB.ToArray()[..N];                 // first n entries hold the solution
    Console.WriteLine($"  fitted: c0={coefficients[0]:F4}, c1={coefficients[1]:F4}, c2={coefficients[2]:F4}  (true: 1, 2, 0.5)");

    Check("coefficients close to the generating polynomial",
        Math.Max(Math.Abs(coefficients[0] - 1), Math.Max(Math.Abs(coefficients[1] - 2), Math.Abs(coefficients[2] - 0.5))), 0.05);
}

// ── 2. QR factorisation and the explicit Q ───────────────────────────────────
{
    Console.WriteLine("\n--- QR factorisation (4 x 3) ---");
    const int M = 4, N = 3;
    double[] a = [1, 2, 3, 4, 2, 1, 0, 1, 0, 3, 1, 2];
    using var dA = new DeviceBuffer<double>(M * N);
    using var dTau = new DeviceBuffer<double>(N);
    dA.CopyFrom(a);

    await DenseSolver.QrAsync(dA, dTau, M, N);            // R in the upper triangle, reflectors below
    var factored = dA.ToArray();
    await DenseSolver.QrFormQAsync(dA, dTau, M, N);       // overwrite with the explicit orthonormal Q
    var q = dA.ToArray();

    double orthonormality = 0, reconstruction = 0;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            double dot = 0;
            for (int r = 0; r < M; r++)
            {
                dot += q[r + M * i] * q[r + M * j];
            }

            orthonormality = Math.Max(orthonormality, Math.Abs(dot - (i == j ? 1 : 0)));
        }
    }

    for (int r = 0; r < M; r++)
    {
        for (int c = 0; c < N; c++)
        {
            double sum = 0;
            for (int k = 0; k <= c; k++)
            {
                sum += q[r + M * k] * factored[k + M * c];
            }

            reconstruction = Math.Max(reconstruction, Math.Abs(sum - a[r + M * c]));
        }
    }

    Check("QᵀQ = I", orthonormality);
    Check("Q·R = A", reconstruction);
}

// ── 3. LU: factor once, solve several right-hand sides ───────────────────────
{
    Console.WriteLine("\n--- LU factor once, solve twice (3x3) ---");
    double[] a = [2, 4, 8, 1, 3, 7, 1, 3, 9];             // [[2,1,1],[4,3,3],[8,7,9]] column-major
    using var dA = new DeviceBuffer<double>(9);
    using var dPiv = new DeviceBuffer<int>(3);
    dA.CopyFrom(a);

    int info = await DenseSolver.LuFactorAsync(dA, dPiv, 3, 3);
    Console.WriteLine($"  factorisation status: {info} (0 = success, >0 = singular)");

    double worst = 0;
    foreach (double[] expected in new[] { new double[] { 1, 1, 1 }, new double[] { 1, -2, 3 } })
    {
        // b = A·expected
        double[] b = new double[3];
        for (int r = 0; r < 3; r++)
        {
            for (int c = 0; c < 3; c++)
            {
                b[r] += a[r + 3 * c] * expected[c];
            }
        }

        using var dB = new DeviceBuffer<double>(3);
        dB.CopyFrom(b);
        await DenseSolver.LuSolveAsync(dA, dPiv, dB, 3);
        var x = dB.ToArray();
        worst = Math.Max(worst, Enumerable.Range(0, 3).Max(i => Math.Abs(x[i] - expected[i])));
    }

    Check("both solves recover the solution", worst);
}

// ── 4. Batched solvers: many small independent problems in one call ──────────
{
    Console.WriteLine("\n--- Batched LU and Jacobi SVD (4 systems of 3x3) ---");
    const int N = 3, Batch = 4;
    double[] a = new double[N * N * Batch];
    double[] b = new double[N * Batch];
    for (int m = 0; m < Batch; m++)
    {
        for (int c = 0; c < N; c++)
        {
            for (int r = 0; r < N; r++)
            {
                a[m * N * N + r + N * c] = r == c ? 5 + m : 0.3 * ((r + 2 * c + m) % 5);
            }
        }

        for (int r = 0; r < N; r++)
        {
            b[m * N + r] = (m + 1) * (r + 1);
        }
    }

    using var dA = new DeviceBuffer<double>(a.Length);
    using var dB = new DeviceBuffer<double>(b.Length);
    using var dPiv = new DeviceBuffer<int>(N * Batch);
    dA.CopyFrom(a);
    dB.CopyFrom(b);

    int[] statuses = await DenseSolver.LuFactorBatchedAsync(dA, dPiv, N, Batch);
    await DenseSolver.LuSolveBatchedAsync(dA, dPiv, dB, N, nrhs: 1, Batch);
    var x = dB.ToArray();

    double worst = 0;
    for (int m = 0; m < Batch; m++)
    {
        for (int r = 0; r < N; r++)
        {
            double sum = 0;
            for (int c = 0; c < N; c++)
            {
                sum += a[m * N * N + r + N * c] * x[m * N + c];
            }

            worst = Math.Max(worst, Math.Abs(sum - b[m * N + r]));
        }
    }

    Check($"batched LU solves all {Batch} systems (statuses: {string.Join(",", statuses)})", worst);

    // Batched SVD of small matrices: the singular values of diag(3,4) and diag(1,2) are known.
    double[] diag = [3, 0, 0, 4, 1, 0, 0, 2];            // two 2x2 matrices, column-major
    using var dM = new DeviceBuffer<double>(diag.Length);
    using var dS = new DeviceBuffer<double>(4);
    dM.CopyFrom(diag);
    await DenseSolver.SvdBatchedAsync<double>(dM, dS, null, null, m: 2, n: 2, batchCount: 2, computeVectors: false);
    var s = dS.ToArray();
    Console.WriteLine($"  singular values: [{string.Join(", ", s.Select(v => v.ToString("F4")))}]  (expected 4, 3, 2, 1)");
    Check("batched SVD singular values",
        Math.Max(Math.Max(Math.Abs(s[0] - 4), Math.Abs(s[1] - 3)), Math.Max(Math.Abs(s[2] - 2), Math.Abs(s[3] - 1))));
}

Console.WriteLine(failures == 0 ? "\nAll checks passed." : $"\n{failures} check(s) FAILED.");
return failures == 0 ? 0 : 1;
