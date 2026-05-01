// SolverSample — demonstrates cuSOLVER dense linear algebra operations.
// All matrices use column-major layout.
// Requires a CUDA-capable GPU at runtime.

using NVMathNet;
using NVMathNet.Solver;

Console.WriteLine("=== NVMath.Net Solver Sample ===");

// ── 1. LU Solve: A * x = b ───────────────────────────────────────────────────
{
    Console.WriteLine("\n--- LU Solve (3×3, double) ---");

    // A = [[2, 1, 1],
    //      [4, 3, 3],
    //      [8, 7, 9]]
    // Column-major storage:
    double[] hostA =
    [
        2, 4, 8,   // col 0
        1, 3, 7,   // col 1
        1, 3, 9,   // col 2
    ];

    // b = [4, 10, 24]  →  solution x = [1, 1, 1]
    double[] hostB = [4, 10, 24];

    using var dA = new DeviceBuffer<double>(9);
    using var dB = new DeviceBuffer<double>(3);
    dA.CopyFrom(hostA);
    dB.CopyFrom(hostB);

    await DenseSolver.SolveAsync(dA, dB, n: 3, nrhs: 1);

    var x = dB.ToArray();
    Console.WriteLine($"  x = [{x[0]:F4}, {x[1]:F4}, {x[2]:F4}]");
    Console.WriteLine($"  Expected: [1.0000, 1.0000, 1.0000]");
}

// ── 2. Cholesky Factorisation ─────────────────────────────────────────────────
{
    Console.WriteLine("\n--- Cholesky (3×3 SPD, float) ---");

    // A = [[4, 2, 0],
    //      [2, 5, 1],
    //      [0, 1, 6]]   (symmetric positive-definite)
    float[] hostA =
    [
        4, 2, 0,   // col 0
        2, 5, 1,   // col 1
        0, 1, 6,   // col 2
    ];

    using var dA = new DeviceBuffer<float>(9);
    dA.CopyFrom(hostA);

    await DenseSolver.CholeskyAsync(dA, n: 3, upper: false);

    var L = dA.ToArray();
    Console.WriteLine("  L (lower Cholesky factor):");
    for (int row = 0; row < 3; row++)
    {
        Console.Write("    [");
        for (int col = 0; col < 3; col++)
        {
            float val = col <= row ? L[row + 3 * col] : 0f;
            Console.Write($" {val,8:F4}");
        }
        Console.WriteLine(" ]");
    }
}

// ── 3. SVD: A = U * diag(S) * VT ─────────────────────────────────────────────
{
    Console.WriteLine("\n--- SVD (3×2, double) ---");

    // A = [[1, 0],
    //      [0, 1],
    //      [1, 1]]
    double[] hostA =
    [
        1, 0, 1,   // col 0
        0, 1, 1,   // col 1
    ];

    const int M = 3, N = 2;
    int minMN = Math.Min(M, N);

    using var dA  = new DeviceBuffer<double>(M * N);
    using var dS  = new DeviceBuffer<double>(minMN);
    using var dU  = new DeviceBuffer<double>(M * M);
    using var dVT = new DeviceBuffer<double>(N * N);
    dA.CopyFrom(hostA);

    await DenseSolver.SvdAsync(dA, dS, dU, dVT, M, N);

    var s = dS.ToArray();
    Console.WriteLine($"  Singular values: [{s[0]:F4}, {s[1]:F4}]");
    Console.WriteLine($"  Expected: ~[1.7321, 1.0000]");
}

// ── 4. Eigenvalue Decomposition ───────────────────────────────────────────────
{
    Console.WriteLine("\n--- Eigenvalues (2×2 symmetric, float) ---");

    // A = [[2, 1],
    //      [1, 2]]
    // Eigenvalues: 1 and 3
    float[] hostA = [2, 1, 1, 2]; // col-major

    using var dA = new DeviceBuffer<float>(4);
    using var dW = new DeviceBuffer<float>(2);
    dA.CopyFrom(hostA);

    await DenseSolver.EigAsync(dA, dW, n: 2, computeVectors: true);

    var w = dW.ToArray();
    Console.WriteLine($"  Eigenvalues: [{w[0]:F4}, {w[1]:F4}]");
    Console.WriteLine($"  Expected: [1.0000, 3.0000]");

    var vecs = dA.ToArray();
    Console.WriteLine("  Eigenvectors (columns):");
    for (int row = 0; row < 2; row++)
    {
        Console.Write("    [");
        for (int col = 0; col < 2; col++)
        {
            Console.Write($" {vecs[row + 2 * col],8:F4}");
        }
        Console.WriteLine(" ]");
    }
}

Console.WriteLine("\nAll Solver samples completed successfully.");
