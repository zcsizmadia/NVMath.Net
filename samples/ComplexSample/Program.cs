// ComplexSample — complex (double-precision, System.Numerics.Complex) linear algebra: dense solvers, a
// Hermitian eigenproblem, complex sparse products and, when cuDSS is installed, a Hermitian sparse direct solve.
// Matrices are column-major. Requires a CUDA-capable GPU at runtime; the cuDSS part is skipped if cuDSS is missing.
// The sample verifies its own results and returns a non-zero exit code on failure.

using System.Numerics;

using NVMathNet;
using NVMathNet.Solver;
using NVMathNet.Sparse;

Console.WriteLine("=== NVMath.Net Complex Sample ===");
int failures = 0;

void Check(string what, double error, double tolerance = 1e-9)
{
    bool ok = error < tolerance;
    failures += ok ? 0 : 1;
    Console.WriteLine($"  [{(ok ? "PASS" : "FAIL")}] {what} (max error {error:E2})");
}

// A well-conditioned complex test matrix (diagonally dominant when `boost` is true).
static Complex[] TestMatrix(int n, int seed, bool boost)
{
    var a = new Complex[n * n];
    for (int c = 0; c < n; c++)
    {
        for (int r = 0; r < n; r++)
        {
            a[r + n * c] = new Complex(Math.Sin(seed + r * 1.3 + c * 0.7) + (boost && r == c ? 4 : 0), Math.Cos(seed + r * 0.9 - c * 1.1));
        }
    }

    return a;
}

static Complex[] MatVec(Complex[] a, int n, Complex[] x)
{
    var y = new Complex[n];
    for (int r = 0; r < n; r++)
    {
        for (int c = 0; c < n; c++)
        {
            y[r] += a[r + n * c] * x[c];
        }
    }

    return y;
}

// ── 1. LU solve of a complex system ──────────────────────────────────────────
{
    Console.WriteLine("\n--- Complex LU solve (5x5) ---");
    const int N = 5;
    var a = TestMatrix(N, 1, boost: true);
    var xTrue = Enumerable.Range(0, N).Select(i => new Complex(i + 1, 0.5 * i - 1)).ToArray();
    var b = MatVec(a, N, xTrue);

    using var dA = new DeviceBuffer<Complex>(N * N);
    using var dB = new DeviceBuffer<Complex>(N);
    using var dPiv = new DeviceBuffer<int>(N);
    dA.CopyFrom(a);
    dB.CopyFrom(b);

    await DenseSolver.LuFactorAsync(dA, dPiv, N, N);
    await DenseSolver.LuSolveAsync(dA, dPiv, dB, N);
    var x = dB.ToArray();
    Check("A x = b recovered", Enumerable.Range(0, N).Max(i => (x[i] - xTrue[i]).Magnitude));
}

// ── 2. Least squares and SVD ─────────────────────────────────────────────────
{
    Console.WriteLine("\n--- Complex least squares and SVD (6x3) ---");
    const int M = 6, N = 3;
    var a = TestMatrix(M, 3, boost: false)[..(M * N)];
    var xTrue = new[] { new Complex(1, -1), new Complex(0.5, 2), new Complex(-2, 0.25) };
    var b = new Complex[M];
    for (int r = 0; r < M; r++)
    {
        for (int c = 0; c < N; c++)
        {
            b[r] += a[r + M * c] * xTrue[c];
        }
    }

    using var dA = new DeviceBuffer<Complex>(M * N);
    using var dB = new DeviceBuffer<Complex>(M);
    dA.CopyFrom(a);
    dB.CopyFrom(b);
    await DenseSolver.LeastSquaresAsync(dA, dB, M, N);
    var x = dB.ToArray();
    Check("least-squares solution of a consistent system", Enumerable.Range(0, N).Max(i => (x[i] - xTrue[i]).Magnitude));

    using var dA2 = new DeviceBuffer<Complex>(M * N);
    using var dS = new DeviceBuffer<double>(N);
    using var dU = new DeviceBuffer<Complex>(M * M);
    using var dVh = new DeviceBuffer<Complex>(N * N);
    dA2.CopyFrom(a);
    await DenseSolver.SvdAsync(dA2, dS, dU, dVh, M, N);       // A = U · diag(S) · Vᴴ  (returns Vᴴ)
    var s = dS.ToArray();
    var u = dU.ToArray();
    var vh = dVh.ToArray();
    Console.WriteLine($"  singular values: [{string.Join(", ", s.Select(v => v.ToString("F4")))}]");
    double worst = 0;
    for (int r = 0; r < M; r++)
    {
        for (int c = 0; c < N; c++)
        {
            Complex sum = 0;
            for (int k = 0; k < N; k++)
            {
                sum += u[r + M * k] * s[k] * vh[k + N * c];
            }

            worst = Math.Max(worst, (sum - a[r + M * c]).Magnitude);
        }
    }

    Check("U · diag(S) · Vᴴ reconstructs A", worst);
}

// ── 3. Hermitian eigenproblem ────────────────────────────────────────────────
{
    Console.WriteLine("\n--- Hermitian eigenvalues (3x3) ---");
    Complex[,] src =
    {
        { 2, new(1, 1), 0 },
        { new(1, -1), 3, new(0, 2) },
        { 0, new(0, -2), 1 },
    };
    var h = new Complex[9];
    for (int c = 0; c < 3; c++)
    {
        for (int r = 0; r < 3; r++)
        {
            h[r + 3 * c] = src[r, c];
        }
    }

    using var dA = new DeviceBuffer<Complex>(9);
    using var dW = new DeviceBuffer<double>(3);
    dA.CopyFrom(h);
    await DenseSolver.HermitianEigAsync(dA, dW, 3);           // eigenvalues are real and ascending; dA holds the vectors
    var w = dW.ToArray();
    var v = dA.ToArray();
    Console.WriteLine($"  eigenvalues: [{string.Join(", ", w.Select(e => e.ToString("F4")))}]");

    double worst = 0;
    for (int k = 0; k < 3; k++)
    {
        var vec = new[] { v[3 * k], v[3 * k + 1], v[3 * k + 2] };
        var av = MatVec(h, 3, vec);
        for (int i = 0; i < 3; i++)
        {
            worst = Math.Max(worst, (av[i] - w[k] * vec[i]).Magnitude);   // A v = λ v
        }
    }

    Check("A v = λ v for every eigenpair", worst);
}

// ── 4. Complex sparse matrix-vector product with a complex alpha ─────────────
{
    Console.WriteLine("\n--- Complex SpMV: y = α·A·x + β·y ---");
    // A = [[1+i, 0, 2], [0, 3, -i]]
    var dRow = new DeviceBuffer<int>(3);
    var dCol = new DeviceBuffer<int>(4);
    var dVal = new DeviceBuffer<Complex>(4);
    dRow.CopyFrom([0, 2, 4]);
    dCol.CopyFrom([0, 2, 1, 2]);
    dVal.CopyFrom([new Complex(1, 1), new Complex(2, 0), new Complex(3, 0), new Complex(0, -1)]);
    using var a = new CsrMatrix<Complex>(2, 3, dRow, dCol, dVal);

    Complex[] hx = [new(1, 0), new(0, 1), new(2, -1)];
    Complex[] hy = [new(1, 1), new(-1, 2)];
    using var x = new DeviceBuffer<Complex>(3);
    using var y = new DeviceBuffer<Complex>(2);
    x.CopyFrom(hx);
    y.CopyFrom(hy);

    var alpha = new Complex(0, 2);
    var beta = new Complex(1, -1);
    await SparseLinearAlgebra.SpMVAsync(a, x, y, alpha, beta);

    var ax0 = new Complex(1, 1) * hx[0] + 2 * hx[2];
    var ax1 = 3 * hx[1] + new Complex(0, -1) * hx[2];
    var r = y.ToArray();
    Check("matches the host computation",
        Math.Max((r[0] - (alpha * ax0 + beta * hy[0])).Magnitude, (r[1] - (alpha * ax1 + beta * hy[1])).Magnitude));
}

// ── 5. Hermitian positive-definite sparse direct solve (cuDSS) ───────────────
try
{
    Console.WriteLine("\n--- cuDSS: Hermitian positive-definite tridiagonal system (50x50) ---");
    const int N = 50;
    var lower = new Complex(-1, 0.5);
    var rowPtr = new List<int> { 0 };
    var cols = new List<int>();
    var vals = new List<Complex>();
    var dense = new Complex[N * N];
    for (int r = 0; r < N; r++)
    {
        for (int c = Math.Max(0, r - 1); c <= Math.Min(N - 1, r + 1); c++)
        {
            var value = r == c ? new Complex(4, 0) : c < r ? lower : Complex.Conjugate(lower);
            dense[r + N * c] = value;
            if (c <= r)                                            // store the lower triangle only
            {
                cols.Add(c);
                vals.Add(value);
            }
        }

        rowPtr.Add(cols.Count);
    }

    var dRow = new DeviceBuffer<int>(rowPtr.Count);
    var dCol = new DeviceBuffer<int>(cols.Count);
    var dVal = new DeviceBuffer<Complex>(vals.Count);
    dRow.CopyFrom(rowPtr.ToArray());
    dCol.CopyFrom(cols.ToArray());
    dVal.CopyFrom(vals.ToArray());
    using var matrix = new CsrMatrix<Complex>(N, N, dRow, dCol, dVal);

    var xTrue = Enumerable.Range(0, N).Select(i => new Complex(1 + i * 0.1, -i * 0.05)).ToArray();
    using var dB = new DeviceBuffer<Complex>(N);
    using var dX = new DeviceBuffer<Complex>(N);
    dB.CopyFrom(MatVec(dense, N, xTrue));

    var options = new DirectSolverOptions
    {
        Structure = SparseMatrixStructure.HermitianPositiveDefinite,
        View = SparseMatrixView.Lower,
    };
    using var solver = new SparseDirectSolver<Complex>(matrix, options);
    solver.Solve(dB, dX);
    var x = dX.ToArray();
    Check("A x = b recovered", Enumerable.Range(0, N).Max(i => (x[i] - xTrue[i]).Magnitude));
}
catch (DllNotFoundException ex)
{
    Console.WriteLine($"  cuDSS is not installed or not on the library path ({ex.Message.Split('\n')[0]}). Skipping.");
}

Console.WriteLine(failures == 0 ? "\nAll checks passed." : $"\n{failures} check(s) FAILED.");
return failures == 0 ? 0 : 1;
