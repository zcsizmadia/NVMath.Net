// RandSample — demonstrates cuRAND GPU random number generation.
// Requires a CUDA-capable GPU at runtime.

using NVMathNet;
using NVMathNet.Interop;
using NVMathNet.Rand;

Console.WriteLine("=== NVMath.Net Rand Sample ===");

// ── 1. Uniform distribution ──────────────────────────────────────────────────
{
    Console.WriteLine("\n--- Uniform Distribution (float) ---");

    const int N = 1_000_000;
    using var rng = new CudaRandom(CuRandRngType.PseudoDefault, seed: 42);
    using var buf = new DeviceBuffer<float>(N);

    rng.FillUniform(buf);

    var data = buf.ToArray();
    float min = data.Min();
    float max = data.Max();
    float mean = data.Average();
    Console.WriteLine($"  {N:N0} samples — min: {min:F6}, max: {max:F6}, mean: {mean:F6}");
    Console.WriteLine($"  Expected: min ≈ 0, max ≈ 1, mean ≈ 0.5");
}

// ── 2. Normal distribution ───────────────────────────────────────────────────
{
    Console.WriteLine("\n--- Normal Distribution (double, μ=5.0, σ=2.0) ---");

    const int N = 1_000_000;
    using var rng = new CudaRandom(CuRandRngType.PseudoPhilox4_32_10, seed: 123);
    using var buf = new DeviceBuffer<double>(N);

    rng.FillNormal(buf, mean: 5.0, stddev: 2.0);

    var data = buf.ToArray();
    double mean = data.Average();
    double variance = data.Select(x => (x - mean) * (x - mean)).Average();
    double stddev = Math.Sqrt(variance);
    Console.WriteLine($"  {N:N0} samples — mean: {mean:F4}, stddev: {stddev:F4}");
    Console.WriteLine($"  Expected: mean ≈ 5.0, stddev ≈ 2.0");
}

// ── 3. Seed reproducibility ──────────────────────────────────────────────────
{
    Console.WriteLine("\n--- Seed Reproducibility ---");

    const int N = 1024;
    using var buf1 = new DeviceBuffer<float>(N);
    using var buf2 = new DeviceBuffer<float>(N);

    using var rng1 = new CudaRandom(CuRandRngType.PseudoPhilox4_32_10, seed: 999);
    rng1.FillUniform(buf1);

    using var rng2 = new CudaRandom(CuRandRngType.PseudoPhilox4_32_10, seed: 999);
    rng2.FillUniform(buf2);

    var a = buf1.ToArray();
    var b = buf2.ToArray();
    bool identical = a.Zip(b).All(pair => pair.First == pair.Second);
    Console.WriteLine($"  Same seed produces identical results: {identical}");
}

// ── 4. Async fill + stream ───────────────────────────────────────────────────
{
    Console.WriteLine("\n--- Async Fill with Stream ---");

    const int N = 500_000;
    await using var stream = new CudaStream();
    using var rng = new CudaRandom(CuRandRngType.PseudoDefault, seed: 7, stream: stream);
    using var buf = new DeviceBuffer<float>(N);

    await rng.FillUniformAsync(buf);

    var data = buf.ToArray();
    float mean = data.Average();
    Console.WriteLine($"  {N:N0} async samples — mean: {mean:F6} (expected ≈ 0.5)");
}

// ── 5. Log-normal distribution ───────────────────────────────────────────────
{
    Console.WriteLine("\n--- Log-Normal Distribution (float) ---");

    const int N = 100_000;
    using var rng = new CudaRandom(CuRandRngType.PseudoDefault, seed: 55);
    using var buf = new DeviceBuffer<float>(N);

    rng.FillLogNormal(buf, mean: 0f, stddev: 0.5f);

    var data = buf.ToArray();
    float min = data.Min();
    float max = data.Max();
    double mean = data.Select(x => (double)x).Average();
    Console.WriteLine($"  {N:N0} samples — min: {min:F4}, max: {max:F4}, mean: {mean:F4}");
    Console.WriteLine($"  All values > 0: {data.All(x => x > 0)}");
}

Console.WriteLine("\nAll Rand samples completed successfully.");
