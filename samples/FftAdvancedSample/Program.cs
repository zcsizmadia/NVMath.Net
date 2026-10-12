// FftAdvancedSample — N-D, batched and leading-axis FFTs, real-to-complex transforms and output normalization.
//
// Layout note: FFT shapes follow cuFFT's convention — row-major, the LAST axis is contiguous in memory
// (unlike the column-major BLAS-style modules).
// Requires a CUDA-capable GPU at runtime. The sample verifies its own results and returns a non-zero exit code on failure.

using System.Numerics;

using NVMathNet;
using NVMathNet.Fft;

Console.WriteLine("=== NVMath.Net Advanced FFT Sample ===");
int failures = 0;

void Check(string what, double error, double tolerance = 1e-9)
{
    bool ok = error < tolerance;
    failures += ok ? 0 : 1;
    Console.WriteLine($"  [{(ok ? "PASS" : "FAIL")}] {what} (max error {error:E2})");
}

static Complex[] Ramp(int count)
{
    var data = new Complex[count];
    for (int i = 0; i < count; i++)
    {
        data[i] = new Complex(Math.Sin(i * 0.7) + 0.1 * i, Math.Cos(i * 1.3));
    }

    return data;
}

// ── 1. 2-D transform with NumPy-style "backward" normalization ───────────────
// cuFFT never scales, so a forward + inverse pair multiplies the data by N. With
// FftNormalization.Backward the inverse divides by N and the round trip is exact.
{
    Console.WriteLine("\n--- 2-D C2C round trip (4 x 8, Backward normalization) ---");
    long[] shape = [4, 8];
    var host = Ramp(32);
    var options = new FftOptions { FftType = FftType.C2C, Blocking = true, Normalization = FftNormalization.Backward };

    using var input = new DeviceBuffer<Complex>(32);
    using var spectrum = new DeviceBuffer<Complex>(32);
    using var back = new DeviceBuffer<Complex>(32);
    input.CopyFrom(host);

    using (var forward = new Fft(shape, options: options, doublePrecision: true))
    {
        forward.ResetOperand(input.PointerAsInt, spectrum.PointerAsInt);
        await forward.PlanAsync();
        await forward.ExecuteAsync(FftDirection.Forward);
    }

    using (var inverse = new Fft(shape, options: options, doublePrecision: true))
    {
        inverse.ResetOperand(spectrum.PointerAsInt, back.PointerAsInt);
        await inverse.PlanAsync();
        await inverse.ExecuteAsync(FftDirection.Inverse);
    }

    var result = back.ToArray();
    Check("inverse(forward(x)) == x", host.Zip(result, (a, b) => (a - b).Magnitude).Max());
}

// ── 2. Batched real-to-complex / complex-to-real along the last axis ─────────
{
    Console.WriteLine("\n--- Batched R2C -> C2R (3 signals of 16 samples) ---");
    const int Rows = 3, N = 16;
    double[] signal = new double[Rows * N];
    for (int i = 0; i < signal.Length; i++)
    {
        signal[i] = Math.Sin(i * 0.37) + 0.05 * i;
    }

    using var real = new DeviceBuffer<double>(signal.Length);
    using var half = new DeviceBuffer<Complex>(Rows * (N / 2 + 1));   // half spectrum: N/2+1 bins per signal
    using var recovered = new DeviceBuffer<double>(signal.Length);
    real.CopyFrom(signal);

    var fwdOptions = new FftOptions { FftType = FftType.R2C, Blocking = true, Normalization = FftNormalization.Backward };
    var invOptions = new FftOptions { FftType = FftType.C2R, Blocking = true, Normalization = FftNormalization.Backward };

    // shape = [batch, samples]; transform axis 1 only
    using (var fwd = new Fft([Rows, N], [1], fwdOptions, doublePrecision: true))
    {
        fwd.ResetOperand(real.PointerAsInt, half.PointerAsInt);
        await fwd.PlanAsync();
        await fwd.ExecuteAsync(FftDirection.Forward);
    }

    using (var inv = new Fft([Rows, N], [1], invOptions, doublePrecision: true))
    {
        inv.ResetOperand(half.PointerAsInt, recovered.PointerAsInt);
        await inv.PlanAsync();
        await inv.ExecuteAsync(FftDirection.Inverse);
    }

    var back = recovered.ToArray();
    Check("batched round trip", signal.Zip(back, (a, b) => Math.Abs(a - b)).Max());
}

// ── 3. Transform along a LEADING axis (strided layout) ───────────────────────
{
    Console.WriteLine("\n--- Leading-axis C2C (shape 8 x 3, transform axis 0) ---");
    const int N = 8, Cols = 3;
    var host = Ramp(N * Cols);
    using var input = new DeviceBuffer<Complex>(host.Length);
    using var output = new DeviceBuffer<Complex>(host.Length);
    input.CopyFrom(host);

    var options = new FftOptions { FftType = FftType.C2C, Blocking = true };
    using var fft = new Fft([N, Cols], [0], options, doublePrecision: true);
    fft.ResetOperand(input.PointerAsInt, output.PointerAsInt);
    await fft.PlanAsync();
    await fft.ExecuteAsync(FftDirection.Forward);

    var result = output.ToArray();
    double worst = 0;
    for (int col = 0; col < Cols; col++)
    {
        for (int k = 0; k < N; k++)
        {
            Complex expected = 0;
            for (int j = 0; j < N; j++)
            {
                expected += host[j * Cols + col] * Complex.FromPolarCoordinates(1, -2 * Math.PI * j * k / N);
            }

            worst = Math.Max(worst, (result[k * Cols + col] - expected).Magnitude);
        }
    }

    Check("each column matches a direct DFT", worst);
}

// ── 4. Orthonormal normalization preserves energy (Parseval) ─────────────────
{
    Console.WriteLine("\n--- Orthonormal normalization (energy preservation) ---");
    var host = Ramp(256);
    using var input = new DeviceBuffer<Complex>(host.Length);
    using var output = new DeviceBuffer<Complex>(host.Length);
    input.CopyFrom(host);

    var options = new FftOptions { FftType = FftType.C2C, Blocking = true, Normalization = FftNormalization.Orthonormal };
    using var fft = new Fft([256], [0], options, doublePrecision: true);
    fft.ResetOperand(input.PointerAsInt, output.PointerAsInt);
    await fft.PlanAsync();
    await fft.ExecuteAsync(FftDirection.Forward);

    double before = host.Sum(c => c.Real * c.Real + c.Imaginary * c.Imaginary);
    double after = output.ToArray().Sum(c => c.Real * c.Real + c.Imaginary * c.Imaginary);
    Console.WriteLine($"  energy before = {before:F6}, after = {after:F6}");
    Check("energy preserved", Math.Abs(after - before) / before);
}

Console.WriteLine(failures == 0 ? "\nAll checks passed." : $"\n{failures} check(s) FAILED.");
return failures == 0 ? 0 : 1;
