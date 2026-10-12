// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the MIT License.

using System.Numerics;
using NVMathNet.Fft;
using FftPlan = NVMathNet.Fft.Fft;

namespace NVMathNet.Tests;

/// <summary>
/// Tests for the FFT module. Requires a CUDA-capable GPU.
/// </summary>
public sealed class FftTests
{
    [Test]
    public async Task Fft_C2C_ForwardAndInverse_RoundTrip()
    {
        const int N = 512;

        // Populate input: a simple real signal embedded in complex
        var host = new Complex[N];
        for (int i = 0; i < N; i++)
        {
            host[i] = new Complex(Math.Sin(2 * Math.PI * 4 * i / N), 0);
        }

        using var stream = new CudaStream();
        using var inputBuf = new DeviceBuffer<Complex>(N);
        inputBuf.CopyFrom(host.AsSpan());

        // Forward FFT
        long[] shape = [N];
        var opts = new FftOptions { FftType = FftType.C2C, Blocking = true };
        using var fwdFft = new FftPlan(shape, options: opts, stream: stream, doublePrecision: true);
        using var outputBuf = new DeviceBuffer<Complex>(N);
        fwdFft.ResetOperand(inputBuf.PointerAsInt, outputBuf.PointerAsInt);
        await fwdFft.PlanAsync();
        await fwdFft.ExecuteAsync(FftDirection.Forward);

        // Inverse FFT
        using var invFft = new FftPlan(shape, options: opts, stream: stream, doublePrecision: true);
        using var recovBuf = new DeviceBuffer<Complex>(N);
        invFft.ResetOperand(outputBuf.PointerAsInt, recovBuf.PointerAsInt);
        await invFft.PlanAsync();
        await invFft.ExecuteAsync(FftDirection.Inverse);

        // Normalize
        var result = recovBuf.ToArray();
        for (int i = 0; i < N; i++)
        {
            double re = result[i].Real / N;
            double err = Math.Abs(re - host[i].Real);
            await Assert.That(err).IsLessThan(1e-4);
        }
    }

    [Test]
    public async Task Fft_R2C_ProducesCorrectOutputLength()
    {
        const int N = 128;
        double[] real = new double[N];
        for (int i = 0; i < N; i++)
        {
            real[i] = Math.Cos(2 * Math.PI * 8 * i / N);
        }

        using var stream = new CudaStream();
        using var inputBuf = new DeviceBuffer<double>(N);
        inputBuf.CopyFrom(real.AsSpan());

        long[] shape = [N];
        var opts = new FftOptions { FftType = FftType.R2C, Blocking = true };
        using var fft = new FftPlan(shape, options: opts, stream: stream, doublePrecision: true);
        using var outputBuf = new DeviceBuffer<Complex>(N / 2 + 1);
        fft.ResetOperand(inputBuf.PointerAsInt, outputBuf.PointerAsInt);
        await fft.PlanAsync();
        await fft.ExecuteAsync(FftDirection.Forward);

        var result = outputBuf.ToArray();
        // The dominant frequency component should be at index 8
        double maxMag = 0;
        int maxIdx = 0;
        for (int i = 0; i < result.Length; i++)
        {
            double mag = result[i].Magnitude;
            if (mag > maxMag) { maxMag = mag; maxIdx = i; }
        }

        await Assert.That(maxIdx).IsEqualTo(8);
    }

    [Test]
    public async Task Fft_Static_FftAsync()
    {
        const int N = 64;
        var host = new Complex[N];
        for (int i = 0; i < N; i++)
        {
            host[i] = new Complex(i % 4 == 0 ? 1.0 : 0.0, 0);
        }

        using var inputBuf = new DeviceBuffer<Complex>(N);
        inputBuf.CopyFrom(host.AsSpan());

        using var output = await FftPlan.FftAsync(inputBuf, [N]);
        var result = output.ToArray();

        // Sum of all magnitudes^2 should equal N * sum of input^2 (Parseval)
        double inputPower = host.Sum(c => c.Real * c.Real);
        double outputPower = result.Sum(c => c.Magnitude * c.Magnitude) / N;
        double err = Math.Abs(inputPower - outputPower);
        await Assert.That(err).IsLessThan(1e-3);
    }

    // Host reference DFT of `n` points read from `src` starting at `offset` with `stride`.
    private static Complex[] HostDft(Complex[] src, int offset, int stride, int n)
    {
        var result = new Complex[n];
        for (int k = 0; k < n; k++)
        {
            Complex sum = Complex.Zero;
            for (int j = 0; j < n; j++)
            {
                sum += src[offset + j * stride] * Complex.FromPolarCoordinates(1, -2 * Math.PI * j * k / n);
            }

            result[k] = sum;
        }

        return result;
    }

    private static Complex[] RampSignal(int count)
    {
        var host = new Complex[count];
        for (int i = 0; i < count; i++)
        {
            host[i] = new Complex(Math.Sin(i * 0.7) + 0.1 * i, Math.Cos(i * 1.3));
        }

        return host;
    }

    private static async Task<Complex[]> RunC2C(long[] shape, int[] axes, Complex[] host)
    {
        using var input = new DeviceBuffer<Complex>(host.Length);
        using var output = new DeviceBuffer<Complex>(host.Length);
        input.CopyFrom(host.AsSpan());
        using var fft = new FftPlan(shape, axes, new FftOptions { FftType = FftType.C2C, Blocking = true }, doublePrecision: true);
        fft.ResetOperand(input.PointerAsInt, output.PointerAsInt);
        await fft.PlanAsync();
        await fft.ExecuteAsync(FftDirection.Forward);
        return output.ToArray();
    }

    [Test]
    public async Task Fft_C2C_Batched_TrailingAxis()
    {
        // 3 rows of 8 points; transform along the last axis.
        var host = RampSignal(24);
        var result = await RunC2C([3, 8], [1], host);

        for (int row = 0; row < 3; row++)
        {
            var expected = HostDft(host, row * 8, 1, 8);
            for (int k = 0; k < 8; k++)
            {
                await Assert.That((result[row * 8 + k] - expected[k]).Magnitude).IsLessThan(1e-9);
            }
        }
    }

    [Test]
    public async Task Fft_C2C_Batched_LeadingAxis()
    {
        // 8 x 3 tensor; transform along axis 0 (stride 3 between elements).
        var host = RampSignal(24);
        var result = await RunC2C([8, 3], [0], host);

        for (int col = 0; col < 3; col++)
        {
            var expected = HostDft(host, col, 3, 8);
            for (int k = 0; k < 8; k++)
            {
                await Assert.That((result[k * 3 + col] - expected[k]).Magnitude).IsLessThan(1e-9);
            }
        }
    }

    [Test]
    public async Task Fft_C2C_NegativeAxis_MatchesPositive()
    {
        var host = RampSignal(24);
        var a = await RunC2C([3, 8], [-1], host);
        var b = await RunC2C([3, 8], [1], host);
        for (int i = 0; i < a.Length; i++)
        {
            await Assert.That((a[i] - b[i]).Magnitude).IsLessThan(1e-12);
        }
    }

    [Test]
    public async Task Fft_InvalidAxes_Throw()
    {
        await Assert.That(() => new FftPlan([2, 3, 4], [1], doublePrecision: true)).Throws<NotSupportedException>();
        await Assert.That(() => new FftPlan([2, 3, 4], [0, 2], doublePrecision: true)).Throws<ArgumentException>();
        await Assert.That(() => new FftPlan([2, 3], [5], doublePrecision: true)).Throws<ArgumentOutOfRangeException>();
    }

    [Test]
    public async Task Fft_R2C_C2R_Batched_RoundTrip()
    {
        // 3 rows of 16 real samples; R2C -> [3, 9] half-spectrum -> C2R -> 16 * original.
        const int Rows = 3, N = 16;
        double[] real = new double[Rows * N];
        for (int i = 0; i < real.Length; i++)
        {
            real[i] = Math.Sin(i * 0.37) + 0.05 * i;
        }

        using var input = new DeviceBuffer<double>(real.Length);
        using var spectrum = new DeviceBuffer<Complex>(Rows * (N / 2 + 1));
        using var recovered = new DeviceBuffer<double>(real.Length);
        input.CopyFrom(real.AsSpan());

        using (var fwd = new FftPlan([Rows, N], [1], new FftOptions { FftType = FftType.R2C, Blocking = true }, doublePrecision: true))
        {
            fwd.ResetOperand(input.PointerAsInt, spectrum.PointerAsInt);
            await fwd.PlanAsync();
            await fwd.ExecuteAsync(FftDirection.Forward);
        }

        // First row, bin 0 equals the row sum.
        var half = spectrum.ToArray();
        double rowSum = real[..N].Sum();
        await Assert.That(Math.Abs(half[0].Real - rowSum)).IsLessThan(1e-9);

        using (var inv = new FftPlan([Rows, N], [1], new FftOptions { FftType = FftType.C2R, Blocking = true }, doublePrecision: true))
        {
            inv.ResetOperand(spectrum.PointerAsInt, recovered.PointerAsInt);
            await inv.PlanAsync();
            await inv.ExecuteAsync(FftDirection.Inverse);
        }

        var back = recovered.ToArray();
        for (int i = 0; i < real.Length; i++)
        {
            await Assert.That(Math.Abs(back[i] / N - real[i])).IsLessThan(1e-9);
        }
    }

    // ── Normalization ────────────────────────────────────────────────────────

    private static async Task<Complex[]> RoundTripC2C(long[] shape, int[] axes, Complex[] host, FftNormalization norm)
    {
        var opts = new FftOptions { FftType = FftType.C2C, Blocking = true, Normalization = norm };
        using var input = new DeviceBuffer<Complex>(host.Length);
        using var freq = new DeviceBuffer<Complex>(host.Length);
        using var back = new DeviceBuffer<Complex>(host.Length);
        input.CopyFrom(host.AsSpan());

        using var fwd = new FftPlan(shape, axes, opts, doublePrecision: true);
        fwd.ResetOperand(input.PointerAsInt, freq.PointerAsInt);
        await fwd.PlanAsync();
        await fwd.ExecuteAsync(FftDirection.Forward);

        using var inv = new FftPlan(shape, axes, opts, doublePrecision: true);
        inv.ResetOperand(freq.PointerAsInt, back.PointerAsInt);
        await inv.PlanAsync();
        await inv.ExecuteAsync(FftDirection.Inverse);
        return back.ToArray();
    }

    [Test]
    public async Task Fft_Normalization_Backward_RoundTripNeedsNoManualScaling()
    {
        var host = RampSignal(64);
        var back = await RoundTripC2C([64], [0], host, FftNormalization.Backward);
        for (int i = 0; i < host.Length; i++)
        {
            await Assert.That((back[i] - host[i]).Magnitude).IsLessThan(1e-9);
        }
    }

    [Test]
    public async Task Fft_Normalization_None_RoundTripIsScaledByN()
    {
        var host = RampSignal(64);
        var back = await RoundTripC2C([64], [0], host, FftNormalization.None);
        await Assert.That((back[5] - host[5] * 64).Magnitude).IsLessThan(1e-7);
    }

    [Test]
    public async Task Fft_Normalization_Orthonormal_PreservesEnergy_Double()
    {
        var host = RampSignal(128);
        var freq = await RunC2C([128], [0], host);
        double scaledEnergy;
        {
            var opts = new FftOptions { FftType = FftType.C2C, Blocking = true, Normalization = FftNormalization.Orthonormal };
            using var input = new DeviceBuffer<Complex>(host.Length);
            using var output = new DeviceBuffer<Complex>(host.Length);
            input.CopyFrom(host.AsSpan());
            using var fft = new FftPlan([128], [0], opts, doublePrecision: true);
            fft.ResetOperand(input.PointerAsInt, output.PointerAsInt);
            await fft.PlanAsync();
            await fft.ExecuteAsync(FftDirection.Forward);
            scaledEnergy = output.ToArray().Sum(c => c.Real * c.Real + c.Imaginary * c.Imaginary);
        }

        double inputEnergy = host.Sum(c => c.Real * c.Real + c.Imaginary * c.Imaginary);
        await Assert.That(Math.Abs(scaledEnergy - inputEnergy) / inputEnergy).IsLessThan(1e-9);

        // And the unscaled transform has N times the energy (sanity check of the baseline).
        double rawEnergy = freq.Sum(c => c.Real * c.Real + c.Imaginary * c.Imaginary);
        await Assert.That(Math.Abs(rawEnergy / inputEnergy - 128)).IsLessThan(1e-6);
    }

    [Test]
    public async Task Fft_Normalization_Orthonormal_PreservesEnergy_Single()
    {
        const int N = 256;
        float[] interleaved = new float[2 * N];
        for (int i = 0; i < N; i++)
        {
            interleaved[2 * i] = MathF.Sin(i * 0.3f) + 0.01f * i;
            interleaved[2 * i + 1] = MathF.Cos(i * 0.9f);
        }

        var opts = new FftOptions { FftType = FftType.C2C, Blocking = true, Normalization = FftNormalization.Orthonormal };
        using var input = new DeviceBuffer<float>(2 * N);
        using var output = new DeviceBuffer<float>(2 * N);
        input.CopyFrom(interleaved.AsSpan());
        using var fft = new FftPlan([N], [0], opts, doublePrecision: false);
        fft.ResetOperand(input.PointerAsInt, output.PointerAsInt);
        await fft.PlanAsync();
        await fft.ExecuteAsync(FftDirection.Forward);

        double inE = interleaved.Sum(v => (double)v * v);
        double outE = output.ToArray().Sum(v => (double)v * v);
        await Assert.That(Math.Abs(outE - inE) / inE).IsLessThan(1e-4);
    }

    [Test]
    public async Task Fft_Normalization_Backward_LeadingAxisBatch_RoundTrips()
    {
        var host = RampSignal(24);
        var back = await RoundTripC2C([8, 3], [0], host, FftNormalization.Backward);
        for (int i = 0; i < host.Length; i++)
        {
            await Assert.That((back[i] - host[i]).Magnitude).IsLessThan(1e-9);
        }
    }

    [Test]
    public async Task Fft_Normalization_Backward_R2C_C2R_RecoversRealSignal()
    {
        const int Rows = 3, N = 16;
        double[] real = new double[Rows * N];
        for (int i = 0; i < real.Length; i++)
        {
            real[i] = Math.Sin(i * 0.37) + 0.05 * i;
        }

        using var input = new DeviceBuffer<double>(real.Length);
        using var spectrum = new DeviceBuffer<Complex>(Rows * (N / 2 + 1));
        using var recovered = new DeviceBuffer<double>(real.Length);
        input.CopyFrom(real.AsSpan());

        using (var fwd = new FftPlan([Rows, N], [1], new FftOptions { FftType = FftType.R2C, Blocking = true, Normalization = FftNormalization.Backward }, doublePrecision: true))
        {
            fwd.ResetOperand(input.PointerAsInt, spectrum.PointerAsInt);
            await fwd.PlanAsync();
            await fwd.ExecuteAsync(FftDirection.Forward);
        }

        using (var inv = new FftPlan([Rows, N], [1], new FftOptions { FftType = FftType.C2R, Blocking = true, Normalization = FftNormalization.Backward }, doublePrecision: true))
        {
            inv.ResetOperand(spectrum.PointerAsInt, recovered.PointerAsInt);
            await inv.PlanAsync();
            await inv.ExecuteAsync(FftDirection.Inverse);
        }

        double[] back = recovered.ToArray();
        for (int i = 0; i < real.Length; i++)
        {
            await Assert.That(Math.Abs(back[i] - real[i])).IsLessThan(1e-9);
        }
    }
}
