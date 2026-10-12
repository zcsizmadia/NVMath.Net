// Copyright (c) 2026 Zoltan Csizmadia. All rights reserved.
// Licensed under the MIT License.

using System.Numerics;
using NVMathNet.Fft;
using FftPlan = NVMathNet.Fft.Fft;

namespace NVMathNet.Tests;

/// <summary>
/// Regression tests for issues found in the pre-1.0 review of the FFT wrapper.
/// Requires a CUDA-capable GPU.
/// </summary>
public sealed class FftReviewTests
{
    private const int N = 16;

    private static double[] Signal() =>
        [.. Enumerable.Range(0, N).Select(i => Math.Sin(i * 0.9) + 0.25 * i)];

    private static Complex Dft(double[] x, int k)
    {
        Complex sum = 0;
        for (int j = 0; j < x.Length; j++)
        {
            sum += x[j] * Complex.FromPolarCoordinates(1, -2 * Math.PI * j * k / x.Length);
        }

        return sum;
    }

    [Test]
    public async Task RFft_Double_MatchesDft()
    {
        double[] x = Signal();
        using var input = new DeviceBuffer<double>(N);
        input.CopyFrom(x);
        using var spec = await FftPlan.RFftAsync(input, [N]);
        Complex[] got = spec.ToArray();

        await Assert.That(got.Length).IsEqualTo(N / 2 + 1);
        for (int k = 0; k <= N / 2; k++)
        {
            await Assert.That((got[k] - Dft(x, k)).Magnitude).IsLessThan(1e-9);
        }
    }

    [Test]
    public async Task RFft_Float_ReturnsInterleavedAndMatchesDft()
    {
        double[] x = Signal();
        using var input = new DeviceBuffer<float>(N);
        input.CopyFrom([.. x.Select(v => (float)v)]);
        using var spec = await FftPlan.RFftAsync(input, [N]);
        float[] got = spec.ToArray();

        await Assert.That(got.Length).IsEqualTo(2 * (N / 2 + 1));
        for (int k = 0; k <= N / 2; k++)
        {
            var value = new Complex(got[2 * k], got[2 * k + 1]);
            await Assert.That((value - Dft(x, k)).Magnitude).IsLessThan(1e-3);
        }
    }

    [Test]
    public async Task RFftThenIRFft_RoundTrips_BothPrecisions()
    {
        double[] x = Signal();
        var options = new FftOptions { Normalization = FftNormalization.Backward };

        using (var inD = new DeviceBuffer<double>(N))
        {
            inD.CopyFrom(x);
            using var spec = await FftPlan.RFftAsync(inD, [N], options: options);
            using var back = await FftPlan.IRFftAsync(spec, [N], options: options);
            double[] y = back.ToArray();
            for (int i = 0; i < N; i++)
            {
                await Assert.That(Math.Abs(y[i] - x[i])).IsLessThan(1e-9);
            }
        }

        using (var inF = new DeviceBuffer<float>(N))
        {
            inF.CopyFrom([.. x.Select(v => (float)v)]);
            using var spec = await FftPlan.RFftAsync(inF, [N], options: options);
            using var back = await FftPlan.IRFftAsync(spec, [N], options: options);
            float[] y = back.ToArray();
            for (int i = 0; i < N; i++)
            {
                await Assert.That(Math.Abs(y[i] - x[i])).IsLessThan(1e-3);
            }
        }
    }

    [Test]
    public async Task IRFft_TooSmallInput_Throws()
    {
        using var spec = new DeviceBuffer<Complex>(2);
        await Assert.That(async () => await FftPlan.IRFftAsync(spec, [N])).Throws<ArgumentException>();
    }

    [Test]
    public async Task InPlace_RealTransform_IsRejected()
    {
        var options = new FftOptions { FftType = FftType.R2C, InPlace = true };
        await Assert.That(() => new FftPlan([N], null, options)).Throws<NotSupportedException>();
    }

    [Test]
    public async Task StaticHelpers_HonourNormalization_AndDoNotMutateOptions()
    {
        var options = new FftOptions { Normalization = FftNormalization.Backward };
        Complex[] x = [.. Enumerable.Range(0, N).Select(i => new Complex(i, -i * 0.5))];
        using var input = new DeviceBuffer<Complex>(N);
        input.CopyFrom(x);

        using var fwd = await FftPlan.FftAsync(input, [N], options: options);
        using var back = await FftPlan.IFftAsync(fwd, [N], options: options);
        Complex[] y = back.ToArray();

        await Assert.That(options.FftType).IsNull();
        for (int i = 0; i < N; i++)
        {
            await Assert.That((y[i] - x[i]).Magnitude).IsLessThan(1e-9);
        }
    }
}
