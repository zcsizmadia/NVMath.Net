// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the Apache 2.0 License.

using System.Numerics;
using NVMathNet;
using NVMathNet.Fft;
using FftPlan = NVMathNet.Fft.Fft;

namespace NVMathNet.Net.Tests;

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
            host[i] = new Complex(Math.Sin(2 * Math.PI * 4 * i / N), 0);

        using var stream = new CudaStream();
        using var inputBuf = new DeviceBuffer<Complex>(N);
        inputBuf.CopyFrom(host.AsSpan());

        // Forward FFT
        long[] shape = [N];
        var opts = new FftOptions { FftType = FftType.C2C, Blocking = true };
        using var fwdFft = new FftPlan(shape, options: opts, stream: stream);
        using var outputBuf = new DeviceBuffer<Complex>(N);
        fwdFft.ResetOperand(inputBuf.PointerAsInt, outputBuf.PointerAsInt);
        await fwdFft.PlanAsync();
        await fwdFft.ExecuteAsync(FftDirection.Forward);

        // Inverse FFT
        using var invFft = new FftPlan(shape, options: opts, stream: stream);
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
        float[] real = new float[N];
        for (int i = 0; i < N; i++) real[i] = (float)Math.Cos(2 * Math.PI * 8 * i / N);

        using var stream = new CudaStream();
        using var inputBuf = new DeviceBuffer<float>(N);
        inputBuf.CopyFrom(real.AsSpan());

        long[] shape = [N];
        var opts = new FftOptions { FftType = FftType.R2C, Blocking = true };
        using var fft = new FftPlan(shape, options: opts, stream: stream);
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
            host[i] = new Complex(i % 4 == 0 ? 1.0 : 0.0, 0);

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
}
