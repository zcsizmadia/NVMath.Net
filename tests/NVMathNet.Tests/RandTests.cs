// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the MIT License.

using NVMathNet.Interop;
using NVMathNet.Rand;

namespace NVMathNet.Tests;

/// <summary>
/// Tests for the cuRAND random number generator module. Requires a CUDA-capable GPU.
/// </summary>
public sealed class RandTests
{
    [Test]
    public async Task Uniform_Float_InRange()
    {
        const int N = 10000;
        using var rng = new CudaRandom(seed: 42);
        using var dBuf = new DeviceBuffer<float>(N);

        await rng.FillUniformAsync(dBuf);

        float[] result = dBuf.ToArray();
        // cuRAND uniform produces values in (0, 1]
        float min = result.Min();
        float max = result.Max();
        await Assert.That(min).IsGreaterThan(0f);
        await Assert.That(max).IsLessThanOrEqualTo(1f);

        // Mean should be close to 0.5
        float mean = result.Average();
        await Assert.That(Math.Abs(mean - 0.5f)).IsLessThan(0.05f);
    }

    [Test]
    public async Task Uniform_Double_InRange()
    {
        const int N = 10000;
        using var rng = new CudaRandom(seed: 123);
        using var dBuf = new DeviceBuffer<double>(N);

        await rng.FillUniformAsync(dBuf);

        double[] result = dBuf.ToArray();
        double min = result.Min();
        double max = result.Max();
        await Assert.That(min).IsGreaterThan(0.0);
        await Assert.That(max).IsLessThanOrEqualTo(1.0);

        double mean = result.Average();
        await Assert.That(Math.Abs(mean - 0.5)).IsLessThan(0.05);
    }

    [Test]
    public async Task Normal_Float_MeanAndStddev()
    {
        const int N = 100000; // large enough for stable statistics
        using var rng = new CudaRandom(seed: 7);
        using var dBuf = new DeviceBuffer<float>(N);

        await rng.FillNormalAsync(dBuf, mean: 5f, stddev: 2f);

        float[] result = dBuf.ToArray();
        float mean = result.Average();
        float variance = result.Select(x => (x - mean) * (x - mean)).Average();
        float stddev = MathF.Sqrt(variance);

        await Assert.That(Math.Abs(mean - 5f)).IsLessThan(0.1f);
        await Assert.That(Math.Abs(stddev - 2f)).IsLessThan(0.1f);
    }

    [Test]
    public async Task Seed_Reproducibility()
    {
        const int N = 1024;
        using var rng1 = new CudaRandom(seed: 999);
        using var dBuf1 = new DeviceBuffer<float>(N);
        await rng1.FillUniformAsync(dBuf1);
        float[] result1 = dBuf1.ToArray();

        using var rng2 = new CudaRandom(seed: 999);
        using var dBuf2 = new DeviceBuffer<float>(N);
        await rng2.FillUniformAsync(dBuf2);
        float[] result2 = dBuf2.ToArray();

        // Same seed should produce identical results
        for (int i = 0; i < N; i++)
        {
            await Assert.That(result1[i]).IsEqualTo(result2[i]);
        }
    }

    [Test]
    public async Task DifferentSeeds_ProduceDifferentResults()
    {
        const int N = 1024;
        using var rng1 = new CudaRandom(seed: 100);
        using var dBuf1 = new DeviceBuffer<float>(N);
        await rng1.FillUniformAsync(dBuf1);
        float[] result1 = dBuf1.ToArray();

        using var rng2 = new CudaRandom(seed: 200);
        using var dBuf2 = new DeviceBuffer<float>(N);
        await rng2.FillUniformAsync(dBuf2);
        float[] result2 = dBuf2.ToArray();

        // Different seeds should produce different results
        bool anyDifferent = false;
        for (int i = 0; i < N; i++)
        {
            if (result1[i] != result2[i])
            {
                anyDifferent = true;
                break;
            }
        }

        await Assert.That(anyDifferent).IsTrue();
    }

    [Test]
    public async Task Philox_Generator()
    {
        const int N = 1024;
        using var rng = new CudaRandom(rngType: CuRandRngType.PseudoPhilox4_32_10, seed: 42);
        using var dBuf = new DeviceBuffer<float>(N);

        await rng.FillUniformAsync(dBuf);

        float[] result = dBuf.ToArray();
        float mean = result.Average();
        await Assert.That(Math.Abs(mean - 0.5f)).IsLessThan(0.1f);
    }
}
