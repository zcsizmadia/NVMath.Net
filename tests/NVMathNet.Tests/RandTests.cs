// Copyright (c) 2026 Zoltan Csizmadia. All rights reserved.
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

    // ── Integers, Poisson, skip-ahead, quasirandom ───────────────────────────

    [Test]
    public async Task Poisson_MeanAndVarianceMatchLambda()
    {
        const int N = 200_000;
        const double Lambda = 4.0;
        using var rng = new CudaRandom(seed: 7);
        using var dBuf = new DeviceBuffer<uint>(N);

        await rng.FillPoissonAsync(dBuf, Lambda);

        uint[] r = dBuf.ToArray();
        double mean = r.Average(v => (double)v);
        double variance = r.Average(v => (v - mean) * (v - mean));
        await Assert.That(Math.Abs(mean - Lambda)).IsLessThan(0.05);
        await Assert.That(Math.Abs(variance - Lambda)).IsLessThan(0.1);
    }

    [Test]
    public async Task UInt32_IsRoughlyUniform()
    {
        const int N = 100_000;
        using var rng = new CudaRandom(seed: 11);
        using var dBuf = new DeviceBuffer<uint>(N);

        await rng.FillUInt32Async(dBuf);

        double mean = dBuf.ToArray().Average(v => v / 4294967296.0);
        await Assert.That(Math.Abs(mean - 0.5)).IsLessThan(0.01);
    }

    [Test]
    public async Task SetOffset_Philox_SkipsAhead()
    {
        const int Skip = 128, N = 128;
        using var full = new DeviceBuffer<uint>(Skip + N);
        using var tail = new DeviceBuffer<uint>(N);

        using (var a = new CudaRandom(CuRandRngType.PseudoPhilox4_32_10, seed: 99))
        {
            await a.FillUInt32Async(full);
        }

        using (var b = new CudaRandom(CuRandRngType.PseudoPhilox4_32_10, seed: 99))
        {
            b.SetOffset(Skip);
            await b.FillUInt32Async(tail);
        }

        uint[] expected = full.ToArray()[Skip..];
        uint[] actual = tail.ToArray();
        await Assert.That(actual.SequenceEqual(expected)).IsTrue();
    }

    [Test]
    public async Task Sobol32_IsDeterministicAndLowDiscrepancy()
    {
        const int N = 4096;
        float[] Run()
        {
            using var rng = new CudaRandom(CuRandRngType.QuasiSobol32);
            using var buf = new DeviceBuffer<float>(N);
            rng.FillUniform(buf);
            return buf.ToArray();
        }

        float[] first = Run();
        float[] second = Run();
        await Assert.That(first.SequenceEqual(second)).IsTrue();
        await Assert.That(Math.Abs(first.Average() - 0.5f)).IsLessThan(0.005f);
    }

    [Test]
    public async Task Sobol64_FillUInt64_IsRoughlyUniform()
    {
        const int N = 4096;
        using var rng = new CudaRandom(CuRandRngType.QuasiSobol64);
        using var buf = new DeviceBuffer<ulong>(N);

        await rng.FillUInt64Async(buf);

        double mean = buf.ToArray().Average(v => v / 18446744073709551616.0);
        await Assert.That(Math.Abs(mean - 0.5)).IsLessThan(0.02);
    }

    [Test]
    public async Task QuasiRandom_Dimensions_AndSeedValidation()
    {
        using (var rng = new CudaRandom(CuRandRngType.QuasiSobol32))
        {
            rng.SetQuasiRandomDimensions(2);
            using var buf = new DeviceBuffer<float>(2048);
            await rng.FillUniformAsync(buf);
            await Assert.That(Math.Abs(buf.ToArray().Average() - 0.5f)).IsLessThan(0.02f);
        }

        await Assert.That(() => new CudaRandom(CuRandRngType.QuasiSobol32, seed: 5)).Throws<ArgumentException>();
        using var pseudo = new CudaRandom(seed: 1);
        await Assert.That(() => pseudo.SetQuasiRandomDimensions(2)).Throws<InvalidOperationException>();
    }
}
