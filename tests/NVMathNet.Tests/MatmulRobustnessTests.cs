// Copyright (c) 2026 Zoltan Csizmadia. All rights reserved.
// Licensed under the MIT License.

using NVMathNet.LinAlg;

namespace NVMathNet.Tests;

/// <summary>
/// Regression tests for issues found in the pre-1.0 review of <see cref="Matmul"/>.
/// Requires a CUDA-capable GPU.
/// </summary>
public sealed class MatmulRobustnessTests
{
    [Test]
    public async Task FastAccum_CanBeEnabled()
    {
        // CUBLASLT_MATMUL_DESC_FAST_ACCUM is an int8_t; writing it as an int made construction throw InvalidValue.
        using var matmul = new Matmul(8, 8, 8, options: new MatmulOptions { FastAccum = true });
        await Assert.That(matmul.AlgorithmCount).IsEqualTo(0); // not planned yet; construction succeeded
    }

    [Test]
    [Arguments(0, 4, 4, 1)]
    [Arguments(4, 0, 4, 1)]
    [Arguments(4, 4, 0, 1)]
    [Arguments(4, -3, 4, 1)]
    [Arguments(4, 4, 4, 0)]
    [Arguments(4, 4, 4, -2)]
    public async Task NonPositiveDimensionsOrBatch_Throw(int m, int n, int k, int batch)
    {
        await Assert.That(() => new Matmul(m, n, k, batch)).Throws<ArgumentOutOfRangeException>();
    }

    [Test]
    public async Task RequestedAlgoCountBelowOne_Throws()
    {
        using var matmul = new Matmul(8, 8, 8);
        await Assert.That(() => matmul.Plan(new MatmulPlanPreferences { RequestedAlgoCount = 0 })).Throws<ArgumentOutOfRangeException>();
    }

    [Test]
    public async Task Options_AreSnapshotAtConstruction()
    {
        // alpha is encoded with ScaleType at every Execute. If the caller's options object were read live,
        // changing it afterwards would make cuBLASLt read a double where a float is expected.
        var options = new MatmulOptions { Blocking = true };
        using var matmul = new Matmul(2, 2, 2, options: options);
        options.ScaleType = MatrixDataType.Float64;
        options.ComputeType = MatmulComputeType.Compute64F;
        await matmul.PlanAsync();

        float[] identity = [1, 0, 0, 1];
        float[] b = [1, 2, 3, 4];
        using var dA = new DeviceBuffer<float>(4);
        using var dB = new DeviceBuffer<float>(4);
        using var dC = new DeviceBuffer<float>(4);
        using var dD = new DeviceBuffer<float>(4);
        dA.CopyFrom(identity);
        dB.CopyFrom(b);
        dC.Clear();
        await matmul.ExecuteAsync<float>(dA, dB, dC, dD, alpha: 2.0, beta: 0.0);

        float[] d = dD.ToArray();
        for (int i = 0; i < 4; i++)
        {
            await Assert.That(Math.Abs(d[i] - 2 * b[i])).IsLessThan(1e-6f);
        }
    }

    [Test]
    public async Task Dispose_WithWorkStillQueued_IsSafe()
    {
        // Blocking = false: the kernel may still be running when Dispose is called; Dispose must wait for it
        // before the workspace goes back to its manager.
        const int N = 512;
        using var a = new DeviceBuffer<float>(N * N);
        using var b = new DeviceBuffer<float>(N * N);
        using var c = new DeviceBuffer<float>(N * N);
        using var d = new DeviceBuffer<float>(N * N);
        a.Clear();
        b.Clear();
        c.Clear();

        var matmul = new Matmul(N, N, N, options: new MatmulOptions { Blocking = false });
        await matmul.PlanAsync();
        for (int i = 0; i < 4; i++)
        {
            await matmul.ExecuteAsync<float>(a, b, c, d);
        }

        matmul.Dispose();                    // must not fault or corrupt anything
        CudaDevice.Get(0).Synchronize();     // would surface a deferred CUDA error
        await Assert.That(d.ToArray().All(v => v == 0f)).IsTrue();
    }

    [Test]
    [NotInParallel("global-log")]
    public async Task PlanCache_IsBounded_ByMaxEntries()
    {
        int previousMax = MatmulPlanCache.MaxEntries;
        try
        {
            MatmulPlanCache.Clear();
            MatmulPlanCache.MaxEntries = 2;
            for (int shape = 101; shape < 106; shape++)
            {
                using var matmul = new Matmul(shape, 16, 16);
                await matmul.PlanAsync();
                await Assert.That(MatmulPlanCache.Count).IsLessThanOrEqualTo(2);
            }
        }
        finally
        {
            MatmulPlanCache.MaxEntries = previousMax;
        }
    }
}
