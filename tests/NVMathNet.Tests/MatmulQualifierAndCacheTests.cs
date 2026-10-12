// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the MIT License.

using System.Collections.Concurrent;
using System.Numerics;

using NVMathNet.LinAlg;

namespace NVMathNet.Tests;

/// <summary>
/// Tests for conjugate-transpose / complex matmul and the plan cache. Requires a CUDA-capable GPU.
/// </summary>
public sealed class MatmulQualifierAndCacheTests
{
    private static MatmulOptions ComplexOptions(bool conjugateA = false, bool conjugateB = false) => new()
    {
        ComputeType = MatmulComputeType.Compute64F,
        TypeA = MatrixDataType.Complex128,
        TypeB = MatrixDataType.Complex128,
        TypeC = MatrixDataType.Complex128,
        TypeD = MatrixDataType.Complex128,
        ScaleType = MatrixDataType.Complex128,
        ConjugateTransposeA = conjugateA,
        ConjugateTransposeB = conjugateB,
        Blocking = true,
    };

    [Test]
    public async Task ComplexMatmul_ConjugateTransposeA_WithComplexScalars()
    {
        const int N = 3; // square so that op(A) = A^H has the same shape as A
        var a = new Complex[N * N];
        var b = new Complex[N * N];
        var c = new Complex[N * N];
        for (int i = 0; i < N * N; i++)
        {
            a[i] = new Complex(i + 1, 0.5 * i - 2);
            b[i] = new Complex(Math.Sin(i), Math.Cos(i * 1.5));
            c[i] = new Complex(1 - i, i * 0.25);
        }

        var alpha = new Complex(1, 1);
        var beta = new Complex(0.5, -2);
        using var dA = new DeviceBuffer<Complex>(N * N);
        using var dB = new DeviceBuffer<Complex>(N * N);
        using var dC = new DeviceBuffer<Complex>(N * N);
        using var dD = new DeviceBuffer<Complex>(N * N);
        dA.CopyFrom(a.AsSpan());
        dB.CopyFrom(b.AsSpan());
        dC.CopyFrom(c.AsSpan());

        using var matmul = new Matmul(N, N, N, options: ComplexOptions(conjugateA: true));
        await matmul.PlanAsync();
        await matmul.ExecuteAsync(dA, dB, dC, dD, alpha, beta);

        var d = dD.ToArray();
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                Complex sum = 0;
                for (int k = 0; k < N; k++)
                {
                    sum += Complex.Conjugate(a[k + N * i]) * b[k + N * j]; // (A^H)[i,k] = conj(A[k,i])
                }

                var expected = alpha * sum + beta * c[i + N * j];
                await Assert.That((d[i + N * j] - expected).Magnitude).IsLessThan(1e-9);
            }
        }
    }

    [Test]
    public async Task ComplexMatmul_PlainProduct_WithComplexAlpha()
    {
        const int N = 2;
        Complex[] a = [new(1, 1), new(2, 0), new(0, -1), new(3, 2)];
        Complex[] b = [new(1, 0), new(0, 1), new(2, -1), new(1, 1)];
        using var dA = new DeviceBuffer<Complex>(4);
        using var dB = new DeviceBuffer<Complex>(4);
        using var dC = new DeviceBuffer<Complex>(4);
        using var dD = new DeviceBuffer<Complex>(4);
        dA.CopyFrom(a.AsSpan());
        dB.CopyFrom(b.AsSpan());
        dC.Clear();

        using var matmul = new Matmul(N, N, N, options: ComplexOptions());
        await matmul.PlanAsync();
        await matmul.ExecuteAsync(dA, dB, dC, dD, new Complex(0, 2), Complex.Zero);

        var d = dD.ToArray();
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                Complex sum = 0;
                for (int k = 0; k < N; k++)
                {
                    sum += a[i + N * k] * b[k + N * j];
                }

                await Assert.That((d[i + N * j] - new Complex(0, 2) * sum).Magnitude).IsLessThan(1e-12);
            }
        }
    }

    // ── Plan cache ───────────────────────────────────────────────────────────

    private sealed record Entry(NVMathLogLevel Level, string Category, string Message);

    private static async Task<List<Entry>> CaptureDebug(Func<Task> action)
    {
        var entries = new ConcurrentQueue<Entry>();
        var previousHandler = NVMathLog.Handler;
        var previousLevel = NVMathLog.MinimumLevel;
        NVMathLog.MinimumLevel = NVMathLogLevel.Debug;
        NVMathLog.Handler = (level, category, message) => entries.Enqueue(new Entry(level, category, message));
        try
        {
            await action();
        }
        finally
        {
            NVMathLog.Handler = previousHandler;
            NVMathLog.MinimumLevel = previousLevel;
        }

        return [.. entries];
    }

    [Test]
    [NotInParallel("global-log")]
    public async Task PlanCache_SecondIdenticalPlanIsServedFromTheCache()
    {
        // A shape no other test uses, so only this test touches the cache entry.
        const int M = 37, N = 41, K = 43;
        var entries = await CaptureDebug(async () =>
        {
            using var first = new Matmul(M, N, K);
            await first.PlanAsync();
            using var second = new Matmul(M, N, K);
            await second.PlanAsync();
            using var other = new Matmul(M + 1, N, K);
            await other.PlanAsync();
        });

        var mine = entries.Where(e => e.Category == "Matmul" && e.Message.Contains($"{M}x{N}x{K}")).ToList();
        // Exactly one cache hit for the repeated shape (the second plan), and none for the different shape.
        await Assert.That(mine.Count(e => e.Message.StartsWith("Plan cache hit"))).IsEqualTo(1);
        await Assert.That(entries.Any(e => e.Message.Contains($"{M + 1}x{N}x{K}") && e.Message.StartsWith("Plan cache hit"))).IsFalse();
        await Assert.That(entries.Any(e => e.Message.Contains($"{M + 1}x{N}x{K}") && e.Message.StartsWith("Planned"))).IsTrue();
        await Assert.That(MatmulPlanCache.Count).IsGreaterThanOrEqualTo(2);
        await Assert.That(MatmulPlanCache.Hits).IsGreaterThanOrEqualTo(1L);
    }

    [Test]
    [NotInParallel("global-log")]
    public async Task PlanCache_ReusesTheAutotunedAlgorithm_AndStaysCorrect()
    {
        const int M = 53, N = 59, K = 61;
        var prefs = new MatmulPlanPreferences { RequestedAlgoCount = 4 };
        using var a = new DeviceBuffer<float>(M * K);
        using var b = new DeviceBuffer<float>(K * N);
        using var c = new DeviceBuffer<float>(M * N);
        using var d = new DeviceBuffer<float>(M * N);
        a.CopyFrom(Enumerable.Range(0, M * K).Select(i => (i % 7) * 0.5f).ToArray().AsSpan());
        b.CopyFrom(Enumerable.Range(0, K * N).Select(i => (i % 5) * 0.25f).ToArray().AsSpan());
        c.Clear();

        int tuned;
        using (var first = new Matmul(M, N, K, options: new MatmulOptions { Blocking = true }))
        {
            await first.PlanAsync(prefs);
            first.Autotune(a.PointerAsInt, b.PointerAsInt, c.PointerAsInt, d.PointerAsInt);
            tuned = first.SelectedAlgorithm;
        }

        using var second = new Matmul(M, N, K, options: new MatmulOptions { Blocking = true });
        await second.PlanAsync(prefs);
        await Assert.That(second.SelectedAlgorithm).IsEqualTo(tuned);

        // The reused algorithm must still compute the right result.
        await second.ExecuteAsync<float>(a, b, c, d);
        float[] ha = a.ToArray(), hb = b.ToArray(), hd = d.ToArray();
        int row = 5, col = 7;
        double expected = 0;
        for (int k = 0; k < K; k++)
        {
            expected += ha[row + M * k] * (double)hb[k + K * col];
        }

        await Assert.That(Math.Abs(hd[row + M * col] - expected)).IsLessThan(1e-2);
    }

    [Test]
    [NotInParallel("global-log")]
    public async Task PlanCache_CanBeDisabled()
    {
        const int M = 67, N = 71, K = 73;
        bool previous = MatmulPlanCache.Enabled;
        MatmulPlanCache.Enabled = false;
        try
        {
            var entries = await CaptureDebug(async () =>
            {
                using var first = new Matmul(M, N, K);
                await first.PlanAsync();
                using var second = new Matmul(M, N, K);
                await second.PlanAsync();
            });

            var mine = entries.Where(e => e.Category == "Matmul" && e.Message.Contains($"{M}x{N}x{K}")).ToList();
            await Assert.That(mine.Count(e => e.Message.StartsWith("Planned"))).IsEqualTo(2);
            await Assert.That(mine.Any(e => e.Message.StartsWith("Plan cache hit"))).IsFalse();
        }
        finally
        {
            MatmulPlanCache.Enabled = previous;
        }
    }
}
