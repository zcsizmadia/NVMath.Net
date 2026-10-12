// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the MIT License.

using System.Collections.Concurrent;

using NVMathNet.Interop;
using NVMathNet.LinAlg;
using FftPlan = NVMathNet.Fft.Fft;

namespace NVMathNet.Tests;

/// <summary>
/// Tests for the <see cref="NVMathLog"/> hook. They change process-wide state, so they do not run in parallel
/// with each other; messages that other tests produce while a handler is installed are tolerated.
/// </summary>
public sealed class LogTests
{
    private sealed record Entry(NVMathLogLevel Level, string Category, string Message);

    private static async Task<List<Entry>> Capture(NVMathLogLevel minimum, Func<Task> action)
    {
        var entries = new ConcurrentQueue<Entry>();
        var previousHandler = NVMathLog.Handler;
        var previousLevel = NVMathLog.MinimumLevel;
        NVMathLog.MinimumLevel = minimum;
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
    public async Task NativeErrors_AreLoggedBeforeTheExceptionIsThrown()
    {
        var entries = await Capture(NVMathLogLevel.Error, async () =>
        {
            await Assert.That(() => CuSolverNative.Check((CuSolverStatus)3, "unit-test-context")).Throws<CuSolverException>();
        });

        var match = entries.FirstOrDefault(e => e.Category == "cuSOLVER" && e.Message.Contains("unit-test-context"));
        await Assert.That(match).IsNotNull();
        await Assert.That(match!.Level).IsEqualTo(NVMathLogLevel.Error);
    }

    [Test]
    [NotInParallel("global-log")]
    public async Task PlanDetails_AreLoggedAtDebugLevel()
    {
        var entries = await Capture(NVMathLogLevel.Debug, async () =>
        {
            using var fft = new FftPlan([64], doublePrecision: true);
            await fft.PlanAsync();
            using var matmul = new Matmul(8, 8, 8);
            await matmul.PlanAsync();
        });

        await Assert.That(entries.Any(e => e.Category == "Fft" && e.Level == NVMathLogLevel.Debug && e.Message.Contains("Planned"))).IsTrue();
        await Assert.That(entries.Any(e => e.Category == "Matmul" && e.Message.Contains("candidate algorithm"))).IsTrue();
    }

    [Test]
    [NotInParallel("global-log")]
    public async Task MinimumLevel_FiltersLowerSeverities()
    {
        var entries = await Capture(NVMathLogLevel.Warning, async () =>
        {
            using var fft = new FftPlan([64], doublePrecision: true);
            await fft.PlanAsync();
        });

        await Assert.That(entries.All(e => e.Level >= NVMathLogLevel.Warning)).IsTrue();
    }

    [Test]
    [NotInParallel("global-log")]
    public async Task Autotune_IsLoggedAtInformationLevel()
    {
        var entries = await Capture(NVMathLogLevel.Information, async () =>
        {
            const int N = 64;
            using var a = new DeviceBuffer<float>(N * N);
            using var b = new DeviceBuffer<float>(N * N);
            using var c = new DeviceBuffer<float>(N * N);
            using var d = new DeviceBuffer<float>(N * N);
            a.Clear();
            b.Clear();
            c.Clear();
            using var matmul = new Matmul(N, N, N);
            await matmul.PlanAsync(new MatmulPlanPreferences { RequestedAlgoCount = 4 });
            matmul.Autotune(a.PointerAsInt, b.PointerAsInt, c.PointerAsInt, d.PointerAsInt);
        });

        await Assert.That(entries.Any(e => e.Category == "Matmul" && e.Level == NVMathLogLevel.Information && e.Message.Contains("Autotuned"))).IsTrue();
    }

    [Test]
    [NotInParallel("global-log")]
    public async Task FaultyHandler_DoesNotBreakTheOperation()
    {
        var previousHandler = NVMathLog.Handler;
        var previousLevel = NVMathLog.MinimumLevel;
        NVMathLog.MinimumLevel = NVMathLogLevel.Error;
        NVMathLog.Handler = (_, _, _) => throw new InvalidOperationException("handler failure");
        try
        {
            // The original native error must surface, not the handler's exception.
            await Assert.That(() => CuSolverNative.Check((CuSolverStatus)3, "ctx")).Throws<CuSolverException>();
        }
        finally
        {
            NVMathLog.Handler = previousHandler;
            NVMathLog.MinimumLevel = previousLevel;
        }
    }

    [Test]
    [NotInParallel("global-log")]
    public async Task WithoutHandler_NothingIsEnabled()
    {
        var previous = NVMathLog.Handler;
        NVMathLog.Handler = null;
        try
        {
            await Assert.That(NVMathLog.IsEnabled(NVMathLogLevel.Error)).IsFalse();
        }
        finally
        {
            NVMathLog.Handler = previous;
        }
    }
}
