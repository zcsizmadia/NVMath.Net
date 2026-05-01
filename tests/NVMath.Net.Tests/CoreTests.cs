// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the Apache 2.0 License.

using NVMathNet;
using NVMathNet.Interop;

namespace NVMathNet.Net.Tests;

/// <summary>
/// Tests for core CUDA infrastructure: DeviceBuffer, CudaStream, CudaEvent.
/// These tests require a CUDA-capable GPU to run.
/// </summary>
public sealed class CoreTests
{
    [Test]
    public async Task DeviceBuffer_AllocateAndFree()
    {
        using var buf = new DeviceBuffer<float>(1024);
        await Assert.That(buf.Length).IsEqualTo(1024L);
        await Assert.That(buf.SizeInBytes).IsEqualTo((nuint)(1024 * sizeof(float)));
        await Assert.That(buf.PointerAsInt).IsNotEqualTo(0);
    }

    [Test]
    public async Task DeviceBuffer_CopyFromAndToHost()
    {
        const int N = 256;
        float[] host = new float[N];
        for (int i = 0; i < N; i++) host[i] = i * 1.5f;

        using var buf = new DeviceBuffer<float>(N);
        buf.CopyFrom(host.AsSpan());

        float[] result = new float[N];
        buf.CopyTo(result.AsSpan());

        for (int i = 0; i < N; i++)
            await Assert.That(result[i]).IsEqualTo(host[i]);
    }

    [Test]
    public async Task DeviceBuffer_AsyncCopyRoundtrip()
    {
        const int N = 512;
        double[] host = new double[N];
        for (int i = 0; i < N; i++) host[i] = i * 2.0;

        using var stream = new CudaStream();
        using var buf = new DeviceBuffer<double>(N);

        await buf.CopyFromAsync(host, stream);

        double[] result = new double[N];
        await buf.CopyToAsync(result, stream);

        for (int i = 0; i < N; i++)
            await Assert.That(result[i]).IsEqualTo(host[i]);
    }

    [Test]
    public async Task DeviceBuffer_Clear()
    {
        const int N = 64;
        float[] host = Enumerable.Repeat(1.0f, N).ToArray();

        using var buf = new DeviceBuffer<float>(N);
        buf.CopyFrom(host.AsSpan());
        buf.Clear();

        float[] result = buf.ToArray();
        foreach (float v in result)
            await Assert.That(v).IsEqualTo(0.0f);
    }

    [Test]
    public async Task CudaStream_Create()
    {
        using var stream = new CudaStream();
        await Assert.That(stream.Handle).IsNotEqualTo(0);
    }

    [Test]
    public async Task CudaStream_Synchronize()
    {
        using var stream = new CudaStream();
        await stream.SynchronizeAsync();
    }

    [Test]
    public async Task CudaEvent_RecordAndElapsed()
    {
        using var stream = new CudaStream();
        using var start  = new CudaEvent();
        using var stop   = new CudaEvent();

        start.Record(stream);

        // Small workload: copy 1 MB
        using var buf = new DeviceBuffer<byte>(1024 * 1024);
        buf.Clear(stream);

        stop.Record(stream);
        await stream.SynchronizeAsync();

        float ms = stop.ElapsedMilliseconds(start);
        await Assert.That(ms).IsGreaterThan(0.0f);
    }

    [Test]
    public async Task PinnedBuffer_AllocateAndSpan()
    {
        using var pinned = new PinnedBuffer<int>(128);
        await Assert.That(pinned.Length).IsEqualTo(128L);

        var span = pinned.Span;
        for (int i = 0; i < span.Length; i++)
            span[i] = i * 3;

        await Assert.That(span[42]).IsEqualTo(126);
    }

    [Test]
    public async Task PinnedBuffer_CopyToDeviceAndBack()
    {
        const int N = 256;
        using var pinned = new PinnedBuffer<float>(N);
        var span = pinned.Span;
        for (int i = 0; i < N; i++) span[i] = i * 0.5f;

        using var stream = new CudaStream();
        using var device = new DeviceBuffer<float>(N);

        pinned.CopyToDevice(device, stream);
        await stream.SynchronizeAsync();

        using var out_pinned = new PinnedBuffer<float>(N);
        out_pinned.CopyFromDevice(device, stream);
        await stream.SynchronizeAsync();

        var outArr = out_pinned.Span.ToArray();
        for (int i = 0; i < N; i++)
            await Assert.That(outArr[i]).IsEqualTo(i * 0.5f);
    }
}
