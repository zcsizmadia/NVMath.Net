// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the MIT License.

namespace NVMathNet.Tests;

/// <summary>
/// Tests for multi-GPU support. Requires at least 1 CUDA-capable GPU.
/// Tests with peer access require 2+ GPUs.
/// </summary>
public sealed class MultiGpuTests
{
    [Test]
    public async Task MultiGpuContext_SingleDevice_Works()
    {
        using var ctx = new MultiGpuContext([0]);
        await Assert.That(ctx.DeviceCount).IsEqualTo(1);

        string name = ctx.GetDeviceName(0);
        await Assert.That(name).IsNotNull();
        await Assert.That(name.Length).IsGreaterThan(0);
    }

    [Test]
    public async Task MultiGpuContext_ForEachDevice_Executes()
    {
        using var ctx = new MultiGpuContext([0]);
        int executionCount = 0;

        ctx.ForEachDevice((index, deviceId, stream) => Interlocked.Increment(ref executionCount));

        await Assert.That(executionCount).IsEqualTo(1);
    }

    [Test]
    public async Task MultiGpuContext_ForEachDeviceAsync_Executes()
    {
        using var ctx = new MultiGpuContext([0]);
        int executionCount = 0;

        await ctx.ForEachDeviceAsync(async (index, deviceId, stream) =>
        {
            Interlocked.Increment(ref executionCount);
            await Task.CompletedTask;
        });

        await Assert.That(executionCount).IsEqualTo(1);
    }

    [Test]
    public async Task MultiGpuContext_SynchronizeAll()
    {
        using var ctx = new MultiGpuContext([0]);

        // Enqueue some work
        using var buf = new DeviceBuffer<float>(1024);
        buf.Clear();

        ctx.SynchronizeAll();

        // No exception means success
        await Assert.That(ctx.DeviceCount).IsGreaterThanOrEqualTo(1);
    }

    [Test]
    public async Task MultiGpuContext_DeviceId_Correct()
    {
        using var ctx = new MultiGpuContext([0]);
        await Assert.That(ctx.GetDeviceId(0)).IsEqualTo(0);
    }

    [Test]
    public async Task MultiGpuContext_AllDevices()
    {
        // Test with all available devices
        int deviceCount = CudaDevice.GetDeviceCount();
        int[] devices = [.. Enumerable.Range(0, deviceCount)];
        using var ctx = new MultiGpuContext(devices);

        await Assert.That(ctx.DeviceCount).IsEqualTo(deviceCount);

        // Each device should be executable
        int count = 0;
        ctx.ForEachDevice((index, deviceId, stream) => Interlocked.Increment(ref count));

        await Assert.That(count).IsEqualTo(deviceCount);
    }

    [Test]
    public async Task MultiGpuContext_CanAllocatePerDevice()
    {
        using var ctx = new MultiGpuContext([0]);

        ctx.ForEachDevice((index, deviceId, stream) =>
        {
            using var buf = new DeviceBuffer<float>(256);
            buf.Clear();
            float[] data = buf.ToArray();
            // All zeros
            foreach (float v in data)
            {
                if (v != 0f)
                {
                    throw new Exception("Expected zero");
                }
            }
        });

        await Assert.That(ctx.DeviceCount).IsGreaterThanOrEqualTo(1);
    }
}
