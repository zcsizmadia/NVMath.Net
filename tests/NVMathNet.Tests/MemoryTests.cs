// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the MIT License.

using NVMathNet.Memory;

namespace NVMathNet.Tests;

/// <summary>
/// Tests for pluggable device memory managers. Requires a CUDA-capable GPU.
/// </summary>
public sealed class MemoryTests
{
    private sealed class CountingManager : IDeviceMemoryManager
    {
        public int Allocations;
        public int Frees;
        public nuint LastSize;

        public nint Allocate(nuint sizeInBytes)
        {
            Allocations++;
            LastSize = sizeInBytes;
            return CudaMallocMemoryManager.Instance.Allocate(sizeInBytes);
        }

        public void Free(nint pointer)
        {
            Frees++;
            CudaMallocMemoryManager.Instance.Free(pointer);
        }
    }

    [Test]
    public async Task DeviceBuffer_UsesCustomManager_ForAllocateAndFree()
    {
        var manager = new CountingManager();

        using (var buf = new DeviceBuffer<float>(1024, manager))
        {
            await Assert.That(manager.Allocations).IsEqualTo(1);
            await Assert.That(manager.LastSize).IsEqualTo((nuint)(1024 * sizeof(float)));
            await Assert.That(manager.Frees).IsEqualTo(0);
        }

        await Assert.That(manager.Frees).IsEqualTo(1);
    }

    [Test]
    public async Task CachingManager_ReusesFreedBlocks_AndTrimReleasesThem()
    {
        using var cache = new CachingMemoryManager();

        nint first;
        using (var a = new DeviceBuffer<float>(1000, cache)) // 4000 B -> 4096 B block
        {
            first = a.PointerAsInt;
            await Assert.That(cache.InUseBytes).IsEqualTo(4096L);
        }

        await Assert.That(cache.CachedBytes).IsEqualTo(4096L);
        await Assert.That(cache.InUseBytes).IsEqualTo(0L);

        using (var b = new DeviceBuffer<float>(1024, cache)) // same 4096 B bucket
        {
            await Assert.That(b.PointerAsInt).IsEqualTo(first);
            await Assert.That(cache.CachedBytes).IsEqualTo(0L);
        }

        cache.Trim();
        await Assert.That(cache.CachedBytes).IsEqualTo(0L);
    }

    [Test]
    public async Task CachingManager_ReusedBuffer_IsFullyUsable()
    {
        using var cache = new CachingMemoryManager();
        float[] data = [.. Enumerable.Range(0, 256).Select(i => (float)i)];

        using (var a = new DeviceBuffer<float>(256, cache))
        {
            a.CopyFrom(data.AsSpan());
        }

        using var b = new DeviceBuffer<float>(256, cache);
        b.CopyFrom(data.AsSpan());
        float[] result = b.ToArray();
        await Assert.That(result.SequenceEqual(data)).IsTrue();
    }

    [Test]
    public async Task CachingManager_FreeingUnknownPointer_Throws()
    {
        using var cache = new CachingMemoryManager();
        await Assert.That(() => cache.Free(12345)).Throws<ArgumentException>();
    }
}
