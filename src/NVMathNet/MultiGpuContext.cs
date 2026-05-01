// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the MIT License.

using System.Numerics;

using NVMathNet.Interop;

namespace NVMathNet;

/// <summary>
/// Manages a set of CUDA devices for single-process multi-GPU execution.
/// Enables peer-to-peer access between GPUs and provides helpers
/// to run work concurrently across devices using one thread per GPU.
/// </summary>
public sealed class MultiGpuContext : IDisposable
{
    private readonly int[] _deviceIds;
    private readonly CudaStream[] _streams;
    private bool _disposed;

    /// <summary>The device IDs managed by this context.</summary>
    public IReadOnlyList<int> DeviceIds => _deviceIds;

    /// <summary>Number of GPUs in this context.</summary>
    public int DeviceCount => _deviceIds.Length;

    /// <summary>
    /// Creates a multi-GPU context managing the specified devices.
    /// Enables peer-to-peer access where supported.
    /// </summary>
    /// <param name="deviceIds">Device ordinals to manage. If null, uses all available devices.</param>
    public MultiGpuContext(int[]? deviceIds = null)
    {
        int totalDevices = CudaRuntime.GetDeviceCount();
        _deviceIds = deviceIds ?? [.. Enumerable.Range(0, totalDevices)];

        if (_deviceIds.Length == 0)
        {
            throw new InvalidOperationException("No CUDA devices available.");
        }

        foreach (int id in _deviceIds)
        {
            if (id < 0 || id >= totalDevices)
            {
                throw new ArgumentOutOfRangeException(nameof(deviceIds), $"Device {id} is out of range (0..{totalDevices - 1}).");
            }
        }

        // Create a stream per device
        _streams = new CudaStream[_deviceIds.Length];
        for (int i = 0; i < _deviceIds.Length; i++)
        {
            CudaRuntime.SetDevice(_deviceIds[i]);
            _streams[i] = new CudaStream();
        }

        // Enable peer-to-peer access between all device pairs
        EnablePeerAccess();
    }

    /// <summary>Returns the stream associated with the given device index (0-based within this context).</summary>
    public CudaStream GetStream(int index)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return _streams[index];
    }

    /// <summary>Returns the CUDA device ID for the given index (0-based within this context).</summary>
    public int GetDeviceId(int index) => _deviceIds[index];

    /// <summary>Returns the device name for the given device index.</summary>
    public string GetDeviceName(int index) => CudaRuntime.GetDeviceName(_deviceIds[index]);

    /// <summary>
    /// Sets the active CUDA device for the calling thread to the device at <paramref name="index"/>
    /// (0-based within this context).
    /// </summary>
    public void SetDevice(int index)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        CudaRuntime.SetDevice(_deviceIds[index]);
    }

    /// <summary>
    /// Runs an action on each device in parallel, one thread per GPU.
    /// Each thread has its device set as current before the action is invoked.
    /// </summary>
    /// <param name="action">
    /// Action receiving (deviceIndex, deviceId, stream) where deviceIndex is 0-based
    /// within this context.
    /// </param>
    public void ForEachDevice(Action<int, int, CudaStream> action)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(action);

        if (_deviceIds.Length == 1)
        {
            CudaRuntime.SetDevice(_deviceIds[0]);
            action(0, _deviceIds[0], _streams[0]);
            return;
        }

        Parallel.For(0, _deviceIds.Length, i =>
        {
            CudaRuntime.SetDevice(_deviceIds[i]);
            action(i, _deviceIds[i], _streams[i]);
        });
    }

    /// <summary>
    /// Runs an async action on each device concurrently, one task per GPU.
    /// </summary>
    public async Task ForEachDeviceAsync(Func<int, int, CudaStream, Task> action, CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(action);

        var tasks = new Task[_deviceIds.Length];
        for (int i = 0; i < _deviceIds.Length; i++)
        {
            int idx = i;
            tasks[i] = Task.Run(async () =>
            {
                CudaRuntime.SetDevice(_deviceIds[idx]);
                await action(idx, _deviceIds[idx], _streams[idx]).ConfigureAwait(false);
            }, ct);
        }

        await Task.WhenAll(tasks).ConfigureAwait(false);
    }

    /// <summary>
    /// Synchronises all streams in this context, blocking until all GPU work completes.
    /// </summary>
    public void SynchronizeAll()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        for (int i = 0; i < _deviceIds.Length; i++)
        {
            CudaRuntime.SetDevice(_deviceIds[i]);
            _streams[i].Synchronize();
        }
    }

    /// <summary>
    /// Synchronises all streams asynchronously.
    /// </summary>
    public async Task SynchronizeAllAsync(CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var tasks = new Task[_deviceIds.Length];
        for (int i = 0; i < _deviceIds.Length; i++)
        {
            tasks[i] = _streams[i].SynchronizeAsync(ct);
        }

        await Task.WhenAll(tasks).ConfigureAwait(false);
    }

    /// <summary>
    /// Copies data from a device buffer on one GPU to a device buffer on another GPU
    /// using peer-to-peer transfer.
    /// </summary>
    public unsafe void CopyPeerAsync<T>(
        DeviceBuffer<T> dst, int dstDeviceIndex,
        DeviceBuffer<T> src, int srcDeviceIndex,
        CudaStream? stream = null) where T : unmanaged, INumberBase<T>
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        long count = Math.Min(dst.Length, src.Length);
        nuint bytes = (nuint)(count * sizeof(T));
        nint streamHandle = (stream ?? _streams[srcDeviceIndex]).Handle;
        CudaRuntime.MemcpyPeerAsync(
            (void*)dst.Pointer, _deviceIds[dstDeviceIndex],
            (void*)src.Pointer, _deviceIds[srcDeviceIndex],
            bytes, streamHandle);
    }

    /// <summary>Returns whether peer access is possible between two device indices in this context.</summary>
    public bool CanAccessPeer(int fromIndex, int toIndex) =>
        CudaRuntime.DeviceCanAccessPeer(_deviceIds[fromIndex], _deviceIds[toIndex]);

    private void EnablePeerAccess()
    {
        for (int i = 0; i < _deviceIds.Length; i++)
        {
            for (int j = 0; j < _deviceIds.Length; j++)
            {
                if (i == j)
                {
                    continue;
                }

                if (CudaRuntime.DeviceCanAccessPeer(_deviceIds[i], _deviceIds[j]))
                {
                    CudaRuntime.SetDevice(_deviceIds[i]);
                    try
                    {
                        CudaRuntime.DeviceEnablePeerAccess(_deviceIds[j]);
                    }
                    catch (CudaException ex) when (ex.ErrorCode == CudaError.PeerAccessNotEnabled)
                    {
                        // Already enabled, ignore
                    }
                }
            }
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;

        for (int i = 0; i < _streams.Length; i++)
        {
            CudaRuntime.SetDevice(_deviceIds[i]);
            _streams[i].Dispose();
        }
    }
}
