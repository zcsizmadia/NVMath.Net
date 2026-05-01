// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the Apache 2.0 License.

using NVMathNet.Interop;

namespace NVMathNet;

/// <summary>
/// Wraps a CUDA event handle.  Supports async waiting via <see cref="SynchronizeAsync"/>.
/// </summary>
public sealed class CudaEvent : IAsyncDisposable, IDisposable
{
    private nint _event;
    private bool _disposed;

    /// <summary>The raw CUDA event handle (cudaEvent_t).</summary>
    public nint Handle => _event;

    /// <summary>
    /// Creates a new CUDA event.
    /// </summary>
    /// <param name="disableTiming">
    /// When <c>true</c> (default) timing is disabled, resulting in lower overhead.
    /// Set to <c>false</c> to enable <see cref="ElapsedMilliseconds"/>.
    /// </param>
    public CudaEvent(bool disableTiming = true)
    {
        uint flags = disableTiming ? CudaRuntime.EventDisableTiming : 0u;
        _event = CudaRuntime.EventCreateWithFlags(flags);
    }

    /// <summary>Records this event on <paramref name="stream"/>.</summary>
    public void Record(CudaStream stream)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        CudaRuntime.EventRecord(_event, stream.Handle);
    }

    /// <summary>
    /// Records this event on the null (default) stream.
    /// </summary>
    public void Record()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        CudaRuntime.EventRecord(_event, default);
    }

    /// <summary>
    /// Returns a <see cref="Task"/> that completes once all work preceding the last
    /// <see cref="Record()"/> call has finished on the GPU.
    /// </summary>
    public Task SynchronizeAsync(CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return Task.Run(() =>
        {
            cancellationToken.ThrowIfCancellationRequested();
            CudaRuntime.EventSynchronize(_event);
        }, cancellationToken);
    }

    /// <summary>
    /// Blocks the calling thread until this event completes.
    /// Prefer <see cref="SynchronizeAsync"/> in async code.
    /// </summary>
    public void Synchronize()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        CudaRuntime.EventSynchronize(_event);
    }

    /// <summary>
    /// Returns the elapsed time in milliseconds between <paramref name="start"/>
    /// and this event.  Both events must have been created with timing enabled.
    /// </summary>
    public float ElapsedMilliseconds(CudaEvent start) =>
        CudaRuntime.EventElapsedTime(start.Handle, _event);

    /// <inheritdoc/>
    public ValueTask DisposeAsync()
    {
        Dispose();
        return ValueTask.CompletedTask;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        CudaRuntime.EventDestroy(_event);
        _event = default;
    }
}
