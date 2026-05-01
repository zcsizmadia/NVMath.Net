// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the Apache 2.0 License.

using NVMathNet.Interop;

namespace NVMathNet;

/// <summary>
/// Wraps a CUDA stream handle and provides async-friendly synchronization.
/// Owns the stream handle; call <see cref="DisposeAsync"/> (or use <c>await using</c>)
/// to destroy it.
/// </summary>
public sealed class CudaStream : IAsyncDisposable, IDisposable
{
    private nint _stream;
    private bool _disposed;

    /// <summary>The raw CUDA stream handle (cudaStream_t).</summary>
    public nint Handle => _stream;

    /// <summary>
    /// Creates a new CUDA stream.
    /// </summary>
    /// <param name="nonBlocking">
    /// When <c>true</c> (default) the stream does not synchronize with the default stream.
    /// </param>
    public CudaStream(bool nonBlocking = true)
    {
        uint flags = nonBlocking ? CudaRuntime.StreamNonBlocking : 0u;
        _stream = CudaRuntime.StreamCreateWithFlags(flags);
    }

    /// <summary>
    /// Creates a <see cref="CudaStream"/> that wraps an externally-owned handle.
    /// Disposal of this instance does <em>not</em> destroy the handle.
    /// </summary>
    public static CudaStream FromHandle(nint handle) => new(handle, owned: false);

    private CudaStream(nint handle, bool owned)
    {
        _stream = handle;
        _disposed = !owned; // if not owned, mark as already-disposed to skip destroy
    }

    /// <summary>
    /// Enqueues an event record on this stream and returns a <see cref="Task"/> that
    /// completes once the event is reached (using a background thread to avoid blocking
    /// the thread-pool).
    /// </summary>
    public Task SynchronizeAsync(CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return Task.Run(() =>
        {
            cancellationToken.ThrowIfCancellationRequested();
            CudaRuntime.StreamSynchronize(_stream);
        }, cancellationToken);
    }

    /// <summary>
    /// Makes this stream wait until <paramref name="event"/> has been recorded.
    /// This is a GPU-side dependency — the CPU is not blocked.
    /// </summary>
    public void WaitEvent(CudaEvent @event)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        CudaRuntime.StreamWaitEvent(_stream, @event.Handle);
    }

    /// <summary>
    /// Synchronizes this stream on the calling thread.
    /// Prefer <see cref="SynchronizeAsync"/> in async code.
    /// </summary>
    public void Synchronize()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        CudaRuntime.StreamSynchronize(_stream);
    }

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
        CudaRuntime.StreamDestroy(_stream);
        _stream = default;
    }
}
