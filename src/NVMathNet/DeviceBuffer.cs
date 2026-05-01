// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the MIT License.

using System.Numerics;
using System.Runtime.CompilerServices;

using NVMathNet.Interop;

namespace NVMathNet;

/// <summary>
/// A strongly-typed buffer of <typeparamref name="T"/> residing in GPU device memory.
/// Supports zero-copy async transfers with host-pinned memory.
/// Implements <see cref="IDisposable"/> â€” dispose to free the device allocation.
/// </summary>
/// <typeparam name="T">Unmanaged numeric element type (float, double, <see cref="System.Numerics.Complex"/>, etc.).</typeparam>
public sealed class DeviceBuffer<T> : IDisposable, IAsyncDisposable where T : unmanaged, INumberBase<T>
{
    private nint _ptr;
    private bool _disposed;

    /// <summary>Number of elements.</summary>
    public long Length { get; }

    /// <summary>Total size in bytes.</summary>
    public nuint SizeInBytes => (nuint)(Length * Unsafe.SizeOf<T>());

    /// <summary>Raw device pointer as <see cref="nint"/>.</summary>
    public nint PointerAsInt
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _ptr;
        }
    }

    /// <summary>Raw device pointer as <c>void*</c>.</summary>
    public unsafe void* Pointer => (void*)PointerAsInt;

    /// <summary>
    /// Allocates a device buffer of <paramref name="length"/> elements
    /// using <c>cudaMalloc</c>.
    /// </summary>
    public unsafe DeviceBuffer(long length, bool allocate = true)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(length);
        Length = length;
        _ptr = allocate ? (nint)CudaRuntime.Malloc((nuint)(length * Unsafe.SizeOf<T>())) : 0;
    }

    /// <summary>
    /// Allocates a device buffer asynchronously on <paramref name="stream"/>
    /// using <c>cudaMallocAsync</c>.
    /// </summary>
    public static unsafe DeviceBuffer<T> AllocAsync(long length, CudaStream stream)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(length);
        var buf = new DeviceBuffer<T>(length, allocate: false)
        {
            _ptr = (nint)CudaRuntime.MallocAsync((nuint)(length * Unsafe.SizeOf<T>()), stream.Handle)
        };
        return buf;
    }

    /// <summary>
    /// Copies data from a managed <see cref="ReadOnlySpan{T}"/> to this device buffer.
    /// Uses pinned memory for a zero-copy hostâ†’device transfer.
    /// </summary>
    public unsafe void CopyFrom(ReadOnlySpan<T> source, CudaStream? stream = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (source.Length > Length)
        {
            throw new ArgumentException($"Source span ({source.Length}) is larger than the device buffer ({Length}).");
        }

        nuint byteCount = (nuint)(source.Length * Unsafe.SizeOf<T>());
        fixed (T* pSrc = source)
        {
            if (stream is null)
            {
                CudaRuntime.Memcpy((void*)_ptr, pSrc, byteCount, CudaMemcpyKind.HostToDevice);
            }
            else
            {
                CudaRuntime.MemcpyAsync((void*)_ptr, pSrc, byteCount, CudaMemcpyKind.HostToDevice, stream.Handle);
            }
        }
    }

    /// <summary>
    /// Copies data from this device buffer into a managed <see cref="Span{T}"/>.
    /// Uses pinned memory for a zero-copy deviceâ†’host transfer.
    /// </summary>
    public unsafe void CopyTo(Span<T> destination, CudaStream? stream = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (destination.Length < Length)
        {
            throw new ArgumentException($"Destination span ({destination.Length}) is smaller than the device buffer ({Length}).");
        }

        nuint byteCount = (nuint)(Length * Unsafe.SizeOf<T>());
        fixed (T* pDst = destination)
        {
            if (stream is null)
            {
                CudaRuntime.Memcpy(pDst, (void*)_ptr, byteCount, CudaMemcpyKind.DeviceToHost);
            }
            else
            {
                CudaRuntime.MemcpyAsync(pDst, (void*)_ptr, byteCount, CudaMemcpyKind.DeviceToHost, stream.Handle);
            }
        }
    }

    /// <summary>
    /// Asynchronously copies from a managed array to device, returning a
    /// <see cref="Task"/> that completes when the GPU has consumed the data.
    /// </summary>
    public async Task CopyFromAsync(T[] source, CudaStream stream, CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        CopyFrom(source.AsSpan(), stream);
        await stream.SynchronizeAsync(ct).ConfigureAwait(false);
    }

    /// <summary>
    /// Asynchronously copies from device to a managed array, returning a
    /// <see cref="Task"/> that completes when the data is available on the host.
    /// </summary>
    public async Task CopyToAsync(T[] destination, CudaStream stream, CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        CopyTo(destination.AsSpan(), stream);
        await stream.SynchronizeAsync(ct).ConfigureAwait(false);
    }

    /// <summary>Copies from another device buffer using <c>cudaMemcpyAsync</c>.</summary>
    public unsafe void CopyFrom(DeviceBuffer<T> other, CudaStream stream)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ObjectDisposedException.ThrowIf(other._disposed, other);
        if (other.Length > Length)
        {
            throw new ArgumentException("Source buffer is larger than destination.");
        }

        nuint byteCount = (nuint)(other.Length * Unsafe.SizeOf<T>());
        CudaRuntime.MemcpyAsync((void*)_ptr, (void*)other._ptr, byteCount, CudaMemcpyKind.DeviceToDevice, stream.Handle);
    }

    /// <summary>Fills the buffer with zero bytes.</summary>
    public unsafe void Clear(CudaStream? stream = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (stream is null)
        {
            CudaRuntime.Memset((void*)_ptr, 0, SizeInBytes);
        }
        else
        {
            CudaRuntime.MemsetAsync((void*)_ptr, 0, SizeInBytes, stream.Handle);
        }
    }

    /// <summary>
    /// Returns a new T[] containing a snapshot of the device data.
    /// Blocks until transfer completes.
    /// </summary>
    public T[] ToArray()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        T[] result = new T[Length];
        CopyTo(result.AsSpan());
        return result;
    }

    /// <summary>
    /// Releases the <see langword="unmanaged"/> CUDA memory allocated by this instance.
    /// </summary>
    public unsafe void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        if (_ptr != 0)
        {
            CudaRuntime.Free((void*)_ptr);
            _ptr = 0;
        }
    }

    /// <summary>
    /// Asynchronously disposes the device buffer, freeing the underlying device memory.
    /// </summary>
    /// <returns>A <see cref="ValueTask"/> that represents the asynchronous dispose operation.</returns>
    public ValueTask DisposeAsync()
    {
        Dispose();
        return ValueTask.CompletedTask;
    }
}

