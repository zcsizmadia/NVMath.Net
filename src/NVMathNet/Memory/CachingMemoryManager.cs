// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the MIT License.

namespace NVMathNet.Memory;

/// <summary>
/// A memory manager that keeps freed blocks and hands them out again, avoiding repeated
/// <c>cudaMalloc</c>/<c>cudaFree</c> calls (which are slow and synchronise the device).
/// Requests up to 1 GiB are rounded up to a power of two (minimum 512 bytes) so blocks are
/// interchangeable; larger requests are cached by exact size.
/// </summary>
public sealed class CachingMemoryManager : IDeviceMemoryManager, IDisposable
{
    private const nuint MinBlock = 512;
    private const nuint MaxRounded = 1u << 30;

    private readonly IDeviceMemoryManager _inner;
    private readonly object _gate = new();
    private readonly Dictionary<nuint, Stack<nint>> _free = [];
    private readonly Dictionary<nint, nuint> _live = [];
    private long _cachedBytes;
    private long _inUseBytes;
    private bool _disposed;

    /// <summary>Creates a caching manager on top of <paramref name="inner"/>.</summary>
    /// <param name="inner">The underlying allocator; <c>null</c> uses <see cref="CudaMallocMemoryManager.Instance"/>.</param>
    public CachingMemoryManager(IDeviceMemoryManager? inner = null)
    {
        _inner = inner ?? CudaMallocMemoryManager.Instance;
    }

    /// <summary>Bytes currently held in the cache (allocated from the inner manager but not in use).</summary>
    public long CachedBytes
    {
        get
        {
            lock (_gate)
            {
                return _cachedBytes;
            }
        }
    }

    /// <summary>Bytes currently handed out to callers (including rounding).</summary>
    public long InUseBytes
    {
        get
        {
            lock (_gate)
            {
                return _inUseBytes;
            }
        }
    }

    /// <inheritdoc/>
    public nint Allocate(nuint sizeInBytes)
    {
        nuint block = RoundUp(sizeInBytes);
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_free.TryGetValue(block, out var stack) && stack.TryPop(out nint cached))
            {
                _cachedBytes -= (long)block;
                _inUseBytes += (long)block;
                _live[cached] = block;
                return cached;
            }
        }

        // Allocate outside the lock: cudaMalloc can be slow.
        nint ptr = _inner.Allocate(block);
        lock (_gate)
        {
            _live[ptr] = block;
            _inUseBytes += (long)block;
        }

        return ptr;
    }

    /// <inheritdoc/>
    public void Free(nint pointer)
    {
        lock (_gate)
        {
            if (!_live.Remove(pointer, out nuint block))
            {
                throw new ArgumentException("The pointer was not allocated by this manager.", nameof(pointer));
            }

            _inUseBytes -= (long)block;
            if (_disposed)
            {
                _inner.Free(pointer);
                return;
            }

            if (!_free.TryGetValue(block, out var stack))
            {
                _free[block] = stack = new Stack<nint>();
            }

            stack.Push(pointer);
            _cachedBytes += (long)block;
        }
    }

    /// <summary>Returns all cached (unused) blocks to the inner manager.</summary>
    public void Trim()
    {
        lock (_gate)
        {
            foreach (var stack in _free.Values)
            {
                while (stack.TryPop(out nint ptr))
                {
                    _inner.Free(ptr);
                }
            }

            _free.Clear();
            _cachedBytes = 0;
        }
    }

    /// <summary>
    /// Trims the cache and stops caching. Blocks still in use are released to the inner manager
    /// when they are freed.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        Trim();
        lock (_gate)
        {
            _disposed = true;
        }
    }

    private static nuint RoundUp(nuint size)
    {
        if (size > MaxRounded)
        {
            return size;
        }

        nuint block = MinBlock;
        while (block < size)
        {
            block <<= 1;
        }

        return block;
    }
}
