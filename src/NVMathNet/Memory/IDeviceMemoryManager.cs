// Copyright (c) 2026 Zoltan Csizmadia. All rights reserved.
// Licensed under the MIT License.

namespace NVMathNet.Memory;

/// <summary>
/// Allocates and frees GPU device memory for <see cref="DeviceBuffer{T}"/>.
/// Implement this to plug in a pooling or externally managed allocator.
/// Implementations must be thread-safe.
/// </summary>
public interface IDeviceMemoryManager
{
    /// <summary>Allocates <paramref name="sizeInBytes"/> bytes of device memory.</summary>
    /// <param name="sizeInBytes">Requested size; always greater than zero.</param>
    /// <returns>A device pointer that stays valid until passed to <see cref="Free"/>.</returns>
    nint Allocate(nuint sizeInBytes);

    /// <summary>
    /// Releases memory previously returned by <see cref="Allocate"/> on this manager. A pooling manager may hand the
    /// block out again immediately, so callers must make sure no queued GPU work still uses it (synchronise the stream first).
    /// </summary>
    /// <param name="pointer">The device pointer to release.</param>
    void Free(nint pointer);
}
