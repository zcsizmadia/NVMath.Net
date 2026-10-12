// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
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

    /// <summary>Releases memory previously returned by <see cref="Allocate"/> on this manager.</summary>
    /// <param name="pointer">The device pointer to release.</param>
    void Free(nint pointer);
}
