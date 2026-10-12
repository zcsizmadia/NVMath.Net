// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the MIT License.

namespace NVMathNet.Memory;

/// <summary>
/// A scratch allocation taken from <see cref="DeviceMemory.Default"/> and returned to the manager
/// that provided it. <c>default</c> is a valid empty workspace, and disposing twice is harmless only
/// if each copy is disposed once, so keep it in a single local or field.
/// </summary>
internal readonly unsafe struct Workspace : IDisposable
{
    private readonly IDeviceMemoryManager? _manager;
    private readonly nint _pointer;

    /// <summary>Allocates <paramref name="sizeInBytes"/> bytes; a size of zero yields an empty workspace.</summary>
    public Workspace(nuint sizeInBytes)
    {
        if (sizeInBytes == 0)
        {
            _manager = null;
            _pointer = 0;
        }
        else
        {
            _manager = DeviceMemory.Default;
            _pointer = _manager.Allocate(sizeInBytes);
        }
    }

    /// <summary>The device pointer, or null for an empty workspace.</summary>
    public void* Pointer => (void*)_pointer;

    /// <summary>The device pointer as an integer.</summary>
    public nint PointerAsInt => _pointer;

    /// <summary>Returns the memory to its manager.</summary>
    public void Dispose()
    {
        if (_pointer != 0)
        {
            _manager!.Free(_pointer);
        }
    }
}
