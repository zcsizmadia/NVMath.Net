// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the Apache 2.0 License.

using System.Runtime.InteropServices;

namespace NVMathNet.Interop;

/// <summary>
/// Cross-platform native library loader using <see cref="NativeLibrary"/> so that
/// both Windows (<c>LoadLibrary</c>) and Linux (<c>dlopen</c>) are supported
/// without any additional code.
/// </summary>
public static class NativeLibraryLoader
{
    private static readonly Dictionary<string, nint> _handles = new();
    private static readonly object _lock = new();

    /// <summary>
    /// Returns a function pointer for <paramref name="symbol"/> inside
    /// <paramref name="libraryName"/>, loading the library on first use.
    /// </summary>
    public static nint GetExport(string libraryName, string symbol)
    {
        nint lib = GetOrLoad(libraryName);
        if (!NativeLibrary.TryGetExport(lib, symbol, out nint ptr))
            throw new EntryPointNotFoundException(
                $"Symbol '{symbol}' not found in '{libraryName}'.");
        return ptr;
    }

    /// <summary>Loads or retrieves a cached handle for <paramref name="libraryName"/>.</summary>
    public static nint GetOrLoad(string libraryName)
    {
        lock (_lock)
        {
            if (_handles.TryGetValue(libraryName, out nint cached))
                return cached;

            nint handle = NativeLibrary.Load(libraryName);
            _handles[libraryName] = handle;
            return handle;
        }
    }

    /// <summary>Releases all loaded library handles.</summary>
    public static void UnloadAll()
    {
        lock (_lock)
        {
            foreach (nint h in _handles.Values)
                NativeLibrary.Free(h);
            _handles.Clear();
        }
    }
}
