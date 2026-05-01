// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the MIT License.

using System.Runtime.InteropServices;

namespace NVMathNet.Interop;

/// <summary>
/// Cross-platform native library loader using <see cref="NativeLibrary"/> so that
/// both Windows (<c>LoadLibrary</c>) and Linux (<c>dlopen</c>) are supported
/// without any additional code.
/// </summary>
public static class NativeLibraryLoader
{
    private static readonly Dictionary<string, nint> _handles = [];
    private static readonly HashSet<string> _addedDirs = new(StringComparer.OrdinalIgnoreCase);
#if NET9_0_OR_GREATER    
    private static readonly Lock _lock = new();
#else
    private static readonly object _lock = new();
#endif

    /// <summary>
    /// Returns a function pointer for <paramref name="symbol"/> inside
    /// <paramref name="libraryName"/>, loading the library on first use.
    /// </summary>
    public static nint GetExport(string libraryName, string symbol)
    {
        nint lib = GetOrLoad(libraryName);
        return !NativeLibrary.TryGetExport(lib, symbol, out nint ptr) ?
            throw new EntryPointNotFoundException($"Symbol '{symbol}' not found in '{libraryName}'.") :
            ptr;
    }

    /// <summary>Loads or retrieves a cached handle for <paramref name="libraryName"/>.</summary>
    public static nint GetOrLoad(string libraryName)
    {
        lock (_lock)
        {
            if (_handles.TryGetValue(libraryName, out nint cached))
            {
                return cached;
            }

            if (!NativeLibrary.TryLoad(libraryName, out nint handle))
            {
                handle = LoadFromKnownPaths(libraryName);
            }

            _handles[libraryName] = handle;
            return handle;
        }
    }

    /// <summary>
    /// Probes well-known CUDA Toolkit installation directories when the library
    /// is not on the default search path.
    /// </summary>
    private static nint LoadFromKnownPaths(string libraryName)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            // Register all known CUDA directories so transitive dependencies resolve
            RegisterWindowsDllDirectories();

            // CUDA Toolkit on Windows: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.Y\bin\x64\
            string baseDir = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles),
                "NVIDIA GPU Computing Toolkit", "CUDA");

            if (Directory.Exists(baseDir))
            {
                foreach (string versionDir in Directory.GetDirectories(baseDir, "v*").OrderDescending())
                {
                    // Try bin/x64 first (CUDA 13+), then bin (older layouts)
                    string[] subDirs = [Path.Combine(versionDir, "bin", "x64"), Path.Combine(versionDir, "bin")];
                    foreach (string dir in subDirs)
                    {
                        string fullPath = Path.Combine(dir, libraryName);
                        if (NativeLibrary.TryLoad(fullPath, out nint handle))
                        {
                            return handle;
                        }
                    }
                }
            }

            // cuTENSOR standalone install: C:\Program Files\NVIDIA cuTENSOR\vX.Y\bin\<cuda-major>
            string cuTensorBase = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles),
                "NVIDIA cuTENSOR");

            if (Directory.Exists(cuTensorBase))
            {
                foreach (string versionDir in Directory.GetDirectories(cuTensorBase, "v*").OrderDescending())
                {
                    string binDir = Path.Combine(versionDir, "bin");
                    if (Directory.Exists(binDir))
                    {
                        // Try versioned subdirs (e.g. bin/13) then bin itself
                        foreach (string subDir in Directory.GetDirectories(binDir).OrderDescending())
                        {
                            string fullPath = Path.Combine(subDir, libraryName);
                            if (NativeLibrary.TryLoad(fullPath, out nint handle))
                            {
                                return handle;
                            }
                        }

                        string binPath = Path.Combine(binDir, libraryName);
                        if (NativeLibrary.TryLoad(binPath, out nint binHandle))
                        {
                            return binHandle;
                        }
                    }
                }
            }

            // Also check CUTENSOR_ROOT environment variable
            string? cudaPath = Environment.GetEnvironmentVariable("CUTENSOR_ROOT");
            if (cudaPath is not null)
            {
                string fullPath = Path.Combine(cudaPath, "lib", libraryName);
                if (NativeLibrary.TryLoad(fullPath, out nint handle))
                {
                    return handle;
                }
            }
        }
        else
        {
            // Linux: try /usr/local/cuda/lib64
            string[] linuxPaths =
            [
                "/usr/local/cuda/lib64",
                "/usr/lib/x86_64-linux-gnu",
            ];
            foreach (string dir in linuxPaths)
            {
                string fullPath = Path.Combine(dir, libraryName);
                if (NativeLibrary.TryLoad(fullPath, out nint handle))
                {
                    return handle;
                }
            }
        }

        // Fall back to the original load which will throw a descriptive exception
        return NativeLibrary.Load(libraryName);
    }

    /// <summary>
    /// Registers well-known CUDA directories via <c>AddDllDirectory</c> so that
    /// transitive native dependencies (e.g. cublas, cudart) can be resolved
    /// when loading libraries like cuTENSOR by full path.
    /// </summary>
    private static void RegisterWindowsDllDirectories()
    {
        string programFiles = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles);

        // CUDA Toolkit: bin/x64 and bin
        string cudaBase = Path.Combine(programFiles, "NVIDIA GPU Computing Toolkit", "CUDA");
        if (Directory.Exists(cudaBase))
        {
            foreach (string versionDir in Directory.GetDirectories(cudaBase, "v*"))
            {
                TryAddDllDir(Path.Combine(versionDir, "bin", "x64"));
                TryAddDllDir(Path.Combine(versionDir, "bin"));
            }
        }

        // cuTENSOR: bin/<cuda-major> and bin
        string cuTensorBase = Path.Combine(programFiles, "NVIDIA cuTENSOR");
        if (Directory.Exists(cuTensorBase))
        {
            foreach (string versionDir in Directory.GetDirectories(cuTensorBase, "v*"))
            {
                string binDir = Path.Combine(versionDir, "bin");
                if (Directory.Exists(binDir))
                {
                    foreach (string subDir in Directory.GetDirectories(binDir))
                    {
                        TryAddDllDir(subDir);
                    }

                    TryAddDllDir(binDir);
                }
            }
        }
    }

    private static void TryAddDllDir(string dir)
    {
        if (Directory.Exists(dir) && _addedDirs.Add(dir))
        {
            string path = Environment.GetEnvironmentVariable("PATH") ?? string.Empty;
            Environment.SetEnvironmentVariable("PATH", dir + Path.PathSeparator + path);
        }
    }

    /// <summary>Releases all loaded library handles.</summary>
    public static void UnloadAll()
    {
        lock (_lock)
        {
            foreach (nint h in _handles.Values)
            {
                NativeLibrary.Free(h);
            }

            _handles.Clear();
        }
    }
}
