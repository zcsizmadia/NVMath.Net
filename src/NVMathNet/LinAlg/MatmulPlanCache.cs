// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the MIT License.

using System.Collections.Concurrent;

namespace NVMathNet.LinAlg;

/// <summary>
/// Process-wide cache of <see cref="Matmul"/> planning results. When a <see cref="Matmul"/> is planned with the same
/// shape, data types, options, epilog configuration and preferences as an earlier one on the same device, the
/// cuBLASLt heuristic is not queried again and the algorithm chosen by <see cref="Matmul.Autotune"/> (if any) is
/// reused. Caching is enabled by default.
/// </summary>
public static class MatmulPlanCache
{
    internal sealed record Entry(byte[] Algorithms, int Count, int Selected);

    private static readonly ConcurrentDictionary<string, Entry> _entries = new();
    private static long _hits;
    private static long _misses;

    /// <summary>Whether planning results are cached. Default: <c>true</c>. Changing it does not clear the cache.</summary>
    public static bool Enabled { get; set; } = true;

    /// <summary>
    /// Maximum number of cached results (default 1024). When the limit is reached the cache is cleared, so programs
    /// with unboundedly many distinct shapes do not grow memory without limit.
    /// </summary>
    public static int MaxEntries { get; set; } = 1024;

    /// <summary>Number of distinct planning results currently cached.</summary>
    public static int Count => _entries.Count;

    /// <summary>Number of plans that were satisfied from the cache since the process started or <see cref="Clear"/> was called.</summary>
    public static long Hits => Interlocked.Read(ref _hits);

    /// <summary>Number of plans that had to query cuBLASLt while caching was enabled.</summary>
    public static long Misses => Interlocked.Read(ref _misses);

    /// <summary>Removes all cached results and resets the hit and miss counters.</summary>
    public static void Clear()
    {
        _entries.Clear();
        Interlocked.Exchange(ref _hits, 0);
        Interlocked.Exchange(ref _misses, 0);
    }

    internal static bool TryGet(string key, out Entry entry)
    {
        if (_entries.TryGetValue(key, out entry!))
        {
            Interlocked.Increment(ref _hits);
            return true;
        }

        Interlocked.Increment(ref _misses);
        return false;
    }

    internal static void Store(string key, byte[] algorithms, int count)
    {
        if (_entries.Count >= Math.Max(MaxEntries, 1))
        {
            _entries.Clear();
        }

        // Two threads can miss on the same key; the first result wins so an autotuned choice is never overwritten.
        _entries.AddOrUpdate(key, new Entry(algorithms, count, 0), (_, existing) => existing);
    }

    internal static void UpdateSelected(string key, int selected)
    {
        while (_entries.TryGetValue(key, out var existing))
        {
            if (_entries.TryUpdate(key, existing with { Selected = selected }, existing))
            {
                return;
            }
        }
    }
}
