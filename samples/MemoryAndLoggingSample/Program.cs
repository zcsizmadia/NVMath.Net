// MemoryAndLoggingSample — pluggable device memory managers and the logging hook.
// Requires a CUDA-capable GPU at runtime.
// The sample verifies its own results and returns a non-zero exit code on failure.

using NVMathNet;
using NVMathNet.Fft;
using NVMathNet.LinAlg;
using NVMathNet.Memory;

Console.WriteLine("=== NVMath.Net Memory Manager & Logging Sample ===");
int failures = 0;

void Check(string what, bool ok)
{
    failures += ok ? 0 : 1;
    Console.WriteLine($"  [{(ok ? "PASS" : "FAIL")}] {what}");
}

// ── 1. A caching memory manager ──────────────────────────────────────────────
// cudaMalloc/cudaFree are slow and synchronise the device. CachingMemoryManager keeps freed blocks
// (rounded up to a power of two) and hands them out again.
{
    Console.WriteLine("\n--- CachingMemoryManager ---");
    using var cache = new CachingMemoryManager();

    nint firstPointer;
    using (var a = new DeviceBuffer<float>(1000, cache))          // 4000 bytes -> a 4096-byte block
    {
        firstPointer = a.PointerAsInt;
        Console.WriteLine($"  in use: {cache.InUseBytes} bytes, cached: {cache.CachedBytes} bytes");
    }

    Console.WriteLine($"  after dispose -> in use: {cache.InUseBytes} bytes, cached: {cache.CachedBytes} bytes");
    using (var b = new DeviceBuffer<float>(1024, cache))          // same 4096-byte bucket
    {
        Check("the freed block was reused (same device pointer)", b.PointerAsInt == firstPointer);
    }

    cache.Trim();                                                  // give cached blocks back to CUDA
    Check("Trim() released the cache", cache.CachedBytes == 0);
}

// ── 2. Installing a manager process-wide ─────────────────────────────────────
// DeviceMemory.Default is used by `new DeviceBuffer<T>(n)` AND by the library's internal workspaces
// (cuFFT, cuBLASLt, cuTENSOR, cuSPARSE scratch memory). A buffer always returns memory to the manager it came from.
{
    Console.WriteLine("\n--- DeviceMemory.Default ---");
    var previous = DeviceMemory.Default;
    using var cache = new CachingMemoryManager();
    DeviceMemory.Default = cache;
    try
    {
        for (int round = 0; round < 3; round++)
        {
            using var buffer = new DeviceBuffer<double>(1 << 16);
        }

        Console.WriteLine($"  three allocate/free rounds left {cache.CachedBytes} bytes cached (one reusable block)");
        Check("repeated allocations are served from the cache", cache.CachedBytes == (1 << 16) * sizeof(double));
    }
    finally
    {
        DeviceMemory.Default = previous;                            // buffers created above stay valid
    }
}

// ── 3. The logging hook ──────────────────────────────────────────────────────
// Nothing is logged until a handler is installed. Bridge the callback to ILogger, Serilog, ... as needed;
// here it simply collects messages and prints them.
{
    Console.WriteLine("\n--- NVMathLog ---");
    var lines = new List<string>();
    NVMathLog.MinimumLevel = NVMathLogLevel.Debug;                 // Debug shows plan details; Information is the default
    NVMathLog.Handler = (level, category, message) =>
    {
        lock (lines)
        {
            lines.Add($"[{level}] {category}: {message}");
        }
    };

    try
    {
        using (var fft = new Fft([1024], options: new FftOptions { Blocking = true }))
        {
            await fft.PlanAsync();
        }

        using var matmul = new Matmul(64, 64, 64);
        await matmul.PlanAsync(new MatmulPlanPreferences { RequestedAlgoCount = 4 });
    }
    finally
    {
        NVMathLog.Handler = null;                                   // stop logging
    }

    foreach (string line in lines)
    {
        Console.WriteLine("  " + line);
    }

    Check("plan details were reported", lines.Any(l => l.Contains("Fft:")) && lines.Any(l => l.Contains("Matmul:")));
}

Console.WriteLine(failures == 0 ? "\nAll checks passed." : $"\n{failures} check(s) FAILED.");
return failures == 0 ? 0 : 1;
