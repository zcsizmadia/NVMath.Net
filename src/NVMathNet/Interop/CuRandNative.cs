// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the MIT License.

using System.Runtime.InteropServices;

namespace NVMathNet.Interop;

/// <summary>cuRAND status codes.</summary>
public enum CuRandStatus : int
{
    /// <summary>No errors.</summary>
    Success = 0,
    /// <summary>Header/library version mismatch.</summary>
    VersionMismatch = 100,
    /// <summary>Generator not initialised.</summary>
    NotInitialized = 101,
    /// <summary>Memory allocation failed.</summary>
    AllocationFailed = 102,
    /// <summary>Generator is wrong type.</summary>
    TypeError = 103,
    /// <summary>Argument out of range.</summary>
    OutOfRange = 104,
    /// <summary>Length not a multiple of dimension.</summary>
    LengthNotMultiple = 105,
    /// <summary>GPU does not have double precision required.</summary>
    DoublePrecisionRequired = 106,
    /// <summary>Kernel launch failure.</summary>
    LaunchFailure = 201,
    /// <summary>Pre-existing failure on library entry.</summary>
    PreexistingFailure = 202,
    /// <summary>CUDA initialisation failed.</summary>
    InitializationFailed = 203,
    /// <summary>Architecture mismatch.</summary>
    ArchMismatch = 204,
    /// <summary>Internal library error.</summary>
    InternalError = 999,
}

/// <summary>cuRAND random number generator types.</summary>
public enum CuRandRngType : int
{
    /// <summary>Default pseudorandom generator (XORWOW).</summary>
    PseudoDefault = 100,
    /// <summary>XORWOW pseudorandom generator.</summary>
    PseudoXorwow = 101,
    /// <summary>MRG32k3a pseudorandom generator.</summary>
    PseudoMrg32k3a = 121,
    /// <summary>Mersenne Twister MTGP32 pseudorandom generator.</summary>
    PseudoMtgp32 = 141,
    /// <summary>Mersenne Twister MT19937 pseudorandom generator.</summary>
    PseudoMt19937 = 142,
    /// <summary>PHILOX-4x32-10 pseudorandom generator.</summary>
    PseudoPhilox4_32_10 = 161,
    /// <summary>Default quasirandom generator (Sobol32).</summary>
    QuasiDefault = 200,
    /// <summary>Sobol32 quasirandom generator.</summary>
    QuasiSobol32 = 201,
    /// <summary>Scrambled Sobol32 quasirandom generator.</summary>
    QuasiScrambledSobol32 = 202,
    /// <summary>Sobol64 quasirandom generator.</summary>
    QuasiSobol64 = 203,
    /// <summary>Scrambled Sobol64 quasirandom generator.</summary>
    QuasiScrambledSobol64 = 204,
}

/// <summary>
/// Raw P/Invoke bindings for cuRAND.
/// </summary>
public static unsafe class CuRandNative
{
    private const string LibWindows = "curand64_10.dll";
    private const string LibLinux   = "libcurand.so.10";

    private static readonly string LibName =
        RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? LibWindows : LibLinux;

    // ── Delegate types ───────────────────────────────────────────────────────

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuRandStatus curandCreateGeneratorDelegate(nint* generator, CuRandRngType rngType);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuRandStatus curandDestroyGeneratorDelegate(nint generator);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuRandStatus curandSetStreamDelegate(nint generator, nint stream);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuRandStatus curandSetPseudoRandomGeneratorSeedDelegate(nint generator, ulong seed);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuRandStatus curandSetGeneratorOffsetDelegate(nint generator, ulong offset);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuRandStatus curandGenerateUniformDelegate(nint generator, float* outputPtr, nuint num);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuRandStatus curandGenerateUniformDoubleDelegate(nint generator, double* outputPtr, nuint num);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuRandStatus curandGenerateNormalDelegate(nint generator, float* outputPtr, nuint n, float mean, float stddev);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuRandStatus curandGenerateNormalDoubleDelegate(nint generator, double* outputPtr, nuint n, double mean, double stddev);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuRandStatus curandGenerateLogNormalDelegate(nint generator, float* outputPtr, nuint n, float mean, float stddev);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuRandStatus curandGenerateLogNormalDoubleDelegate(nint generator, double* outputPtr, nuint n, double mean, double stddev);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CuRandStatus curandGenerateDelegate(nint generator, uint* outputPtr, nuint num);

    // ── Lazy-loaded delegates ────────────────────────────────────────────────

    private static T Load<T>(string name) where T : Delegate =>
        Marshal.GetDelegateForFunctionPointer<T>(NativeLibraryLoader.GetExport(LibName, name));

    private static readonly Lazy<curandCreateGeneratorDelegate>                _create       = new(() => Load<curandCreateGeneratorDelegate>("curandCreateGenerator"));
    private static readonly Lazy<curandDestroyGeneratorDelegate>               _destroy      = new(() => Load<curandDestroyGeneratorDelegate>("curandDestroyGenerator"));
    private static readonly Lazy<curandSetStreamDelegate>                      _setStream    = new(() => Load<curandSetStreamDelegate>("curandSetStream"));
    private static readonly Lazy<curandSetPseudoRandomGeneratorSeedDelegate>   _setSeed      = new(() => Load<curandSetPseudoRandomGeneratorSeedDelegate>("curandSetPseudoRandomGeneratorSeed"));
    private static readonly Lazy<curandSetGeneratorOffsetDelegate>             _setOffset    = new(() => Load<curandSetGeneratorOffsetDelegate>("curandSetGeneratorOffset"));
    private static readonly Lazy<curandGenerateUniformDelegate>                _genUniform   = new(() => Load<curandGenerateUniformDelegate>("curandGenerateUniform"));
    private static readonly Lazy<curandGenerateUniformDoubleDelegate>          _genUniformD  = new(() => Load<curandGenerateUniformDoubleDelegate>("curandGenerateUniformDouble"));
    private static readonly Lazy<curandGenerateNormalDelegate>                 _genNormal    = new(() => Load<curandGenerateNormalDelegate>("curandGenerateNormal"));
    private static readonly Lazy<curandGenerateNormalDoubleDelegate>           _genNormalD   = new(() => Load<curandGenerateNormalDoubleDelegate>("curandGenerateNormalDouble"));
    private static readonly Lazy<curandGenerateLogNormalDelegate>              _genLogNormal  = new(() => Load<curandGenerateLogNormalDelegate>("curandGenerateLogNormal"));
    private static readonly Lazy<curandGenerateLogNormalDoubleDelegate>        _genLogNormalD = new(() => Load<curandGenerateLogNormalDoubleDelegate>("curandGenerateLogNormalDouble"));
    private static readonly Lazy<curandGenerateDelegate>                       _generate      = new(() => Load<curandGenerateDelegate>("curandGenerate"));

    // ── Public helpers ───────────────────────────────────────────────────────

    /// <summary>
    /// Throws <see cref="CuRandException"/> if <paramref name="status"/> is not
    /// <see cref="CuRandStatus.Success"/>.
    /// </summary>
    public static void Check(CuRandStatus status, string? context = null)
    {
        if (status == CuRandStatus.Success)
        {
            return;
        }

        string msg = $"cuRAND error {status} ({(int)status})";
        throw new CuRandException((int)status, context is null ? msg : $"{context}: {msg}");
    }

    // ── Public API ───────────────────────────────────────────────────────────

    /// <summary>Creates a cuRAND generator.</summary>
    public static nint CreateGenerator(CuRandRngType rngType = CuRandRngType.PseudoDefault)
    {
        nint gen;
        Check(_create.Value(&gen, rngType), "curandCreateGenerator");
        return gen;
    }

    /// <summary>Destroys a cuRAND generator.</summary>
    public static void DestroyGenerator(nint generator) =>
        Check(_destroy.Value(generator), "curandDestroyGenerator");

    /// <summary>Associates a CUDA stream with the generator.</summary>
    public static void SetStream(nint generator, nint stream) =>
        Check(_setStream.Value(generator, stream), "curandSetStream");

    /// <summary>Sets the seed for a pseudorandom generator.</summary>
    public static void SetSeed(nint generator, ulong seed) =>
        Check(_setSeed.Value(generator, seed), "curandSetPseudoRandomGeneratorSeed");

    /// <summary>Sets the offset for a generator.</summary>
    public static void SetOffset(nint generator, ulong offset) =>
        Check(_setOffset.Value(generator, offset), "curandSetGeneratorOffset");

    /// <summary>Generates uniformly distributed single-precision floats in (0,1].</summary>
    public static void GenerateUniform(nint generator, float* output, nuint count) =>
        Check(_genUniform.Value(generator, output, count), "curandGenerateUniform");

    /// <summary>Generates uniformly distributed double-precision floats in (0,1].</summary>
    public static void GenerateUniformDouble(nint generator, double* output, nuint count) =>
        Check(_genUniformD.Value(generator, output, count), "curandGenerateUniformDouble");

    /// <summary>Generates normally distributed single-precision floats.</summary>
    public static void GenerateNormal(nint generator, float* output, nuint count, float mean, float stddev) =>
        Check(_genNormal.Value(generator, output, count, mean, stddev), "curandGenerateNormal");

    /// <summary>Generates normally distributed double-precision floats.</summary>
    public static void GenerateNormalDouble(nint generator, double* output, nuint count, double mean, double stddev) =>
        Check(_genNormalD.Value(generator, output, count, mean, stddev), "curandGenerateNormalDouble");

    /// <summary>Generates log-normally distributed single-precision floats.</summary>
    public static void GenerateLogNormal(nint generator, float* output, nuint count, float mean, float stddev) =>
        Check(_genLogNormal.Value(generator, output, count, mean, stddev), "curandGenerateLogNormal");

    /// <summary>Generates log-normally distributed double-precision floats.</summary>
    public static void GenerateLogNormalDouble(nint generator, double* output, nuint count, double mean, double stddev) =>
        Check(_genLogNormalD.Value(generator, output, count, mean, stddev), "curandGenerateLogNormalDouble");

    /// <summary>Generates 32-bit unsigned integers.</summary>
    public static void Generate(nint generator, uint* output, nuint count) =>
        Check(_generate.Value(generator, output, count), "curandGenerate");
}
