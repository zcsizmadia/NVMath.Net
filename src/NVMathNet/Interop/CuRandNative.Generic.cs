// Copyright (c) 2026 Zoltan Csizmadia. All rights reserved.
// Licensed under the MIT License.

namespace NVMathNet.Interop;

// Additional cuRAND host-API bindings. Signatures verified against curand.h (CUDA 13.x).
public static unsafe partial class CuRandNative
{
    private static nint Fn(string name) => NativeLibraryLoader.GetExport(LibName, name);

    /// <summary>Generates 64-bit unsigned integers (64-bit quasirandom generators).</summary>
    public static void GenerateLongLong(nint generator, ulong* output, nuint count) =>
        Check(((delegate* unmanaged[Cdecl]<nint, ulong*, nuint, CuRandStatus>)Fn("curandGenerateLongLong"))(
            generator, output, count), "curandGenerateLongLong");

    /// <summary>Generates Poisson-distributed 32-bit unsigned integers.</summary>
    public static void GeneratePoisson(nint generator, uint* output, nuint count, double lambda) =>
        Check(((delegate* unmanaged[Cdecl]<nint, uint*, nuint, double, CuRandStatus>)Fn("curandGeneratePoisson"))(
            generator, output, count, lambda), "curandGeneratePoisson");

    /// <summary>Sets the number of dimensions of a quasirandom generator.</summary>
    public static void SetQuasiRandomGeneratorDimensions(nint generator, uint dimensions) =>
        Check(((delegate* unmanaged[Cdecl]<nint, uint, CuRandStatus>)Fn("curandSetQuasiRandomGeneratorDimensions"))(
            generator, dimensions), "curandSetQuasiRandomGeneratorDimensions");
}
