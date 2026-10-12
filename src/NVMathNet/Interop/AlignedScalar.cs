// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the MIT License.

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace NVMathNet.Interop;

/// <summary>
/// Places host scalars (alpha, beta, ...) at 16-byte aligned addresses before they are passed to native code.
/// <c>cuDoubleComplex</c> is a <c>double2</c> and native routines may read it with aligned 128-bit loads, but
/// <see cref="System.Numerics.Complex"/> only guarantees 8-byte alignment, so a scalar that merely lives in a
/// local variable can crash the library depending on where the JIT puts it on the stack.
/// </summary>
internal static unsafe class AlignedScalar
{
    /// <summary>The number of bytes a caller must <c>stackalloc</c> for one scalar of up to 16 bytes.</summary>
    public const int StorageSize = 32;

    /// <summary>
    /// Writes <paramref name="value"/> to a 16-byte aligned address inside <paramref name="storage"/>
    /// (at least <see cref="StorageSize"/> bytes, for example <c>stackalloc byte[32]</c> in the caller's frame)
    /// and returns that address. The pointer is valid as long as the storage is.
    /// </summary>
    public static T* Store<T>(Span<byte> storage, T value) where T : unmanaged
    {
        byte* raw = (byte*)Unsafe.AsPointer(ref MemoryMarshal.GetReference(storage));
        T* aligned = (T*)(((nint)raw + 15) & ~(nint)15);
        *aligned = value;
        return aligned;
    }
}
