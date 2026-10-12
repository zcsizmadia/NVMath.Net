// Copyright (c) 2026 Zoltan Csizmadia. All rights reserved.
// Licensed under the MIT License.

using NVMathNet.Interop;

namespace NVMathNet.Tensor;

internal static unsafe class TensorScalars
{
    /// <summary>
    /// Stores <paramref name="value"/> in 16-byte aligned memory inside <paramref name="storage"/>
    /// (at least <see cref="AlignedScalar.StorageSize"/> bytes) using the scalar type that cuTENSOR expects for
    /// <paramref name="computeDesc"/>, and returns the address. The scalar type follows the data type: complex tensors
    /// take a complex scalar (16 bytes for double complex), which is written as <c>value + 0i</c>; the whole slot is
    /// zeroed so the imaginary part and any padding are defined.
    /// </summary>
    public static byte* Store(Span<byte> storage, double value, nint computeDesc)
    {
        byte* p = AlignedScalar.Store(storage, (byte)0);
        *(ulong*)p = 0;
        *((ulong*)p + 1) = 0;
        if (computeDesc == CuTensorNative.ComputeDesc64F)
        {
            *(double*)p = value;
        }
        else if (computeDesc == CuTensorNative.ComputeDesc16F)
        {
            *(Half*)p = (Half)value;
        }
        else
        {
            *(float*)p = (float)value;
        }

        return p;
    }
}
