// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the MIT License.

using NVMathNet.Interop;

namespace NVMathNet.Tensor;

internal static unsafe class TensorScalars
{
    /// <summary>
    /// Writes <paramref name="value"/> into <paramref name="dst"/> (at least 8 bytes) using the
    /// scalar type that cuTENSOR expects for <paramref name="computeDesc"/>: half for 16F,
    /// double for 64F and float otherwise.
    /// </summary>
    public static void Encode(double value, nint computeDesc, byte* dst)
    {
        *(ulong*)dst = 0;
        if (computeDesc == CuTensorNative.ComputeDesc64F)
        {
            *(double*)dst = value;
        }
        else if (computeDesc == CuTensorNative.ComputeDesc16F)
        {
            *(Half*)dst = (Half)value;
        }
        else
        {
            *(float*)dst = (float)value;
        }
    }
}
