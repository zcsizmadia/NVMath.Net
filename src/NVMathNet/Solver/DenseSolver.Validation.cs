// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the MIT License.

namespace NVMathNet.Solver;

// Argument checks shared by the dense solvers. cuSOLVER does no bounds checking: an undersized buffer or a
// non-positive dimension corrupts memory or leaves the CUDA context in a sticky error state, so every entry point
// checks before any native call.
public static partial class DenseSolver
{
    private static void Dim(long value, string name)
    {
        if (value < 1)
        {
            throw new ArgumentOutOfRangeException(name, value, "Must be at least 1.");
        }
    }

    // cuSOLVER gesvd supports m >= n only.
    private static void RequireTall(int m, int n)
    {
        if (m < n)
        {
            throw new ArgumentException($"The full SVD requires m >= n but got m = {m}, n = {n}; decompose the transpose instead.", "m");
        }
    }

    private static void Need(long actual, long required, string name)
    {
        if (actual < required)
        {
            throw new ArgumentException($"The buffer holds {actual} elements but {required} are required.", name);
        }
    }
}
