// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the Apache 2.0 License.

namespace NVMathNet.LinAlg;

/// <summary>cuBLASLt compute types available for matrix multiplication.</summary>
public enum MatmulComputeType
{
    /// <summary>16-bit floating-point accumulation.</summary>
    Compute16F           = 64,
    /// <summary>16-bit floating-point, strict IEEE compliance.</summary>
    Compute16FPedantic   = 65,
    /// <summary>32-bit floating-point accumulation (default for SGEMM).</summary>
    Compute32F           = 68,
    /// <summary>32-bit floating-point, strict IEEE compliance.</summary>
    Compute32FPedantic   = 69,
    /// <summary>32-bit float compute accelerated by 16-bit tensor cores.</summary>
    Compute32FFast16F    = 74,
    /// <summary>32-bit float compute accelerated by bfloat16 tensor cores.</summary>
    Compute32FFast16BF   = 75,
    /// <summary>32-bit float compute using TF32 tensor cores.</summary>
    Compute32FFastTF32   = 77,
    /// <summary>64-bit floating-point accumulation (DGEMM).</summary>
    Compute64F           = 70,
    /// <summary>64-bit floating-point, strict IEEE compliance.</summary>
    Compute64FPedantic   = 71,
    /// <summary>32-bit integer accumulation (IGEMM).</summary>
    Compute32I           = 72,
    /// <summary>32-bit integer, strict mode.</summary>
    Compute32IPedantic   = 73,
}

/// <summary>Epilog operations applied after the matrix multiplication.</summary>
public enum MatmulEpilog
{
    /// <summary>No epilog.</summary>
    Default         = 1,
    /// <summary>Apply ReLU activation.</summary>
    Relu            = 2,
    /// <summary>Add a per-row bias vector.</summary>
    Bias            = 4,
    /// <summary>Add bias then apply ReLU.</summary>
    ReluBias        = 6,
    /// <summary>Apply GELU activation.</summary>
    Gelu            = 32,
    /// <summary>Apply GELU and save the pre-GELU output for backward pass.</summary>
    GeluAux         = 96,
    /// <summary>Add bias and apply GELU.</summary>
    GeluBias        = 100,
}

/// <summary>Element data types for matrices.</summary>
public enum MatrixDataType
{
    /// <summary>16-bit IEEE floating-point (half precision).</summary>
    Float16  = 2,
    /// <summary>16-bit brain floating-point.</summary>
    BFloat16 = 14,
    /// <summary>32-bit IEEE floating-point (single precision).</summary>
    Float32  = 0,
    /// <summary>64-bit IEEE floating-point (double precision).</summary>
    Float64  = 1,
    /// <summary>8-bit signed integer.</summary>
    Int8     = 3,
    /// <summary>32-bit signed integer.</summary>
    Int32    = 10,
    /// <summary>8-bit floating-point with 4-bit exponent, 3-bit mantissa (E4M3).</summary>
    FP8E4M3 = 28,
    /// <summary>8-bit floating-point with 5-bit exponent, 2-bit mantissa (E5M2).</summary>
    FP8E5M2 = 29,
}

/// <summary>
/// Options controlling matrix multiplication behaviour.
/// </summary>
public sealed class MatmulOptions
{
    /// <summary>
    /// Compute type.  Default: <see cref="MatmulComputeType.Compute32F"/>.
    /// </summary>
    public MatmulComputeType ComputeType { get; set; } = MatmulComputeType.Compute32F;

    /// <summary>Data type of matrix A.  Default: <see cref="MatrixDataType.Float32"/>.</summary>
    public MatrixDataType TypeA { get; set; } = MatrixDataType.Float32;

    /// <summary>Data type of matrix B.  Default: <see cref="MatrixDataType.Float32"/>.</summary>
    public MatrixDataType TypeB { get; set; } = MatrixDataType.Float32;

    /// <summary>Data type of matrix C (accumulator).  Default: <see cref="MatrixDataType.Float32"/>.</summary>
    public MatrixDataType TypeC { get; set; } = MatrixDataType.Float32;

    /// <summary>Data type of matrix D (output).  Default: <see cref="MatrixDataType.Float32"/>.</summary>
    public MatrixDataType TypeD { get; set; } = MatrixDataType.Float32;

    /// <summary>Scale type for alpha/beta scalars.  Default: <see cref="MatrixDataType.Float32"/>.</summary>
    public MatrixDataType ScaleType { get; set; } = MatrixDataType.Float32;

    /// <summary>
    /// Epilog to apply after the core GEMM.
    /// Default: <see cref="MatmulEpilog.Default"/> (no epilog).
    /// </summary>
    public MatmulEpilog Epilog { get; set; } = MatmulEpilog.Default;

    /// <summary>Maximum workspace in bytes offered to the heuristic.  Default: 4 MiB.</summary>
    public nuint MaxWorkspaceBytes { get; set; } = 4 * 1024 * 1024;

    /// <summary>Whether to transpose matrix A.  Default: <c>false</c>.</summary>
    public bool TransposeA { get; set; }

    /// <summary>Whether to transpose matrix B.  Default: <c>false</c>.</summary>
    public bool TransposeB { get; set; }

    /// <summary>
    /// If <c>true</c> (default), execution waits for the GPU to finish before returning.
    /// Set to <c>false</c> for asynchronous overlapped execution.
    /// </summary>
    public bool Blocking { get; set; } = true;
}

/// <summary>
/// Fine-grained control over the cuBLASLt algorithm selection used during
/// <see cref="Matmul.Plan(MatmulPlanPreferences)"/> / <see cref="Matmul.PlanAsync"/>.
/// Pass an instance to the optional <c>preferences</c> parameter on those methods.
/// </summary>
public sealed class MatmulPlanPreferences
{
    /// <summary>
    /// Maximum number of algorithm candidates to request from the cuBLASLt heuristic.
    /// The first candidate (highest score) is used.  Increase to let the heuristic
    /// consider more options when the default choice is sub-optimal.
    /// Default: 4.
    /// </summary>
    public int RequestedAlgoCount { get; set; } = 4;

    /// <summary>
    /// Bit-mask controlling which numerical implementations are allowed.
    /// <c>0</c> (default) means all implementations are permitted.
    /// Set bit 0 (<c>1</c>) to allow tensor-core variants only,
    /// bit 1 (<c>2</c>) to allow SIMD variants only, etc.
    /// Maps to <c>CUBLASLT_MATMUL_PREF_IMPL_MASK</c>.
    /// </summary>
    public ulong NumericalImplMask { get; set; }

    /// <summary>
    /// Override the maximum workspace in bytes offered to the algorithm search,
    /// replacing the value set in <see cref="MatmulOptions.MaxWorkspaceBytes"/>.
    /// <c>null</c> (default) keeps the options value unchanged.
    /// </summary>
    public nuint? MaxWorkspaceBytes { get; set; }
}
