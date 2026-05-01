// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the MIT License.

namespace NVMathNet.Tensor;

/// <summary>Options for a tensor contraction operation.</summary>
public sealed class TensorContractionOptions
{
    /// <summary>
    /// Preferred workspace strategy.
    /// Default: <see cref="WorksizePreference.Recommended"/>.
    /// </summary>
    public WorksizePreference WorksizePreference { get; set; } = WorksizePreference.Recommended;

    /// <summary>
    /// cuTENSOR algorithm to use.
    /// Default: <see cref="TensorAlgorithm.Default"/> (auto-select).
    /// When <see cref="AutotuneMode"/> is not <see cref="ContractionAutotuneMode.None"/>,
    /// the autotune setting overrides this value.
    /// </summary>
    public TensorAlgorithm Algorithm { get; set; } = TensorAlgorithm.Default;

    /// <summary>
    /// Algorithm search strategy used when building the contraction plan.
    /// Default: <see cref="ContractionAutotuneMode.None"/> (use <see cref="Algorithm"/> directly).
    /// </summary>
    public ContractionAutotuneMode AutotuneMode { get; set; } = ContractionAutotuneMode.None;

    /// <summary>
    /// If <c>true</c> (default), <c>Execute</c> blocks until the GPU completes.
    /// </summary>
    public bool Blocking { get; set; } = true;
}

/// <summary>
/// Controls the algorithm search strategy used when building a tensor contraction plan.
/// </summary>
public enum ContractionAutotuneMode
{
    /// <summary>
    /// No autotuning — use the algorithm specified by
    /// <see cref="TensorContractionOptions.Algorithm"/> directly.
    /// </summary>
    None = 0,
    /// <summary>
    /// Patient auto-select: cuTENSOR tries more heuristic candidates than
    /// <see cref="TensorAlgorithm.Default"/> before picking the best one.
    /// Low overhead; good default when performance matters.
    /// </summary>
    Auto = 1,
    /// <summary>
    /// Full benchmark: cuTENSOR times every candidate algorithm and picks the fastest.
    /// Has measurable overhead; best reserved for shapes that will be executed many times.
    /// </summary>
    Benchmark = 2,
}

/// <summary>Workspace size preference hint for cuTENSOR.</summary>
public enum WorksizePreference
{
    /// <summary>Minimal workspace.</summary>
    Min = 1,
    /// <summary>Default workspace; high performance with reduced memory (default).</summary>
    Recommended = 2,
    /// <summary>Maximum workspace for best performance.</summary>
    Max = 3,
}

/// <summary>Algorithm selection hint for tensor contraction.</summary>
public enum TensorAlgorithm
{
    /// <summary>Auto-select (cuTENSOR default).</summary>
    Default   = -1,
    /// <summary>Pure GEMM-based algorithm (no transpositions before/after).</summary>
    Gett      = -4,
    /// <summary>Transpose input, use GEMM, then transpose output.</summary>
    Tgett     = -3,
    /// <summary>Transpose inputs, use GEMM, then transpose output.</summary>
    Ttgt      = -2,
    /// <summary>Patient algorithm selection (tries more algorithms).</summary>
    DefaultPatient = -6,
}

/// <summary>Data types supported by cuTENSOR operations.</summary>
public enum TensorDataType
{
    /// <summary>16-bit IEEE floating-point (half precision).</summary>
    Float16   = 2,
    /// <summary>16-bit brain floating-point.</summary>
    BFloat16  = 14,
    /// <summary>32-bit IEEE floating-point (single precision).</summary>
    Float32   = 0,
    /// <summary>64-bit IEEE floating-point (double precision).</summary>
    Float64   = 1,
    /// <summary>64-bit complex number (two 32-bit floats).</summary>
    Complex64  = 4,
    /// <summary>128-bit complex number (two 64-bit doubles).</summary>
    Complex128 = 5,
}
