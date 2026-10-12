// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the MIT License.

using System.Numerics;

using NVMathNet.Interop;

namespace NVMathNet.Sparse;

/// <summary>Structure of the system matrix; selects the factorisation cuDSS uses.</summary>
public enum SparseMatrixStructure
{
    /// <summary>General (non-symmetric) matrix: LU factorisation.</summary>
    General = 0,
    /// <summary>Symmetric matrix: LDL^T factorisation.</summary>
    Symmetric = 1,
    /// <summary>Symmetric positive-definite matrix: Cholesky factorisation.</summary>
    SymmetricPositiveDefinite = 3,
}

/// <summary>Which part of the matrix is stored in the CSR arrays.</summary>
public enum SparseMatrixView
{
    /// <summary>The full matrix.</summary>
    Full = 0,
    /// <summary>Only the lower triangle (symmetric and SPD matrices).</summary>
    Lower = 1,
    /// <summary>Only the upper triangle (symmetric and SPD matrices).</summary>
    Upper = 2,
}

/// <summary>Options for <see cref="SparseDirectSolver{T}"/>.</summary>
public sealed class DirectSolverOptions
{
    /// <summary>Structure of the matrix. Default: <see cref="SparseMatrixStructure.General"/>.</summary>
    public SparseMatrixStructure Structure { get; set; } = SparseMatrixStructure.General;

    /// <summary>Stored triangle. Default: <see cref="SparseMatrixView.Full"/>.</summary>
    public SparseMatrixView View { get; set; } = SparseMatrixView.Full;

    /// <summary>Fill-reducing reordering algorithm. Default: library default.</summary>
    public CuDssReorderingAlg Reordering { get; set; } = CuDssReorderingAlg.Default;

    /// <summary>Maximum iterative-refinement steps after the solve; 0 (default) disables refinement.</summary>
    public int RefinementSteps { get; set; }
}

/// <summary>
/// Sparse direct solver for <c>A·X = B</c> backed by NVIDIA cuDSS. Mirrors the staged workflow of
/// nvmath-python's <c>DirectSolver</c>:
/// <list type="number">
///   <item><description><see cref="Analyze"/> — reordering and symbolic factorisation (depends only on the sparsity pattern).</description></item>
///   <item><description><see cref="Factorize"/> — numerical factorisation.</description></item>
///   <item><description><see cref="Solve"/> — forward/backward substitution; call repeatedly for new right-hand sides.</description></item>
///   <item><description>Change the values of A in place and call <see cref="Refactorize"/> to reuse the analysis.</description></item>
/// </list>
/// Requires the cuDSS runtime library (<c>cudss64_0.dll</c> / <c>libcudss.so.0</c>) on the library path.
/// The CSR matrix must have sorted column indices and must outlive this solver.
/// </summary>
/// <typeparam name="T"><see cref="float"/> or <see cref="double"/>.</typeparam>
public sealed class SparseDirectSolver<T> : IDisposable where T : unmanaged, INumberBase<T>
{
    private readonly CsrMatrix<T> _matrix;
    private readonly CudaStream _stream;
    private readonly bool _streamOwned;
    private readonly CuDssDataType _valueType;
    private nint _handle;
    private nint _config;
    private nint _data;
    private nint _matrixDesc;
    private bool _analyzed;
    private bool _factorized;
    private bool _disposed;

    /// <summary>Creates a solver for <paramref name="matrix"/>.</summary>
    /// <param name="matrix">Square CSR matrix; must outlive the solver.</param>
    /// <param name="options">Solver options; <c>null</c> uses defaults.</param>
    /// <param name="stream">Optional CUDA stream; a private stream is created if <c>null</c>.</param>
    public unsafe SparseDirectSolver(CsrMatrix<T> matrix, DirectSolverOptions? options = null, CudaStream? stream = null)
    {
        ArgumentNullException.ThrowIfNull(matrix);
        if (matrix.Rows != matrix.Cols)
        {
            throw new ArgumentException("The direct solver requires a square matrix.", nameof(matrix));
        }

        _valueType = typeof(T) == typeof(float) ? CuDssDataType.R32F
            : typeof(T) == typeof(double) ? CuDssDataType.R64F
            : throw new NotSupportedException($"Element type {typeof(T)} is not supported; use float or double.");

        options ??= new DirectSolverOptions();
        _matrix = matrix;
        _streamOwned = stream is null;
        _stream = stream ?? new CudaStream();

        try
        {
            _handle = CuDssNative.Create();
            CuDssNative.SetStream(_handle, _stream.Handle);
            _config = CuDssNative.ConfigCreate();
            _data = CuDssNative.DataCreate(_handle);

            int reordering = (int)options.Reordering;
            CuDssNative.ConfigSet(_config, CuDssConfigParam.ReorderingAlg, &reordering, sizeof(int));
            if (options.RefinementSteps > 0)
            {
                int steps = options.RefinementSteps;
                CuDssNative.ConfigSet(_config, CuDssConfigParam.IrNSteps, &steps, sizeof(int));
            }

            _matrixDesc = CuDssNative.MatrixCreateCsr(
                matrix.Rows, matrix.Cols, matrix.NNZ,
                matrix.RowOffsets.Pointer, matrix.ColIndices.Pointer, matrix.Values.Pointer,
                _valueType, (CuDssMatrixType)(int)options.Structure, (CuDssMatrixView)(int)options.View);
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    /// <summary>
    /// Reordering and symbolic factorisation. Depends only on the sparsity pattern, so it is done
    /// once even if the values change.
    /// </summary>
    public void Analyze()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        RunPhase(CuDssPhase.Analysis);
        NVMathLog.Debug("cuDSS", () => $"Analysis done for {_matrix.Rows}x{_matrix.Cols} matrix with {_matrix.NNZ} non-zeros.");
        _analyzed = true;
        _factorized = false;
    }

    /// <summary>
    /// Numerical factorisation. Runs <see cref="Analyze"/> first if it has not been done.
    /// Throws <see cref="InvalidOperationException"/> if cuDSS reports a failure (for example a
    /// singular matrix).
    /// </summary>
    public void Factorize()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (!_analyzed)
        {
            Analyze();
        }

        RunPhase(CuDssPhase.Factorization);
        CheckInfo("factorization");
        NVMathLog.Debug("cuDSS", () => $"Factorization done: {FactorNonZeros} non-zeros in the factors, {PerturbedPivots} perturbed pivot(s).");
        _factorized = true;
    }

    /// <summary>
    /// Re-factorises after the values of the matrix changed in place (same sparsity pattern),
    /// reusing the analysis.
    /// </summary>
    public void Refactorize()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (!_factorized)
        {
            Factorize();
            return;
        }

        RunPhase(CuDssPhase.Refactorization);
        CheckInfo("refactorization");
    }

    /// <summary>
    /// Points the solver at a different values array (same sparsity pattern). Call
    /// <see cref="Refactorize"/> afterwards.
    /// </summary>
    public unsafe void SetValues(DeviceBuffer<T> values)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(values);
        if (values.Length < _matrix.NNZ)
        {
            throw new ArgumentException("Values buffer is smaller than the number of non-zeros.", nameof(values));
        }

        CuDssNative.MatrixSetValues(_matrixDesc, values.Pointer);
    }

    /// <summary>
    /// Solves <c>A·X = B</c>. Runs <see cref="Factorize"/> first if it has not been done.
    /// </summary>
    /// <param name="b">Right-hand sides, column-major <c>n × nrhs</c> (leading dimension n).</param>
    /// <param name="x">Solution, column-major <c>n × nrhs</c>; must not alias <paramref name="b"/>.</param>
    /// <param name="nrhs">Number of right-hand sides.</param>
    public unsafe void Solve(DeviceBuffer<T> b, DeviceBuffer<T> x, int nrhs = 1)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(b);
        ArgumentNullException.ThrowIfNull(x);
        ArgumentOutOfRangeException.ThrowIfLessThan(nrhs, 1);
        long n = _matrix.Rows;
        if (b.Length < n * nrhs || x.Length < n * nrhs)
        {
            throw new ArgumentException($"B and X must each hold at least {n * nrhs} elements.");
        }

        if (!_factorized)
        {
            Factorize();
        }

        nint bDesc = 0, xDesc = 0;
        try
        {
            bDesc = CuDssNative.MatrixCreateDn(n, nrhs, n, b.Pointer, _valueType);
            xDesc = CuDssNative.MatrixCreateDn(n, nrhs, n, x.Pointer, _valueType);
            CuDssNative.Execute(_handle, CuDssPhase.Solve, _config, _data, _matrixDesc, xDesc, bDesc);
            _stream.Synchronize();
        }
        finally
        {
            if (bDesc != 0)
            {
                CuDssNative.MatrixDestroy(bDesc);
            }

            if (xDesc != 0)
            {
                CuDssNative.MatrixDestroy(xDesc);
            }
        }
    }

    /// <summary>Asynchronous version of <see cref="Analyze"/> (runs on the thread pool).</summary>
    public Task AnalyzeAsync(CancellationToken ct = default) => Task.Run(Analyze, ct);

    /// <summary>Asynchronous version of <see cref="Factorize"/> (runs on the thread pool).</summary>
    public Task FactorizeAsync(CancellationToken ct = default) => Task.Run(Factorize, ct);

    /// <summary>Asynchronous version of <see cref="Refactorize"/> (runs on the thread pool).</summary>
    public Task RefactorizeAsync(CancellationToken ct = default) => Task.Run(Refactorize, ct);

    /// <summary>Asynchronous version of <see cref="Solve"/> (runs on the thread pool).</summary>
    public Task SolveAsync(DeviceBuffer<T> b, DeviceBuffer<T> x, int nrhs = 1, CancellationToken ct = default) =>
        Task.Run(() => Solve(b, x, nrhs), ct);

    /// <summary>Number of non-zeros in the factors; available after <see cref="Factorize"/>.</summary>
    public unsafe long FactorNonZeros
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            long nnz = 0;
            CuDssNative.DataGet(_handle, _data, CuDssDataParam.LuNnz, &nnz, sizeof(long));
            return nnz;
        }
    }

    /// <summary>Number of pivots cuDSS perturbed during the last factorisation.</summary>
    public unsafe int PerturbedPivots
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            int count = 0;
            CuDssNative.DataGet(_handle, _data, CuDssDataParam.NPivots, &count, sizeof(int));
            return count;
        }
    }

    private void RunPhase(CuDssPhase phase)
    {
        // The dense arguments are unused by the analysis and factorisation phases.
        CuDssNative.Execute(_handle, phase, _config, _data, _matrixDesc, 0, 0);
        _stream.Synchronize();
    }

    private unsafe void CheckInfo(string what)
    {
        int info = 0;
        CuDssNative.DataGet(_handle, _data, CuDssDataParam.Info, &info, sizeof(int));
        if (info != 0)
        {
            throw new InvalidOperationException($"cuDSS {what} failed with info = {info} (the matrix may be singular).");
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        if (_matrixDesc != 0)
        {
            CuDssNative.MatrixDestroy(_matrixDesc);
            _matrixDesc = 0;
        }

        if (_data != 0)
        {
            CuDssNative.DataDestroy(_handle, _data);
            _data = 0;
        }

        if (_config != 0)
        {
            CuDssNative.ConfigDestroy(_config);
            _config = 0;
        }

        if (_handle != 0)
        {
            CuDssNative.Destroy(_handle);
            _handle = 0;
        }

        if (_streamOwned)
        {
            _stream.Dispose();
        }
    }
}
