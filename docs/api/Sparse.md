# Sparse — `NVMathNet.Sparse`

## `CsrMatrix<T>`

CSR (Compressed Sparse Row) matrix. Implements `IDisposable`.
Constraint: `T : unmanaged, INumberBase<T>`.

| Member | Description |
|---|---|
| `CsrMatrix(long rows, long cols, DeviceBuffer<int> rowOffsets, DeviceBuffer<int> colIndices, DeviceBuffer<T> values)` | Constructs a CSR matrix. |
| `long Rows, Cols, NNZ` | Matrix dimensions and non-zero count. |
| `DeviceBuffer<int> RowOffsets, ColIndices` | Index arrays. |
| `DeviceBuffer<T> Values` | Non-zero values. |

## `CooMatrix<T>`

COO (Coordinate) sparse matrix. Implements `IDisposable`.

| Member | Description |
|---|---|
| `CooMatrix(long rows, long cols, DeviceBuffer<int> rowIndices, DeviceBuffer<int> colIndices, DeviceBuffer<T> values)` | Constructs a COO matrix. |
| `long Rows, Cols, NNZ` | Matrix dimensions and non-zero count. |

## `BsrMatrix<T>`

BSR (Block Sparse Row) matrix. Implements `IDisposable`.

| Member | Description |
|---|---|
| `BsrMatrix(long blockRows, long blockCols, int rowBlockDim, int colBlockDim, DeviceBuffer<int> rowOffsets, DeviceBuffer<int> colIndices, DeviceBuffer<T> values, bool columnMajorBlocks = false)` | Constructs a BSR matrix. |
| `long BlockRows, BlockCols, BlockNNZ` | Block-level dimensions. |
| `int RowBlockDim, ColBlockDim` | Block sizes. |
| `long Rows, Cols` | Full matrix dimensions. |

## `SparseLinearAlgebra` (static class)

Dense matrices passed to SpMM use **column-major** layout.

| Method | Description |
|---|---|
| `Task SpMVAsync<T>(CsrMatrix<T> A, DeviceBuffer<T> x, y, float alpha = 1f, float beta = 0f, bool transpose = false, CudaStream? stream = null, CancellationToken ct = default)` | Sparse matrix–vector: y = α·op(A)·x + β·y. Set `transpose = true` for A^T. |
| `Task SpMMAsync<T>(CsrMatrix<T> A, DeviceBuffer<T> B, C, long n, float alpha = 1f, float beta = 0f, bool transpose = false, CudaStream? stream = null, CancellationToken ct = default)` | Sparse–dense: C = α·op(A)·B + β·C. `n` = columns in B/C. Set `transpose = true` for A^T. |
| `Task SpSVAsync<T>(CsrMatrix<T> A, DeviceBuffer<T> x, y, SparseTriangle triangle = Lower, SparseDiagonal diagonal = NonUnit, T? alpha = null, bool transpose = false, ...)` | Triangular solve `op(A)·y = alpha·x` (vector RHS). |
| `Task SpSMAsync<T>(CsrMatrix<T> A, DeviceBuffer<T> B, C, long n, SparseTriangle triangle = Lower, SparseDiagonal diagonal = NonUnit, T? alpha = null, bool transpose = false, ...)` | Triangular solve `op(A)·C = alpha·B` (dense matrix RHS, column-major). |
| `Task SddmmAsync<T>(DeviceBuffer<T> A, DeviceBuffer<T> B, CsrMatrix<T> C, long k, T? alpha = null, T? beta = null, ...)` | Sampled dense-dense product `C = alpha·(A·B)∘spy(C) + beta·C`; C's values are updated in place. |
| `Task<CsrMatrix<T>> SpGemmAsync<T>(CsrMatrix<T> A, CsrMatrix<T> B, T? alpha = null, ...)` | Sparse-sparse product; returns a new CSR matrix (throws if the product is empty). |

`CsrMatrix<T>.Transpose()` returns a new CSR matrix holding the transpose (CSR→CSC conversion).
The generic-API methods take scalars as `T` (matching the matrix element type); supported element types are `float`, `double` and `Half`.
| `Task SpMVAsync<T>(CooMatrix<T> A, ...)` / `Task SpMVAsync<T>(BsrMatrix<T> A, ...)` | SpMV overloads for COO and BSR matrices (same parameters as the CSR version). |
| `Task SpMMAsync<T>(CooMatrix<T> A, ...)` / `Task SpMMAsync<T>(BsrMatrix<T> A, ...)` | SpMM overloads for COO and BSR matrices. |

## `SparseDirectSolver<T>` (cuDSS)

Sparse direct solver for `A·X = B` backed by NVIDIA cuDSS (`cudss64_0.dll` / `libcudss.so.0` must be on the library path). `T` is `float` or `double`. The CSR matrix must have sorted column indices and must outlive the solver. Mirrors the staged workflow of nvmath-python's `DirectSolver`.

```csharp
using var solver = new SparseDirectSolver<double>(matrix);
solver.Analyze();           // reordering + symbolic factorisation (pattern only)
solver.Factorize();         // numerical factorisation
solver.Solve(b, x, nrhs);   // call repeatedly for new right-hand sides
// change matrix.Values in place, then:
solver.Refactorize();       // reuses the analysis
```

| Member | Description |
|---|---|
| `SparseDirectSolver(CsrMatrix<T> matrix, DirectSolverOptions? options = null, CudaStream? stream = null)` | Creates the solver (square matrix required). |
| `void Analyze()` / `Factorize()` / `Refactorize()` | The solver phases. `Factorize` and `Solve` run earlier phases implicitly. `Factorize`/`Refactorize` throw `InvalidOperationException` if cuDSS reports a failure. |
| `void Solve(DeviceBuffer<T> b, DeviceBuffer<T> x, int nrhs = 1)` | Solves for column-major `n × nrhs` right-hand sides; `x` must not alias `b`. |
| `void SetValues(DeviceBuffer<T> values)` | Points the solver at a different values array (same pattern); call `Refactorize` afterwards. |
| `Task AnalyzeAsync / FactorizeAsync / RefactorizeAsync / SolveAsync` | Thread-pool versions. |
| `long FactorNonZeros`, `int PerturbedPivots` | Statistics from the last factorisation. |

`DirectSolverOptions`: `Structure` (`General` = LU, `Symmetric` = LDLᵀ, `SymmetricPositiveDefinite` = Cholesky), `View` (`Full`, `Lower`, `Upper` — which triangle is stored for symmetric/SPD matrices), `Reordering` (`CuDssReorderingAlg`), `RefinementSteps` (iterative refinement, default 0).

### CSC, COO triangular solves, empty products

| Member | Description |
|---|---|
| `CscMatrix<T>` | Compressed Sparse Column matrix (`ColOffsets`, `RowIndices`, `Values`). Create from a CSR matrix with `CsrMatrix<T>.ToCsc()`. |
| `SpMVAsync` / `SpMMAsync` overloads for `CscMatrix<T>` | Same parameters as the CSR versions. |
| `SpSVAsync` / `SpSMAsync` overloads for `CooMatrix<T>` | Triangular solves on COO matrices (entries sorted by row). |
| `SpGemmAsync` with an empty product | Returns a matrix with `NNZ == 0` (previously threw). |
| `new CsrMatrix<T>(..., long? nnz)` | Optional explicit non-zero count, for matrices backed by placeholder buffers. |

SpGEMM and SDDMM are only available for CSR matrices — cuSPARSE does not provide COO or BSR variants of these operations.
