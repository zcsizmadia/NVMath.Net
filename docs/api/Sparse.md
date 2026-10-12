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
