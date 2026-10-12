# Dense Solver — `NVMathNet.Solver`

## `DenseSolver` (static)

Stateless helpers for cuSOLVER dense operations.

| Member | Description |
|---|---|
| `Task SolveAsync(DeviceBuffer<float> a, b, int n, int nrhs = 1, ...)` | Single-precision LU solve: A*X = B. A is overwritten with LU factors; B with solution. |
| `Task SolveAsync(DeviceBuffer<double> a, b, int n, int nrhs = 1, ...)` | Double-precision LU solve. |
| `Task CholeskyAsync(DeviceBuffer<float> a, int n, bool upper = false, ...)` | Single-precision Cholesky factorisation of SPD matrix. |
| `Task CholeskyAsync(DeviceBuffer<double> a, int n, bool upper = false, ...)` | Double-precision Cholesky factorisation. |
| `Task SvdAsync(DeviceBuffer<float> a, s, u, vt, int m, n, ...)` | Single-precision SVD: A = U·diag(S)·VT. |
| `Task SvdAsync(DeviceBuffer<double> a, s, u, vt, int m, n, ...)` | Double-precision SVD. |
| `Task EigAsync(DeviceBuffer<float> a, w, int n, bool computeVectors = true, upper = false, ...)` | Single-precision symmetric eigenvalue decomposition. |
| `Task EigAsync(DeviceBuffer<double> a, w, int n, ...)` | Double-precision symmetric eigenvalue decomposition. |
| `Task<int> LuFactorAsync<T>(DeviceBuffer<T> a, DeviceBuffer<int> ipiv, int m, int n, ...)` | LU with partial pivoting `P·A = L·U` in place. Returns 0, or a positive value if `U(i,i)` is exactly zero (singular). |
| `Task LuSolveAsync<T>(DeviceBuffer<T> a, DeviceBuffer<int> ipiv, DeviceBuffer<T> b, int n, int nrhs = 1, bool transpose = false, ...)` | Solves `op(A)·X = B` from an LU factorisation. |
| `Task QrAsync<T>(DeviceBuffer<T> a, DeviceBuffer<T> tau, int m, int n, ...)` | QR factorisation in place (R in the upper triangle, reflectors below and in `tau`). |
| `Task QrFormQAsync<T>(DeviceBuffer<T> a, DeviceBuffer<T> tau, int m, int n, ...)` | Overwrites `a` with the explicit m×n orthonormal Q (`m ≥ n`). |
| `Task LeastSquaresAsync<T>(DeviceBuffer<T> a, DeviceBuffer<T> b, int m, int n, int nrhs = 1, ...)` | Solves `min ‖A·X − B‖` by QR (`m ≥ n`, full column rank). The first `n` rows of each column of `b` hold the solution. |
| `Task<int[]> CholeskyBatchedAsync<T>(DeviceBuffer<T> a, int n, int batchCount, bool upper = false, ...)` | Cholesky of a packed batch of SPD matrices; returns a per-matrix status (positive = not positive definite). |

The `<T>` methods accept `float` or `double`. They check cuSOLVER's `devInfo` and throw `InvalidOperationException` for illegal arguments; the older overloads above do not check `devInfo`.
