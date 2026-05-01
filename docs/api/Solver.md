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
