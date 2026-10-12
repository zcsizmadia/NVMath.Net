// Copyright (c) 2026 Zoltan Csizmadia. All rights reserved.
// Licensed under the MIT License.

using NVMathNet.Interop;
using NVMathNet.Tensor;

namespace NVMathNet.Tests;

/// <summary>
/// Tests for the Tensor contraction module (cuTENSOR). Requires a CUDA-capable GPU.
/// </summary>
public sealed class TensorTests
{
    [Test]
    public async Task TensorContraction_MatrixMultiply()
    {
        // Represent matrix multiplication as a tensor contraction:
        // C[m,n] = A[m,k] * B[k,n]
        // Mode labels: m=0, n=1, k=2
        const int M = 4, N = 4, K = 4;

        float[] hA = new float[M * K];
        float[] hB = new float[K * N];
        for (int i = 0; i < M * K; i++)
        {
            hA[i] = i + 1.0f;
        }

        for (int i = 0; i < K * N; i++)
        {
            hB[i] = (i % K) == (i / K) ? 1.0f : 0.0f; // identity
        }

        using var dA = new DeviceBuffer<float>(M * K);
        using var dB = new DeviceBuffer<float>(K * N);
        using var dC = new DeviceBuffer<float>(M * N);
        using var dD = new DeviceBuffer<float>(M * N);
        dA.CopyFrom(hA.AsSpan());
        dB.CopyFrom(hB.AsSpan());
        dC.Clear();
        dD.Clear();

        long[] extA = [M, K];
        long[] extB = [K, N];
        long[] extC = [M, N];
        long[] extD = [M, N];
        int[] modeA = [0, 2];  // m, k
        int[] modeB = [2, 1];  // k, n
        int[] modeC = [0, 1];  // m, n
        int[] modeD = [0, 1];  // m, n

        var opts = new TensorContractionOptions { Blocking = true };

        using var contraction = new TensorContraction(
            extA, modeA, extB, modeB, extC, modeC, extD, modeD,
            TensorDataType.Float32, opts);
        await contraction.PlanAsync();
        await contraction.ExecuteAsync(
            dA.PointerAsInt, dB.PointerAsInt, dC.PointerAsInt, dD.PointerAsInt,
            alpha: 1.0, beta: 0.0);

        // A * I = A
        float[] result = dD.ToArray();
        for (int i = 0; i < M * N; i++)
        {
            float err = Math.Abs(result[i] - hA[i]);
            await Assert.That(err).IsLessThan(1e-3f);
        }
    }

    [Test]
    public async Task TensorContraction_NonTrivialProduct()
    {
        // A (3x2) * B (2x4) = D (3x4) with non-trivial data
        // Mode labels: m=0, n=1, k=2
        const int M = 3, N = 4, K = 2;

        // A[r,c]: [[1,4],[2,5],[3,6]]  stored row-major in memory
        float[] hA = [1, 2, 3, 4, 5, 6]; // element (r,c) at r + M*c
        // B[r,c]: [[7,9,11,13],[8,10,12,14]]
        float[] hB = [7, 8, 9, 10, 11, 12, 13, 14]; // element (r,c) at r + K*c

        // CPU reference D[r,c] = Σ_k A[r,k]*B[k,c]
        float[] expected = new float[M * N];
        for (int r = 0; r < M; r++)
        {
            for (int c = 0; c < N; c++)
            {
                float sum = 0;
                for (int kk = 0; kk < K; kk++)
                {
                    sum += hA[r + M * kk] * hB[kk + K * c];
                }

                expected[r + M * c] = sum;
            }
        }

        using var dA = new DeviceBuffer<float>(M * K);
        using var dB = new DeviceBuffer<float>(K * N);
        using var dC = new DeviceBuffer<float>(M * N);
        using var dD = new DeviceBuffer<float>(M * N);
        dA.CopyFrom(hA.AsSpan());
        dB.CopyFrom(hB.AsSpan());
        dC.Clear();
        dD.Clear();

        long[] extA = [M, K];
        long[] extB = [K, N];
        long[] extC = [M, N];
        long[] extD = [M, N];
        int[] modeA = [0, 2];  // m, k
        int[] modeB = [2, 1];  // k, n
        int[] modeC = [0, 1];  // m, n
        int[] modeD = [0, 1];  // m, n

        var opts = new TensorContractionOptions { Blocking = true };
        using var contraction = new TensorContraction(
            extA, modeA, extB, modeB, extC, modeC, extD, modeD,
            TensorDataType.Float32, opts);
        await contraction.PlanAsync();
        await contraction.ExecuteAsync(
            dA.PointerAsInt, dB.PointerAsInt, dC.PointerAsInt, dD.PointerAsInt,
            alpha: 1.0, beta: 0.0);

        float[] result = dD.ToArray();
        for (int i = 0; i < M * N; i++)
        {
            float err = Math.Abs(result[i] - expected[i]);
            await Assert.That(err).IsLessThan(1e-2f);
        }
    }

    [Test]
    public async Task TensorContraction_TypedExecuteAsync()
    {
        // Test the DeviceBuffer<T> generic overload
        const int S = 4;
        float[] hA = new float[S * S];
        float[] hB = new float[S * S];
        for (int i = 0; i < S * S; i++)
        {
            hA[i] = i + 1f;
            hB[i] = (S * S) - i;
        }

        // CPU reference
        float[] expected = new float[S * S];
        for (int r = 0; r < S; r++)
        {
            for (int c = 0; c < S; c++)
            {
                float sum = 0;
                for (int kk = 0; kk < S; kk++)
                {
                    sum += hA[r + S * kk] * hB[kk + S * c];
                }

                expected[r + S * c] = sum;
            }
        }

        using var dA = new DeviceBuffer<float>(S * S);
        using var dB = new DeviceBuffer<float>(S * S);
        using var dC = new DeviceBuffer<float>(S * S);
        using var dD = new DeviceBuffer<float>(S * S);
        dA.CopyFrom(hA.AsSpan());
        dB.CopyFrom(hB.AsSpan());
        dC.Clear();
        dD.Clear();

        long[] ext = [S, S];
        int[] modeA = [0, 2];
        int[] modeB = [2, 1];
        int[] modeC = [0, 1];

        var opts = new TensorContractionOptions { Blocking = true };
        using var tc = new TensorContraction(
            ext, modeA, ext, modeB, ext, modeC, ext, modeC,
            TensorDataType.Float32, opts);
        await tc.PlanAsync();
        await tc.ExecuteAsync<float>(dA, dB, dC, dD, alpha: 1.0, beta: 0.0);

        float[] result = dD.ToArray();
        for (int i = 0; i < S * S; i++)
        {
            float err = Math.Abs(result[i] - expected[i]);
            await Assert.That(err).IsLessThan(1e-1f);
        }
    }

    // ── Element-wise trinary and operator values ─────────────────────────────

    private static async Task<T[]> RunTrinary<T>(
        T[] a, T[] b, T[] c, TensorDataType type,
        CuTensorOperator opA, CuTensorOperator opB, CuTensorOperator opC,
        CuTensorOperator opAB, CuTensorOperator opABC,
        double alpha, double beta, double gamma) where T : unmanaged, System.Numerics.INumberBase<T>
    {
        int n = a.Length;
        using var dA = new DeviceBuffer<T>(n);
        using var dB = new DeviceBuffer<T>(n);
        using var dC = new DeviceBuffer<T>(n);
        using var dD = new DeviceBuffer<T>(n);
        dA.CopyFrom(a.AsSpan());
        dB.CopyFrom(b.AsSpan());
        dC.CopyFrom(c.AsSpan());
        dD.Clear();

        long[] ext = [4, n / 4];
        int[] modes = [0, 1];
        using var op = new ElementwiseTrinary(
            ext, modes, ext, modes, ext, modes, ext, modes,
            type, opA, opB, opC, opAB, opABC,
            new TensorContractionOptions { Blocking = true });
        await op.ExecuteAsync(dA, dB, dC, dD, alpha, beta, gamma);
        return dD.ToArray();
    }

    private static readonly float[] SampleA = [.. Enumerable.Range(0, 16).Select(i => (i - 6) * 0.5f)];
    private static readonly float[] SampleB = [.. Enumerable.Range(0, 16).Select(i => (i % 5) * 0.25f)];
    private static readonly float[] SampleC = [.. Enumerable.Range(0, 16).Select(i => 1f - i * 0.125f)];

    [Test]
    public async Task ElementwiseTrinary_Add_ComputesWeightedSum()
    {
        float[] d = await RunTrinary(SampleA, SampleB, SampleC, TensorDataType.Float32,
            CuTensorOperator.Identity, CuTensorOperator.Identity, CuTensorOperator.Identity,
            CuTensorOperator.Add, CuTensorOperator.Add, 2.0, 3.0, 4.0);

        for (int i = 0; i < d.Length; i++)
        {
            await Assert.That(Math.Abs(d[i] - (2 * SampleA[i] + 3 * SampleB[i] + 4 * SampleC[i]))).IsLessThan(1e-5f);
        }
    }

    [Test]
    public async Task ElementwiseTrinary_MaxThenAdd_UsesBinaryOperatorValues()
    {
        // D = max(A, B) + C
        float[] d = await RunTrinary(SampleA, SampleB, SampleC, TensorDataType.Float32,
            CuTensorOperator.Identity, CuTensorOperator.Identity, CuTensorOperator.Identity,
            CuTensorOperator.Max, CuTensorOperator.Add, 1.0, 1.0, 1.0);

        for (int i = 0; i < d.Length; i++)
        {
            await Assert.That(Math.Abs(d[i] - (Math.Max(SampleA[i], SampleB[i]) + SampleC[i]))).IsLessThan(1e-5f);
        }
    }

    [Test]
    public async Task ElementwiseTrinary_UnaryOperators_Sin_Abs_Sigmoid()
    {
        // D = sin(A) + |B_shifted| + sigmoid(C)
        float[] bShift = [.. SampleB.Select(v => v - 0.5f)];
        float[] d = await RunTrinary(SampleA, bShift, SampleC, TensorDataType.Float32,
            CuTensorOperator.Sin, CuTensorOperator.Abs, CuTensorOperator.Sigmoid,
            CuTensorOperator.Add, CuTensorOperator.Add, 1.0, 1.0, 1.0);

        for (int i = 0; i < d.Length; i++)
        {
            double expected = Math.Sin(SampleA[i]) + Math.Abs(bShift[i]) + 1.0 / (1.0 + Math.Exp(-SampleC[i]));
            await Assert.That(Math.Abs(d[i] - expected)).IsLessThan(1e-4);
        }
    }

    [Test]
    public async Task ElementwiseTrinary_Double_KeepsFullScalarPrecision()
    {
        double[] a = [.. SampleA.Select(v => (double)v)];
        double[] b = [.. SampleB.Select(v => (double)v)];
        double[] c = [.. SampleC.Select(v => (double)v)];

        double[] d = await RunTrinary(a, b, c, TensorDataType.Float64,
            CuTensorOperator.Identity, CuTensorOperator.Identity, CuTensorOperator.Identity,
            CuTensorOperator.Add, CuTensorOperator.Add, 0.1, 0.2, 0.3);

        for (int i = 0; i < d.Length; i++)
        {
            await Assert.That(Math.Abs(d[i] - (0.1 * a[i] + 0.2 * b[i] + 0.3 * c[i]))).IsLessThan(1e-13);
        }
    }

    [Test]
    public async Task TensorContraction_Double_UsesDoubleScalars()
    {
        // C = 0.1 * A * I with 4x4 doubles; a float alpha would be off by ~1e-9.
        const int N = 4;
        double[] hA = [.. Enumerable.Range(1, N * N).Select(i => (double)i)];
        double[] hI = new double[N * N];
        for (int i = 0; i < N; i++)
        {
            hI[i + N * i] = 1.0;
        }

        using var dA = new DeviceBuffer<double>(N * N);
        using var dB = new DeviceBuffer<double>(N * N);
        using var dC = new DeviceBuffer<double>(N * N);
        using var dD = new DeviceBuffer<double>(N * N);
        dA.CopyFrom(hA.AsSpan());
        dB.CopyFrom(hI.AsSpan());
        dC.Clear();

        using var contraction = new TensorContraction(
            [N, N], [0, 2], [N, N], [2, 1], [N, N], [0, 1], [N, N], [0, 1],
            TensorDataType.Float64, new TensorContractionOptions { Blocking = true });
        await contraction.PlanAsync();
        await contraction.ExecuteAsync(dA.PointerAsInt, dB.PointerAsInt, dC.PointerAsInt, dD.PointerAsInt, alpha: 0.1, beta: 0.0);

        double[] r = dD.ToArray();
        for (int i = 0; i < r.Length; i++)
        {
            await Assert.That(Math.Abs(r[i] - 0.1 * hA[i])).IsLessThan(1e-13);
        }
    }

    // ── Permutation, element-wise binary and reduction (column-major tensors) ──

    private static DeviceBuffer<T> Upload<T>(T[] host) where T : unmanaged, System.Numerics.INumberBase<T>
    {
        var buf = new DeviceBuffer<T>(host.Length);
        buf.CopyFrom(host.AsSpan());
        return buf;
    }

    [Test]
    public async Task TensorPermutation_Transpose2D()
    {
        const int I = 2, J = 3;
        float[] a = new float[I * J];
        for (int i = 0; i < I; i++)
        {
            for (int j = 0; j < J; j++)
            {
                a[i + I * j] = i * 10 + j;
            }
        }

        using var dA = Upload(a);
        using var dB = new DeviceBuffer<float>(I * J);
        using var perm = new TensorPermutation([I, J], [0, 1], [1, 0], options: new TensorContractionOptions { Blocking = true });
        await perm.ExecuteAsync(dA, dB);

        float[] b = dB.ToArray(); // B is J x I: B[j + J*i] = A[i,j]
        for (int i = 0; i < I; i++)
        {
            for (int j = 0; j < J; j++)
            {
                await Assert.That(b[j + J * i]).IsEqualTo(a[i + I * j]);
            }
        }
    }

    [Test]
    public async Task TensorPermutation_3D_WithScaleAndAbs()
    {
        // A[a,b,c] with extents [2,3,4]; B modes [2,0,1] => B[c,a,b], scaled by 2 with abs applied.
        const int A0 = 2, A1 = 3, A2 = 4;
        float[] a = new float[A0 * A1 * A2];
        for (int i = 0; i < a.Length; i++)
        {
            a[i] = (i % 5 - 2) * 1.5f;
        }

        using var dA = Upload(a);
        using var dB = new DeviceBuffer<float>(a.Length);
        using var perm = new TensorPermutation(
            [A0, A1, A2], [0, 1, 2], [2, 0, 1], TensorDataType.Float32, CuTensorOperator.Abs,
            new TensorContractionOptions { Blocking = true });
        await perm.ExecuteAsync(dA, dB, alpha: 2.0);

        float[] b = dB.ToArray();
        for (int x = 0; x < A0; x++)
        {
            for (int y = 0; y < A1; y++)
            {
                for (int z = 0; z < A2; z++)
                {
                    float expected = 2f * MathF.Abs(a[x + A0 * (y + A1 * z)]);
                    await Assert.That(Math.Abs(b[z + A2 * (x + A0 * y)] - expected)).IsLessThan(1e-6f);
                }
            }
        }
    }

    [Test]
    public async Task TensorPermutation_InvalidModes_Throw()
    {
        await Assert.That(() => new TensorPermutation([2, 3], [0, 1], [0, 2])).Throws<ArgumentException>();
        await Assert.That(() => new TensorPermutation([2, 3], [0, 0], [0, 0])).Throws<ArgumentException>();
    }

    [Test]
    public async Task ElementwiseBinary_AddsTransposedOperand()
    {
        // D[i,j] = A[i,j] + 2 * C[j,i]
        const int I = 2, J = 3;
        float[] a = [.. Enumerable.Range(0, I * J).Select(v => (float)v)];
        float[] c = [.. Enumerable.Range(0, I * J).Select(v => 100f + v)]; // C is J x I
        using var dA = Upload(a);
        using var dC = Upload(c);
        using var dD = new DeviceBuffer<float>(I * J);

        using var op = new ElementwiseBinary(
            [J, I], [1, 0], [I, J], [0, 1], [I, J], [0, 1],
            options: new TensorContractionOptions { Blocking = true });
        await op.ExecuteAsync(dA, dC, dD, alpha: 1.0, gamma: 2.0);

        float[] d = dD.ToArray();
        for (int i = 0; i < I; i++)
        {
            for (int j = 0; j < J; j++)
            {
                float expected = a[j + J * i] + 2f * c[i + I * j];
                await Assert.That(Math.Abs(d[i + I * j] - expected)).IsLessThan(1e-5f);
            }
        }
    }

    [Test]
    public async Task ElementwiseBinary_MaxOperator()
    {
        float[] a = [.. Enumerable.Range(0, 12).Select(v => (v - 6) * 0.5f)];
        float[] c = [.. Enumerable.Range(0, 12).Select(v => 2f - v * 0.4f)];
        using var dA = Upload(a);
        using var dC = Upload(c);
        using var dD = new DeviceBuffer<float>(12);

        using var op = new ElementwiseBinary(
            [3, 4], [0, 1], [3, 4], [0, 1], [3, 4], [0, 1],
            opAC: CuTensorOperator.Max, options: new TensorContractionOptions { Blocking = true });
        await op.ExecuteAsync(dA, dC, dD);

        float[] d = dD.ToArray();
        for (int i = 0; i < d.Length; i++)
        {
            await Assert.That(Math.Abs(d[i] - Math.Max(a[i], c[i]))).IsLessThan(1e-6f);
        }
    }

    [Test]
    public async Task TensorReduction_SumOverModes()
    {
        // A is 3 x 4 (modes 0,1). Reduce mode 1 -> D[i] = sum_j A[i,j]; reduce mode 0 -> D[j] = sum_i A[i,j].
        const int I = 3, J = 4;
        float[] a = new float[I * J];
        for (int i = 0; i < I; i++)
        {
            for (int j = 0; j < J; j++)
            {
                a[i + I * j] = (i + 1) * (j + 2);
            }
        }

        using var dA = Upload(a);

        using (var dRows = new DeviceBuffer<float>(I))
        {
            dRows.Clear();
            using var red = new TensorReduction([I, J], [0, 1], [I], [0], options: new TensorContractionOptions { Blocking = true });
            await red.ExecuteAsync(dA, dRows);
            float[] d = dRows.ToArray();
            for (int i = 0; i < I; i++)
            {
                float expected = 0;
                for (int j = 0; j < J; j++)
                {
                    expected += a[i + I * j];
                }

                await Assert.That(Math.Abs(d[i] - expected)).IsLessThan(1e-4f);
            }
        }

        using (var dCols = new DeviceBuffer<float>(J))
        {
            dCols.Clear();
            using var red = new TensorReduction([I, J], [0, 1], [J], [1], options: new TensorContractionOptions { Blocking = true });
            await red.ExecuteAsync(dA, dCols);
            float[] d = dCols.ToArray();
            for (int j = 0; j < J; j++)
            {
                float expected = 0;
                for (int i = 0; i < I; i++)
                {
                    expected += a[i + I * j];
                }

                await Assert.That(Math.Abs(d[j] - expected)).IsLessThan(1e-4f);
            }
        }
    }

    [Test]
    public async Task TensorReduction_MaxAndAlphaBeta_Double()
    {
        const int I = 3, J = 4;
        double[] a = [.. Enumerable.Range(0, I * J).Select(v => Math.Sin(v) * 5)];
        using var dA = Upload(a);
        using var dD = Upload(new double[] { 10, 20, 30 });

        // D = 0.5 * max_j(A[i,j]) + 0.25 * D
        using var red = new TensorReduction(
            [I, J], [0, 1], [I], [0], TensorDataType.Float64, CuTensorOperator.Max,
            options: new TensorContractionOptions { Blocking = true });
        await red.ExecuteAsync(dA, dD, alpha: 0.5, beta: 0.25);

        double[] d = dD.ToArray();
        double[] init = [10, 20, 30];
        for (int i = 0; i < I; i++)
        {
            double max = double.MinValue;
            for (int j = 0; j < J; j++)
            {
                max = Math.Max(max, a[i + I * j]);
            }

            await Assert.That(Math.Abs(d[i] - (0.5 * max + 0.25 * init[i]))).IsLessThan(1e-12);
        }
    }

    [Test]
    public async Task TensorReduction_ModeNotInInput_Throws()
    {
        await Assert.That(() => new TensorReduction([3, 4], [0, 1], [3], [7])).Throws<ArgumentException>();
    }
}
