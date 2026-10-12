// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the MIT License.

using NVMathNet.Tensor;

namespace NVMathNet.Tests;

/// <summary>
/// Regression tests for issues found in the pre-1.0 review of the cuTENSOR wrapper.
/// Requires a CUDA-capable GPU.
/// </summary>
public sealed class TensorReviewTests
{
    [Test]
    public async Task Reduction_ExecuteInPlace_AccumulatesIntoD()
    {
        const int I = 3, J = 4;
        float[] a = [.. Enumerable.Range(0, I * J).Select(v => (float)v)];
        using var dA = new DeviceBuffer<float>(I * J);
        dA.CopyFrom(a);
        using var dD = new DeviceBuffer<float>(I);
        dD.CopyFrom([10f, 20f, 30f]);

        using var red = new TensorReduction([I, J], [0, 1], [I], [0], options: new TensorContractionOptions { Blocking = true });
        // D = 2 * rowsum(A) + 1 * D
        red.ExecuteInPlace(dA.PointerAsInt, dD.PointerAsInt, alpha: 2.0, beta: 1.0);

        float[] d = dD.ToArray();
        float[] start = [10f, 20f, 30f];
        for (int i = 0; i < I; i++)
        {
            float sum = 0;
            for (int j = 0; j < J; j++)
            {
                sum += a[i + I * j];
            }

            await Assert.That(Math.Abs(d[i] - (2 * sum + start[i]))).IsLessThan(1e-4f);
        }
    }

    [Test]
    public async Task Contraction_Float16_UsesFloatScalars()
    {
        const int M = 4, N = 4, K = 4;
        using var dA = new DeviceBuffer<Half>(M * K);
        using var dB = new DeviceBuffer<Half>(K * N);
        using var dC = new DeviceBuffer<Half>(M * N);
        using var dD = new DeviceBuffer<Half>(M * N);
        Half[] a = [.. Enumerable.Range(0, M * K).Select(v => (Half)(v % 5))];
        Half[] eye = new Half[K * N];
        for (int i = 0; i < K; i++)
        {
            eye[i + K * i] = (Half)1;
        }

        dA.CopyFrom(a);
        dB.CopyFrom(eye);
        dC.Clear();
        dD.Clear();

        using var tc = new TensorContraction(
            [M, K], [0, 2], [K, N], [2, 1], [M, N], [0, 1], [M, N], [0, 1],
            TensorDataType.Float16, new TensorContractionOptions { Blocking = true });
        await tc.PlanAsync();
        await tc.ExecuteAsync(dA.PointerAsInt, dB.PointerAsInt, dC.PointerAsInt, dD.PointerAsInt, alpha: 1.0, beta: 0.0);

        Half[] d = dD.ToArray();
        for (int i = 0; i < M * N; i++)
        {
            await Assert.That(Math.Abs((float)d[i] - (float)a[i])).IsLessThan(1e-2f);
        }
    }
}
