// FftSample — demonstrates 1-D complex FFT and round-trip accuracy.
// Requires a CUDA-capable GPU at runtime.

using System.Numerics;
using NVMathNet;
using FftPlan = NVMath.Fft.Fft;

Console.WriteLine("=== NVMath.Net FFT Sample ===");

// ── 1. 1-D C2C forward + inverse round-trip ───────────────────────────────────
const int N = 512;
var host = new Complex[N];
for (int i = 0; i < N; i++)
    host[i] = new Complex(Math.Sin(2 * Math.PI * i / N), 0);

await using var stream = new CudaStream();
using var buf = new DeviceBuffer<Complex>(N);
buf.CopyFrom(host);

// Forward FFT
using var fwd = await FftPlan.FftAsync(buf, new long[] { N }, stream: stream);
await stream.SynchronizeAsync();
Console.WriteLine("Forward FFT done.");

// Inverse FFT (in-place)
using var inv = await FftPlan.IFftAsync(buf, new long[] { N }, stream: stream);
await stream.SynchronizeAsync();

// Normalise
var result = buf.ToArray();
double maxErr = 0;
for (int i = 0; i < N; i++)
{
    double normalised = result[i].Real / N;
    maxErr = Math.Max(maxErr, Math.Abs(normalised - host[i].Real));
}
Console.WriteLine($"Round-trip max error: {maxErr:E3}");

// ── 2. Batch FFT (16 independent transforms) ─────────────────────────────────
const int Batch = 16;
var batchHost = new Complex[N * Batch];
for (int b = 0; b < Batch; b++)
for (int i = 0; i < N; i++)
    batchHost[b * N + i] = new Complex(Math.Cos(2 * Math.PI * i / N * (b + 1)), 0);

using var batchBuf = new DeviceBuffer<Complex>(N * Batch);
batchBuf.CopyFrom(batchHost);

using var batchFft = new FftPlan(new long[] { N },
    options: new NVMath.Fft.FftOptions { },
    stream: stream);
batchFft.Plan();
batchFft.Execute(NVMath.Fft.FftDirection.Forward);
await stream.SynchronizeAsync();
Console.WriteLine($"Batch FFT ({Batch}×{N}) done.");

Console.WriteLine("All FFT samples completed successfully.");
