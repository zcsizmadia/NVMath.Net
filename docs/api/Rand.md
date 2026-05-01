# Random — `NVMathNet.Rand`

## `CudaRandom` : `IDisposable`

GPU random number generator backed by cuRAND.

| Member | Description |
|---|---|
| `CudaRandom(CuRandRngType rngType = PseudoDefault, ulong seed = 0, CudaStream? stream = null)` | Creates a generator. |
| `void SetSeed(ulong seed)` | Sets the seed. |
| `void FillUniform(DeviceBuffer<float> buffer)` | Fills with uniform (0, 1]. |
| `void FillUniform(DeviceBuffer<double> buffer)` | Double-precision uniform. |
| `Task FillUniformAsync(DeviceBuffer<float> buffer, CancellationToken ct = default)` | Async uniform fill. |
| `Task FillUniformAsync(DeviceBuffer<double> buffer, CancellationToken ct = default)` | Async double uniform fill. |
| `void FillNormal(DeviceBuffer<float> buffer, float mean = 0, float stddev = 1)` | Normal distribution. |
| `void FillNormal(DeviceBuffer<double> buffer, double mean = 0, double stddev = 1)` | Double-precision normal. |
| `Task FillNormalAsync(DeviceBuffer<float> buffer, float mean = 0, float stddev = 1, ...)` | Async normal fill. |
| `void FillLogNormal(DeviceBuffer<float> buffer, float mean = 0, float stddev = 1)` | Log-normal distribution. |

## `CuRandRngType` (enum)

| Value | Description |
|---|---|
| `PseudoDefault` (100) | Default PRNG (XORWOW). |
| `PseudoXorwow` (101) | XORWOW generator. |
| `PseudoMrg32k3a` (121) | MRG32k3a generator. |
| `PseudoMtgp32` (141) | Mersenne Twister MTGP32. |
| `PseudoMt19937` (142) | Mersenne Twister MT19937. |
| `PseudoPhilox4_32_10` (161) | PHILOX-4×32-10 generator. |
| `QuasiDefault` (200) | Default quasi-random (Sobol32). |
