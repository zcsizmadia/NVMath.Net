// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the MIT License.

using NVMathNet.Interop;

namespace NVMathNet;

/// <summary>
/// Represents a CUDA device and exposes device-level operations.
/// </summary>
public sealed class CudaDevice
{
    /// <summary>The CUDA device ordinal.</summary>
    public int DeviceId { get; }

    private CudaDevice(int deviceId) => DeviceId = deviceId;

    /// <summary>Returns the number of CUDA-capable devices installed in the system.</summary>
    public static int GetDeviceCount() => CudaRuntime.GetDeviceCount();

    /// <summary>Returns a <see cref="CudaDevice"/> for the given ordinal.</summary>
    public static CudaDevice Get(int deviceId = 0)
    {
        int count = GetDeviceCount();
        return deviceId < 0 || deviceId >= count ?
            throw new ArgumentOutOfRangeException(nameof(deviceId), $"Device {deviceId} is out of range (0..{count - 1}).") :
            new CudaDevice(deviceId);
    }

    /// <summary>Returns a <see cref="CudaDevice"/> that wraps the currently active CUDA device.</summary>
    public static CudaDevice GetCurrent() => new(CudaRuntime.GetDevice());

    /// <summary>Sets this device as the current CUDA device for the calling thread.</summary>
    public void SetAsCurrent() => CudaRuntime.SetDevice(DeviceId);

    /// <summary>
    /// Blocks until all operations enqueued on this device are complete.
    /// For async alternatives prefer <see cref="CudaStream.SynchronizeAsync"/>.
    /// </summary>
    public void Synchronize()
    {
        CudaRuntime.SetDevice(DeviceId);
        CudaRuntime.DeviceSynchronize();
    }

    /// <summary>Returns the CUDA driver version (e.g. 12050 means 12.5).</summary>
    public static int GetDriverVersion() => CudaRuntime.GetDriverVersion();

    /// <summary>Returns the CUDA runtime version.</summary>
    public static int GetRuntimeVersion() => CudaRuntime.GetRuntimeVersion();

    /// <inheritdoc/>
    public override string ToString() => $"CudaDevice({DeviceId})";
}
