// Copyright (c) 2026 Zoltan Csizmadia. All rights reserved.
// Licensed under the MIT License.

namespace NVMathNet;

/// <summary>Severity of a log message emitted through <see cref="NVMathLog"/>.</summary>
public enum NVMathLogLevel
{
    /// <summary>Very detailed diagnostics (per-call information).</summary>
    Trace = 0,
    /// <summary>Diagnostics useful when investigating behaviour: plan details, workspace sizes.</summary>
    Debug = 1,
    /// <summary>Noteworthy events: native libraries loaded, autotune results.</summary>
    Information = 2,
    /// <summary>Unexpected but recoverable situations.</summary>
    Warning = 3,
    /// <summary>A native call failed. An exception is thrown right after the message is logged.</summary>
    Error = 4,
    /// <summary>Disables logging when used as <see cref="NVMathLog.MinimumLevel"/>.</summary>
    None = 5,
}

/// <summary>
/// Dependency-free logging hook. By default nothing is logged. Install a <see cref="Handler"/> to receive messages
/// and bridge them to a logging framework, for example:
/// <code>
/// NVMathLog.MinimumLevel = NVMathLogLevel.Information;
/// NVMathLog.Handler = (level, category, message) =>
///     logger.Log(level switch { NVMathLogLevel.Error => LogLevel.Error, NVMathLogLevel.Warning => LogLevel.Warning,
///                               NVMathLogLevel.Information => LogLevel.Information, _ => LogLevel.Debug },
///                "{Category}: {Message}", category, message);
/// </code>
/// The handler is invoked synchronously on the thread that produced the event, possibly from several threads at
/// once, so it must be thread-safe and fast. Exceptions thrown by the handler are swallowed.
/// </summary>
public static class NVMathLog
{
    private static Action<NVMathLogLevel, string, string>? _handler;
    private static int _minimumLevel = (int)NVMathLogLevel.Information;

    /// <summary>
    /// Receives <c>(level, category, message)</c> for every message at or above <see cref="MinimumLevel"/>.
    /// Categories are short library or component names such as <c>cuFFT</c>, <c>Matmul</c> or <c>Loader</c>.
    /// <c>null</c> (the default) disables logging.
    /// </summary>
    public static Action<NVMathLogLevel, string, string>? Handler
    {
        get => Volatile.Read(ref _handler);
        set => Volatile.Write(ref _handler, value);
    }

    /// <summary>Messages below this level are not delivered to <see cref="Handler"/>. Default: <see cref="NVMathLogLevel.Information"/>.</summary>
    public static NVMathLogLevel MinimumLevel
    {
        get => (NVMathLogLevel)Volatile.Read(ref _minimumLevel);
        set => Volatile.Write(ref _minimumLevel, (int)value);
    }

    /// <summary>Returns <c>true</c> if a message at <paramref name="level"/> would be delivered.</summary>
    public static bool IsEnabled(NVMathLogLevel level) => Handler is not null && level >= MinimumLevel;

    internal static void Log(NVMathLogLevel level, string category, string message)
    {
        var handler = Handler;
        if (handler is null || level < MinimumLevel)
        {
            return;
        }

        try
        {
            handler(level, category, message);
        }
        catch
        {
            // A faulty handler must never break a GPU operation.
        }
    }

    /// <summary>Logs lazily: <paramref name="message"/> is only evaluated when the level is enabled.</summary>
    internal static void Log(NVMathLogLevel level, string category, Func<string> message)
    {
        if (IsEnabled(level))
        {
            Log(level, category, message());
        }
    }

    internal static void Debug(string category, Func<string> message) => Log(NVMathLogLevel.Debug, category, message);

    internal static void Info(string category, Func<string> message) => Log(NVMathLogLevel.Information, category, message);

    internal static void Error(string category, string message) => Log(NVMathLogLevel.Error, category, message);
}
