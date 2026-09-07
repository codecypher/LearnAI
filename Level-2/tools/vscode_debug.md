## Visual Studio Debugger

Visual Studio (VS) is a powerful debugging tools, but most developers only use the basics. Mastering the advanced features can reduce debugging time [1].

### VS Debugging Features:

- Conditional Breakpoints
- Tracepoints
- DebuggerDisplay Attribute

### NET CLI Debugging Tools

The .NET CLI provides powerful tools to debug outside of Visual Studio, especially in production or containerized environments.

- dotnet-dump
- dotnet-trace
- dotnet-counters

### Debugging Asynchronous Code

Async/await can simplify code but it also presents unique debugging issues such as unawaited tasks, deadlocks or “lost” exceptions can make debugging difficult.

Techniques for Debugging Async Code:

1. Inspect the Call Stack for Async Methods

The Call Stack window is your main tool for tracking asynchronous workflows. Look for methods marked with [Async], which indicate the current state of async calls.

These [Async] frames show you where the code is paused, including the awaited methods and their origins.

TIP: Double-click on a [Async] frame to navigate directly to the corresponding code.

2. Breakpoints and Async Code

Use breakpoints strategically before and after await statements. Async workflows "hop" between threads, so execution won’t follow a linear stack.

Use conditional breakpoints to stop execution only when specific conditions are met (e.g. someVariable == null).

3. Enable “Enable Parallel Stacks for Tasks”:
Enabling this option provides insight into active tasks.

Go to Tools > Options > Debugging > General and enable Show parallel tasks in the Threads window.

Once enabled, open the Threads window (Debug > Windows > Threads) to see a list of tasks, including their status and current execution point.

4. Diagnose Deadlocks and Stuck Tasks:
Symptoms: Debugging stops but doesn’t advance, or the app hangs indefinitely.

Solution: Look for synchronous calls to Task.Wait() or Task.Result, which can cause deadlocks.
Replace these calls with await.

TIP: Use Exception Settings (Debug > Windows > Exception Settings) to break on specific exceptions in async workflows. This is helpful when unhandled exceptions "disappear" due to async continuations.

### Debugging in Production: Tools for the Real World

Some bugs only show up in production. You can’t connect Visual Studio to a live environment, but modern tools make remote debugging much easier.

1. Application Insights (Azure)

Collect logs, traces, and exception details from live applications.

Set up alerts for specific exceptions or performance issues.

2. Seq and Serilog:

Use Serilog for structured logging and integrate it with Seq to explore logs with advanced queries.

```zsh
  dotnet add package Serilog.Sinks.Seq
```

3. dotnet-monitor

Capture diagnostics from a live .NET process without disrupting it.

```zsh
  dotnet tool install --global dotnet-monitor
  dotnet monitor collect
```

TIP: Use exception filters to catch unexpected exceptions in production and automatically log detailed information.

### Memory and Performance Debugging: Solve the Hard Bugs

Memory leaks and performance issues can be the most difficult bugs to find.

1. Visual Studio Profiler:

Use Debug > Performance Profiler to analyze CPU and memory usage.

2. dotMemory (JetBrains): A powerful tool for detecting memory leaks and analyzing allocations.

3. BenchmarkDotNet: Identify slow code with precise benchmarks.

dotnet add package BenchmarkDotNet

Common Issues to Look For:

Memory Leaks:

- Missing Dispose() calls.
- Retaining objects unnecessarily in static fields.

Excessive Allocations:

- Use `Span<T>` or `ArrayPool<T>` to reduce GC pressure for performance-critical code.

TIP: Always profile in an environment close to production. Debug builds can hide performance issues that only surface in release mode.


## References

[1]: [Debug Smarter, Not Harder: Advanced .NET Debugging](https://medium.com/codex/debug-smarter-not-harder-advanced-net-debugging-tools-and-techniques-%EF%B8%8F-%EF%B8%8F-65a9f4f10cd6)

