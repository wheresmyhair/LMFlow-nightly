# Process sandbox backend

`ProcessSandbox` is the zero-container execution backend for lightweight
Agentic workflows. It runs an argument vector below a declared workspace root
on a POSIX host and provides the process-level guards needed by future tool
agents and rollout workers:

- a clean child environment with per-run `HOME` and temporary directories;
- working-directory containment, including rejection of symlink escapes;
- wall-clock timeouts and process-group cleanup;
- per-stream bounded in-memory stdout and stderr capture without pipe
  backpressure; and
- optional POSIX limits for CPU time, address space, file size, open files, and
  process count.

The backend exposes its capabilities and rejects a task when any required
capability is unavailable. Commands are accepted only as an argument sequence;
shell command strings are deliberately unsupported.

## Security boundary

This backend supervises processes and limits accidental resource loss. It does
not isolate filesystem access or network access, and its workspace-root check
only constrains the initial working directory. Code executed by this backend
must therefore be trusted to the same degree as the parent training process.
Callers that handle untrusted code must require stronger capabilities and use a
future backend that provides OS-level filesystem and network isolation.

Resource-limit behavior follows the host kernel. In particular, some limits
may be unavailable or ineffective for privileged users. Capability reporting
lets higher-level Agentic code fail closed instead of silently assuming a
security property that the selected backend does not provide.
