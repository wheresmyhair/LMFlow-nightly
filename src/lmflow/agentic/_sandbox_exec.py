"""Apply POSIX resource limits and replace this helper with a sandbox command."""

import argparse
import json
import os
import resource
import sys

_RESOURCE_KEYS = {
    "cpu_seconds": resource.RLIMIT_CPU,
    "file_size_bytes": resource.RLIMIT_FSIZE,
    "memory_bytes": resource.RLIMIT_AS,
    "open_files": resource.RLIMIT_NOFILE,
    "processes": resource.RLIMIT_NPROC,
}


def _parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--limits", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    return args


def _apply_limits(encoded_limits):
    limits = json.loads(encoded_limits)
    if not isinstance(limits, dict):
        raise ValueError("limits must decode to an object")
    unknown = set(limits).difference(_RESOURCE_KEYS)
    if unknown:
        raise ValueError(f"unsupported resource limits: {sorted(unknown)}")
    for name, value in limits.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        resource.setrlimit(_RESOURCE_KEYS[name], (value, value))


def main():
    args = _parse_args()
    try:
        _apply_limits(args.limits)
        os.execvpe(args.command[0], args.command, os.environ)
    except (OSError, ValueError) as exc:
        print(f"LMFlow ProcessSandbox could not execute the command: {exc}", file=sys.stderr)
        return 127


if __name__ == "__main__":
    sys.exit(main())
