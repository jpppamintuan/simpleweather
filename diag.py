"""
Run diagnostics for the ingest job. Standard library only; Linux-oriented
(reads /proc) and every call is best-effort -- nothing here raises.

Why it exists: ingest runs have been ending with only "The operation was
canceled" and no traceback, so the log can't say whether the process was
cancelled from outside (timeout / concurrency / manual), killed for lack of
memory, or crashed natively. This makes each of those leave a different
fingerprint in the log:

  * a periodic "[diag ...]" line (memory, disk, load, threads) -- the last
    one before the end shows what the machine looked like when it died;
  * SIGINT/SIGTERM (what GitHub sends on cancel/timeout) -> a
    "RECEIVED SIGTERM" line plus every thread's stack, i.e. what the code
    was doing at that moment;
  * an OOM kill is SIGKILL, which can't be caught -- so "no RECEIVED line,
    and the last [diag] line shows RAM nearly gone" means memory;
  * faulthandler also dumps the stack on a native crash (SIGSEGV etc.) and,
    every 5 minutes regardless, so a hang shows where it is hanging.
"""

from __future__ import annotations

import faulthandler
import os
import shutil
import signal
import sys
import threading
import time

_T0 = time.time()


def _read_kb_fields(path: str, wanted: set) -> dict:
    """Parses '/proc' style 'Key:   123 kB' lines into {key: MB}."""
    out = {}
    try:
        with open(path) as f:
            for line in f:
                key, _, rest = line.partition(":")
                if key in wanted:
                    out[key] = int(rest.split()[0]) / 1024.0
    except Exception:
        pass
    return out


def snapshot(tag: str = "") -> str:
    mem = _read_kb_fields("/proc/meminfo", {"MemTotal", "MemAvailable", "SwapFree"})
    me = _read_kb_fields("/proc/self/status", {"VmRSS", "VmHWM"})
    try:
        disk_free = shutil.disk_usage("/").free / 1e9
    except Exception:
        disk_free = float("nan")
    try:
        load1 = os.getloadavg()[0]
    except Exception:
        load1 = float("nan")
    return (
        f"[diag +{time.time() - _T0:5.0f}s] {tag} | "
        f"rss={me.get('VmRSS', 0):.0f}MB peak={me.get('VmHWM', 0):.0f}MB | "
        f"ram_avail={mem.get('MemAvailable', 0):.0f}/{mem.get('MemTotal', 0):.0f}MB "
        f"swap_free={mem.get('SwapFree', 0):.0f}MB | disk_free={disk_free:.1f}GB | "
        f"load1={load1:.1f} cpus={os.cpu_count()} threads={threading.active_count()}"
    )


def log(tag: str = "") -> None:
    try:
        print(snapshot(tag), flush=True)
    except Exception:
        pass


def start_monitor(interval_seconds: float = 20.0) -> None:
    """Background thread printing a snapshot every `interval_seconds`."""
    def loop():
        while True:
            time.sleep(interval_seconds)
            log("monitor")

    threading.Thread(target=loop, name="diag-monitor", daemon=True).start()


def _on_signal(signum, frame) -> None:
    name = signal.Signals(signum).name
    print(snapshot(f"RECEIVED {name} -- the process was told to stop from OUTSIDE "
                   f"(cancel / timeout / concurrency); a memory kill would be SIGKILL and print nothing"),
          flush=True)
    faulthandler.dump_traceback(all_threads=True)
    sys.stderr.flush()
    # Restore the default behaviour and re-deliver, so the exit status is unchanged.
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


def install(stack_dump_every_seconds: int = 300) -> None:
    """Call once, early, from the main thread."""
    try:
        faulthandler.enable(all_threads=True)  # native crash -> stack on stderr
        faulthandler.dump_traceback_later(stack_dump_every_seconds, repeat=True)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, _on_signal)
    except Exception as e:
        print(f"[diag] could not install handlers: {e}", flush=True)
