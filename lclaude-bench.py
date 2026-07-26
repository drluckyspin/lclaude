#!/usr/bin/env python3
"""Benchmark local Anthropic Messages API backends used by lclaude.

The benchmark measures raw streaming inference, not Claude Code's tool loop or
terminal UI. It reads lclaude's last-used preferences but never writes config,
patches Claude settings, or launches Claude Code.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import lclaude


__version__ = "0.4.0"


QUICK_PROMPT = (
    "Explain, in two concise paragraphs, how a command-line tool should "
    "measure local LLM inference performance. Include one practical caveat."
)


@dataclass(frozen=True)
class Workload:
    """A fixed prompt and generation cap for one benchmark stage."""

    name: str
    label: str
    prompt: str
    max_tokens: int


MAKEFILE_PROMPT = """\
Create a complete Makefile for a small Python 3.11+ project containing
`lclaude.py` and `lclaude-bench.py`. Both scripts are stdlib-only command-line
tools and must run directly with `python3`; do not add dependency installation,
virtualenv, package, or network targets.

The Makefile must be portable to macOS/Linux POSIX make and shell. Include
.PHONY, a helpful default/help target, and a `check` target that syntax-checks
both scripts with `python3 -c "import ast; ast.parse(open(...).read())"`.
Include a `bench-dry-run` target that runs `python3 lclaude-bench.py --dry-run`.
Output only a valid Makefile with concise comments, no prose.
"""

MAKEFILE_FULL_PROMPT = """\
Create a complete, production-quality Makefile for this local-LLM CLI project.
The repository has two executable Python 3.11+ scripts:

- `lclaude.py` launches Claude Code against a local Ollama, llama.cpp, or
  Managed llama.cpp Messages API endpoint.
- `lclaude-bench.py` directly benchmarks streaming `POST /v1/messages` on
  Ollama and Managed llama.cpp. It is read-only: it must never write lclaude
  config, patch Claude settings, or launch Claude Code.

Use only portable POSIX make and shell features that work on macOS and Linux.
Do not add package-install, virtualenv, lint-tool, download, Docker, or network
targets: the project deliberately has no third-party Python dependencies.

Provide:
1. `.DEFAULT_GOAL := help`, `.PHONY`, and a self-documenting `help` target.
2. `check`, which syntax-checks each script with
   `python3 -c "import ast; ast.parse(open('SCRIPT').read())"`.
3. `bench-dry-run`, which prints the resolved benchmark workload/targets without
   contacting a backend.
4. `bench`, configurable with `MODEL`, `BACKENDS`, `WARMUP`, `REPEATS`, and
   `BENCH_ARGS` variables, passing them safely to lclaude-bench.
5. `bench-json`, which writes JSON to a configurable `BENCH_OUTPUT` path while
   preserving lclaude-bench progress on stderr.
6. A `test` alias for `check`, and concise comments explaining the no-dependency
   and local-backend assumptions.

The benchmark’s default workload includes increasingly substantial Makefile
generation tasks, while `--quick` runs only its short latency request. Make
targets must not modify user configuration or Claude settings. Output only a
valid Makefile; do not explain it.
"""

DEFAULT_WORKLOADS = (
    Workload("quick", "Quick latency check", QUICK_PROMPT, 256),
    Workload("makefile", "Makefile generation", MAKEFILE_PROMPT, 768),
    Workload("makefile-full", "Full Makefile generation", MAKEFILE_FULL_PROMPT, 1536),
)
DEFAULT_BACKENDS = (lclaude.BACKEND_OLLAMA, lclaude.BACKEND_MANAGED)
VALID_BACKENDS = (
    lclaude.BACKEND_OLLAMA,
    lclaude.BACKEND_MANAGED,
    lclaude.BACKEND_LLAMACPP,
)


def backend_label(backend: str) -> str:
    """Return the unambiguous user-facing name for a backend keyword."""
    if backend == lclaude.BACKEND_MANAGED:
        return "Managed llama.cpp"
    if backend == lclaude.BACKEND_LLAMACPP:
        return "llama.cpp"
    return "Ollama"


@dataclass
class RunResult:
    """Metrics for one streaming Messages API request."""

    run: int
    ok: bool
    ttft_ms: float | None = None
    total_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    decode_tok_per_s: float | None = None
    error: str | None = None


@dataclass
class WorkloadResult:
    """All measured requests for one workload on one backend."""

    workload: Workload
    runs: list[RunResult]
    error: str | None = None


@dataclass
class BackendResult:
    """All results and metadata for a benchmarked backend."""

    name: str
    port: int
    version: str = "unknown"
    startup_s: float | None = None
    blob_path: str | None = None
    error: str | None = None
    workloads: list[WorkloadResult] | None = None


class Progress:
    """Write step-by-step stderr feedback, with a spinner on interactive TTYs."""

    _FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

    def __init__(self, *, quiet: bool, verbose: bool, no_color: bool) -> None:
        self.quiet = quiet
        self.verbose = verbose
        self.interactive = sys.stderr.isatty()
        self.color = lclaude._use_color(sys.stderr) and not no_color
        self.accent = lclaude.HEADER_ACCENT if self.color else ""
        self.error = lclaude.ERROR_ACCENT if self.color else ""
        self.reset = lclaude.ANSI_RESET if self.color else ""

    def note(self, message: str, *, detail: bool = True) -> None:
        """Print a milestone, hiding detailed steps outside verbose mode."""
        if not self.quiet and (self.verbose or not detail):
            print(f"{self.accent}•{self.reset} {message}", file=sys.stderr, flush=True)

    @contextmanager
    def activity(self, message: str, *, detail: bool = True) -> Any:
        """Show a spinner while a blocking operation is in progress."""
        if self.quiet or (detail and not self.verbose):
            yield
            return
        if not self.interactive:
            self.note(f"{message} …", detail=False)
            try:
                yield
            except BaseException:
                print(
                    f"{self.error}✗{self.reset} {message} failed",
                    file=sys.stderr,
                    flush=True,
                )
                raise
            else:
                print(
                    f"{self.accent}✓{self.reset} {message} complete",
                    file=sys.stderr,
                    flush=True,
                )
            return

        done = threading.Event()

        def render() -> None:
            frame = 0
            while not done.is_set():
                sys.stderr.write(
                    f"\r\x1b[2K{self.accent}{self._FRAMES[frame]}"
                    f"{self.reset} {message}"
                )
                sys.stderr.flush()
                frame = (frame + 1) % len(self._FRAMES)
                done.wait(0.1)

        thread = threading.Thread(target=render, daemon=True)
        thread.start()
        try:
            yield
        except BaseException:
            done.set()
            thread.join()
            sys.stderr.write(
                f"\r\x1b[2K{self.error}✗{self.reset} {message} failed\n"
            )
            sys.stderr.flush()
            raise
        else:
            done.set()
            thread.join()
            sys.stderr.write(
                f"\r\x1b[2K{self.accent}✓{self.reset} {message}\n"
            )
            sys.stderr.flush()


class RunSeries:
    """Render repeated benchmark requests as one live horizontal status line."""

    def __init__(self, progress: Progress, label: str, total: int) -> None:
        self.progress = progress
        self.label = label
        self.total = total
        self.results: list[bool] = []
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.thread: threading.Thread | None = None

        if progress.quiet or not progress.verbose:
            return
        if progress.interactive:
            self.thread = threading.Thread(target=self._render_loop, daemon=True)
            self.thread.start()
        else:
            sys.stderr.write(
                f"  {progress.accent}•{progress.reset} {label}: "
            )
            sys.stderr.flush()

    def _line(self, *, spinner: str = "") -> str:
        with self.lock:
            symbols = [
                (
                    f"{self.progress.accent}✓{self.progress.reset}"
                    if ok
                    else f"{self.progress.error}✗{self.progress.reset}"
                )
                for ok in self.results
            ]
            completed = len(self.results)
        if completed < self.total and spinner:
            symbols.append(
                f"{self.progress.accent}{spinner}{self.progress.reset}"
            )
        if completed < self.total and self.done.is_set():
            symbols.extend("—" for _ in range(self.total - completed))
        return (
            f"  {self.progress.accent}•{self.progress.reset} "
            f"{self.label}: {' '.join(symbols)}"
        )

    def _render_loop(self) -> None:
        frame = 0
        while not self.done.is_set():
            sys.stderr.write(f"\r\x1b[2K{self._line(spinner=Progress._FRAMES[frame])}")
            sys.stderr.flush()
            frame = (frame + 1) % len(Progress._FRAMES)
            self.done.wait(0.1)

    def record(self, ok: bool) -> None:
        """Append one completed request to the horizontal progress line."""
        with self.lock:
            self.results.append(ok)
        if (
            not self.progress.quiet
            and self.progress.verbose
            and not self.progress.interactive
        ):
            symbol = (
                f"{self.progress.accent}✓{self.progress.reset}"
                if ok
                else f"{self.progress.error}✗{self.progress.reset}"
            )
            sys.stderr.write(f"{symbol} ")
            sys.stderr.flush()

    def close(self) -> None:
        """Finish the status line once all requests have settled."""
        if self.progress.quiet or not self.progress.verbose:
            return
        self.done.set()
        if self.thread is not None:
            self.thread.join()
            sys.stderr.write(f"\r\x1b[2K{self._line()}\n")
        else:
            sys.stderr.write("\n")
        sys.stderr.flush()


def _env_int(name: str) -> int | None:
    """Use lclaude's validation without changing the parent environment."""
    return lclaude._env_int(name)


def _config_string(config: dict[str, Any], key: str) -> str | None:
    value = config.get(key)
    return value if isinstance(value, str) else None


def _config_int(config: dict[str, Any], key: str) -> int | None:
    value = config.get(key)
    return value if isinstance(value, int) else None


def parse_backends(raw: str) -> tuple[str, ...]:
    """Parse a comma-separated unique backend list."""
    backends = tuple(item.strip() for item in raw.split(",") if item.strip())
    invalid = [name for name in backends if name not in VALID_BACKENDS]
    if not backends or invalid:
        valid = ", ".join(VALID_BACKENDS)
        raise argparse.ArgumentTypeError(
            f"backends must be a comma-separated subset of: {valid}"
        )
    return tuple(dict.fromkeys(backends))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-h", "--help", action="store_true")
    parser.add_argument("--model", default=None, help="Model name to benchmark")
    parser.add_argument(
        "--backends",
        type=parse_backends,
        default=None,
        metavar="LIST",
        help="Comma-separated: ollama,managed,llamacpp",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override all selected backend ports (mirrors lclaude)",
    )
    parser.add_argument("--ollama-port", type=int, default=None)
    parser.add_argument("--managed-port", type=int, default=None)
    parser.add_argument("--llamacpp-port", type=int, default=None)
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt", default=None, help="Fixed benchmark prompt")
    prompt_group.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Read the fixed benchmark prompt from a UTF-8 file",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only the short 256-token latency workload",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum generated tokens for an explicit --prompt/--prompt-file",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--cooldown",
        type=float,
        default=0,
        help="Seconds to wait between backend suites",
    )
    parser.add_argument(
        "--unload-between",
        action="store_true",
        help="Stop Ollama's model between suites to reduce GPU residency bias",
    )
    parser.add_argument(
        "--keep-managed",
        action="store_true",
        help="Leave the managed llama.cpp server running after its suite",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600,
        help="Per-request socket timeout in seconds",
    )
    parser.add_argument("--json", action="store_true", help="Write full JSON to stdout")
    parser.add_argument("--no-color", action="store_true", help="Accepted for CLI consistency")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show setup, warmup, and individual-run progress",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show resolved targets without contacting local backends",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress messages (results still print)")
    return parser


def print_help(*, no_color: bool) -> None:
    """Print lclaude-style benchmark help without touching local backends."""
    heading = lclaude.HEADER_ACCENT if lclaude._use_color() and not no_color else ""
    reset = lclaude.ANSI_RESET if heading else ""
    help_text = (
        f"{heading}Usage:{reset} lclaude-bench [OPTIONS]\n"
        "\n"
        "Compare raw streaming inference through Ollama and llama.cpp\n"
        "Reads lclaude preferences but never saves them\n"
        "\n"
        f"{heading}Options:{reset}\n"
        "  -h, --help             Show this help message and exit\n"
        "  --model MODEL          Model to benchmark (default: ornith:35b or last-used)\n"
        "  --backends LIST        Comma-separated: ollama,managed,llamacpp\n"
        "                         managed = Managed llama.cpp\n"
        "                         Default: saved backend or ollama,managed for auto\n"
        "  --port PORT            Override all selected backend ports (mirrors lclaude)\n"
        "  --ollama-port PORT     Override Ollama's inference port\n"
        "  --managed-port PORT    Override Managed llama.cpp port\n"
        "  --llamacpp-port PORT   Override user-managed llama-server port\n"
        "\n"
        f"{heading}Workload:{reset}\n"
        "  Default: staged quick, Makefile, and full Makefile workloads\n"
        "  --quick                Run only the short 256-token latency workload\n"
        "  --prompt TEXT          Run one fixed custom benchmark prompt\n"
        "  --prompt-file PATH     Run one fixed custom UTF-8 prompt from a file\n"
        "  --max-tokens N         Generation cap for an explicit prompt (default: 256)\n"
        "  --warmup N             Discarded warmup requests per backend (default: 1)\n"
        "  --repeats N            Measured requests per backend (default: 5)\n"
        "  --timeout SECONDS      Per-request socket timeout (default: 600)\n"
        "  --cooldown SECONDS     Wait between backend suites (default: 0)\n"
        "  --unload-between       Stop the Ollama model between backend suites\n"
        "  --keep-managed         Leave the owned managed llama.cpp server running\n"
        "\n"
        f"{heading}Output:{reset}\n"
        "  --json                 Write individual runs and summaries as JSON\n"
        "  --dry-run              Show resolved targets without contacting local backends\n"
        "  --no-color             Disable colored section headings\n"
        "  -v, --verbose          Show setup, warmup, and individual-run progress\n"
        "  --quiet                Suppress progress messages (results still print)\n"
        "\n"
        f"{heading}Config:{reset}\n"
        f"  {lclaude.CONFIG_FILE}\n"
        "  CLI > LCLAUDE_* env > config > built-in defaults (read-only)\n"
        "\n"
        f"{heading}Examples:{reset}\n"
        "  lclaude-bench                              # staged Ollama vs Managed llama.cpp\n"
        "  lclaude-bench --quick                      # short latency comparison\n"
        "  lclaude-bench --model ornith:35b --repeats 5\n"
        "  lclaude-bench --backends ollama,managed --unload-between --cooldown 10\n"
        "  lclaude-bench --backends llamacpp --llamacpp-port 8080\n"
        "  lclaude-bench --dry-run                   # inspect resolved targets only\n"
        "  lclaude-bench --json > benchmark.json\n"
        "\n"
        f"{heading}Prerequisites:{reset}\n"
        "  brew install ollama                        # Ollama benchmark target\n"
        "  brew install llama.cpp                     # Managed llama.cpp / llamacpp target\n"
        "  ollama pull ornith:35b                     # benchmark model\n"
    )
    print(help_text, file=sys.stdout)


def _bench_header_row(label: str, value: str, width: int, *, dim: str, accent: str, reset: str) -> str:
    """Build one clipped, styled row for the benchmark status box."""
    prefix = f" {dim}{label:<8}:{reset} {accent}"
    available = width - 2 - lclaude._visible_len(prefix) - lclaude._visible_len(reset)
    if len(value) > available:
        value = value[: max(0, available - 1)] + "…"
    return lclaude._box_row(f"{prefix}{value}{reset}", width, accent, reset)


def print_bench_header(
    model: str,
    backends: tuple[str, ...],
    ports: dict[str, int],
    workloads: tuple[Workload, ...],
    args: argparse.Namespace,
    *,
    stream: Any,
) -> None:
    """Print a lclaude-style box with the current benchmark parameters."""
    columns, _ = shutil.get_terminal_size((80, 20))
    width = max(columns, 52)
    color = lclaude._use_color(stream) and not args.no_color
    dim = lclaude.DIM_ACCENT if color else ""
    accent = lclaude.HEADER_ACCENT if color else ""
    reset = lclaude.ANSI_RESET if color else ""
    targets = "  ·  ".join(
        f"{backend_label(backend)} port {ports[backend]}" for backend in backends
    )
    workload_names = "  ·  ".join(
        f"{item.name} (max {item.max_tokens})" for item in workloads
    )
    workload = (
        f"{len(workloads)} workload{'s' if len(workloads) != 1 else ''}: {workload_names}"
    )
    runs = f"{args.warmup} warmup + {args.repeats} measured per workload"
    controls: list[str] = []
    if args.unload_between:
        controls.append("unload between")
    if args.cooldown:
        controls.append(f"{args.cooldown:g}s cooldown")
    if controls:
        workload += "  ·  " + "  ·  ".join(controls)

    print(
        lclaude._box_top(f"LCLAUDE-BENCH v{__version__}", width, accent, reset),
        file=stream,
    )
    print(
        _bench_header_row("Model", model, width, dim=dim, accent=accent, reset=reset),
        file=stream,
    )
    print(
        _bench_header_row(
            "Targets", targets, width, dim=dim, accent=accent, reset=reset
        ),
        file=stream,
    )
    print(
        _bench_header_row(
            "Workload", workload, width, dim=dim, accent=accent, reset=reset
        ),
        file=stream,
    )
    print(
        _bench_header_row("Runs", runs, width, dim=dim, accent=accent, reset=reset),
        file=stream,
    )
    print(lclaude._box_bottom(width, accent, reset), file=stream, flush=True)


def dry_run_document(
    model: str,
    backends: tuple[str, ...],
    ports: dict[str, int],
    workloads: tuple[Workload, ...],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Return the side-effect-free benchmark plan for human or JSON output."""
    return {
        "tool": "lclaude-bench",
        "dry_run": True,
        "model": model,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "workloads": [
            {
                "name": workload.name,
                "label": workload.label,
                "prompt_sha256": hashlib.sha256(workload.prompt.encode()).hexdigest(),
                "prompt_chars": len(workload.prompt),
                "max_tokens": workload.max_tokens,
            }
            for workload in workloads
        ],
        "backends": [
            {
                "name": backend,
                "label": backend_label(backend),
                "port": ports[backend],
            }
            for backend in backends
        ],
    }


def resolve_options(args: argparse.Namespace) -> tuple[str, tuple[str, ...], dict[str, int]]:
    """Resolve model, selected targets, and ports with lclaude-like precedence."""
    config = lclaude.load_config()
    cfg_model = _config_string(config, "model")
    cfg_backend = _config_string(config, "backend")
    cfg_port = _config_int(config, "port")
    env_model = os.environ.get("LCLAUDE_MODEL") or None
    env_backend = os.environ.get("LCLAUDE_BACKEND") or None
    env_port = _env_int("LCLAUDE_PORT")

    model = args.model or env_model or cfg_model or lclaude.DEFAULT_MODEL
    requested_backend = env_backend or cfg_backend or lclaude.BACKEND_AUTO
    if requested_backend not in (*VALID_BACKENDS, lclaude.BACKEND_AUTO):
        print(
            f"Warning: ignoring invalid benchmark backend preference: {requested_backend!r}",
            file=sys.stderr,
        )
        requested_backend = lclaude.BACKEND_AUTO

    if args.backends is not None:
        backends = args.backends
    elif requested_backend == lclaude.BACKEND_AUTO:
        backends = DEFAULT_BACKENDS
    else:
        backends = (requested_backend,)

    explicit_port = args.port if args.port is not None else env_port or cfg_port
    ports = {
        lclaude.BACKEND_OLLAMA: (
            args.ollama_port
            or explicit_port
            or lclaude.DEFAULT_PORTS[lclaude.BACKEND_OLLAMA]
        ),
        lclaude.BACKEND_MANAGED: (
            args.managed_port
            or explicit_port
            or lclaude.DEFAULT_PORTS[lclaude.BACKEND_MANAGED]
        ),
        lclaude.BACKEND_LLAMACPP: (
            args.llamacpp_port
            or explicit_port
            or lclaude.DEFAULT_PORTS[lclaude.BACKEND_LLAMACPP]
        ),
    }
    return model, backends, ports


def resolve_workloads(args: argparse.Namespace) -> tuple[Workload, ...]:
    """Resolve the default suite or one explicit benchmark workload."""
    if args.quick and (args.prompt is not None or args.prompt_file is not None):
        raise ValueError("--quick cannot be combined with --prompt or --prompt-file")
    if args.quick and args.max_tokens is not None:
        raise ValueError("--quick uses its fixed 256-token cap; omit --max-tokens")
    if args.quick:
        return (DEFAULT_WORKLOADS[0],)

    if args.prompt_file is not None:
        try:
            prompt = args.prompt_file.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"could not read prompt file {args.prompt_file}: {exc}") from exc
        if not prompt.strip():
            raise ValueError("benchmark prompt must not be empty")
        return (
            Workload(
                "custom-file",
                "Custom prompt file",
                prompt,
                args.max_tokens or DEFAULT_WORKLOADS[0].max_tokens,
            ),
        )
    if args.prompt is not None:
        if not args.prompt.strip():
            raise ValueError("benchmark prompt must not be empty")
        return (
            Workload(
                "custom",
                "Custom prompt",
                args.prompt,
                args.max_tokens or DEFAULT_WORKLOADS[0].max_tokens,
            ),
        )
    if args.max_tokens is not None:
        raise ValueError("--max-tokens requires --prompt or --prompt-file")
    return DEFAULT_WORKLOADS


def stream_messages(
    port: int, model: str, prompt: str, max_tokens: int, timeout: float, run: int
) -> RunResult:
    """POST one streaming Anthropic Messages request and collect timing metrics."""
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "messages": [{"role": "user", "content": prompt}],
    }
    started = time.perf_counter()
    first_token_at: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    conn = http.client.HTTPConnection("localhost", port, timeout=timeout)
    try:
        conn.request(
            "POST",
            "/v1/messages",
            body=json.dumps(payload),
            headers={
                "Content-Type": "application/json",
                "x-api-key": "lclaude",
                "anthropic-version": "2023-06-01",
            },
        )
        response = conn.getresponse()
        if response.status != 200:
            body = response.read(2048).decode("utf-8", errors="replace").strip()
            return RunResult(
                run=run, ok=False, error=f"HTTP {response.status}: {body or response.reason}"
            )

        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            event_type = event.get("type")
            if event_type == "message_start":
                usage = event.get("message", {}).get("usage", {})
                if isinstance(usage.get("input_tokens"), int):
                    input_tokens = usage["input_tokens"]
            elif event_type == "content_block_delta":
                delta = event.get("delta", {})
                token_text = (
                    delta.get("text")
                    or delta.get("thinking")
                    or delta.get("partial_json")
                )
                if isinstance(token_text, str) and token_text:
                    first_token_at = first_token_at or time.perf_counter()
            elif event_type == "message_delta":
                usage = event.get("usage", {})
                if isinstance(usage.get("output_tokens"), int):
                    output_tokens = usage["output_tokens"]

        completed = time.perf_counter()
    except (OSError, http.client.HTTPException, TimeoutError) as exc:
        return RunResult(run=run, ok=False, error=str(exc))
    finally:
        conn.close()

    total_ms = (completed - started) * 1000
    ttft_ms = ((first_token_at or completed) - started) * 1000
    decode_seconds = max((completed - (first_token_at or completed)), 0)
    decode_tok_per_s = (
        output_tokens / decode_seconds
        if output_tokens is not None and decode_seconds > 0
        else None
    )
    return RunResult(
        run=run,
        ok=True,
        ttft_ms=ttft_ms,
        total_ms=total_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        decode_tok_per_s=decode_tok_per_s,
    )


def prepare_backend(
    name: str, model: str, port: int, progress: Progress
) -> tuple[str, str | None, Any | None, float]:
    """Pre-flight a backend and return its version, blob, and owned process."""
    started = time.perf_counter()
    proc: Any | None = None
    blob_path: str | None = None
    if name == lclaude.BACKEND_OLLAMA:
        with progress.activity(f"Connecting to Ollama on port {port}"):
            ok, reason = lclaude.check_ollama(port, auto_start=True)
            if not ok:
                raise RuntimeError(reason or "Ollama pre-flight failed")
        with progress.activity(f"Checking Ollama model {model}"):
            lclaude.ensure_model_in_ollama(model)
    elif name == lclaude.BACKEND_MANAGED:
        ollama_port = lclaude.DEFAULT_PORTS[lclaude.BACKEND_OLLAMA]
        with progress.activity(f"Connecting to Ollama on port {ollama_port}"):
            ok, reason = lclaude.check_ollama(ollama_port, auto_start=True)
            if not ok:
                raise RuntimeError(reason or "Ollama pre-flight failed")
        with progress.activity(f"Checking Ollama model {model}"):
            lclaude.ensure_model_in_ollama(model)
        with progress.activity(f"Resolving the GGUF blob for {model}"):
            blob_path = lclaude.resolve_ollama_blob(model)
        with progress.activity(f"Loading Managed llama.cpp on port {port}"):
            proc = lclaude.prepare_managed_backend(model, port)
    elif name == lclaude.BACKEND_LLAMACPP:
        with progress.activity(f"Connecting to llama-server on port {port}"):
            ok, reason = lclaude.check_llamacpp(port)
            if not ok:
                raise RuntimeError(reason or "llama-server pre-flight failed")
    else:
        raise RuntimeError(f"unsupported backend: {name}")
    return lclaude._BACKEND_VERSION, blob_path, proc, time.perf_counter() - started


def unload_ollama_model(model: str) -> None:
    """Ask Ollama to unload a model; errors are non-fatal cleanup noise."""
    try:
        subprocess.run(
            ["ollama", "stop", model],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=lclaude.OLLAMA_LIST_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass


def benchmark_backend(
    name: str,
    model: str,
    port: int,
    workloads: tuple[Workload, ...],
    args: argparse.Namespace,
    progress: Progress,
) -> BackendResult:
    """Start one backend, then warm up and measure every requested workload."""
    result = BackendResult(name=name, port=port, workloads=[])
    proc: Any | None = None
    try:
        result.version, result.blob_path, proc, result.startup_s = prepare_backend(
            name, model, port, progress
        )
        progress.note(f"{backend_label(name)} ready on port {port} ({result.version})")
        for workload in workloads:
            progress.note(
                f"{backend_label(name)} — {workload.label}: "
                f"{args.warmup} warmup{'s' if args.warmup != 1 else ''}, "
                f"{args.repeats} measured, max {workload.max_tokens} tokens"
            )
            workload_result = WorkloadResult(workload=workload, runs=[])
            if args.warmup:
                warmups = RunSeries(
                    progress,
                    f"{backend_label(name)} {workload.name} warmups",
                    args.warmup,
                )
                for index in range(args.warmup):
                    warmup = stream_messages(
                        port,
                        model,
                        workload.prompt,
                        workload.max_tokens,
                        args.timeout,
                        -(index + 1),
                    )
                    warmups.record(warmup.ok)
                    if not warmup.ok:
                        workload_result.error = f"warmup failed: {warmup.error}"
                        break
                warmups.close()
            if workload_result.error is None:
                runs = RunSeries(
                    progress,
                    f"{backend_label(name)} {workload.name} runs",
                    args.repeats,
                )
                for index in range(args.repeats):
                    workload_result.runs.append(
                        stream_messages(
                            port,
                            model,
                            workload.prompt,
                            workload.max_tokens,
                            args.timeout,
                            index + 1,
                        )
                    )
                    runs.record(workload_result.runs[-1].ok)
                runs.close()
            result.workloads.append(workload_result)
            successful = sum(run.ok for run in workload_result.runs)
            progress.note(
                f"{backend_label(name)} — {workload.label} complete: "
                f"{successful}/{len(workload_result.runs)} runs succeeded"
            )
        return result
    except SystemExit:
        result.error = "backend setup failed (see diagnostic above)"
        return result
    except (OSError, RuntimeError) as exc:
        result.error = str(exc)
        return result
    finally:
        if name == lclaude.BACKEND_MANAGED and not args.keep_managed:
            with progress.activity("Stopping Managed llama.cpp"):
                lclaude._stop_process(proc)


def summary(values: list[float | int | None]) -> dict[str, float] | None:
    numbers = [float(value) for value in values if value is not None]
    if not numbers:
        return None
    return {
        "median": statistics.median(numbers),
        "min": min(numbers),
        "max": max(numbers),
    }


def _backend_document(
    result: BackendResult, workload: Workload, workload_result: WorkloadResult | None
) -> dict[str, Any]:
    """Serialize one backend's outcome for one workload."""
    runs = workload_result.runs if workload_result is not None else []
    successful = [run for run in runs if run.ok]
    return {
        "name": result.name,
        "port": result.port,
        "version": result.version,
        "startup_s": result.startup_s,
        "blob_path": result.blob_path,
        "error": (workload_result.error if workload_result else None) or result.error,
        "runs": [asdict(run) for run in runs],
        "successes": len(successful),
        "attempts": len(runs),
        "metrics": {
            "ttft_ms": summary([run.ttft_ms for run in successful]),
            "total_ms": summary([run.total_ms for run in successful]),
            "output_tokens": summary([run.output_tokens for run in successful]),
            "decode_tok_per_s": summary(
                [run.decode_tok_per_s for run in successful]
            ),
        },
    }


def _metric_comparison(
    left: dict[str, Any],
    right: dict[str, Any],
    metric: str,
    *,
    lower_is_better: bool,
) -> dict[str, Any] | None:
    """Compare two median metrics, returning the winner and relative delta."""
    left_summary = left["metrics"][metric]
    right_summary = right["metrics"][metric]
    if left_summary is None or right_summary is None:
        return None
    left_value = left_summary["median"]
    right_value = right_summary["median"]
    if left_value == right_value:
        return {
            "winner": None,
            "relative_difference_pct": 0.0,
            "ollama_median": left_value,
            "managed_median": right_value,
        }
    if (left_value < right_value) == lower_is_better:
        winner, winner_value, other_value = "ollama", left_value, right_value
    else:
        winner, winner_value, other_value = "managed", right_value, left_value
    difference = (
        (other_value - winner_value) / other_value
        if lower_is_better
        else (winner_value - other_value) / other_value
    )
    return {
        "winner": winner,
        "relative_difference_pct": difference * 100,
        "ollama_median": left_value,
        "managed_median": right_value,
    }


def workload_comparison(backends: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return steady-state Ollama vs Managed llama.cpp comparisons when possible."""
    by_name = {backend["name"]: backend for backend in backends}
    ollama = by_name.get(lclaude.BACKEND_OLLAMA)
    managed = by_name.get(lclaude.BACKEND_MANAGED)
    if ollama is None or managed is None:
        return None
    comparison = {
        "ollama_backend": lclaude.BACKEND_OLLAMA,
        "managed_backend": lclaude.BACKEND_MANAGED,
        "total_latency": _metric_comparison(
            ollama, managed, "total_ms", lower_is_better=True
        ),
        "decode_throughput": _metric_comparison(
            ollama, managed, "decode_tok_per_s", lower_is_better=False
        ),
    }
    return (
        comparison
        if comparison["total_latency"] is not None
        or comparison["decode_throughput"] is not None
        else None
    )


def _workload_document(
    workload: Workload, results: list[BackendResult]
) -> dict[str, Any]:
    """Serialize one workload across all benchmarked backends."""
    backends = [
        _backend_document(
            result,
            workload,
            next(
                (
                    item
                    for item in result.workloads or []
                    if item.workload.name == workload.name
                ),
                None,
            ),
        )
        for result in results
    ]
    return {
        "name": workload.name,
        "label": workload.label,
        "prompt_sha256": hashlib.sha256(workload.prompt.encode()).hexdigest(),
        "prompt_chars": len(workload.prompt),
        "max_tokens": workload.max_tokens,
        "backends": backends,
        "comparison": workload_comparison(backends),
    }


def result_document(
    model: str,
    workloads: tuple[Workload, ...],
    args: argparse.Namespace,
    results: list[BackendResult],
) -> dict[str, Any]:
    """Build stable machine-readable output, grouped by comparable workload."""
    return {
        "tool": "lclaude-bench",
        "model": model,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "workloads": [_workload_document(workload, results) for workload in workloads],
    }


def _format_metric(document: dict[str, Any], metric: str, suffix: str) -> str:
    value = document["metrics"][metric]
    if value is None:
        return "—"
    return f"{value['median']:.1f}{suffix}"


def _format_comparison(comparison: dict[str, Any] | None) -> str | None:
    """Format paired steady-state deltas for the human result table."""
    if comparison is None:
        return None
    parts: list[str] = []
    for key, description in (
        ("total_latency", "lower total latency"),
        ("decode_throughput", "higher decode throughput"),
    ):
        metric = comparison[key]
        if metric is None:
            continue
        if metric["winner"] is None:
            parts.append(f"equal {description}")
        else:
            parts.append(
                f"{backend_label(metric['winner'])} has "
                f"{metric['relative_difference_pct']:.1f}% {description}"
            )
    return "; ".join(parts) if parts else None


def print_human(document: dict[str, Any], *, no_color: bool) -> None:
    """Print all workload result tables inside a lclaude-style box."""
    color = lclaude._use_color() and not no_color
    heading = lclaude.HEADER_ACCENT if color else ""
    value = lclaude.HEADER_ACCENT if color else ""
    error = lclaude.ERROR_ACCENT if color else ""
    warning = lclaude.WARN_ACCENT if color else ""
    reset = lclaude.ANSI_RESET if color else ""
    columns, _ = shutil.get_terminal_size((80, 20))
    width = max(columns, 106)

    def row(content: str = "") -> None:
        print(lclaude._box_row(content, width, heading, reset))

    print()
    print(
        lclaude._box_top("BENCHMARK RESULTS", width, heading, reset)
    )
    row(
        f" {lclaude.DIM_ACCENT if color else ''}Model:{reset} {value}{document['model']}{reset}"
        f"  {lclaude.DIM_ACCENT if color else ''}Runs:{reset} "
        f"{document['warmup']} warmup + {document['repeats']} measured"
    )
    for index, workload in enumerate(document["workloads"]):
        row()
        row(
            f" {heading}{workload['label']}:{reset} "
            f"prompt={workload['prompt_chars']} chars  max={workload['max_tokens']} tokens"
        )
        row(
            f" {heading}{'Backend':<18} {'Version':<16} {'TTFT Med':>10} "
            f"{'Total Med':>10} {'Decode tok/s':>13} {'Out tokens':>10} {'Result':>8}{reset}"
        )
        for backend in workload["backends"]:
            attempts = backend["attempts"]
            successes = backend["successes"]
            result_color = value if successes == attempts and attempts else warning
            if successes == 0:
                result_color = error
            result_status = f"{result_color}{f'{successes}/{attempts}':>8}{reset}"
            row(
                f" {value}{backend_label(backend['name']):<18}{reset} "
                f"{backend['version']:<16} "
                f"{_format_metric(backend, 'ttft_ms', 'ms'):>10} "
                f"{_format_metric(backend, 'total_ms', 'ms'):>10} "
                f"{_format_metric(backend, 'decode_tok_per_s', ''):>13} "
                f"{_format_metric(backend, 'output_tokens', ''):>10} "
                f"{result_status}"
            )
            if backend["error"]:
                for message in backend["error"].splitlines():
                    row(f" {error}{backend_label(backend['name'])}: {message}{reset}")
            for run in backend["runs"]:
                if not run["ok"]:
                    row(
                        f" {error}{backend_label(backend['name'])} run {run['run']}: "
                        f"{run['error']}{reset}"
                    )
        comparison = _format_comparison(workload["comparison"])
        if comparison:
            row()
            row(f" {heading}Comparison:{reset} {comparison}")
        if index < len(document["workloads"]) - 1:
            row(" " + "-" * (width - 4))
    print(lclaude._box_bottom(width, heading, reset))


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    if args.help:
        print_help(no_color=args.no_color)
        return 0
    if (
        (args.max_tokens is not None and args.max_tokens <= 0)
        or args.warmup < 0
        or args.repeats <= 0
    ):
        print(
            "Error: --max-tokens and --repeats must be positive; --warmup non-negative",
            file=sys.stderr,
        )
        return 2
    if args.timeout <= 0 or args.cooldown < 0:
        print("Error: --timeout must be positive and --cooldown non-negative", file=sys.stderr)
        return 2
    try:
        workloads = resolve_workloads(args)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    model, backends, ports = resolve_options(args)
    if not args.quiet:
        print_bench_header(
            model,
            backends,
            ports,
            workloads,
            args,
            stream=sys.stderr if args.json else sys.stdout,
        )
    if args.dry_run:
        plan = dry_run_document(model, backends, ports, workloads, args)
        if args.json:
            print(json.dumps(plan, indent=2))
        else:
            print(
                "Dry run: resolved benchmark targets; no backend was contacted"
            )
        return 0

    progress = Progress(
        quiet=args.quiet, verbose=args.verbose, no_color=args.no_color
    )
    progress.note(
        "Running benchmarks"
        f" — {len(backends)} backend{'s' if len(backends) != 1 else ''},"
        f" {len(workloads)} workload{'s' if len(workloads) != 1 else ''},"
        f" {args.warmup} warmup + {args.repeats} measured each",
        detail=False,
    )
    results: list[BackendResult] = []
    for index, backend in enumerate(backends):
        if args.verbose:
            result = benchmark_backend(
                backend, model, ports[backend], workloads, args, progress
            )
        else:
            try:
                with progress.activity(
                    f"Running {backend_label(backend)} benchmarks", detail=False
                ):
                    result = benchmark_backend(
                        backend, model, ports[backend], workloads, args, progress
                    )
                    if result.error:
                        raise RuntimeError("backend benchmark failed")
            except RuntimeError:
                pass
        results.append(result)
        if args.unload_between and index < len(backends) - 1:
            with progress.activity(f"Unloading Ollama model {model}"):
                unload_ollama_model(model)
        if index < len(backends) - 1 and args.cooldown:
            with progress.activity(
                f"Cooling down for {args.cooldown:g} seconds before the next backend",
                detail=False,
            ):
                time.sleep(args.cooldown)

    document = result_document(model, workloads, args, results)
    progress.note("Rendering results", detail=False)
    if args.json:
        print(json.dumps(document, indent=2))
    else:
        print_human(document, no_color=args.no_color)
    return (
        0
        if any(
            backend["successes"]
            for workload in document["workloads"]
            for backend in workload["backends"]
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
