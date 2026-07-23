#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# lclaude.py — Run Claude Code against a local LLM server (Ollama or llama.cpp)
# ---------------------------------------------------------------------------
#
# What it does
#   Wraps the ``claude`` CLI (Anthropic's Claude Code) and routes its
#   API traffic to a local LLM server instead of the Anthropic cloud.
#   Supports three backends:
#     - Ollama — managed model registry with auto-start
#     - llama.cpp (llama-server) — direct GGUF inference, maximum control
#     - hybrid — Ollama for model management, llama-server for inference
#   It also temporarily patches ``~/.claude/settings.json`` to suppress
#   the attribution header while running.
#
# Prerequisites
#   1.  claude CLI (Claude Code) — from the official Anthropic installer:
#           - curl -L https://raw.githubusercontent.com/anthropics/claude-code/... | bash
#         Other options:
#           - brew install claude-code
#           - npm install -g @anthropic-ai/claude-code
#   2.  A local LLM backend (one of):
#       a)  ollama:
#               brew install ollama
#           After installation, pull at least one model (see Usage below).
#       b)  llama.cpp (llama-server):
#               brew install llama.cpp
#           Download a GGUF model from HuggingFace, then start:
#               llama-server -m model.gguf --port 8080
#   3.  Python 3.11+ (stdlib only — no third-party dependencies).
#
# Setup
#   1.  Put this script somewhere on your PATH, e.g.:
#           cp lclaude.py ~/bin/lclaude
#           chmod +x ~/bin/lclaude
#   2.  Pull/download a model:
#       Ollama:     ollama pull ornith:35b
#       llama.cpp:  Download a GGUF from https://huggingface.co/models
#   3.  Add an alias to your shell config (~/.zshrc, ~/.bashrc, etc.):
#           alias lclaude='/full/path/to/lclaude.py'
#       Then reload: ``source ~/.zshrc`` (or restart your terminal).
#
# Usage
#   Basic (auto-detects backend, uses default model ornith:35b):
#       lclaude
#
#   Choose a specific model:
#       lclaude --model ornith:35b
#       lclaude --model ornith
#
#   Explicitly choose a backend:
#       lclaude --backend ollama
#       lclaude --backend llamacpp
#       lclaude --backend hybrid        # Ollama models + llama-server
#       lclaude --backend auto          (default — probes both)
#
#   Custom port:
#       lclaude --backend llamacpp --port 9090
#
#   List models available in Ollama:
#       lclaude --list
#
#   Pass arguments through to ``claude``:
#       lclaude --system "You are a helpful assistant" --message "Hello"
#
#   Show help (prints this script's docstring + --model info):
#       lclaude --help
#
# How it works
#   1.  Detects or selects the backend (Ollama on :11434, llama.cpp on :8080,
#       hybrid on :9090).
#   2.  For Ollama: verifies installed & running (auto-starts if needed),
#       validates the model is pulled.
#   3.  For llama.cpp: verifies the server is reachable (no auto-start,
#       no model validation — the server already has a model loaded).
#   4.  For hybrid: resolves the Ollama GGUF blob, auto-starts llama-server
#       (with a patched chat template when needed), then routes Claude to it.
#   5.  Backs up ``~/.claude/settings.json`` to ``settings.json.off``,
#       then sets ``CLAUDE_CODE_ATTRIBUTION_HEADER=0`` in the settings file.
#   6.  Sets ``ANTHROPIC_AUTH_TOKEN`` and ``ANTHROPIC_BASE_URL`` in the
#       environment, disables Claude's alternate-screen (fullscreen) TUI so
#       the header stays visible, then runs ``claude --model <model>`` with
#       the remaining args.
#   7.  On exit (normal, signal, or interrupt) the original settings file is
#       restored, any owned llama-server is stopped, and the backup is deleted.
#
# Environment overrides
#   No env vars need to be set beforehand — this script manages everything
#   internally.  If ``ANTHROPIC_API_KEY`` is present in the parent env it
#   is deliberately stripped so traffic goes to the local server, not the cloud.
#
# Copyright (c) 2026 Todd Papaioannou
# License: MIT
# ---------------------------------------------------------------------------

from __future__ import annotations

import argparse
import http.client
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger("lclaude")

# Path to Claude Code's settings file and its backup copy
SETTINGS = Path.home() / ".claude" / "settings.json"
SETTINGS_OFF = SETTINGS.with_name(SETTINGS.name + ".off")

# Supported backends and their default ports
BACKEND_OLLAMA = "ollama"
BACKEND_LLAMACPP = "llamacpp"
BACKEND_HYBRID = "hybrid"
BACKEND_AUTO = "auto"
DEFAULT_PORTS = {
    BACKEND_OLLAMA: 11434,
    BACKEND_LLAMACPP: 8080,
    BACKEND_HYBRID: 9090,
}

# Seconds to wait for ``ollama serve`` to accept connections after launch
OLLAMA_STARTUP_TIMEOUT = 30

# Seconds to wait for ``ollama list`` / ``ollama show`` to complete
OLLAMA_LIST_TIMEOUT = 15

# Seconds to wait for hybrid-mode ``llama-server`` to become healthy
LLAMACPP_STARTUP_TIMEOUT = 120

# Patched chat template for Ornith/Qwen 3.6 models (Claude Code-compatible)
TEMPLATE_CACHE_DIR = Path.home() / ".cache" / "lclaude"
QWEN36_TEMPLATE_URL = (
    "https://huggingface.co/spiritbuun/buun-Qwen3.6-chat_template/"
    "raw/main/chat_template.jinja"
)
QWEN36_TEMPLATE_FILE = TEMPLATE_CACHE_DIR / "qwen3.6-claude.jinja"
# Hybrid-mode llama-server stdout/stderr (truncated each launch)
LLAMACPP_LOG_FILE = TEMPLATE_CACHE_DIR / "llama-server.log"

# Models known to embed templates that reject late system messages
MODELS_NEEDING_TEMPLATE_PATCH = ("ornith", "qwen3")

# ANSI accents for the status box, help headings, and failure messages
HEADER_ACCENT = "\x1b[92m"  # bright green
ERROR_ACCENT = "\x1b[91m"  # bright red
WARN_ACCENT = "\x1b[93m"  # bright yellow
DIM_ACCENT = "\x1b[2m"
BOLD_ACCENT = "\x1b[1m"
ANSI_RESET = "\x1b[0m"

# Cached backend version from the last pre-flight check (module-level singleton)
_BACKEND_VERSION: str = "unknown"


def setup_logging() -> None:
    """Configure lclaude's own logger: stderr only, CRITICAL level by default
    so user-visible messages must use ``print()`` to stderr."""
    handler = logging.StreamHandler(sys.stderr)
    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))

    logger.addHandler(handler)
    logger.setLevel(logging.CRITICAL)
    # Suppress noisy loggers (urllib3, http.client, etc.)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("http.client").setLevel(logging.WARNING)


def load_settings(path: Path) -> dict[str, Any]:
    """Read a JSON settings file; returns empty dict for missing / empty files."""
    result: dict[str, Any] = {}
    if path.is_file():
        raw = path.read_text().strip()
        if raw:
            data = json.loads(raw)
            if isinstance(data, dict):
                result = data
    return result


def save_settings(path: Path, data: dict[str, Any]) -> None:
    """Write *data* as indented JSON, ensuring the parent directory exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def ensure_settings_file() -> None:
    """Create ``~/.claude/settings.json`` as an empty JSON object if it doesn't
    already exist — ``claude`` expects it to be present."""
    SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    if not SETTINGS.is_file():
        save_settings(SETTINGS, {})


def backup_settings() -> None:
    """Save the current ``settings.json`` as ``settings.json.off`` so we can
    restore the user's original configuration after claude exits."""
    if SETTINGS.is_file():
        shutil.copy(SETTINGS, SETTINGS_OFF)
    else:
        SETTINGS_OFF.write_text("{}\n")


def restore_settings() -> None:
    """Overwrite ``settings.json`` with the backup and delete the backup.
    This is the cleanup step — the user's settings are restored to their
    pre-run state so subsequent non-lclaude ``claude`` calls aren't affected."""
    if not SETTINGS_OFF.is_file():
        return
    shutil.copy(SETTINGS_OFF, SETTINGS)
    SETTINGS_OFF.unlink(missing_ok=True)


def apply_attribution_patch() -> None:
    """Temporarily set ``CLAUDE_CODE_ATTRIBUTION_HEADER=0`` in the settings file
    so Claude Code doesn't append its attribution banner to output."""
    data = load_settings(SETTINGS)
    env = data.get("env")
    if not isinstance(env, dict):
        env = {}
    env["CLAUDE_CODE_ATTRIBUTION_HEADER"] = "0"
    data["env"] = env
    save_settings(SETTINGS, data)


def build_child_env(backend: str, port: int) -> dict[str, str]:
    """Build the environment for the claude subprocess.

    Start from the parent env, inject backend routing vars, and strip
    ``ANTHROPIC_API_KEY`` so requests don't accidentally hit the cloud.
    """
    env = os.environ.copy()
    env.update({
        "ANTHROPIC_AUTH_TOKEN": "lclaude",
        "ANTHROPIC_BASE_URL": f"http://localhost:{port}",
        # Since v2.1.89 Claude Code defaults to a fullscreen alternate-screen TUI
        # that wipes anything printed on the main buffer (our header) as soon as
        # claude starts.  CLAUDE_CODE_DISABLE_ALTERNATE_SCREEN keeps Claude on the
        # main screen so the lclaude box stays in scrollback and native
        # selection / copy-paste / Cmd-F keep working.
        "CLAUDE_CODE_DISABLE_ALTERNATE_SCREEN": "1",
    })
    env.pop("ANTHROPIC_API_KEY", None)
    return env


def _is_print_mode(claude_argv: list[str]) -> bool:
    """True when Claude is in -p/--print mode (stdout must stay clean)."""
    for arg in claude_argv:
        if arg in ("-p", "--print") or arg.startswith("--print="):
            return True
    return False


def _set_terminal_title(title: str) -> None:
    """Set the terminal tab/window title (survives Claude's full-screen TUI)."""
    if not sys.stdout.isatty():
        return
    sys.stdout.write(f"\x1b]0;{title}\x07")
    sys.stdout.flush()


def run_claude(
    claude_argv: list[str],
    model: str,
    *,
    backend: str,
    port: int,
    backend_ver: str | None = None,
    server_proc: subprocess.Popen[Any] | None = None,
) -> int:
    """Run ``claude --model <model>`` with local backend env vars, settings
    backup, and signal-safe cleanup.

    Claude Code owns the TTY directly (no PTY shim) so mouse selection and
    copy/paste keep working. Session context is kept in the terminal tab title
    because Claude's redraw clears anything printed above it.

    If *server_proc* is provided (hybrid mode), it is terminated on exit.
    """
    ensure_settings_file()
    backup_settings()

    if backend in (BACKEND_HYBRID, BACKEND_LLAMACPP):
        backend_label = "llama.cpp"
    else:
        backend_label = "Ollama"
    interactive = (
        sys.stdin.isatty()
        and sys.stdout.isatty()
        and not _is_print_mode(claude_argv)
        and backend_ver is not None
    )

    def _stop_owned_server() -> None:
        if server_proc is None or server_proc.poll() is not None:
            return
        try:
            server_proc.terminate()
            server_proc.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            try:
                server_proc.kill()
            except OSError:
                pass

    def on_signal(signum: int, _frame: Any) -> None:
        restore_settings()
        _stop_owned_server()
        if interactive:
            try:
                os.write(sys.stdout.fileno(), b"\x1b]0;\x07")
            except OSError:
                pass
        os._exit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, on_signal)

    try:
        apply_attribution_patch()
        if interactive:
            _set_terminal_title(
                f"lclaude · {model} · {backend_label} v{backend_ver}"
            )
        return subprocess.run(
            ["claude", "--model", model, *claude_argv],
            env=build_child_env(backend, port),
            check=False,
        ).returncode
    finally:
        if interactive:
            _set_terminal_title("")
        restore_settings()
        _stop_owned_server()


def check_ollama(port: int, auto_start: bool = True) -> tuple[bool, str | None]:
    """Verify ollama is installed and the server is running.

    If *auto_start* is True and ollama is installed but not running, tries to
    launch it in the background and waits for it to come up.

    Returns:
        (ok, reason): ok=True means healthy; ok=False means reason explains why.
    """
    # 1. Is ollama installed?
    ollama_bin = shutil.which("ollama")
    if ollama_bin is None:
        return (False, "ollama not found in PATH — install it from https://ollama.com")

    # 2. Is the server reachable?
    ok, ver = _is_ollama_reachable(port)
    if ok:
        global _BACKEND_VERSION
        _BACKEND_VERSION = ver
        return (True, None)

    # 3. Not reachable — try to start it
    if auto_start:
        logger.info(
            "ollama is installed but not running — starting it (%s)", ollama_bin
        )
        return _start_and_wait(ollama_bin, port)

    return (False, "ollama is installed but not running — start it with `ollama serve`")


def _is_ollama_reachable(port: int) -> tuple[bool, str]:
    """Probe Ollama's ``/api/version`` endpoint over raw HTTP.

    We use ``http.client`` instead of ``requests`` to avoid any dependency
    — Ollama's version endpoint is a plain GET, so the stdlib is enough.
    """
    try:
        conn = http.client.HTTPConnection("localhost", port, timeout=3)
        conn.request("GET", "/api/version")
        resp = conn.getresponse()
        if resp.status == 200:
            raw = resp.read().decode()
            conn.close()
            if raw:
                try:
                    body = json.loads(raw)
                    ver = body.get("version", "unknown")
                    logger.debug("ollama reachable — version %s", ver)
                    return (True, ver)
                except json.JSONDecodeError:
                    pass
            logger.debug("ollama reachable (HTTP 200)")
            return (True, "unknown")
        conn.close()
        logger.warning("ollama responded with HTTP %d", resp.status)
        return (False, "unknown")
    except OSError as exc:
        logger.debug("ollama not reachable: %s", exc)
        return (False, "unknown")


def _start_and_wait(ollama_bin: str, port: int) -> tuple[bool, str | None]:
    """Launch ``ollama serve`` in the background and poll until it accepts
    connections (or we time out).

    ``start_new_session=True`` detaches the child from our terminal group so
    it doesn't receive Ctrl-C from the parent.

    When ``port`` differs from Ollama's default (11434), the ``OLLAMA_HOST``
    environment variable is set so the server binds to the requested port.
    """
    env = None
    if port != DEFAULT_PORTS[BACKEND_OLLAMA]:
        env = {**os.environ, "OLLAMA_HOST": f"http://localhost:{port}"}
    try:
        proc = subprocess.Popen(
            [ollama_bin, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env=env,
        )
        logger.info(
            "launched `ollama serve` (pid %d) — waiting for it to come up …", proc.pid
        )
    except OSError as exc:
        return (False, f"failed to start ollama: {exc}")

    # Poll the version endpoint until it responds or we time out
    deadline = time.monotonic() + OLLAMA_STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        ok, ver = _is_ollama_reachable(port)
        if ok:
            global _BACKEND_VERSION
            _BACKEND_VERSION = ver
            logger.info("ollama is now ready")
            return (True, None)
        time.sleep(0.5)

    # Timed out — the child may still be starting, clean up our stale process
    if proc.poll() is None:
        proc.terminate()
        proc.wait(timeout=5)
    return (False, f"ollama did not become ready within {OLLAMA_STARTUP_TIMEOUT}s")


# ---------------------------------------------------------------------------
# llama.cpp backend
# ---------------------------------------------------------------------------


def check_llamacpp(port: int) -> tuple[bool, str | None]:
    """Verify llama-server is reachable on the given port.

    Unlike Ollama, llama-server is not auto-started — the user must launch it
    themselves with the desired model already loaded.

    Returns:
        (ok, reason): ok=True means healthy; ok=False means reason explains why.
    """
    ok, ver = _is_llamacpp_reachable(port)
    if ok:
        if _llamacpp_template_rejects_late_system_messages(port):
            cache = QWEN36_TEMPLATE_FILE
            return (
                False,
                "llama-server's embedded chat template rejects system messages "
                "outside the first turn, which is incompatible with Claude Code.\n"
                "Restart llama-server with a patched Qwen 3.6 template:\n"
                f"  mkdir -p {TEMPLATE_CACHE_DIR}\n"
                f"  curl -L {QWEN36_TEMPLATE_URL} -o {cache}\n"
                f"  llama-server -m model.gguf --chat-template-file {cache} "
                f"--port {port}\n"
                "\n"
                "Or use hybrid mode (auto-fixes this):\n"
                "  lclaude --backend hybrid --model <model>",
            )
        global _BACKEND_VERSION
        _BACKEND_VERSION = ver
        return (True, None)
    if ver == "loading":
        return (
            False,
            f"llama-server on port {port} is still loading the model.\n"
            "Wait a moment and try again.",
        )
    return (
        False,
        f"llama-server not reachable on port {port}.\n"
        f"Start it first: llama-server -m model.gguf --port {port}\n"
        f"Or install: brew install llama.cpp\n"
        f"Or use hybrid mode: lclaude --backend hybrid",
    )


def _is_llamacpp_reachable(port: int) -> tuple[bool, str]:
    """Probe llama-server's ``/health`` endpoint.

    Returns (reachable, version_string).  llama-server's /health returns
    ``{"status":"ok"}`` when ready, or HTTP 503 while loading a model.
    The /props endpoint provides build info for display.
    """
    try:
        conn = http.client.HTTPConnection("localhost", port, timeout=3)
        conn.request("GET", "/health")
        resp = conn.getresponse()
        body_raw = resp.read().decode()
        conn.close()

        if resp.status == 503:
            try:
                body = json.loads(body_raw)
                msg = body.get("error", {}).get("message", "")
                if "loading" in msg.lower():
                    return (False, "loading")
            except (json.JSONDecodeError, AttributeError):
                pass
            logger.debug("llama-server returned 503 (not ready)")
            return (False, "unknown")

        if resp.status != 200:
            logger.warning("llama-server /health returned HTTP %d", resp.status)
            return (False, "unknown")

        logger.debug("llama-server reachable (HTTP 200)")
    except OSError as exc:
        logger.debug("llama-server not reachable: %s", exc)
        return (False, "unknown")

    # Try to get version/build info from /props
    ver = _get_llamacpp_version(port)
    return (True, ver)


def _get_llamacpp_version(port: int) -> str:
    """Fetch llama-server's build info from ``/props`` for display."""
    props = _get_llamacpp_props(port)
    if props is None:
        return "unknown"

    build_info = props.get("build_info")
    if isinstance(build_info, str) and build_info:
        # llama-server reports e.g. "b10090-7347430f4"; the header adds "v".
        return build_info.removeprefix("b")

    build_number = props.get("build_number", "")
    build_commit = props.get("build_commit", "")
    if build_number:
        return str(build_number)
    if build_commit:
        return str(build_commit)[:8]
    return "unknown"


def _llamacpp_template_rejects_late_system_messages(port: int) -> bool:
    """Return whether llama-server's model template is incompatible with Claude.

    Claude Code sends system/developer messages after the opening turn. The
    stock Ornith/Qwen 3.6 template raises an exception for that valid API
    pattern, causing llama-server to return HTTP 400 before inference begins.
    """
    props = _get_llamacpp_props(port)
    if props is None:
        return False
    template = props.get("chat_template", "")
    return (
        isinstance(template, str)
        and "System message must be at the beginning" in template
    )


def _get_llamacpp_props(port: int) -> dict[str, Any] | None:
    """Fetch and decode llama-server's ``/props`` response."""
    try:
        conn = http.client.HTTPConnection("localhost", port, timeout=3)
        conn.request("GET", "/props")
        resp = conn.getresponse()
        if resp.status == 200:
            raw = resp.read().decode()
            conn.close()
            body = json.loads(raw)
            return body if isinstance(body, dict) else None
        conn.close()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        pass
    return None


# ---------------------------------------------------------------------------
# Hybrid backend (Ollama models + llama-server inference)
# ---------------------------------------------------------------------------


def resolve_ollama_blob(model: str) -> str:
    """Resolve an Ollama model name to its on-disk GGUF blob path.

    Runs ``ollama show <model> --modelfile`` and parses the ``FROM`` line.
    Raises ``SystemExit`` with a clear error if the model or blob is missing.
    """
    try:
        result = subprocess.run(
            ["ollama", "show", model, "--modelfile"],
            capture_output=True,
            text=True,
            timeout=OLLAMA_LIST_TIMEOUT,
        )
    except FileNotFoundError:
        print(
            "Error: ollama not found in PATH — install it from https://ollama.com",
            file=sys.stderr,
        )
        raise SystemExit(1)
    except subprocess.TimeoutExpired:
        print(
            f"Error: `ollama show {model}` timed out after {OLLAMA_LIST_TIMEOUT}s",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()
        print(
            f"Error: model '{model}' not found in Ollama.\n"
            f"Run `ollama pull {model}` first.\n"
            f"{err}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    blob_path: str | None = None
    for line in result.stdout.splitlines():
        if line.startswith("FROM "):
            blob_path = line[5:].strip()
            break

    if not blob_path:
        print(
            f"Error: could not find FROM path in `ollama show {model} --modelfile`.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    # Relative/registry refs (e.g. "llama3") are not usable blob paths
    path = Path(blob_path)
    if not path.is_file():
        print(
            f"Error: blob file missing for model '{model}': {blob_path}\n"
            f"Try `ollama pull {model}` to re-download.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    return str(path)


def _model_needs_template_patch(model: str) -> bool:
    """Return True if *model* is known to need a Claude-compatible chat template."""
    name = model.lower()
    return any(token in name for token in MODELS_NEEDING_TEMPLATE_PATCH)


def ensure_patched_template() -> Path:
    """Download the Claude-compatible Qwen 3.6 template into the local cache.

    Returns the path to the cached template file. Raises ``SystemExit`` if the
    download fails and the cache does not already exist.

    Tries ``urllib`` first, then falls back to ``curl`` (common on macOS where
    Python's SSL cert store can fail hostname/CA verification).
    """
    if QWEN36_TEMPLATE_FILE.is_file() and QWEN36_TEMPLATE_FILE.stat().st_size > 0:
        return QWEN36_TEMPLATE_FILE

    TEMPLATE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"Downloading Claude-compatible chat template to {QWEN36_TEMPLATE_FILE} …",
        file=sys.stderr,
    )

    errors: list[str] = []

    # 1. urllib (stdlib)
    try:
        with urllib.request.urlopen(QWEN36_TEMPLATE_URL, timeout=60) as resp:
            data = resp.read()
        if not data:
            raise ValueError("empty response")
        QWEN36_TEMPLATE_FILE.write_bytes(data)
        return QWEN36_TEMPLATE_FILE
    except (urllib.error.URLError, OSError, ValueError) as exc:
        errors.append(f"urllib: {exc}")

    # 2. curl — uses the system CA bundle (works when Python SSL does not)
    curl_bin = shutil.which("curl")
    if curl_bin is not None:
        try:
            result = subprocess.run(
                [
                    curl_bin,
                    "-fsSL",
                    "--connect-timeout",
                    "30",
                    "-o",
                    str(QWEN36_TEMPLATE_FILE),
                    QWEN36_TEMPLATE_URL,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if (
                result.returncode == 0
                and QWEN36_TEMPLATE_FILE.is_file()
                and QWEN36_TEMPLATE_FILE.stat().st_size > 0
            ):
                return QWEN36_TEMPLATE_FILE
            err = (result.stderr or result.stdout or f"exit {result.returncode}").strip()
            errors.append(f"curl: {err}")
            QWEN36_TEMPLATE_FILE.unlink(missing_ok=True)
        except (OSError, subprocess.TimeoutExpired) as exc:
            errors.append(f"curl: {exc}")
            QWEN36_TEMPLATE_FILE.unlink(missing_ok=True)
    else:
        errors.append("curl: not found in PATH")

    detail = "\n".join(f"  • {e}" for e in errors)
    print(
        f"Error: failed to download patched chat template:\n{detail}\n"
        "Download it manually and re-run:\n"
        f"  mkdir -p {TEMPLATE_CACHE_DIR}\n"
        f"  curl -L {QWEN36_TEMPLATE_URL} -o {QWEN36_TEMPLATE_FILE}",
        file=sys.stderr,
    )
    raise SystemExit(1)


def start_llamacpp_server(
    blob_path: str,
    port: int,
    template_file: Path | None = None,
    *,
    log_file: Path = LLAMACPP_LOG_FILE,
) -> subprocess.Popen[Any]:
    """Launch ``llama-server`` with the given GGUF blob and optional template.

    stdout/stderr are written to *log_file* (truncated each launch) so the TUI
    stays clean while users can ``tail -f`` the log.

    Returns the process handle. The caller owns the process lifecycle.
    """
    llama_bin = shutil.which("llama-server")
    if llama_bin is None:
        print(
            "Error: llama-server not found in PATH.\n"
            "Install: brew install llama.cpp",
            file=sys.stderr,
        )
        raise SystemExit(1)

    cmd = [
        llama_bin,
        "-m",
        blob_path,
        "--port",
        str(port),
        "-ngl",
        "99",
    ]
    if template_file is not None:
        cmd.extend(["--chat-template-file", str(template_file)])
    else:
        cmd.append("--jinja")

    log_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Truncate so each hybrid session starts a fresh log for easy tail -f.
        log_fh = open(log_file, "w", encoding="utf-8")
    except OSError as exc:
        print(
            f"Error: cannot write llama-server log to {log_file}: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except OSError as exc:
        log_fh.close()
        print(f"Error: failed to start llama-server: {exc}", file=sys.stderr)
        raise SystemExit(1)
    finally:
        # Child has its own dup of the fd; parent can close.
        log_fh.close()

    logger.info(
        "launched llama-server (pid %d) on port %d — log: %s",
        proc.pid,
        port,
        log_file,
    )
    return proc


def wait_for_llamacpp(
    port: int,
    timeout: int = LLAMACPP_STARTUP_TIMEOUT,
    *,
    proc: subprocess.Popen[Any] | None = None,
) -> bool:
    """Poll llama-server ``/health`` until ready or *timeout* seconds elapse.

    If *proc* is given and exits before becoming healthy, return False
    immediately instead of waiting out the full timeout.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            logger.info(
                "llama-server exited early (code %s) before becoming healthy",
                proc.returncode,
            )
            return False
        ok, ver = _is_llamacpp_reachable(port)
        if ok:
            global _BACKEND_VERSION
            _BACKEND_VERSION = ver
            logger.info("llama-server is now ready")
            return True
        time.sleep(0.5)
    return False


def _parse_llamacpp_load_error() -> str | None:
    """Extract a concise cause from the hybrid llama-server log, if present."""
    try:
        if not LLAMACPP_LOG_FILE.is_file():
            return None
        text = LLAMACPP_LOG_FILE.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    # Prefer the first concrete hyperparameter / load error over generic wrappers.
    patterns = (
        r"error loading model hyperparameters:\s*(.+)",
        r"error loading model:\s*(.+)",
        r"failed to load model[,\s]*(.+)?",
        r"exiting due to (.+)",
    )
    for line in text.splitlines():
        # Strip llama.cpp timestamp / level / component prefixes
        cleaned = re.sub(
            r"^\s*[\d.]+\s+[EIW]\s+\S+\s+\S+:\s*",
            "",
            line,
        ).strip()
        if not cleaned:
            continue
        for pat in patterns:
            m = re.search(pat, cleaned, re.IGNORECASE)
            if m:
                detail = (m.group(1) or cleaned).strip().rstrip(".")
                # Drop redundant leading "error loading model:" if still present
                detail = re.sub(
                    r"^error loading model:\s*",
                    "",
                    detail,
                    flags=re.IGNORECASE,
                ).strip()
                if detail:
                    return detail
    return None


def _hybrid_startup_failure_message(
    model: str,
    *,
    exited: bool,
) -> str:
    """Build a short, user-facing error for hybrid llama-server startup failure."""
    cause = _parse_llamacpp_load_error()
    if exited:
        headline = f"could not load model '{model}' via llama-server (hybrid mode)."
    else:
        headline = (
            f"llama-server did not become ready within "
            f"{LLAMACPP_STARTUP_TIMEOUT}s for model '{model}'."
        )

    if _use_color(sys.stderr):
        err = ERROR_ACCENT + BOLD_ACCENT
        label = WARN_ACCENT
        dim = DIM_ACCENT
        reset = ANSI_RESET
    else:
        err = label = dim = reset = ""

    lines = [
        f"{err}Error:{reset} {headline}",
        "",
    ]
    if cause:
        lines.append(f"  {label}Cause:{reset} {cause}")
    else:
        lines.append(
            f"  {label}Cause:{reset} model failed to load (see log for details)"
        )
    lines.append(
        f"  {label}Hint:{reset}  This Ollama blob may be incompatible with "
        f"your llama.cpp build."
    )
    lines.append(
        f"  {label}Try:{reset}   {dim}lclaude --backend ollama --model {model}{reset}"
    )
    lines.append(f"  {label}Log:{reset}   {dim}{LLAMACPP_LOG_FILE}{reset}")
    return "\n".join(lines)


def _stop_process(proc: subprocess.Popen[Any] | None) -> None:
    """Terminate *proc* if it is still running."""
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        try:
            proc.kill()
        except OSError:
            pass


def prepare_hybrid_backend(
    model: str, port: int
) -> subprocess.Popen[Any]:
    """Resolve the Ollama blob, start llama-server, and wait until healthy.

    Caller must have already verified Ollama is running and *model* is installed.
    Returns the owned ``llama-server`` process. Raises ``SystemExit`` on failure.
    """
    blob_path = resolve_ollama_blob(model)

    template_file: Path | None = None
    if _model_needs_template_patch(model):
        template_file = ensure_patched_template()

    server_proc = start_llamacpp_server(blob_path, port, template_file)
    if not wait_for_llamacpp(port, proc=server_proc):
        exited = server_proc.poll() is not None
        _stop_process(server_proc)
        print(
            _hybrid_startup_failure_message(model, exited=exited),
            file=sys.stderr,
        )
        raise SystemExit(1)

    # Belt-and-suspenders: if the live template is still incompatible, restart
    # with the patched file (covers models whose names weren't in the heuristic).
    if _llamacpp_template_rejects_late_system_messages(port):
        if template_file is None:
            template_file = ensure_patched_template()
            _stop_process(server_proc)
            server_proc = start_llamacpp_server(blob_path, port, template_file)
            if not wait_for_llamacpp(port, proc=server_proc):
                _stop_process(server_proc)
                print(
                    _hybrid_startup_failure_message(
                        model, exited=server_proc.poll() is not None
                    ),
                    file=sys.stderr,
                )
                raise SystemExit(1)
        if _llamacpp_template_rejects_late_system_messages(port):
            _stop_process(server_proc)
            print(
                "Error: llama-server still has an incompatible chat template "
                "after applying the patch.\n"
                f"  Try:   lclaude --backend ollama --model {model}\n"
                f"  Log:   {LLAMACPP_LOG_FILE}",
                file=sys.stderr,
            )
            raise SystemExit(1)

    return server_proc


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------


def detect_backend(ollama_port: int, llamacpp_port: int) -> str | None:
    """Probe both backends and return whichever is reachable first.

    Returns BACKEND_OLLAMA, BACKEND_LLAMACPP, or None if neither responds.
    Ollama is checked first (it is the more common/default setup).
    """
    ok_ollama, _ = _is_ollama_reachable(ollama_port)
    if ok_ollama:
        return BACKEND_OLLAMA

    ok_cpp, _ = _is_llamacpp_reachable(llamacpp_port)
    if ok_cpp:
        return BACKEND_LLAMACPP

    return None


def list_ollama_models() -> int:
    """Print the raw ``ollama list`` output and return its exit status."""
    try:
        return subprocess.run(
            ["ollama", "list"],
            check=False,
            timeout=OLLAMA_LIST_TIMEOUT,
        ).returncode
    except subprocess.TimeoutExpired:
        print(
            f"Error: `ollama list` timed out after {OLLAMA_LIST_TIMEOUT}s",
            file=sys.stderr,
        )
        return 1


def ensure_model_in_ollama(model: str) -> list[str]:
    """Validate that ollama has the requested model installed.

    Returns the list of installed model names (without tags) for display.

    Matching logic (in order of precedence):
      1. Exact full match against ``ollama list`` output (e.g. "ornith:35b")
      2. Base name match — strips the tag from *model* and checks against
         the base name column (e.g. "ornith" matches "ornith:35b")
    """
    result = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True,
        timeout=OLLAMA_LIST_TIMEOUT,
    )
    if result.returncode != 0:
        print(
            "Error: `ollama list` failed — is ollama running?",
            file=sys.stderr,
        )
        raise SystemExit(1)

    installed: list[str] = []
    for line in result.stdout.strip().splitlines():
        parts = line.split()
        if not parts:
            continue
        name = parts[0].rsplit(":", 1)[0]
        # Skip the header row (e.g. "NAME", "NAME:TAG")
        if name.lower() == "name":
            continue
        if name not in installed:
            installed.append(name)

    # Two-pass match: full name first, then base name without tag
    full_output = result.stdout.strip()
    if model not in set(installed) and f"{model}:" not in full_output:
        model_base = model.rsplit(":", 1)[0]
        if model_base not in installed:
            print(
                f"Error: model '{model}' not found in ollama.\n"
                f"Run `ollama pull {model}` first, or use --model to pick an installed model.\n"
                f"Currently available models: {', '.join(installed)}",
                file=sys.stderr,
            )
            raise SystemExit(1)

    return installed


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _use_color(stream: Any = None) -> bool:
    """Return whether *stream* (default stdout) supports and allows ANSI color."""
    out = sys.stdout if stream is None else stream
    return out.isatty() and not os.environ.get("NO_COLOR")


def _visible_len(text: str) -> int:
    return len(_ANSI_RE.sub("", text))


def _pad_visible(text: str, width: int) -> str:
    return text + " " * max(0, width - _visible_len(text))


def _box_top(title: str, width: int, border: str, reset: str) -> str:
    """╭─── Title ───...──╮"""
    inner = f"─── {title} "
    fill = max(0, width - 2 - len(inner))
    return f"{border}╭{inner}{'─' * fill}╮{reset}"


def _box_row(content: str, width: int, border: str, reset: str) -> str:
    """│ content          │"""
    inner_w = width - 2
    return f"{border}│{reset}{_pad_visible(content, inner_w)}{border}│{reset}"


def _box_bottom(width: int, border: str, reset: str) -> str:
    return f"{border}╰{'─' * (width - 2)}╯{reset}"


def _print_header(
    backend: str,
    backend_ver: str,
    model: str,
    port: int,
    installed_models: list[str] | None = None,
    *,
    log_file: Path | None = None,
) -> None:
    """Print a boxed header (Claude Code-style) with backend info, model, and
    optionally installed models (Ollama only) and a server log path."""
    cols, _ = shutil.get_terminal_size((80, 20))
    width = max(cols, 40)
    inner_w = width - 2

    if _use_color():
        dim = "\x1b[2m"
        reset = ANSI_RESET
        border = HEADER_ACCENT
        value = HEADER_ACCENT
    else:
        dim = reset = border = value = ""

    title = "LCLAUDE"
    # Hybrid runs inference via llama-server; show that engine (and its build) in the UI.
    if backend in (BACKEND_HYBRID, BACKEND_LLAMACPP):
        backend_label = "llama.cpp"
    else:
        backend_label = "Ollama"
    info_line = (
        f" {dim}{backend_label}:{reset} {value}v{backend_ver}{reset}"
        f"  {dim}Model:{reset} {value}{model}{reset}"
        f"  {dim}Port:{reset} {value}{port}{reset}"
    )

    print(_box_top(title, width, border, reset), file=sys.stdout)
    print(_box_row(info_line, width, border, reset), file=sys.stdout)

    if installed_models:
        models_str = ", ".join(installed_models)
        models_prefix = f" {dim}Models:{reset} "
        max_models = inner_w - _visible_len(models_prefix)
        if _visible_len(models_str) > max_models:
            models_str = models_str[: max(0, max_models - 1)] + "…"
        models_line = models_prefix + models_str
        print(_box_row(models_line, width, border, reset), file=sys.stdout)

    if log_file is not None:
        log_line = f" {dim}Log:{reset} {value}{log_file}{reset}"
        print(_box_row(log_line, width, border, reset), file=sys.stdout)

    print(_box_bottom(width, border, reset), file=sys.stdout)


def main(argv: list[str]) -> int:
    setup_logging()

    # Parse only our flags; pass the rest through verbatim to claude
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", default="ornith:35b")
    parser.add_argument(
        "--backend",
        choices=[BACKEND_OLLAMA, BACKEND_LLAMACPP, BACKEND_HYBRID, BACKEND_AUTO],
        default=BACKEND_AUTO,
        help="LLM backend: ollama, llamacpp, hybrid, or auto (default: auto).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help=(
            "Override the backend port "
            "(default: 11434 Ollama / 8080 llama.cpp / 9090 hybrid)."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List models available in Ollama and exit.",
    )
    args, claude_argv = parser.parse_known_args(argv)
    model = args.model

    is_help = any(arg in ("-h", "--help") for arg in argv)

    # --- Help early-exit (no backend required) ---
    if is_help:
        heading = HEADER_ACCENT if _use_color() else ""
        reset = ANSI_RESET if heading else ""
        help_text = (
            f"{heading}Usage:{reset} lclaude [OPTIONS] [ARGS passed to claude]\n"
            "\n"
            "Run Claude Code against a local LLM (Ollama, llama.cpp, or hybrid)\n"
            "\n"
            f"{heading}Options:{reset}\n"
            "  -h, --help           Show this help message and exit\n"
            "  --backend BACKEND    ollama, llamacpp, hybrid, or auto (default: auto)\n"
            "  --port PORT          Override backend port (default: 11434/8080/9090)\n"
            "  --list               List models available in Ollama and exit\n"
            "  --model MODEL        Model to use (default: ornith:35b)\n"
            "                       Ollama/hybrid: must be pulled. llama.cpp: label.\n"
            "\n"
            f"{heading}Examples (Ollama):{reset}\n"
            "  lclaude                          # auto-detect, default model\n"
            "  lclaude --model ornith           # use the model's base name\n"
            "  lclaude --model ornith:35b       # with a specific tag\n"
            "  lclaude --list                   # list available Ollama models\n"
            "\n"
            f"{heading}Examples (hybrid — Ollama models + llama-server):{reset}\n"
            "  lclaude --backend hybrid         # auto-start llama-server from Ollama blob\n"
            "  lclaude --backend hybrid --model ornith:35b\n"
            f"  tail -f {LLAMACPP_LOG_FILE}   # watch hybrid llama-server logs\n"
            "\n"
            f"{heading}Examples (llama.cpp):{reset}\n"
            "  lclaude --backend llamacpp       # use llama-server on :8080\n"
            "  lclaude --backend llamacpp --port 9090  # custom port\n"
            "  lclaude --backend llamacpp --model my-model  # cosmetic name\n"
            "\n"
            f"{heading}Prerequisites:{reset}\n"
            "  brew install claude-code         # install Claude Code CLI\n"
            "  brew install ollama              # install Ollama\n"
            "  ollama pull ornith:35b           # pull a model for Ollama/hybrid\n"
            "  brew install llama.cpp           # install llama.cpp (llamacpp/hybrid)\n"
            "  llama-server -m model.gguf       # start llama-server with a GGUF\n"
        )
        print(help_text, file=sys.stdout)
        return 0

    # --- Backend resolution ---
    backend = args.backend

    if backend == BACKEND_AUTO:
        ollama_port = args.port or DEFAULT_PORTS[BACKEND_OLLAMA]
        llamacpp_port = args.port or DEFAULT_PORTS[BACKEND_LLAMACPP]
        detected = detect_backend(ollama_port, llamacpp_port)
        if detected is None:
            print(
                "Error: no local LLM backend detected.\n"
                f"  • Ollama not responding on port {ollama_port}\n"
                f"  • llama-server not responding on port {llamacpp_port}\n"
                "\n"
                "Start one of:\n"
                "  ollama serve                             # then: ollama pull ornith:35b\n"
                "  llama-server -m model.gguf --port 8080  # llama.cpp direct\n"
                "  lclaude --backend hybrid                 # Ollama models + llama-server\n"
                "\n"
                "Or specify explicitly: lclaude --backend ollama | llamacpp | hybrid",
                file=sys.stderr,
            )
            return 1
        backend = detected
        # If user gave --port with auto, use it; otherwise use the detected default
        port = args.port or DEFAULT_PORTS[backend]
    else:
        port = args.port or DEFAULT_PORTS[backend]

    # --- Pre-flight checks (backend-specific) ---
    installed_models: list[str] | None = None
    server_proc: subprocess.Popen[Any] | None = None
    show_header = (
        sys.stdout.isatty()
        and not _is_print_mode(claude_argv)
    )

    if backend == BACKEND_OLLAMA:
        ok, reason = check_ollama(port, auto_start=True)
        if not ok:
            print(f"Error: {reason}", file=sys.stderr)
            return 1

        if args.list:
            return list_ollama_models()

        installed_models = ensure_model_in_ollama(model)

    elif backend == BACKEND_LLAMACPP:
        if args.list:
            print(
                "Note: --list is not supported with llama.cpp backend.\n"
                "llama-server serves a single model specified at startup.\n"
                "Check which model is loaded: curl http://localhost:{}/props".format(
                    port
                ),
                file=sys.stderr,
            )
            return 1

        ok, reason = check_llamacpp(port)
        if not ok:
            print(f"Error: {reason}", file=sys.stderr)
            return 1

    elif backend == BACKEND_HYBRID:
        # --list only needs Ollama; do not start llama-server
        ok, reason = check_ollama(DEFAULT_PORTS[BACKEND_OLLAMA], auto_start=True)
        if not ok:
            print(f"Error: {reason}", file=sys.stderr)
            return 1

        if args.list:
            return list_ollama_models()

        installed_models = ensure_model_in_ollama(model)

        # Show the banner before starting llama-server so failures still
        # appear under a familiar LCLAUDE header.
        if show_header:
            _print_header(
                backend,
                "…",
                model,
                port,
                installed_models,
                log_file=LLAMACPP_LOG_FILE,
            )
        server_proc = prepare_hybrid_backend(model, port)

    # Show a startup banner only when connected to a real terminal.
    # Session context also lives in the terminal tab title while Claude runs.
    # Hybrid already printed a preliminary banner before llama-server start;
    # reprint with the real llama.cpp build once the server is healthy.
    if show_header:
        _print_header(
            backend,
            _BACKEND_VERSION,
            model,
            port,
            installed_models,
            log_file=(
                LLAMACPP_LOG_FILE
                if backend == BACKEND_HYBRID and server_proc is not None
                else None
            ),
        )
    return run_claude(
        claude_argv,
        model,
        backend=backend,
        port=port,
        backend_ver=_BACKEND_VERSION,
        server_proc=server_proc,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
