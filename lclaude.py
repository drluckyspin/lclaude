#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# lclaude.py — Run Claude Code against a local Ollama instance
# ---------------------------------------------------------------------------
#
# What it does
#   Wraps the ``claude`` CLI (Anthropic's Claude Code) and routes its
#   API traffic to a local Ollama server instead of the Anthropic cloud.
#   It also temporarily patches ``~/.claude/settings.json`` to suppress
#   the attribution header while running.
#
# Prerequisites
#   1.  claude CLI (Claude Code) — from the official Anthropic installer:
#           - curl -L https://raw.githubusercontent.com/anthropics/claude-code/... | bash
#         Other options:
#           - brew install claude-code
#           - npm install -g @anthropic-ai/claude-code
#   2.  ollama:
#           brew install ollama
#       After installation, pull at least one model (see Usage below).
#   3.  Python 3.11+ (stdlib only — no third-party dependencies).
#
# Setup
#   1.  Put this script somewhere on your PATH, e.g.:
#           cp lclaude.py ~/bin/lclaude
#           chmod +x ~/bin/lclaude
#   2.  Pull a model you want to use:
#           ollama pull ornith:35b
#       See ``ollama list`` to check which models are already installed.
#       Other suitable models here: https://ollama.com/search?q=coding
#   3.  Add an alias to your shell config (~/.zshrc, ~/.bashrc, etc.):
#           alias lclaude='/full/path/to/lclaude.py'
#       Then reload: ``source ~/.zshrc`` (or restart your terminal).
#
# Usage
#   Basic (uses default model ornith:35b):
#       lclaude
#
#   Choose a specific model:
#       lclaude --model ornith:35b
#       lclaude --model ornith
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
#   1.  Verifies ollama is installed and running (auto-starts it if needed).
#   2.  Validates that the requested model is pulled into ollama (exits with an
#       error listing installed models if not).
#   3.  Backs up ``~/.claude/settings.json`` to ``settings.json.off``,
#       then sets ``CLAUDE_CODE_ATTRIBUTION_HEADER=0`` in the settings file.
#   4.  Sets ``ANTHROPIC_AUTH_TOKEN=ollama`` and
#       ``ANTHROPIC_BASE_URL=http://localhost:11434`` in the environment,
#       disables Claude's alternate-screen (fullscreen) TUI so the header
#       stays visible, then runs ``claude --model <model>`` with the
#       remaining args.
#   5.  On exit (normal, signal, or interrupt) the original settings file is
#       restored and the backup is deleted.
#
# Environment overrides
#   No env vars need to be set beforehand — this script manages everything
#   internally.  If ``ANTHROPIC_API_KEY`` is present in the parent env it
#   is deliberately stripped so traffic goes to Ollama, not the cloud.
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
from pathlib import Path
from typing import Any

logger = logging.getLogger("lclaude")

# Path to Claude Code's settings file and its backup copy
SETTINGS = Path.home() / ".claude" / "settings.json"
SETTINGS_OFF = SETTINGS.with_name(SETTINGS.name + ".off")

# Environment injected into the ``claude`` subprocess to route traffic to Ollama
CLAUDE_ENV = {
    "ANTHROPIC_AUTH_TOKEN": "ollama",
    "ANTHROPIC_BASE_URL": "http://localhost:11434",
    # Since v2.1.89 Claude Code defaults to a fullscreen alternate-screen TUI
    # that wipes anything printed on the main buffer (our header) as soon as
    # claude starts.  CLAUDE_CODE_DISABLE_ALTERNATE_SCREEN keeps Claude on the
    # main screen so the lclaude box stays in scrollback and native
    # selection / copy-paste / Cmd-F keep working.
    "CLAUDE_CODE_DISABLE_ALTERNATE_SCREEN": "1",
}

# Seconds to wait for ``ollama serve`` to accept connections after launch
OLLAMA_STARTUP_TIMEOUT = 30

# Seconds to wait for ``ollama list`` to complete
OLLAMA_LIST_TIMEOUT = 15

# ANSI accent shared by the status box and help-section headings
HEADER_ACCENT = "\x1b[92m"
ANSI_RESET = "\x1b[0m"

# Cached ollama version from the last pre-flight check (module-level singleton)
_OLLAMA_VERSION: str = "unknown"


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


def build_child_env() -> dict[str, str]:
    """Build the environment for the claude subprocess.

    Start from the parent env, inject Ollama routing vars, and strip
    ``ANTHROPIC_API_KEY`` so requests don't accidentally hit the cloud.
    """
    env = os.environ.copy()
    env.update(CLAUDE_ENV)
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
    ollama_ver: str | None = None,
) -> int:
    """Run ``claude --model <model>`` with Ollama env vars, settings backup, and
    signal-safe cleanup.

    Claude Code owns the TTY directly (no PTY shim) so mouse selection and
    copy/paste keep working. Session context is kept in the terminal tab title
    because Claude's redraw clears anything printed above it.
    """
    ensure_settings_file()
    backup_settings()

    interactive = (
        sys.stdin.isatty()
        and sys.stdout.isatty()
        and not _is_print_mode(claude_argv)
        and ollama_ver is not None
    )

    def on_signal(signum: int, _frame: Any) -> None:
        # Settings restore must happen here — ``finally`` may never run
        # if the process is killed by the signal.
        restore_settings()
        if interactive:
            # Clear terminal title using os.write (async-signal-safe) instead
            # of sys.stdout.write/flush which can deadlock in a signal handler.
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
            _set_terminal_title(f"lclaude · {model} · Ollama v{ollama_ver}")
        return subprocess.run(
            ["claude", "--model", model, *claude_argv],
            env=build_child_env(),
            check=False,
        ).returncode
    finally:
        if interactive:
            _set_terminal_title("")
        restore_settings()


def check_ollama(auto_start: bool = True) -> tuple[bool, str | None]:
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
    ok, ver = _is_ollama_reachable()
    if ok:
        global _OLLAMA_VERSION
        _OLLAMA_VERSION = ver
        return (True, None)

    # 3. Not reachable — try to start it
    if auto_start:
        logger.info(
            "ollama is installed but not running — starting it (%s)", ollama_bin
        )
        return _start_and_wait(ollama_bin)

    return (False, "ollama is installed but not running — start it with `ollama serve`")


def _is_ollama_reachable() -> tuple[bool, str]:
    """Probe Ollama's ``/api/version`` endpoint over raw HTTP.

    We use ``http.client`` instead of ``requests`` to avoid any dependency
    — Ollama's version endpoint is a plain GET, so the stdlib is enough.
    """
    try:
        conn = http.client.HTTPConnection("localhost", 11434, timeout=3)
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


def _start_and_wait(ollama_bin: str) -> tuple[bool, str | None]:
    """Launch ``ollama serve`` in the background and poll until it accepts
    connections (or we time out).

    ``start_new_session=True`` detaches the child from our terminal group so
    it doesn't receive Ctrl-C from the parent.
    """
    try:
        proc = subprocess.Popen(
            [ollama_bin, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        logger.info(
            "launched `ollama serve` (pid %d) — waiting for it to come up …", proc.pid
        )
    except OSError as exc:
        return (False, f"failed to start ollama: {exc}")

    # Poll the version endpoint until it responds or we time out
    deadline = time.monotonic() + OLLAMA_STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        ok, ver = _is_ollama_reachable()
        if ok:
            global _OLLAMA_VERSION
            _OLLAMA_VERSION = ver
            logger.info("ollama is now ready")
            return (True, None)
        time.sleep(0.5)

    # Timed out — the child may still be starting, clean up our stale process
    if proc.poll() is None:
        proc.terminate()
        proc.wait(timeout=5)
    return (False, f"ollama did not become ready within {OLLAMA_STARTUP_TIMEOUT}s")


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


def _use_color() -> bool:
    """Return whether stdout supports and allows ANSI color output."""
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR")


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


def _print_header(ollama_ver: str, model: str, installed_models: list[str]) -> None:
    """Print a boxed header (Claude Code–style) with Ollama version, model, and
    installed models."""
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
    ollama_line = (
        f" {dim}Ollama:{reset} {value}v{ollama_ver}{reset}"
        f"  {dim}Model:{reset} {value}{model}{reset}"
    )

    models_str = ", ".join(installed_models)
    models_prefix = f" {dim}Models:{reset} "
    max_models = inner_w - _visible_len(models_prefix)
    if _visible_len(models_str) > max_models:
        models_str = models_str[: max(0, max_models - 1)] + "…"
    models_line = models_prefix + models_str

    print(_box_top(title, width, border, reset), file=sys.stdout)
    print(_box_row(ollama_line, width, border, reset), file=sys.stdout)
    print(_box_row(models_line, width, border, reset), file=sys.stdout)
    print(_box_bottom(width, border, reset), file=sys.stdout)


def main(argv: list[str]) -> int:
    setup_logging()

    # Parse only our flags; pass the rest through verbatim to claude
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", default="ornith:35b")
    parser.add_argument(
        "--list",
        action="store_true",
        help="List models available in Ollama and exit.",
    )
    args, claude_argv = parser.parse_known_args(argv)
    model = args.model

    is_help = argv[:1] in (["-h"], ["--help"])

    # Pre-flight: verify ollama is installed and running (auto-starts if needed)
    ok, reason = check_ollama(auto_start=True)
    if not ok:
        logger.error("pre-flight failed — %s", reason)
        return 1

    if args.list:
        return list_ollama_models()

    # Validate the model is pulled; also capture installed model list for display
    installed_models = ensure_model_in_ollama(model)

    if is_help:
        logger.info("showing help")
        _print_header(_OLLAMA_VERSION, model, installed_models)
        heading = HEADER_ACCENT if _use_color() else ""
        reset = ANSI_RESET if heading else ""
        help_text = (
            f"{heading}Usage:{reset} lclaude [OPTIONS] [ARGS passed to claude]\n"
            "\n"
            "Run Claude Code against a locally served Ollama LLM\n"
            "\n"
            f"{heading}Options:{reset}\n"
            "  -h, --help           Show this help message and exit\n"
            "  --list               List models available in Ollama and exit\n"
            "  --model MODEL        Ollama model to use (default: ornith:35b)\n"
            "                       Run ``lclaude --list`` to see installed models\n"
            "\n"
            f"{heading}Examples:{reset}\n"
            "  lclaude                          # uses default model (ornith:35b)\n"
            "  lclaude --model ornith           # use the model's base name\n"
            "  lclaude --model ornith:35b       # with a specific tag\n"
            "  lclaude --list                   # list available models\n"
            '  lclaude --system "You are a helpful assistant" --message "Hello"\n'
            "\n"
            f"{heading}Prerequisites:{reset}\n"
            "  brew install ollama              # install Ollama\n"
            "  ollama pull ornith:35b           # pull a model\n"
            "  brew install claude-code         # install Claude Code CLI\n"
        )
        print(help_text, file=sys.stdout)
        return 0

    # Show a startup banner only when connected to a real terminal.
    # Session context also lives in the terminal tab title while Claude runs.
    if sys.stdout.isatty() and not _is_print_mode(claude_argv):
        _print_header(_OLLAMA_VERSION, model, installed_models)
    return run_claude(claude_argv, model, ollama_ver=_OLLAMA_VERSION)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
