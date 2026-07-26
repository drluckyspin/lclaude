# AGENTS.md — lclaude

Guidance for coding agents working on this repository.

## What this project is

`lclaude` (local Claude) is a **single-file** Python 3.11+ launcher that runs Anthropic’s **Claude Code** CLI against a
**local** LLM instead of `api.anthropic.com`.

It does **not** implement chat/completions itself. It:

1. Resolves a local backend (smart `auto`, or an explicit override)
2. Temporarily patches `~/.claude/settings.json`
3. Spawns `claude --model <name> …` with env vars that point Anthropic API traffic at localhost
4. Persists last-used prefs to `~/.config/lclaude/config.toml`

**Stack constraints (do not violate without explicit user approval):**

- One main script: [`lclaude.py`](lclaude.py)
- **stdlib only** — no `requirements.txt`, no third-party packages
- Keep the UX of a small CLI wrapper (header box, pass-through args to `claude`)

User-facing docs: [`README.md`](README.md). License: MIT.

## Architecture

```text
User → lclaude.py → claude CLI → POST /v1/messages → local server
                              ← Anthropic-compatible SSE ←
```

Claude Code speaks the **Anthropic Messages API**. Both **Ollama** (since ~Jan 2026) and **llama-server** expose
`POST /v1/messages`. No Anthropic↔OpenAI proxy is required.

### Backends

| `--backend` | Constant           | Default port | Inference                    | Model source              |
| ----------- | ------------------ | ------------ | ---------------------------- | ------------------------- |
| `auto`      | `BACKEND_AUTO`     | (resolve)    | Smart policy (see below)     | —                         |
| `ollama`    | `BACKEND_OLLAMA`   | 11434        | Ollama daemon                | `ollama pull` / `--model` |
| `llamacpp`  | `BACKEND_LLAMACPP` | 8080         | Existing `llama-server`      | Already loaded in server  |
| `managed`   | `BACKEND_MANAGED`  | **9090**     | lclaude-owned `llama-server` | Ollama blob path          |

`DEFAULT_PORTS` in `lclaude.py` is the source of truth. There is **no** `hybrid` backend (never released; use
`managed`).

### Auto resolve (`resolve_backend`)

1. Healthy or still-loading user `llama-server` on the llamacpp port → `llamacpp`
2. Else if `_model_needs_template_patch(model)` and `llama-server` on `PATH` and Ollama has the model → `managed`
3. Else if Ollama usable (auto-start OK) → `ollama`
4. Else `None` (error)

`--list` with `auto` forces the Ollama path so a warm llama-server does not block listing.

### Config

- Path: `CONFIG_FILE` = `~/.config/lclaude/config.toml`
- Read: stdlib `tomllib`; invalid file → warn + empty
- Write: hand-rolled TOML after **successful** pre-flight (not on `--help` / failed resolve / early `--list` return)
- Save **requested** `backend` (keep `auto` sticky), effective `model`, and `port` only when an explicit override was
  used (CLI / `LCLAUDE_PORT` / config)
- Precedence: CLI → `LCLAUDE_MODEL` / `LCLAUDE_BACKEND` / `LCLAUDE_PORT` → config → built-ins (`DEFAULT_MODEL`, `auto`)

### Env injected into `claude` (`build_child_env`)

- `ANTHROPIC_BASE_URL=http://localhost:<port>`
- `ANTHROPIC_AUTH_TOKEN=lclaude`
- `CLAUDE_CODE_DISABLE_ALTERNATE_SCREEN=1` — required so Claude Code does not wipe the LCLAUDE banner (alt-screen TUI
  since Claude Code ~v2.1.89)
- **`ANTHROPIC_API_KEY` is stripped** so the SDK does not prefer cloud Anthropic over `BASE_URL`

### Settings lifecycle

- Paths: `~/.claude/settings.json` and backup `settings.json.off`
- On run: backup → set `env.CLAUDE_CODE_ATTRIBUTION_HEADER=0` → spawn claude
- On exit / `SIGTERM` / `SIGHUP`: restore settings; if managed, terminate owned `llama-server`
- Signal handler must use async-signal-safe cleanup (`os.write` for title clear, `os._exit`) — do not use buffered
  stdout flush in the handler

## Critical product knowledge

### 1. Chat template incompatibility (Ornith / Qwen 3.6)

Stock Ornith / Qwen 3.6 Jinja templates embed logic equivalent to _“system message must be at the beginning”_. Claude
Code sends additional system/developer messages after the first turn (tool use). With llama-server `--jinja`, that
returns **HTTP 400** before inference.

**Detection:** `_llamacpp_template_rejects_late_system_messages()` reads `/props` → `chat_template` for the string
`System message must be at the beginning`.

**Fix (managed / auto→managed):** download patched template once to `~/.cache/lclaude/qwen3.6-claude.jinja` and start
with `--chat-template-file` (not `--jinja`):

- URL constant: `QWEN36_TEMPLATE_URL` (HuggingFace `spiritbuun/buun-Qwen3.6-chat_template` … `/chat_template.jinja`)
- Heuristic: `_model_needs_template_patch` if model name contains tokens in `MODELS_NEEDING_TEMPLATE_PATCH` (`ornith`,
  `qwen3`)
- Download: `ensure_patched_template()` — try `urllib`, fall back to `curl` (corporate SSL MITM / Zscaler often breaks
  Python’s cert store while curl works)

**Fix (manual llamacpp):** user must pass `--chat-template-file`; `check_llamacpp()` refuses to proceed if the live
template is bad and points them at managed or curl instructions.

### 2. Ollama blob ≠ always loadable in llama.cpp

Ollama stores GGUFs under `~/.ollama/models/blobs/sha256-…`. Managed mode resolves them via:

```bash
ollama show <model> --modelfile   # parse FROM <path>
```

(`resolve_ollama_blob`)

Some Ollama-distributed blobs **fail to load** in a given llama.cpp build (metadata drift). Real example:

```text
qwen35.rope.dimension_sections has wrong array length; expected 4, got 3
```

`ornith:35b` has been observed to work; `qwen3.5` may not. On failure, managed must **not hang** — see below.

### 3. Managed must fail fast if llama-server exits

`wait_for_llamacpp(..., proc=server_proc)` polls `/health` but returns immediately if `proc.poll() is not None`.

Startup failure UX (`_managed_startup_failure_message`):

1. Print LCLAUDE header **first** (fast `llama-server --version` build if available; include Log path) before starting
   the server
2. On failure, print a short colored Cause / Hint / Try / Log block (parse cause via `_parse_llamacpp_load_error`)
3. Do **not** dump raw multi-line llama-server logs to the terminal (log file is enough)

Log file: `LLAMACPP_LOG_FILE` = `~/.cache/lclaude/llama-server.log` (truncated each managed launch).

### 4. Header labeling

The status box and terminal title distinguish the user's selection from the resolved engine:

- **Backend** — requested selection (`auto`, `ollama`, `llamacpp`, or `managed`)
- **Engine** — actual inference engine and build (`Ollama` or `llama.cpp`)
- **Mode** — `Ollama`, `external` (user server), or `managed` (lclaude-owned server)

Managed inference therefore shows `Engine: llama.cpp`, not `Engine: managed`. Distinguish modes by the Mode field, API
endpoint (`http://localhost:9090` vs `:8080`), and Log line.

Version parsing: prefer `/props` field `build_info` (e.g. `b10090-7347430f4`); fall back to legacy `build_number` /
`build_commit`.

### 5. `--help` detection

`is_help = any(arg in ("-h", "--help") for arg in argv)` — must work when flags appear after `--backend`, etc. Help /
`--list` for managed must **not** start llama-server.

### 6. Pass-through CLI

Only parse lclaude flags with `argparse.ArgumentParser(add_help=False)` + `parse_known_args`; remaining argv goes to
`claude` unchanged. Flag defaults are `None` so merge can apply CLI > env > config > built-in.

## Key functions (map)

| Function                           | Role                                                              |
| ---------------------------------- | ----------------------------------------------------------------- |
| `main`                             | CLI, config merge, resolve, pre-flight, save config, `run_claude` |
| `load_config` / `save_config`      | `~/.config/lclaude/config.toml`                                   |
| `resolve_backend`                  | Smart auto policy                                                 |
| `run_claude`                       | Settings backup/patch, spawn claude, cleanup, stop owned server   |
| `build_child_env`                  | Anthropic routing env                                             |
| `check_ollama` / `_start_and_wait` | Ollama health + auto-start                                        |
| `check_llamacpp`                   | External llama-server health + template gate                      |
| `_ollama_has_model`                | Non-fatal model presence check for auto                           |
| `ensure_model_in_ollama`           | Fatal model validation for ollama/managed                         |
| `resolve_ollama_blob`              | Modelfile → blob path                                             |
| `ensure_patched_template`          | Cache Claude-compatible Jinja                                     |
| `start_llamacpp_server`            | Spawn llama-server → log file                                     |
| `_get_llamacpp_binary_version`     | Fast pre-load `llama-server --version` build lookup               |
| `wait_for_llamacpp`                | Health wait + early exit if process died                          |
| `prepare_managed_backend`          | Blob + template + start + wait + template verify                  |
| `_print_header`                    | Startup banner                                                    |

## Benchmark utility

`lclaude-bench.py` measures the raw streaming `POST /v1/messages` API path for `ollama`, owned `managed`, and optional
user-managed `llamacpp` targets. It imports `lclaude` helpers for constants, read-only config, pre-flight, and owned
managed-server cleanup.

- It applies the same CLI → `LCLAUDE_*` env → config → defaults preference order, but **must never call**
  `save_config()`, `main()`, `run_claude()`, or modify Claude settings
- An `auto` backend preference benchmarks Ollama then managed llama.cpp; a pinned preference benchmarks only that
  backend unless `--backends` is supplied; `llamacpp` is an optional user-managed target
- Keep startup timing separate from request metrics; report median TTFT, total latency, output tokens, and decode tokens
  per second across measured repeats. `startup_s` is JSON-only diagnostic metadata and is not comparable between the
  persistent Ollama daemon and benchmark-owned Managed llama.cpp
- Progress belongs on stderr so `--json` keeps stdout machine-readable; show milestones and a TTY spinner unless the
  user passes `--quiet`. Keep normal mode concise (one spinner per backend); `--verbose` may show setup, warmup, and
  per-request progress. Normal milestones are `Running benchmarks`, `Running <backend> benchmarks`, and
  `Rendering results`
- Managed-server ownership belongs to the benchmark only while its suite is running; always terminate it unless the user
  explicitly requests `--keep-managed`
- The default benchmark is a fixed, staged workload suite: quick latency, Makefile generation, and a larger Makefile
  generation task with 256, 768, and 1536-token caps. The prompts must stay self-contained and revision-independent so
  cross-machine results are comparable; `--quick` runs only the latency stage, while `--prompt` / `--prompt-file` select
  one custom workload. `--quick` conflicts with custom prompts and `--max-tokens`
- Human output consists of an `LCLAUDE-BENCH` parameter box followed by a boxed `BENCHMARK RESULTS` panel with
  workload-specific tables and Ollama-vs-Managed total-latency/decode-throughput comparisons when both metrics exist
- Result JSON is grouped by `workloads[]`, each with `backends[]` and an optional `comparison`; dry-run JSON has a
  top-level target list. The benchmark header moves to stderr in JSON mode, and `--quiet` suppresses both that header
  and progress
- Keep documented benchmark controls in sync: global/per-backend ports, `--timeout`, `--keep-managed`, `--no-color`,
  `--quiet`, and `--verbose`
- Do not treat this as a full Claude Code benchmark: tool loops, growing context, and terminal I/O are intentionally out
  of scope

## Cache / filesystem

```text
~/.config/lclaude/config.toml    # last-used prefs
~/.claude/settings.json          # Claude Code settings (temporarily patched)
~/.claude/settings.json.off      # backup during run
~/.cache/lclaude/qwen3.6-claude.jinja
~/.cache/lclaude/llama-server.log
~/.ollama/models/blobs/sha256-…  # Ollama GGUFs (read-only for managed)
```

## Coding conventions

- Match existing style: type hints, short docstrings, `print(..., file=sys.stderr)` for user errors
- Logging logger is CRITICAL by default — user-visible messages use `print`, not `logger.info`
- Prefer extending existing helpers over new abstraction layers / classes unless complexity demands it
- Do not add dependencies
- When changing managed startup, preserve: early header → fail-fast → concise colored error → log file for detail
- Update **both** `README.md` and this `AGENTS.md` when behavior/ports/URLs/constants change
- Do not edit plan files under `.cursor/plans/` unless the user asks

## Manual test checklist

`make test` runs the offline behavioral suite with temporary files and mocks; it does not contact a backend or launch
Claude Code.

```bash
make format
make test
make run ARGS="--version"
make bump-version 0.4.0

python3 -c "import ast; ast.parse(open('lclaude.py').read())"
python3 -c "import ast; ast.parse(open('lclaude-bench.py').read())"

# Help / list (no server start for managed)
python3 lclaude.py --help
python3 lclaude.py --backend managed --help
python3 lclaude.py --list
python3 lclaude.py --backend managed --list
python3 lclaude.py --version  # prints local backend status; never launches Claude Code or a managed server

# Config create/reuse
rm -f ~/.config/lclaude/config.toml
python3 lclaude.py --backend ollama --model ornith:35b   # creates config
cat ~/.config/lclaude/config.toml                        # backend should stay "ollama" if that was requested
LCLAUDE_MODEL=ornith:35b python3 lclaude.py --help       # env merge smoke

# Auto (requires local backends)
python3 lclaude.py --model ornith:35b

# Managed success (requires ollama model + llama-server on PATH)
python3 lclaude.py --backend managed --model ornith:35b
# other terminal: tail -f ~/.cache/lclaude/llama-server.log

# Managed failure path (known-bad blob if present)
python3 lclaude.py --backend managed --model qwen3.5
# Expect: header, then Cause/Hint/Try/Log — no hang

# llamacpp (requires user-started server on 8080 with good template)
python3 lclaude.py --backend llamacpp

# Benchmark help / non-mutating unavailable-server JSON path
python3 lclaude-bench.py --help
python3 lclaude-bench.py --backends llamacpp --port 1 --repeats 1 --warmup 0 --json

# Benchmark both local engines (requires ornith:35b + llama-server)
python3 lclaude-bench.py --backends ollama,managed --warmup 1 --repeats 5
python3 lclaude-bench.py --quick --backends ollama,managed --warmup 0 --repeats 1
```

## Common pitfalls for agents

- Do not reintroduce a `hybrid` backend name
- Do not reintroduce `--jinja` as the only managed template path for Ornith
- Do not block for `LLAMACPP_STARTUP_TIMEOUT` after llama-server has already exited
- Do not dump full llama-server logs into stderr on managed failure
- Do not label the managed header as `managed:` next to a llama.cpp build id
- Do not leave `ANTHROPIC_API_KEY` in the child env
- Do not save the _resolved_ backend when the user requested `auto`
- Do not assume all Ollama models work under managed — document fallbacks
- Port **9090** for managed is intentional (avoid clash with user llama-server on 8080)
