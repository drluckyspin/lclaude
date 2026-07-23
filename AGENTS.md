# AGENTS.md — lclaude

Guidance for coding agents working on this repository.

## What this project is

`lclaude` (local Claude) is a **single-file** Python 3.11+ launcher that runs Anthropic’s **Claude Code** CLI against a
**local** LLM instead of `api.anthropic.com`.

It does **not** implement chat/completions itself. It:

1. Pre-flights a local backend
2. Temporarily patches `~/.claude/settings.json`
3. Spawns `claude --model <name> …` with env vars that point Anthropic API traffic at localhost

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

| `--backend` | Constant | Default port | Inference | Model source |
| ----------- | -------- | ------------ | --------- | ------------ |
| `auto` | `BACKEND_AUTO` | (detect) | First healthy of Ollama / llamacpp | — |
| `ollama` | `BACKEND_OLLAMA` | 11434 | Ollama daemon | `ollama pull` / `--model` |
| `llamacpp` | `BACKEND_LLAMACPP` | 8080 | Existing `llama-server` | Already loaded in server |
| `hybrid` | `BACKEND_HYBRID` | **9090** | lclaude-owned `llama-server` | Ollama blob path |

`DEFAULT_PORTS` in `lclaude.py` is the source of truth.

**Auto-detect** (`detect_backend`): probe Ollama `GET /api/version` on the Ollama port, else llama-server `GET /health`
on the llamacpp port. Hybrid is **never** auto-selected — user must pass `--backend hybrid`.

### Env injected into `claude` (`build_child_env`)

- `ANTHROPIC_BASE_URL=http://localhost:<port>`
- `ANTHROPIC_AUTH_TOKEN=lclaude`
- `CLAUDE_CODE_DISABLE_ALTERNATE_SCREEN=1` — required so Claude Code does not wipe the LCLAUDE banner (alt-screen TUI
  since Claude Code ~v2.1.89)
- **`ANTHROPIC_API_KEY` is stripped** so the SDK does not prefer cloud Anthropic over `BASE_URL`

### Settings lifecycle

- Paths: `~/.claude/settings.json` and backup `settings.json.off`
- On run: backup → set `env.CLAUDE_CODE_ATTRIBUTION_HEADER=0` → spawn claude
- On exit / `SIGTERM` / `SIGHUP`: restore settings; if hybrid, terminate owned `llama-server`
- Signal handler must use async-signal-safe cleanup (`os.write` for title clear, `os._exit`) — do not use buffered
  stdout flush in the handler

## Critical product knowledge

### 1. Chat template incompatibility (Ornith / Qwen 3.6)

Stock Ornith / Qwen 3.6 Jinja templates embed logic equivalent to *“system message must be at the beginning”*. Claude
Code sends additional system/developer messages after the first turn (tool use). With llama-server `--jinja`, that
returns **HTTP 400** before inference.

**Detection:** `_llamacpp_template_rejects_late_system_messages()` reads `/props` → `chat_template` for the string
`System message must be at the beginning`.

**Fix (hybrid):** download patched template once to `~/.cache/lclaude/qwen3.6-claude.jinja` and start with
`--chat-template-file` (not `--jinja`):

- URL constant: `QWEN36_TEMPLATE_URL` (HuggingFace `spiritbuun/buun-Qwen3.6-chat_template` … `/chat_template.jinja`)
- Heuristic: `_model_needs_template_patch` if model name contains tokens in `MODELS_NEEDING_TEMPLATE_PATCH`
  (`ornith`, `qwen3`)
- Download: `ensure_patched_template()` — try `urllib`, fall back to `curl` (corporate SSL MITM / Zscaler often breaks
  Python’s cert store while curl works)

**Fix (manual llamacpp):** user must pass `--chat-template-file`; `check_llamacpp()` refuses to proceed if the live
template is bad and points them at hybrid or curl instructions.

### 2. Ollama blob ≠ always loadable in llama.cpp

Ollama stores GGUFs under `~/.ollama/models/blobs/sha256-…`. Hybrid resolves them via:

```bash
ollama show <model> --modelfile   # parse FROM <path>
```

(`resolve_ollama_blob`)

Some Ollama-distributed blobs **fail to load** in a given llama.cpp build (metadata drift). Real example:

```text
qwen35.rope.dimension_sections has wrong array length; expected 4, got 3
```

`ornith:35b` has been observed to work; `qwen3.5` may not. On failure, hybrid must **not hang** — see below.

### 3. Hybrid must fail fast if llama-server exits

`wait_for_llamacpp(..., proc=server_proc)` polls `/health` but returns immediately if `proc.poll() is not None`.

Startup failure UX (`_hybrid_startup_failure_message`):

1. Print LCLAUDE header **first** (version `…`, include Log path) before starting the server
2. On failure, print a short colored Cause / Hint / Try / Log block (parse cause via `_parse_llamacpp_load_error`)
3. Do **not** dump raw multi-line llama-server logs to the terminal (log file is enough)

Log file: `LLAMACPP_LOG_FILE` = `~/.cache/lclaude/llama-server.log` (truncated each hybrid launch).

### 4. Header labeling

Hybrid **inference** is llama.cpp. The status box and terminal title should show **`llama.cpp`** (and llama-server
`build_info` from `/props`), not the word `hybrid`. Distinguish modes by port (9090 vs 8080) and Log line.

Version parsing: prefer `/props` field `build_info` (e.g. `b10090-7347430f4`); fall back to legacy `build_number` /
`build_commit`.

### 5. `--help` detection

`is_help = any(arg in ("-h", "--help") for arg in argv)` — must work when flags appear after `--backend`, etc. Help /
`--list` for hybrid must **not** start llama-server.

### 6. Pass-through CLI

Only parse lclaude flags with `argparse.ArgumentParser(add_help=False)` + `parse_known_args`; remaining argv goes to
`claude` unchanged.

## Key functions (map)

| Function | Role |
| -------- | ---- |
| `main` | CLI, backend resolution, pre-flight, header, `run_claude` |
| `run_claude` | Settings backup/patch, spawn claude, signal-safe cleanup, stop owned server |
| `build_child_env` | Anthropic routing env |
| `check_ollama` / `_is_ollama_reachable` / `_start_and_wait` | Ollama health + auto-start |
| `check_llamacpp` / `_is_llamacpp_reachable` | External llama-server health + template gate |
| `_get_llamacpp_props` / `_get_llamacpp_version` | `/props` |
| `detect_backend` | auto mode |
| `ensure_model_in_ollama` / `list_ollama_models` | Model registry |
| `resolve_ollama_blob` | Modelfile → blob path |
| `ensure_patched_template` | Cache Claude-compatible Jinja |
| `start_llamacpp_server` | Spawn llama-server → log file |
| `wait_for_llamacpp` | Health wait + early exit if process died |
| `prepare_hybrid_backend` | Blob + template + start + wait + template verify |
| `_print_header` | Startup banner |

## Cache / filesystem

```text
~/.claude/settings.json          # Claude Code settings (temporarily patched)
~/.claude/settings.json.off      # backup during run
~/.cache/lclaude/qwen3.6-claude.jinja
~/.cache/lclaude/llama-server.log
~/.ollama/models/blobs/sha256-…  # Ollama GGUFs (read-only for hybrid)
```

## Coding conventions

- Match existing style: type hints, short docstrings, `print(..., file=sys.stderr)` for user errors
- Logging logger is CRITICAL by default — user-visible messages use `print`, not `logger.info`
- Prefer extending existing helpers over new abstraction layers / classes unless complexity demands it
- Do not add dependencies
- When changing hybrid startup, preserve: early header → fail-fast → concise colored error → log file for detail
- Update **both** `README.md` and this `AGENTS.md` when behavior/ports/URLs/constants change
- Do not edit plan files under `.cursor/plans/` unless the user asks

## Manual test checklist

```bash
python3 -c "import ast; ast.parse(open('lclaude.py').read())"

# Help / list (no server start for hybrid)
python3 lclaude.py --help
python3 lclaude.py --backend hybrid --help
python3 lclaude.py --list
python3 lclaude.py --backend hybrid --list

# Ollama (requires ollama + model)
python3 lclaude.py --backend ollama --model ornith:35b

# Hybrid success (requires ollama model + llama-server on PATH)
python3 lclaude.py --backend hybrid --model ornith:35b
# other terminal: tail -f ~/.cache/lclaude/llama-server.log

# Hybrid failure path (known-bad blob if present)
python3 lclaude.py --backend hybrid --model qwen3.5
# Expect: header, then Cause/Hint/Try/Log — no hang

# llamacpp (requires user-started server on 8080 with good template)
python3 lclaude.py --backend llamacpp
```

## Common pitfalls for agents

- Do not reintroduce `--jinja` as the only hybrid template path for Ornith
- Do not block for `LLAMACPP_STARTUP_TIMEOUT` after llama-server has already exited
- Do not dump full llama-server logs into stderr on hybrid failure
- Do not label the hybrid header as `hybrid:` next to a llama.cpp build id
- Do not leave `ANTHROPIC_API_KEY` in the child env
- Do not assume all Ollama models work under hybrid — document fallbacks
- Port **9090** for hybrid is intentional (avoid clash with user llama-server on 8080)
