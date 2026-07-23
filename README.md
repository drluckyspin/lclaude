# LCLAUDE

![alt text](screenshot.png)

Run [Claude Code](https://docs.anthropic.com/en/docs/claude-code) against a **local** LLM instead of the Anthropic
cloud API.

`lclaude` (local claude) is a single-file Python wrapper (**stdlib only**, **Python 3.11+**) that points the Claude Code
CLI at Ollama, llama.cpp, or both. No proxy process, no extra dependencies — just env vars and a thin pre-flight.

| Backend | Role | Default port |
| ------- | ---- | ------------ |
| **Ollama** | Model registry + inference | 11434 |
| **llama.cpp** (`llama-server`) | User-managed GGUF server | 8080 |
| **hybrid** | Ollama for models, llama-server for inference | 9090 |

With `--backend auto` (default), lclaude probes Ollama then llama-server and uses whichever responds.

## What is Claude Code?

[Claude Code](https://docs.anthropic.com/en/docs/claude-code) is Anthropic’s agentic coding CLI. It reads your repo,
runs tools (edit files, shell, search), and talks to a model over the **Anthropic Messages API**
(`POST /v1/messages`). Out of the box it expects Anthropic’s cloud. Both [Ollama](https://ollama.com/) and
[llama.cpp](https://github.com/ggml-org/llama.cpp)’s `llama-server` can speak that same API on localhost — so Claude
Code can drive a local model if you redirect `ANTHROPIC_BASE_URL` and auth.

## What is Ollama? What is llama.cpp?

[Ollama](https://ollama.com/) is the easy local model manager: `ollama pull`, `ollama list`, and a server on
`:11434`. Great for day-to-day use; chat templates and GGUF details stay mostly hidden.

[llama.cpp](https://github.com/ggml-org/llama.cpp) (`llama-server`) is a fast OpenAI-/Anthropic-compatible HTTP server
over GGUF weights. You control quants, GPU layers, context, and — critically — the **chat template**. Some models
(notably Ornith / Qwen 3.6) need a Claude-friendly template for tool use to work.

**hybrid** mode uses both: Ollama as the model registry, llama-server for inference, with lclaude wiring the blob path
and template automatically.

## Why this project?

Claude Code is excellent, but cloud-only by default. Local runners already speak the Anthropic API; what’s missing is
the glue: pick a backend, validate the model, fix chat-template footguns, set the right env vars, keep the UI banner
visible, and clean up on exit.

lclaude is that glue in one script you can drop on `PATH`. Teams that already use Ollama keep `ollama pull`. People who
want llama.cpp performance (or a working Ornith template) get it without maintaining a separate launcher. Hybrid mode
is the “best of both” path when you want registry convenience and llama.cpp inference.

## How it works

```text
lclaude (setup)  →  claude (CLI)  →  Ollama :11434
                                  ↘  llama-server :8080 / :9090
                       POST /v1/messages
```

1. Resolve backend (`auto` probes Ollama, then llama-server).
2. Pre-flight: start/validate the server, check the model, patch the chat template when needed (hybrid).
3. Point Claude Code at localhost (`ANTHROPIC_BASE_URL` + dummy token; strip `ANTHROPIC_API_KEY`).
4. On exit / signal: restore `~/.claude/settings.json` and stop any llama-server lclaude started.

No proxy — traffic goes straight from `claude` to the local Anthropic-compatible endpoint.

## Quick Start

### Ollama (simplest)

```bash
brew install ollama claude-code
ollama pull ornith:35b
cp lclaude.py ~/bin/lclaude && chmod +x ~/bin/lclaude
lclaude
```

### Hybrid (recommended for Ornith + llama.cpp)

Ollama manages models; lclaude starts llama-server from the Ollama blob, auto-fixes chat templates, and tears the
server down on exit.

```bash
brew install ollama llama.cpp claude-code
ollama pull ornith:35b
lclaude --backend hybrid --model ornith:35b
```

Server log (follow in another terminal):

```bash
tail -f ~/.cache/lclaude/llama-server.log
```

### llama.cpp (you manage the server)

```bash
# Terminal 1 — Ornith needs a Claude-compatible chat template (not plain --jinja)
mkdir -p ~/.cache/lclaude
curl -L https://huggingface.co/spiritbuun/buun-Qwen3.6-chat_template/raw/main/chat_template.jinja \
  -o ~/.cache/lclaude/qwen3.6-claude.jinja
llama-server -hf deepreinforce-ai/Ornith-1.0-35B-GGUF \
  --chat-template-file ~/.cache/lclaude/qwen3.6-claude.jinja --port 8080

# Terminal 2
lclaude --backend llamacpp --model ornith:35b
```

> [!NOTE]
> Ornith’s embedded Qwen 3.6 template rejects system messages after the first turn — that breaks Claude Code tool use.
> **hybrid** patches this automatically. For a manual `llama-server`, always use `--chat-template-file` as above.

## Installation

1. Put the script on your `PATH`:

   ```bash
   cp lclaude.py ~/bin/lclaude
   chmod +x ~/bin/lclaude
   ```

2. Install **Claude Code** (`brew install claude-code` or [docs](https://docs.anthropic.com/en/docs/claude-code)).

3. Install at least one backend:

   - Ollama: `brew install ollama` then `ollama pull ornith:35b`
   - llama.cpp: `brew install llama.cpp` (required for `llamacpp` and `hybrid`)

4. Optional alias in `~/.zshrc` / `~/.bashrc`:

   ```bash
   alias lclaude='/full/path/to/lclaude.py'
   ```

## Usage

```bash
lclaude                                    # auto-detect backend
lclaude --model ornith:35b
lclaude --backend ollama
lclaude --backend hybrid --model ornith:35b
lclaude --backend llamacpp --port 8080
lclaude --list                             # ollama / hybrid: list Ollama models
lclaude --help
lclaude -p "explain this file"             # args after lclaude flags go to claude
```

### Options

| Option | Description | Default |
| ------ | ----------- | ------- |
| `--backend` | `ollama`, `llamacpp`, `hybrid`, or `auto` | `auto` |
| `--port` | Override listen port | 11434 / 8080 / 9090 |
| `--model` | Model name (must exist in Ollama for `ollama`/`hybrid`; cosmetic for `llamacpp`) | `ornith:35b` |
| `--list` | List Ollama models (`ollama` / `hybrid` only) | — |
| `-h`, `--help` | Show help | — |

## Backend comparison

| Feature | Ollama | llama.cpp | hybrid |
| ------- | ------ | --------- | ------ |
| Model management | `ollama pull` | Manual GGUF / `-hf` | Via Ollama |
| Auto-start server | Yes (`ollama serve`) | No | Yes (`llama-server` from blob) |
| Model validation | Yes | No | Yes |
| Template auto-fix | N/A | Manual | Yes (Ornith / Qwen 3.x) |
| Default port | 11434 | 8080 | 9090 |
| Needs | Ollama | llama.cpp | Both |

**Prefer Ollama** for simplest day-to-day use.
**Prefer hybrid** when you want Ollama’s registry but llama.cpp inference (and automatic template fixes).
**Prefer llamacpp** when you already run `llama-server` with custom flags / quants.

## Startup sequence

1. Resolve backend (`auto` probes `:11434/api/version` then `:8080/health`).
2. Pre-flight:
   - **Ollama** — binary + health; auto-start `ollama serve`; validate model via `ollama list`.
   - **llamacpp** — `/health`; reject incompatible chat templates with fix instructions.
   - **hybrid** — validate model in Ollama; print LCLAUDE banner; resolve blob via `ollama show --modelfile`;
     download patched template if needed; start `llama-server` (log file); fail fast with a short colored error if
     load fails; stop server on exit.
3. Backup `~/.claude/settings.json`, set `CLAUDE_CODE_ATTRIBUTION_HEADER=0`.
4. Run `claude --model <model> …` with:
   - `ANTHROPIC_BASE_URL=http://localhost:<port>`
   - `ANTHROPIC_AUTH_TOKEN=lclaude`
   - `ANTHROPIC_API_KEY` removed (so traffic does not hit the cloud)
   - `CLAUDE_CODE_DISABLE_ALTERNATE_SCREEN=1` (keeps the banner in scrollback)
5. Restore settings (and stop owned llama-server) on exit / `SIGTERM` / `SIGHUP`.

## Hybrid mode

```bash
ollama pull ornith:35b
lclaude --backend hybrid --model ornith:35b
```

### Under the hood

1. Ensure Ollama is up; confirm the model is installed
2. Resolve GGUF path: `ollama show <model> --modelfile` → `FROM …/blobs/sha256-…`
3. If the name matches Ornith / Qwen 3.x, ensure `~/.cache/lclaude/qwen3.6-claude.jinja` (download once)
4. Start `llama-server -m <blob> --chat-template-file <patched> -ngl 99 --port 9090`
5. Write stdout/stderr to `~/.cache/lclaude/llama-server.log` (truncated each launch)
6. Poll `/health`; if the process exits, print a short Cause / Hint / Try / Log error and exit
7. Launch Claude Code; stop llama-server on exit

### Cache layout

```text
~/.cache/lclaude/
  qwen3.6-claude.jinja    # Claude-compatible chat template (optional download)
  llama-server.log        # hybrid llama-server log (current session)
```

### Compatibility

- Not every Ollama blob loads in every llama.cpp build. Example: some `qwen3.5` blobs fail with
  `qwen35.rope.dimension_sections` metadata mismatches. Fall back to `--backend ollama`, or use an upstream HuggingFace
  GGUF with `--backend llamacpp`.
- Do not `ollama rm <model>` while hybrid is using that blob.
- Port **9090** avoids colliding with a user-managed server on **8080**.

## llama.cpp setup (manual)

```bash
brew install llama.cpp
# or: build llama-server from https://github.com/ggml-org/llama.cpp

mkdir -p ~/.cache/lclaude
curl -L https://huggingface.co/spiritbuun/buun-Qwen3.6-chat_template/raw/main/chat_template.jinja \
  -o ~/.cache/lclaude/qwen3.6-claude.jinja

llama-server -hf deepreinforce-ai/Ornith-1.0-35B-GGUF \
  --chat-template-file ~/.cache/lclaude/qwen3.6-claude.jinja --port 8080
```

Useful flags: `-ngl 99` (GPU), `-c 262144` (context), `-np 4` (parallel), `--api-key …`, `--port …`.

Reuse an Ollama blob without hybrid:

```bash
ollama show ornith:35b --modelfile | grep '^FROM'
llama-server -m /path/from/FROM --chat-template-file ~/.cache/lclaude/qwen3.6-claude.jinja --port 8080
```

Notes for `--backend llamacpp`:

- `--model` is display-only (server already has one model loaded)
- No auto-start; no `--list`
- If the live template rejects late system messages, lclaude refuses to start and tells you how to fix it (or use hybrid)

## Troubleshooting

| Issue | What to try |
| ----- | ----------- |
| `no local LLM backend detected` | Start Ollama or llama-server, or pass `--backend` |
| `ollama not found` / not running | Install Ollama; `ollama serve` or let lclaude start it |
| `model '…' not found in ollama` | `ollama pull <model>` or `lclaude --list` |
| `llama-server not reachable` | Start server on the expected port |
| Incompatible chat template | Use `--chat-template-file`, or `--backend hybrid` |
| Hybrid: could not load model | See Cause in the error; check log; try `--backend ollama` |
| Hybrid: template download failed | `curl` the URL into `~/.cache/lclaude/qwen3.6-claude.jinja` (SSL proxies like Zscaler can break Python urllib) |
| `claude` not found | Install Claude Code on `PATH` |

> [!TIP]
> First load into memory can be slow; later sessions are faster.

## License

MIT — Copyright (c) 2026 Todd Papaioannou
