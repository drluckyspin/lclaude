# LCLAUDE

![alt text](screenshot.png)

Run [Claude Code](https://docs.anthropic.com/en/docs/claude-code) against a local [Ollama](https://ollama.com) model
instead of the Anthropic cloud API.

`lclaude` (local claude) is a small Python wrapper that:

- Verifies [Ollama](https://ollama.com) is installed and running (and starts `ollama serve` if needed)
- Confirms the requested model is pulled before launching
- Temporarily patches `~/.claude/settings.json` to be optimal for local models, then restores your settings on exit
- Routes Claude Code traffic to `http://localhost:11434` (Ollama’s Anthropic-compatible endpoint)

Requires no third-party Python packages — `stdlib` only (Python 3.11+).

## Prerequisites

1. **Claude Code CLI** — one of:
   - `brew install claude-code`
   - `npm install -g @anthropic-ai/claude-code`
   - [Official installer](https://docs.anthropic.com/en/docs/claude-code)
2. **Ollama** — `brew install ollama` (or install from [ollama.com](https://ollama.com))
3. **Python 3.11+**

## Installation

1. Clone or copy this repo, then put the script on your `PATH`:

   ```bash
   cp lclaude.py ~/bin/lclaude
   chmod +x ~/bin/lclaude
   ```

2. Pull a model (default is `qwen3.6`):

   ```bash
   ollama pull qwen3.6
   ```

   See [coding-oriented models on Ollama](https://ollama.com/search?q=coding) for alternatives. Use `ollama list` to see
   what you already have.

3. Optional: add a shell alias in `~/.zshrc` or `~/.bashrc`:

   ```bash
   alias lclaude='/full/path/to/lclaude.py'
   ```

## Usage

**Default model (`qwen3.6`):**

```bash
lclaude
```

**Choose a model:**

```bash
lclaude --model gemma4
lclaude --model qwen3:latest
```

**Pass arguments through to `claude`:**

```bash
lclaude --system "You are a helpful assistant" --message "Hello"
```

**Help:**

```bash
lclaude --help
```

On startup, `lclaude` prints a status box (Ollama version, selected model, installed models) and sets the terminal tab
title to `lclaude · <model> · Ollama v…`.

Since Claude Code v2.1.89 defaults to a fullscreen alternate-screen TUI that would wipe anything printed above it,
`lclaude` automatically sets `CLAUDE_CODE_DISABLE_ALTERNATE_SCREEN=1` to keep Claude on the main screen buffer. This
preserves the header in scrollback and keeps native text selection, copy/paste, and `Cmd+F` working.

> [!TIP]
> **Tip:** The first time you run `lclaude` it may take some time to load the model into memory. Follow-on calls get
> increasingly faster.

## How it works

1. Checks that `ollama` is on `PATH` and the server responds on port `11434` (starts `ollama serve` in the background if
   needed).
2. Runs `ollama list` and exits with a clear error if `--model` is not installed.
3. Backs up `~/.claude/settings.json` to `settings.json.off`, sets `CLAUDE_CODE_ATTRIBUTION_HEADER=0`, then launches:

   ```text
   claude --model <model> [your args...]
   ```

   with:

   - `ANTHROPIC_AUTH_TOKEN=ollama`
   - `ANTHROPIC_BASE_URL=http://localhost:11434`
   - `ANTHROPIC_API_KEY` removed from the environment (so traffic does not go to the cloud)
   - `CLAUDE_CODE_DISABLE_ALTERNATE_SCREEN=1` to disable the fullscreen TUI (keeps the header visible)

4. On exit (normal, interrupt, or `SIGTERM`/`SIGHUP`), restores the original settings file and removes the backup.

You do not need to set Anthropic env vars yourself before running `lclaude`.

## Troubleshooting

| Issue                                 | What to try                                                |
| ------------------------------------- | ---------------------------------------------------------- |
| `ollama not found in PATH`            | Install Ollama and ensure `ollama` is on your `PATH`.      |
| `ollama is installed but not running` | Run `ollama serve`, or let `lclaude` auto-start it.        |
| `model '…' not found in ollama`       | `ollama pull <model>` or pick a model from `ollama list`.  |
| `claude` not found                    | Install Claude Code and ensure `claude` is on your `PATH`. |

## License

MIT — Copyright (c) 2026 Todd Papaioannou
