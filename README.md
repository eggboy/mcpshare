# mcpshare

CLI tool to share MCP configuration between coding agents.

Synchronizes [Model Context Protocol](https://modelcontextprotocol.io/) server
definitions across **VSCode**, **GitHub Copilot CLI**, **Claude Code**,
**OpenAI Codex**, **Google Gemini CLI**, and **OpenCode**.

## Quick start

### With `uv` (recommended)

```bash
# Run directly with PEP 723 inline metadata
uv run mcpshare.py init   # create default config
uv run mcpshare.py sync   # synchronize MCP settings
uv run mcpshare.py status # show current state
```

### With pip

```bash
pip install .
mcpshare init
mcpshare sync
mcpshare status
```

## Configuration

The config file lives at `~/.config/mcpshare/config.yaml`:

```yaml
source: /path/to/master          # directory for the canonical mcp.json
mode: merge                      # merge | overwrite
targets:
  claude:
    path: ~/.claude
  vscode:
    path: ~/Library/Application Support/Code/User  # macOS; see docs for other OS
  copilot:
    path: ~/.copilot
  codex:
    path: ~/.codex
  gemini:
    path: ~/.gemini
  opencode:
    path: ~/.config/opencode
```

### Modes

| Mode | Behaviour |
|------|-----------|
| `merge` | Collect servers from every target first, merge into master, then distribute |
| `overwrite` | Use the master as the single source of truth and overwrite all targets |

## How it works

1. **`mcpshare init`** – creates the config file and master directory.
2. **`mcpshare sync`** – in *merge* mode, reads MCP server entries from every
   configured target, merges them into the master `mcp.json`, then converts and
   writes the result back to each target in its native format.
3. **`mcpshare status`** – shows the master servers and whether each target
   config file exists.

### Supported formats

| Tool | File | Top-level key | Notes |
|------|------|---------------|-------|
| Claude Code | `.mcp.json` | `mcpServers` | |
| VSCode | `mcp.json` | `servers` | Adds `"type": "stdio"` |
| Copilot CLI | `mcp-config.json` | `mcpServers` | Adds `"type": "stdio"` |
| Codex | `config.toml` | `[mcp_servers.*]` | TOML format |
| Gemini CLI | `settings.json` | `mcpServers` | Preserves non-MCP settings |
| OpenCode | `opencode.json` | `mcp` | Uses `command` array, `environment` |

## Development

```bash
pip install -e .
pip install pytest
python -m pytest tests/ -v
```

