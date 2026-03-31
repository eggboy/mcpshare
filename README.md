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
uv run mcpshare.py update # collect servers from targets into master
uv run mcpshare.py sync   # distribute master to all targets
uv run mcpshare.py status # show current state
```

### With `uv tool install`

```bash
# Install as a global CLI tool
uv tool install git+https://github.com/eggboy/mcpshare.git

# Then run from anywhere
mcpshare init
mcpshare update
mcpshare sync
mcpshare status

# Upgrade to the latest version
uv tool upgrade mcpshare
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
2. **`mcpshare update`** – reads MCP server entries from every configured
   target and merges them into the master `mcp.json` (collect only, does not
   write back to targets).
3. **`mcpshare sync`** – distributes the master `mcp.json` to all configured
   targets, converting to each tool's native format.
4. **`mcpshare status`** – shows the master servers and whether each target
   config file exists.

Typical workflow: `mcpshare update` → `mcpshare sync`.

### Disabling servers

You can disable individual MCP servers without removing them from the master:

```bash
mcpshare disable <server>   # mark a server as disabled
mcpshare enable <server>    # re-enable a disabled server
mcpshare status             # disabled servers show [disabled]
```

On `mcpshare sync`, each target handles disabled servers differently:

| Target | Behaviour |
|--------|-----------|
| Copilot CLI | Writes `"disabled": true` — Copilot natively skips the server |
| Claude Code | Writes `"disabled": true` — Claude ignores unknown fields; use `--disallowedTools` or `--strict-mcp-config` externally to skip |
| VSCode | Strips the `disabled` field (VSCode manages enable/disable state in its own UI) |
| Codex, Gemini, OpenCode | Strips the `disabled` field |

### Supported formats

| Tool | File | Top-level key | Notes |
|------|------|---------------|-------|
| Claude Code | `mcp_servers.json` | `mcpServers` | |
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

