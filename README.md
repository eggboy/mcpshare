# mcpshare

[![Claude Code](https://img.shields.io/badge/Claude_Code-supported-blueviolet.svg)](#supported-formats)
[![Codex](https://img.shields.io/badge/Codex-supported-blueviolet.svg)](#supported-formats)
[![GitHub Copilot](https://img.shields.io/badge/GitHub_Copilot-supported-blueviolet.svg)](#supported-formats)
[![Gemini](https://img.shields.io/badge/Gemini-supported-blueviolet.svg)](#supported-formats)
[![opencode](https://img.shields.io/badge/opencode-supported-blueviolet.svg)](#supported-formats)
[![VSCode](https://img.shields.io/badge/VSCode-supported-blueviolet.svg)](#supported-formats)

CLI tool to share MCP configuration between coding agents.

Synchronizes [Model Context Protocol](https://modelcontextprotocol.io/) server
definitions across **VSCode**, **GitHub Copilot CLI**, **Claude Code**,
**OpenAI Codex**, **Google Gemini CLI**, and **OpenCode**.

## Quick start

### With `uv`

```bash
# Run directly with PEP 723 inline metadata
uv run mcpshare.py init   # create default config
uv run mcpshare.py pull   # pull servers from targets into master
uv run mcpshare.py sync   # distribute master to all targets
uv run mcpshare.py status # show current state
```

### With `uv tool install`

```bash
# Install as a global CLI tool from GitHub
uv tool install git+https://github.com/eggboy/mcpshare.git

# Or install from a local clone of the repo
git clone https://github.com/eggboy/mcpshare.git
cd mcpshare
uv tool install .

# Then run from anywhere
mcpshare init
mcpshare pull
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
2. **`mcpshare pull`** – pulls MCP server entries from every configured
   target and merges them into the master `mcp.json` (collect only, does not
   write back to targets).
3. **`mcpshare sync`** – distributes the master `mcp.json` to all configured
   targets, converting to each tool's native format.
4. **`mcpshare status`** – shows the master servers and whether each target
   config file exists.

Typical workflow: `mcpshare pull` → `mcpshare sync`.

### Disabling servers

Disable state is tracked exclusively in `mcp.disabled.yaml`, colocated with
the master `mcp.json` in your `source` directory. The master itself never
carries a per-entry `disabled` flag — if you add one by hand, mcpshare
strips it on the next `pull` / `sync`. The same applies to any IDE-native
disable flag (Codex `enabled = false`, etc.) found in target files on read:
it is stripped, and disable state is re-derived from `mcp.disabled.yaml`
on the next `sync`.

This means the disabled list survives even when you delete and regenerate
`mcp.json`, and there is exactly one place to look — `mcp.disabled.yaml`.

```bash
mcpshare disable --servers <server>                     # disable in all IDEs
mcpshare disable --servers <server> --ide claude        # disable in one IDE only
mcpshare enable --servers <server>                      # re-enable in every IDE
mcpshare enable --servers <server> --ide claude         # re-enable in one IDE only
mcpshare status                                         # shows disabled lists
```

`mcp.disabled.yaml` format (top-level flat mapping, colocated with `mcp.json`):

```yaml
"*":                 # entries here are disabled in every configured IDE
  - workiq-mcp
claude:              # entries here are disabled only for Claude Code
  - filesystem
codex:
  - filesystem
```

> If you previously used `~/.config/mcpshare/disabled.yaml`, mcpshare moves
> it next to your master `mcp.json` automatically on the first load.

### How disable is applied per IDE

`sync` writes every master server to every configured IDE. For disabled
servers, mcpshare uses each IDE's **native** disable mechanism rather than
omitting the entry (so the server stays visible and can be re-enabled in the
tool's UI without round-tripping through `mcpshare enable`).

| IDE | Native disable mechanism | What `sync` writes |
|-----|--------------------------|--------------------|
| Codex | `enabled = false` in the server's TOML block | Server written with `enabled = false` |
| Claude Code | `disabledMcpServers: [...]` in `~/.claude/settings.json` | Server in `mcp_servers.json` **plus** name in `settings.json` array |
| Copilot CLI | `disabledMcpServers: [...]` in `~/.copilot/settings.json` | Server in `mcp-config.json` **plus** name in `settings.json` array |
| Gemini CLI | `mcp.excluded: [...]` in `~/.gemini/settings.json` (security-grade blocklist) | Server in `mcpServers` **plus** name in `mcp.excluded` (same file) |
| OpenCode | `"enabled": false` per entry in `opencode.json` | Server written with `"enabled": false` |
| VSCode | None (UI-only per profile) | Server is always written enabled. `disable --ide vscode` is recorded but warns and is a no-op on sync. |

**Source of truth: mcpshare-owns.** On each `sync`, mcpshare *replaces*
the `disabledMcpServers` array in Claude/Copilot `settings.json` with its
own view (other settings.json keys are preserved). If you previously ran
`/mcp disable foo` natively inside Claude or Copilot, the next `sync` will
re-enable `foo` and log a warning telling you the exact `mcpshare disable`
command to run if you want to keep it disabled.

The same overwrite-with-mcpshare's-view rule applies to Gemini's
`mcp.excluded` array and to Codex/OpenCode's per-entry `enabled = false` —
all other keys in the target files are preserved.

### Sync semantics

For MCP-dedicated target files (`mcp.json`, `mcp_servers.json`,
`mcp-config.json`), `sync` **deletes the file and recreates it** so removed
servers cannot linger. For files shared with other IDE settings
(`config.toml`, `settings.json`, `opencode.json`), only the MCP section is
replaced — non-MCP content is preserved. For Claude/Copilot, the
`disabledMcpServers` array in `settings.json` is also overwritten in place
while preserving every other key.

### Conflict resolution on pull

When `mcpshare pull` finds the same server name in two IDEs with **different**
commands/args/env, it prompts you to pick one. Identical configs are
silently merged, and any per-entry `disabled` / `enabled` flag found in a
target file is stripped before comparison — so a server that differs only
in disable state never counts as a conflict (its disable state is owned by
`mcp.disabled.yaml`, not by the master or any target file). Pass
`--non-interactive` (or run without a TTY) to fall back to last-write-wins
with a warning.

### Supported formats

| Tool | File | Top-level key | Notes |
|------|------|---------------|-------|
| Claude Code | `mcp_servers.json` | `mcpServers` | MCP-dedicated; recreated on sync. Disable via `settings.json` |
| VSCode | `mcp.json` | `servers` | MCP-dedicated; recreated on sync. Adds `"type": "stdio"`. No native disable |
| Copilot CLI | `mcp-config.json` | `mcpServers` | MCP-dedicated; recreated on sync. Adds `"type": "stdio"`. Disable via `settings.json` |
| Codex | `config.toml` | `[mcp_servers.*]` | TOML; preserves non-MCP sections. Native `enabled = false` |
| Gemini CLI | `settings.json` | `mcpServers` | Preserves non-MCP settings. Disable via `mcp.excluded` (same file) |
| OpenCode | `opencode.json` | `mcp` | Uses `command` array, `environment`. Native `"enabled": false` per entry |

## Development

```bash
pip install -e .
pip install pytest
python -m pytest tests/ -v
```

