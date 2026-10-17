# mcpshare

[![Claude Code](https://img.shields.io/badge/Claude_Code-supported-blueviolet.svg)](#supported-formats)
[![Codex](https://img.shields.io/badge/Codex-supported-blueviolet.svg)](#supported-formats)
[![GitHub Copilot](https://img.shields.io/badge/GitHub_Copilot-supported-blueviolet.svg)](#supported-formats)
[![Gemini](https://img.shields.io/badge/Gemini-supported-blueviolet.svg)](#supported-formats)
[![Antigravity](https://img.shields.io/badge/Antigravity-supported-blueviolet.svg)](#supported-formats)
[![opencode](https://img.shields.io/badge/opencode-supported-blueviolet.svg)](#supported-formats)
[![VSCode](https://img.shields.io/badge/VSCode-supported-blueviolet.svg)](#supported-formats)

CLI tool to share MCP configuration between coding agents.

Synchronizes [Model Context Protocol](https://modelcontextprotocol.io/) server
definitions across **VSCode**, **GitHub Copilot CLI**, **Claude Code**,
**OpenAI Codex**, **Google Gemini CLI**, **Google Antigravity**, and **OpenCode**.

## Quick start

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
  antigravity:
    path: ~/.gemini/config
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
tool's UI without round-tripping through `mcpshare enable`). Claude Code is
the exception: it has no global disable list, so disabled servers are left
out of its config.

| IDE | Native disable mechanism | What `sync` writes |
|-----|--------------------------|--------------------|
| Codex | `enabled = false` in the server's TOML block | Server written with `enabled = false` |
| Claude Code | None global (`/mcp` toggles are stored per project in `~/.claude.json`) | Server **omitted** from `mcpServers` in `~/.claude.json` |
| Copilot CLI | `disabledMcpServers: [...]` in `~/.copilot/settings.json` | Server in `mcp-config.json` **plus** name in `settings.json` array |
| Gemini CLI | `mcp.excluded: [...]` in `~/.gemini/settings.json` (security-grade blocklist) | Server in `mcpServers` **plus** name in `mcp.excluded` (same file) |
| Antigravity | `"disabled": true` per entry in `~/.gemini/config/mcp_config.json` | Server written with `"disabled": true` |
| OpenCode | `"enabled": false` per entry in `opencode.json` | Server written with `"enabled": false` |
| VSCode | None (UI-only per profile) | Server is always written enabled. `disable --ide vscode` is recorded but warns and is a no-op on sync. |

**Source of truth: mcpshare-owns.** On each `sync`, mcpshare *replaces*
the `disabledMcpServers` array in Copilot `settings.json` with its own
view (other settings.json keys are preserved). If you previously ran
`/mcp disable foo` natively inside Copilot, the next `sync` will
re-enable `foo` and log a warning telling you the exact `mcpshare disable`
command to run if you want to keep it disabled.

The same overwrite-with-mcpshare's-view rule applies to Gemini's
`mcp.excluded` array and to Codex/OpenCode's per-entry `enabled = false` —
all other keys in the target files are preserved.

### Sync semantics

For MCP-dedicated target files (`mcp.json`, `mcp-config.json`), `sync`
**deletes the file and recreates it** so removed servers cannot linger.
For files shared with other IDE settings (`.claude.json`, `config.toml`,
`settings.json`, `mcp_config.json`, `opencode.json`), only the MCP section
is replaced — non-MCP content is preserved. For Copilot, the
`disabledMcpServers` array in `settings.json` is also overwritten in place
while preserving every other key.

#### Claude Code

Claude Code reads user-scope MCP servers only from the top-level
`mcpServers` object in `~/.claude.json` (the same place
`claude mcp add --scope user` writes). The configured target stays
`~/.claude`, Claude's config dir; mcpshare resolves it the way Claude does,
so `~/.claude` maps to `~/.claude.json` and any other directory (as with
`CLAUDE_CONFIG_DIR`) maps to `<dir>/.claude.json`.

- Only `mcpServers` is replaced, atomically; the rest of `~/.claude.json`
  is Claude's own state and is left alone.
- Each entry gets `"type": "stdio"` or `"type": "http"`. Claude rejects a
  remote server without a type.
- Codex-only `tools` allow-lists are not written.
- Restart running Claude Code sessions after `sync` to pick up the change.

Versions before 0.17.1 wrote `~/.claude/mcp_servers.json` and a
`disabledMcpServers` array in `~/.claude/settings.json`. Claude reads
neither, so `sync` now deletes that file and removes that one key (other
`settings.json` keys are kept).

#### Antigravity

The Antigravity app, IDE, and CLI (`agy`) share one global file,
`~/.gemini/config/mcp_config.json`, so the target path is
`~/.gemini/config`. Older locations (`~/.gemini/antigravity/` and
`~/.gemini/antigravity-cli/`) are not written.

- Remote servers are written with `serverUrl`, the only key Antigravity
  accepts (`url` fails only when a tool is first called).
- No `type` field is written; Codex-only `tools` allow-lists are dropped.
- Disabled servers stay listed with `"disabled": true`, so they can be
  toggled from Antigravity's UI.
- After `sync`, click **Refresh** under Installed MCP Servers in the app,
  or reload from `/mcp` in `agy`.

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
| Claude Code | `~/.claude.json` | `mcpServers` | Shared with Claude's state; only `mcpServers` is replaced. Adds `"type"`. Disabled servers omitted |
| VSCode | `mcp.json` | `servers` | MCP-dedicated; recreated on sync. Adds `"type": "stdio"`. No native disable |
| Copilot CLI | `mcp-config.json` | `mcpServers` | MCP-dedicated; recreated on sync. Adds `"type": "stdio"`. Disable via `settings.json` |
| Codex | `config.toml` | `[mcp_servers.*]` | TOML; preserves non-MCP sections. Native `enabled = false` |
| Gemini CLI | `settings.json` | `mcpServers` | Preserves non-MCP settings. Disable via `mcp.excluded` (same file) |
| Antigravity | `~/.gemini/config/mcp_config.json` | `mcpServers` | `url` written as `serverUrl`; no `type`; Codex `tools` not written. Native `"disabled": true` per entry |
| OpenCode | `opencode.json` | `mcp` | Uses `command` array, `environment`. Native `"enabled": false` per entry |

## Development

```bash
pip install -e .
pip install pytest
python -m pytest tests/ -v
```

