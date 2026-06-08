#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pyyaml>=6.0",
#     "tyro>=0.9",
# ]
# ///
"""mcpshare - Synchronize MCP configurations between coding agents.

Supports VSCode, Claude Code, GitHub Copilot CLI, OpenAI Codex,
Google Gemini CLI, and OpenCode.
"""

import importlib.metadata
import json
import logging
import os
import re
import shutil
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

try:
    __version__ = importlib.metadata.version("mcpshare")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.9.0"

import tyro
import yaml

logger = logging.getLogger(__name__)


class McpShareError(Exception):
    """Base exception for mcpshare errors."""


class ConfigNotFoundError(McpShareError, FileNotFoundError):
    """Raised when the mcpshare config file does not exist."""


class ConfigParseError(McpShareError, ValueError):
    """Raised when a configuration file cannot be parsed."""


CONFIG_DIR = Path.home() / ".config" / "mcpshare"
CONFIG_FILE = CONFIG_DIR / "config.yaml"
MASTER_FILENAME = "mcp.json"
MASTER_BACKUP_FILENAME = "mcp.json.bak"
DISABLED_FILENAME = "mcp.disabled.yaml"

# Sentinel key inside the disabled file for "all configured IDEs".
DISABLED_ALL = "*"

# Tools whose target file holds nothing but MCP servers. For these we
# delete-and-recreate during sync so stale entries can never linger.
# Tools NOT in this set share the file with unrelated IDE settings, so
# we only overwrite the MCP section within the file.
MCP_DEDICATED_TOOLS = frozenset({"claude", "vscode", "copilot"})

# Filename of the native settings file (relative to the target directory)
# used by Claude Code and Copilot CLI for their ``disabledMcpServers`` array.
NATIVE_SETTINGS_FILENAME = "settings.json"
NATIVE_DISABLED_KEY = "disabledMcpServers"

# Human-readable description of each tool's native disable mechanism,
# shown by ``mcpshare status``. The actual mechanism lives inside each
# writer; this dict is purely cosmetic.
DISABLE_DESCRIPTION = {
    "claude": "settings.json/disabledMcpServers",
    "copilot": "settings.json/disabledMcpServers",
    "codex": "enabled = false in TOML",
    "gemini": "settings.json/mcp.excluded",
    "opencode": '"enabled": false in opencode.json',
    "vscode": "UI only (write everything)",
}

_KEBAB_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


def _sanitize_server_id(name: str) -> str:
    """Normalize a server name to kebab-case.

    Lowercases the name and replaces spaces, underscores, and consecutive
    hyphens with a single hyphen.  Leading/trailing hyphens are stripped.
    """
    name = name.lower()
    name = re.sub(r"[\s_]+", "-", name)
    name = re.sub(r"-{2,}", "-", name)
    return name.strip("-")


def _validate_server_id(name: str) -> list[str]:
    """Return warnings for server IDs that don't follow naming conventions.

    Best practice: kebab-case (lowercase, hyphen-separated) containing ``mcp``.
    """
    warnings: list[str] = []
    if not _KEBAB_RE.match(name) and "." not in name:
        warnings.append(f"Server ID '{name}' is not kebab-case. Consider renaming to '{_sanitize_server_id(name)}'.")
    if "mcp" not in name.lower():
        warnings.append(
            f"Server ID '{name}' does not contain 'mcp'. "
            f"Convention: use kebab-case with 'mcp', e.g. '{name}-mcp' or 'mcp-{name}'."
        )
    return warnings


def _vscode_config_dir() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Code" / "User"
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")
        return Path(appdata) / "Code" / "User"
    # Linux and other Unix-like systems
    return Path.home() / ".config" / "Code" / "User"


# Default config locations for each tool (user-level)
DEFAULT_CONFIG_PATHS = {
    "claude": Path.home() / ".claude" / "mcp_servers.json",
    "vscode": _vscode_config_dir() / "mcp.json",
    "copilot": Path.home() / ".copilot" / "mcp-config.json",
    "codex": Path.home() / ".codex" / "config.toml",
    "gemini": Path.home() / ".gemini" / "settings.json",
    "opencode": Path.home() / ".config" / "opencode" / "opencode.json",
}

# Output filenames when writing to target directories
TARGET_FILENAMES = {
    "claude": "mcp_servers.json",
    "vscode": "mcp.json",
    "copilot": "mcp-config.json",
    "codex": "config.toml",
    "gemini": "settings.json",
    "opencode": "opencode.json",
}


def default_config() -> dict[str, Any]:
    """Return a default configuration."""
    source = str(CONFIG_DIR / "master")
    return {
        "source": source,
        "mode": "merge",
        "targets": {
            "claude": {"path": str(DEFAULT_CONFIG_PATHS["claude"].parent)},
            "vscode": {"path": str(DEFAULT_CONFIG_PATHS["vscode"].parent)},
            "copilot": {"path": str(DEFAULT_CONFIG_PATHS["copilot"].parent)},
            "codex": {"path": str(DEFAULT_CONFIG_PATHS["codex"].parent)},
            "gemini": {"path": str(DEFAULT_CONFIG_PATHS["gemini"].parent)},
            "opencode": {
                "path": str(DEFAULT_CONFIG_PATHS["opencode"].parent),
            },
        },
    }


def load_config() -> dict[str, Any]:
    """Load configuration from the config file.

    Raises:
        ConfigNotFoundError: If the config file does not exist.
    """
    if not CONFIG_FILE.exists():
        raise ConfigNotFoundError(f"Config not found: {CONFIG_FILE}. Run 'mcpshare init' to create one.")
    with open(CONFIG_FILE) as f:
        return yaml.safe_load(f)


def save_config(config: dict[str, Any]) -> None:
    """Save configuration to the config file."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Disabled servers (persistent, per-IDE)
# ---------------------------------------------------------------------------


def _legacy_disabled_path() -> Path:
    """Return the historical location of ``disabled.yaml`` (pre-redesign)."""
    return CONFIG_DIR / "disabled.yaml"


def disabled_file_path(source_dir: str) -> Path:
    """Return the path to the disabled-servers file colocated with the master."""
    return Path(source_dir) / DISABLED_FILENAME


def _migrate_legacy_disabled(new_path: Path) -> None:
    """Move ``~/.config/mcpshare/disabled.yaml`` next to the master, if applicable.

    Runs at most once per source dir: if *new_path* already exists the legacy
    file is left untouched (the user is in charge of resolving any drift).
    """
    legacy = _legacy_disabled_path()
    if not legacy.exists() or new_path.exists():
        return
    new_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(legacy), str(new_path))
    logger.info("Migrated disabled list: %s → %s", legacy, new_path)


def load_disabled(source_dir: str) -> dict[str, list[str]]:
    """Load the per-IDE disabled-servers list colocated with the master.

    Returns a dict keyed by tool name (with ``"*"`` meaning "all IDEs").
    Returns an empty dict if the file doesn't exist.

    Raises:
        ConfigParseError: If the file contains invalid YAML or is malformed.
    """
    path = disabled_file_path(source_dir)
    _migrate_legacy_disabled(path)
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        raise ConfigParseError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigParseError(f"Expected mapping at top of {path}, got {type(data).__name__}")
    result: dict[str, list[str]] = {}
    for key, value in data.items():
        if not isinstance(value, list):
            raise ConfigParseError(f"Expected list of server names under '{key}' in {path}")
        # De-duplicate while preserving order.
        seen: dict[str, None] = {}
        for item in value:
            if not isinstance(item, str):
                raise ConfigParseError(f"Server names must be strings; got {type(item).__name__} under '{key}'")
            seen.setdefault(item, None)
        if seen:
            result[str(key)] = list(seen)
    return result


def save_disabled(source_dir: str, data: dict[str, list[str]]) -> None:
    """Persist the per-IDE disabled-servers list next to the master."""
    path = disabled_file_path(source_dir)
    # Drop empty entries so the file stays tidy.
    clean = {k: sorted(v) for k, v in data.items() if v}
    if not clean:
        if path.exists():
            path.unlink()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(clean, f, default_flow_style=False, sort_keys=True)


def disabled_for_tool(data: dict[str, list[str]], tool: str) -> set[str]:
    """Return the set of servers disabled for *tool* (global ``*`` ∪ tool-specific)."""
    return set(data.get(DISABLED_ALL, [])) | set(data.get(tool, []))


# ---------------------------------------------------------------------------
# Master config (canonical format)
# ---------------------------------------------------------------------------


def load_master(source_dir: str) -> dict[str, Any]:
    """Load the master MCP config from the source directory.

    Raises:
        ConfigParseError: If the master file contains invalid JSON.
    """
    master_path = Path(source_dir) / MASTER_FILENAME
    if not master_path.exists():
        return {"mcpServers": {}}
    try:
        with open(master_path) as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ConfigParseError(f"Invalid JSON in {master_path}: {exc}") from exc
    if "mcpServers" not in data:
        data["mcpServers"] = {}
    # Normalize any multi-word command strings so the executable is separated
    # from its flags/arguments (e.g. "npx -y @azure/mcp@latest" → "npx").
    for cfg in data["mcpServers"].values():
        if "command" in cfg and " " in cfg["command"]:
            parts = cfg["command"].split()
            cfg["command"] = parts[0]
            cfg["args"] = parts[1:] + list(cfg.get("args", []))
    return data


def save_master(source_dir: str, master: dict[str, Any]) -> None:
    """Save the master MCP config to the source directory.

    Server entries are written as-is. Disable state is tracked exclusively in
    ``mcp.disabled.yaml`` (see :func:`load_disabled`); the master never
    carries a per-entry ``disabled`` flag.
    """
    source_path = Path(source_dir)
    source_path.mkdir(parents=True, exist_ok=True)
    master_path = source_path / MASTER_FILENAME
    with open(master_path, "w") as f:
        json.dump(master, f, indent=2)
        f.write("\n")


def read_vscode(path: Path) -> dict[str, Any]:
    """Read MCP servers from VSCode mcp.json format.

    VSCode uses ``{"servers": {"name": {"type": "stdio", ...}}}``.
    Server names are sanitized to kebab-case.

    Raises:
        ConfigParseError: If the file contains invalid JSON.
    """
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ConfigParseError(f"Invalid JSON in {path}: {exc}") from exc
    servers = data.get("servers", {})
    result: dict[str, Any] = {}
    for name, cfg in servers.items():
        safe_name = _sanitize_server_id(name)
        entry = {k: v for k, v in cfg.items() if k not in _READER_DROPPED_KEYS}
        # Normalize command: split multi-word command strings so that only
        # the executable lands in "command" and the rest is prepended to "args".
        if "command" in entry and " " in entry["command"]:
            parts = entry["command"].split()
            entry["command"] = parts[0]
            entry["args"] = parts[1:] + list(entry.get("args", []))
        result[safe_name] = entry
    return result


def read_claude(path: Path) -> dict[str, Any]:
    """Read MCP servers from Claude Code settings.json or .mcp.json.

    Claude uses ``{"mcpServers": {"name": {"command": ...}}}``.

    Raises:
        ConfigParseError: If the file contains invalid JSON.
    """
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ConfigParseError(f"Invalid JSON in {path}: {exc}") from exc
    servers = data.get("mcpServers", {})
    return {name: {k: v for k, v in cfg.items() if k not in _READER_DROPPED_KEYS} for name, cfg in servers.items()}


def read_copilot(path: Path) -> dict[str, Any]:
    """Read MCP servers from Copilot CLI mcp-config.json.

    Copilot CLI uses ``{"mcpServers": {"name": {"type": "stdio", ...}}}``.

    Raises:
        ConfigParseError: If the file contains invalid JSON.
    """
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ConfigParseError(f"Invalid JSON in {path}: {exc}") from exc
    servers = data.get("mcpServers", {})
    result: dict[str, Any] = {}
    for name, cfg in servers.items():
        entry = {k: v for k, v in cfg.items() if k not in _READER_DROPPED_KEYS}
        result[name] = entry
    return result


_READER_DROPPED_KEYS = frozenset({"type", "disabled"})


def _flatten_nested_servers(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts produced by dotted TOML keys into flat server entries.

    When ``tomllib`` parses ``[mcp_servers.microsoft.docs.mcp]``, it creates
    nested dicts: ``{"microsoft": {"docs": {"mcp": {"url": ...}}}}``.
    This function flattens them back to ``{"microsoft.docs.mcp": {"url": ...}}``.
    A dict is considered a server config (leaf) if it contains ``command`` or ``url``.
    """
    result: dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            if any(k in value for k in ("command", "url")):
                result[full_key] = value
            else:
                result.update(_flatten_nested_servers(value, full_key))
    return result


def _parse_toml_mcp_servers(text: str) -> dict[str, Any]:
    """Parse MCP servers from a TOML config string (Codex format)."""
    import tomllib

    data = tomllib.loads(text)
    servers = data.get("mcp_servers", {})
    return _flatten_nested_servers(servers)


def read_codex(path: Path) -> dict[str, Any]:
    """Read MCP servers from Codex config.toml.

    Codex uses TOML with ``[mcp_servers.<name>]`` sections.  Any
    ``enabled`` field is dropped — disable state is tracked exclusively in
    ``mcp.disabled.yaml`` and not propagated through readers.  Codex
    ``tools`` maps are converted back to lists for the canonical format.
    """
    if not path.exists():
        return {}
    text = path.read_text()
    servers = _parse_toml_mcp_servers(text)
    for cfg in servers.values():
        cfg.pop("enabled", None)
        tools = cfg.get("tools")
        if isinstance(tools, dict):
            cfg["tools"] = list(tools.keys())
    return servers


def read_gemini(path: Path) -> dict[str, Any]:
    """Read MCP servers from Gemini settings.json.

    Gemini uses ``{"mcpServers": {"name": {"command": ...}}}``.

    Raises:
        ConfigParseError: If the file contains invalid JSON.
    """
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ConfigParseError(f"Invalid JSON in {path}: {exc}") from exc
    servers = data.get("mcpServers", {})
    return {name: {k: v for k, v in cfg.items() if k not in _READER_DROPPED_KEYS} for name, cfg in servers.items()}


def read_opencode(path: Path) -> dict[str, Any]:
    """Read MCP servers from OpenCode opencode.json.

    OpenCode uses ``{"mcp": {"name": {"type": "local", "command": [...], "environment": {}}}}``.

    Raises:
        ConfigParseError: If the file contains invalid JSON.
    """
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ConfigParseError(f"Invalid JSON in {path}: {exc}") from exc
    mcp = data.get("mcp", {})
    result: dict[str, Any] = {}
    for name, cfg in mcp.items():
        entry: dict = {}
        cmd_list = cfg.get("command", [])
        if isinstance(cmd_list, list) and cmd_list:
            entry["command"] = cmd_list[0]
            if len(cmd_list) > 1:
                entry["args"] = cmd_list[1:]
        env = cfg.get("environment")
        if env is not None:
            entry["env"] = env
        result[name] = entry
    return result


READERS = {
    # "claude": read_claude,
    "vscode": read_vscode,
    "copilot": read_copilot,
    # "codex": read_codex,
    # "gemini": read_gemini,
    # "opencode": read_opencode,
}


def _server_type(cfg: dict[str, Any]) -> str:
    """Return the MCP transport type based on server config."""
    return "http" if "url" in cfg else "stdio"


_VSCODE_ENV_RE = re.compile(r"\$\{env:(\w+)\}")


def _resolve_vscode_vars(value: object) -> object:
    """Recursively resolve VSCode ``${env:VAR}`` placeholders to real env values."""
    if isinstance(value, str):
        return _VSCODE_ENV_RE.sub(lambda m: os.environ.get(m.group(1), m.group(0)), value)
    if isinstance(value, list):
        return [_resolve_vscode_vars(v) for v in value]
    if isinstance(value, dict):
        return {k: _resolve_vscode_vars(v) for k, v in value.items()}
    return value


def write_vscode(path: Path, servers: dict[str, Any], **_: Any) -> tuple[int, list[str]]:
    """Write MCP servers in VSCode format.

    Produces ``{"servers": {"name": {"type": "stdio"|"http", ...}}}``.
    The type is determined by the server configuration: ``"http"`` for
    URL-based servers and ``"stdio"`` for command-based servers.

    Returns a tuple of (server count, list of skipped server descriptions).
    """
    existing = {}
    if path.exists():
        with open(path) as f:
            existing = json.load(f)
    vscode_servers = {}
    for name, cfg in servers.items():
        entry = {"type": _server_type(cfg)}
        entry.update(cfg)
        vscode_servers[name] = entry
    existing["servers"] = vscode_servers
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
        f.write("\n")
    return len(vscode_servers), []


def write_claude(
    path: Path, servers: dict[str, Any], *, disabled_names: Iterable[str] = (), **_: Any
) -> tuple[int, list[str]]:
    """Write MCP servers in Claude Code format.

    Produces ``{"mcpServers": {"name": {"command": ...}}}``.  Claude Code
    silently ignores per-server disable flags in ``mcp_servers.json`` and
    instead reads a top-level ``disabledMcpServers`` array from
    ``settings.json`` (managed via :func:`_emit_native_disabled`, called
    here as the IDE-native disable mechanism).

    Returns a tuple of (server count, list of skipped server descriptions).
    """
    existing = {}
    if path.exists():
        with open(path) as f:
            existing = json.load(f)
    existing["mcpServers"] = dict(servers)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
        f.write("\n")
    _emit_native_disabled(
        "claude",
        path.parent / NATIVE_SETTINGS_FILENAME,
        set(disabled_names),
        set(servers),
    )
    return len(servers), []


def write_copilot(
    path: Path, servers: dict[str, Any], *, disabled_names: Iterable[str] = (), **_: Any
) -> tuple[int, list[str]]:
    """Write MCP servers in Copilot CLI format.

    Produces ``{"mcpServers": {"name": {"type": "stdio"|"http", ...}}}``.
    The type is determined by the server configuration: ``"http"`` for
    URL-based servers and ``"stdio"`` for command-based servers.  Copilot
    CLI silently ignores per-server disable flags in ``mcp-config.json``
    and instead reads a top-level ``disabledMcpServers`` array from
    ``settings.json`` (managed via :func:`_emit_native_disabled`, called
    here as the IDE-native disable mechanism).

    Returns a tuple of (server count, list of skipped server descriptions).
    """
    existing = {}
    if path.exists():
        with open(path) as f:
            existing = json.load(f)
    # Copilot CLI ships with its own GitHub MCP server, so skip any entry
    # pointing at the built-in endpoint (but keep github_spaces and others).
    copilot_servers = {}
    skipped: list[str] = []
    for name, cfg in servers.items():
        if cfg.get("url") == "https://api.githubcopilot.com/mcp":
            skipped.append(f"{name} (built-in)")
            continue
        entry = {"type": _server_type(cfg)}
        entry.update(cfg)
        copilot_servers[name] = entry
    existing["mcpServers"] = copilot_servers
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
        f.write("\n")
    _emit_native_disabled(
        "copilot",
        path.parent / NATIVE_SETTINGS_FILENAME,
        set(disabled_names) & set(copilot_servers),
        set(copilot_servers),
    )
    return len(copilot_servers), skipped


def _format_toml_value(value: object) -> str:
    """Format a Python value as a TOML literal."""
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, list):
        items = ", ".join(_format_toml_value(v) for v in value)
        return f"[{items}]"
    if isinstance(value, dict):
        if not value:
            return "{}"
        pairs = ", ".join(f"{_toml_key(k)} = {_format_toml_value(v)}" for k, v in value.items())
        return f"{{ {pairs} }}"
    return str(value)


_BARE_KEY_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _toml_key(name: str) -> str:
    """Return a TOML-safe key, quoting if not a valid bare key."""
    if _BARE_KEY_RE.match(name):
        return name
    return f'"{name}"'


def write_codex(
    path: Path,
    servers: dict[str, Any],
    *,
    disabled_names: Iterable[str] = (),
    **_: Any,
) -> tuple[int, list[str]]:
    """Write MCP servers in Codex TOML format.

    Produces ``[mcp_servers.<name>]`` sections.  Server names containing
    dots or spaces are quoted to prevent TOML from treating them as nested
    tables.  Non-MCP content in an existing file is preserved.

    Names listed in *disabled_names* receive ``enabled = false`` in their
    section (Codex's native disable mechanism).

    Returns a tuple of (server count, list of skipped server descriptions).
    """
    disabled = set(disabled_names)
    existing_lines: list[str] = []
    if path.exists():
        in_mcp = False
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("[mcp_servers.") or stripped == "[mcp_servers]":
                in_mcp = True
                continue
            if in_mcp and stripped.startswith("[") and not stripped.startswith("[mcp_servers"):
                in_mcp = False
            if not in_mcp:
                existing_lines.append(line)

    while existing_lines and not existing_lines[-1].strip():
        existing_lines.pop()

    parts = existing_lines
    for name, cfg in servers.items():
        parts.append("")
        parts.append(f"[mcp_servers.{_toml_key(name)}]")
        if name in disabled:
            parts.append("enabled = false")
        for key, val in cfg.items():
            # Codex expects 'tools' as a map of {name: McpServerToolConfig}
            if key == "tools" and isinstance(val, list):
                val = {item: {"enabled": True} for item in val}
            parts.append(f"{key} = {_format_toml_value(val)}")
    parts.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts))
    return len(servers), []


def write_gemini(
    path: Path, servers: dict[str, Any], *, disabled_names: Iterable[str] = (), **_: Any
) -> tuple[int, list[str]]:
    """Write MCP servers in Gemini CLI format.

    Produces ``{"mcpServers": {"name": {"command": ...}}}``.  Disabled
    servers stay in ``mcpServers`` but are listed in ``mcp.excluded``
    (Gemini's native security-grade blocklist) under the top-level ``mcp``
    object — both live in the same ``settings.json``.  Other ``mcp.*``
    keys (e.g. ``allowed``, ``serverCommand``) and unrelated top-level
    keys are preserved.

    Returns a tuple of (server count, list of skipped server descriptions).
    """
    existing = {}
    if path.exists():
        with open(path) as f:
            existing = json.load(f)
    existing["mcpServers"] = dict(servers)

    mcp_section = existing.get("mcp")
    if not isinstance(mcp_section, dict):
        mcp_section = {}
    disabled_sorted = sorted(set(disabled_names) & set(servers))
    if disabled_sorted:
        mcp_section["excluded"] = disabled_sorted
    else:
        mcp_section.pop("excluded", None)
    if mcp_section:
        existing["mcp"] = mcp_section
    else:
        existing.pop("mcp", None)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
        f.write("\n")
    return len(servers), []


def write_opencode(
    path: Path, servers: dict[str, Any], *, disabled_names: Iterable[str] = (), **_: Any
) -> tuple[int, list[str]]:
    """Write MCP servers in OpenCode format.

    For stdio servers, converts ``command``/``args`` to a single ``command``
    list, ``env`` to ``environment``, and adds ``"type": "local"``.
    For HTTP/SSE servers (those with ``url``), uses ``"type": "remote"``
    with a ``url`` field.  Names in *disabled_names* receive
    ``"enabled": false`` (OpenCode's native per-entry disable flag).
    Preserves existing non-MCP settings.

    Returns a tuple of (server count, list of skipped server descriptions).
    """
    disabled = set(disabled_names)
    existing = {}
    if path.exists():
        with open(path) as f:
            existing = json.load(f)
    oc_servers = {}
    for name, cfg in servers.items():
        if "url" in cfg:
            entry: dict = {"type": "remote", "url": cfg["url"]}
            headers = cfg.get("headers")
            if headers is not None:
                entry["headers"] = headers
        else:
            entry = {"type": "local"}
            cmd = cfg.get("command", "")
            args = cfg.get("args", [])
            entry["command"] = [cmd] + list(args) if cmd else list(args)
        env = cfg.get("env")
        if env is not None:
            entry["environment"] = env
        if name in disabled:
            entry["enabled"] = False
        oc_servers[name] = entry
    existing["mcp"] = oc_servers
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
        f.write("\n")
    return len(oc_servers), []


WRITERS = {
    "claude": write_claude,
    "vscode": write_vscode,
    "copilot": write_copilot,
    "codex": write_codex,
    "gemini": write_gemini,
    "opencode": write_opencode,
}


def _write_native_disabled_list(settings_path: Path, disabled_names: set[str]) -> set[str]:
    """Set the top-level ``disabledMcpServers`` array in *settings_path*.

    Preserves every other key in the file.  When *disabled_names* is empty
    the key is removed entirely.  Returns the set of names that were
    previously listed (so the caller can warn about anything that mcpshare
    is about to silently re-enable).

    The write is atomic: data is written to a sibling temp file and then
    renamed into place.
    """
    data: dict[str, Any] = {}
    if settings_path.exists():
        with open(settings_path) as f:
            data = json.load(f) or {}
    previous_raw = data.get(NATIVE_DISABLED_KEY, [])
    previous = {str(item) for item in previous_raw} if isinstance(previous_raw, list) else set()
    sorted_names = sorted(disabled_names)
    if sorted_names:
        data[NATIVE_DISABLED_KEY] = sorted_names
    else:
        data.pop(NATIVE_DISABLED_KEY, None)
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = settings_path.with_name(settings_path.name + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    tmp_path.replace(settings_path)
    return previous


def _emit_native_disabled(
    tool: str,
    settings_path: Path,
    disabled_to_emit: set[str],
    known_servers: set[str],
) -> None:
    """Write the native disabled list for *tool* and log drift warnings.

    Helper for :func:`write_claude` and :func:`write_copilot`, whose IDEs
    track disabled servers in ``settings.json`` → ``disabledMcpServers``.
    Warns when mcpshare is about to silently re-enable a server that the
    user had disabled natively in the IDE (mcpshare-owns: it wins, but the
    user gets a clear pointer to the command that would preserve the
    previous state).
    """
    try:
        previous = _write_native_disabled_list(settings_path, disabled_to_emit)
    except json.JSONDecodeError as exc:
        logger.error(
            "Cannot update %s (invalid JSON: %s). Leaving file untouched.",
            settings_path,
            exc,
        )
        return
    re_enabled = (previous & known_servers) - disabled_to_emit
    if re_enabled:
        logger.warning(
            "\033[33m  ↳ %s: re-enabling %s (was in %s; not tracked by mcpshare).\n"
            "      Run 'mcpshare disable --ide %s --servers %s' to keep them disabled.\033[0m",
            tool,
            ", ".join(sorted(re_enabled)),
            NATIVE_DISABLED_KEY,
            tool,
            " ".join(sorted(re_enabled)),
        )


def resolve_target_path(tool: str, target_dir: str) -> Path:
    """Return the full file path for a tool's config inside *target_dir*."""
    return Path(target_dir) / TARGET_FILENAMES[tool]


def _configs_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Compare two server configs for equality."""
    return a == b


def _prompt_conflict(
    name: str,
    current_source: str,
    current_cfg: dict[str, Any],
    new_source: str,
    new_cfg: dict[str, Any],
) -> bool:
    """Ask the user which version of a conflicting server to keep.

    Returns True to use *new_cfg*, False to keep *current_cfg*.
    """
    print(f"\n\033[33m⚠ Conflict for server '{name}':\033[0m", file=sys.stderr)
    print(f"  [1] keep current (from {current_source}):", file=sys.stderr)
    for line in json.dumps(current_cfg, indent=2).splitlines():
        print(f"      {line}", file=sys.stderr)
    print(f"  [2] use new (from {new_source}):", file=sys.stderr)
    for line in json.dumps(new_cfg, indent=2).splitlines():
        print(f"      {line}", file=sys.stderr)
    while True:
        answer = input("Choose [1/2] (default 1): ").strip()
        if answer in ("", "1"):
            return False
        if answer == "2":
            return True
        print("Please enter 1 or 2.", file=sys.stderr)


def collect(config: dict[str, Any], *, interactive: bool = True) -> dict[str, Any]:
    """Read MCP servers from all configured targets and merge into master.

    Disable state lives exclusively in ``mcp.disabled.yaml`` (see
    :func:`load_disabled`); readers strip any legacy per-entry ``disabled``
    field so it never enters the master.

    When two sources define the same server name with different commands or
    arguments, the user is prompted to pick one (when *interactive* is True
    and stdin is a TTY). Otherwise the later source silently wins, matching
    legacy behaviour, but a warning is logged.

    Returns the merged master dict.
    """
    source_dir = config["source"]
    master = load_master(source_dir)
    # Defensive: strip any per-entry 'disabled' that might exist on hand-edited
    # or legacy master files. mcp.disabled.yaml is the only source of truth.
    raw_servers = master.get("mcpServers", {})
    servers = {
        name: {k: v for k, v in cfg.items() if k not in _READER_DROPPED_KEYS} for name, cfg in raw_servers.items()
    }
    sources: dict[str, str] = {name: "master" for name in servers}

    can_prompt = interactive and sys.stdin.isatty()

    targets = config.get("targets", {})
    for tool, tcfg in targets.items():
        if tool not in READERS:
            logger.warning("Skipping unknown tool: %s", tool)
            continue
        target_path = resolve_target_path(tool, tcfg["path"])
        tool_servers = READERS[tool](target_path)
        if not tool_servers:
            continue
        logger.info("Collected %d server(s) from %s", len(tool_servers), tool)
        for name, new_cfg in tool_servers.items():
            if name in servers and not _configs_equal(servers[name], new_cfg):
                current_source = sources.get(name, "?")
                if can_prompt:
                    use_new = _prompt_conflict(name, current_source, servers[name], tool, new_cfg)
                else:
                    use_new = True
                    logger.warning(
                        "Conflict for '%s': %s version overwritten by %s (non-interactive)",
                        name,
                        current_source,
                        tool,
                    )
                if use_new:
                    servers[name] = new_cfg
                    sources[name] = tool
            else:
                servers[name] = new_cfg
                sources[name] = tool

    # Warn about server IDs that don't follow naming conventions.
    for name in sorted(servers):
        for warning in _validate_server_id(name):
            logger.warning("\033[33m  ⚠ %s\033[0m", warning)

    master["mcpServers"] = servers
    return master


def distribute(config: dict[str, Any], master: dict[str, Any]) -> None:
    """Write the master MCP config to all configured targets.

    Every master server is written to every configured tool.  Disable
    state comes exclusively from ``mcp.disabled.yaml`` (see
    :func:`load_disabled`) and is forwarded to each writer via the
    ``disabled_names`` kwarg.  Each writer is responsible for applying it
    using whatever native mechanism its IDE supports:

    * **Claude Code / Copilot CLI** — server written to ``mcp_servers.json``
      / ``mcp-config.json`` as normal; disabled names recorded in
      ``settings.json`` → ``disabledMcpServers`` (mcpshare-owns: the
      array is overwritten to exactly match mcpshare's view).
    * **Codex** — server written with ``enabled = false`` inside its
      ``[mcp_servers.<name>]`` TOML block.
    * **Gemini CLI** — server written to ``mcpServers``; disabled names
      added to ``mcp.excluded`` in the same ``settings.json``.
    * **OpenCode** — server written to ``mcp`` with ``"enabled": false``.
    * **VSCode** — ``disabled_names`` is ignored; VSCode has no JSON-level
      disable mechanism, so every server is written enabled.

    For MCP-dedicated target files (claude, vscode, copilot) the existing
    file is deleted before the writer runs so stale entries can't linger.
    Shared-purpose target files (codex, gemini, opencode) are left in
    place and only their MCP section is overwritten.
    """
    servers = master.get("mcpServers", {})
    disabled_data = load_disabled(config["source"])
    known_servers = set(servers)

    for tool, tcfg in config.get("targets", {}).items():
        if tool not in WRITERS:
            logger.warning("Skipping unknown tool: %s", tool)
            continue
        disabled_set = disabled_for_tool(disabled_data, tool) & known_servers

        target_path = resolve_target_path(tool, tcfg["path"])
        # Wipe MCP-dedicated files so removed servers don't linger.
        if tool in MCP_DEDICATED_TOOLS and target_path.exists():
            target_path.unlink()

        # VSCode handles ${env:VAR} natively; resolve for all other targets.
        resolved = servers if tool == "vscode" else _resolve_vscode_vars(servers)
        count, skipped = WRITERS[tool](target_path, resolved, disabled_names=disabled_set)
        logger.info("Wrote %d server(s) to %s (%s)", count, tool, target_path)
        for name in sorted(disabled_set):
            logger.info("\033[33m  ↳ disabled: %s\033[0m", name)
        for reason in skipped:
            logger.info("\033[31m  ↳ skipped %s\033[0m", reason)


# ---------------------------------------------------------------------------
# CLI subcommand dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Init:
    """Create default config."""

    force: bool = False
    """Overwrite existing config."""
    source: str | None = None
    """Source folder for master MCP config."""


@dataclass
class Pull:
    """Pull MCP server configs from all targets and merge into master."""

    non_interactive: bool = False
    """Skip interactive conflict resolution; the later target silently wins."""


@dataclass
class Sync:
    """Distribute master MCP config to all targets."""


@dataclass
class Status:
    """Show current config status."""


@dataclass
class Disable:
    """Disable one or more MCP servers (persistent, per-IDE)."""

    servers: tuple[str, ...]
    """Names of the servers to disable."""
    ide: str | None = None
    """Restrict disable to a single IDE (e.g. 'vscode'). Default: all IDEs."""


@dataclass
class Enable:
    """Enable one or more previously disabled MCP servers."""

    servers: tuple[str, ...]
    """Names of the servers to enable."""
    ide: str | None = None
    """Restrict enable to a single IDE. Default: clear from all IDEs."""


@dataclass
class Version:
    """Show mcpshare version."""


Command = (
    Annotated[Init, tyro.conf.subcommand("init")]
    | Annotated[Pull, tyro.conf.subcommand("pull")]
    | Annotated[Sync, tyro.conf.subcommand("sync")]
    | Annotated[Status, tyro.conf.subcommand("status")]
    | Annotated[Disable, tyro.conf.subcommand("disable")]
    | Annotated[Enable, tyro.conf.subcommand("enable")]
    | Annotated[Version, tyro.conf.subcommand("version")]
)


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


def cmd_init(args: Init) -> None:
    """Initialise the mcpshare config file."""
    if CONFIG_FILE.exists() and not args.force:
        print(f"Config already exists: {CONFIG_FILE}")
        print("Use --force to overwrite.")
        return
    config = default_config()
    default_source = config["source"]
    if args.source:
        source = args.source
    else:
        answer = input(f"Source folder [{default_source}]: ").strip()
        source = answer if answer else default_source
    config["source"] = source
    save_config(config)
    # Also create the master directory
    Path(config["source"]).mkdir(parents=True, exist_ok=True)
    print(f"Config created: {CONFIG_FILE}")
    print(f"Master directory: {config['source']}")


def cmd_pull(args: Pull) -> None:
    """Pull MCP server configs from all targets and merge into master."""
    config = load_config()
    # Back up the current master before overwriting.
    master_path = Path(config["source"]) / MASTER_FILENAME
    if master_path.exists():
        backup_path = Path(config["source"]) / MASTER_BACKUP_FILENAME
        shutil.copy2(master_path, backup_path)
        logger.info("Backed up master to %s", backup_path)
    logger.info("Collecting MCP servers from targets...")
    master = collect(config, interactive=not args.non_interactive)
    save_master(config["source"], master)
    total = len(master.get("mcpServers", {}))
    logger.info("Master updated with %d server(s).", total)


def cmd_sync(args: Sync) -> None:
    """Distribute master MCP config to all targets."""
    config = load_config()
    master = load_master(config["source"])
    logger.info("Distributing to targets...")
    distribute(config, master)
    logger.info("Sync complete.")


def cmd_status(args: Status) -> None:
    """Show current configuration status."""
    config = load_config()
    source_dir = config["source"]
    master = load_master(source_dir)
    servers = master.get("mcpServers", {})
    disabled_data = load_disabled(source_dir)

    # Status output is user-facing, so print() is intentional here.
    print(f"Config: {CONFIG_FILE}")
    print(f"Source: {source_dir}")
    print(f"Mode:   {config.get('mode', 'merge')}")
    print(f"Master servers ({len(servers)}):")
    global_disabled = set(disabled_data.get(DISABLED_ALL, []))
    for name in sorted(servers):
        cfg = servers[name]
        transport = cfg["url"] if "url" in cfg else cfg.get("command", "?")
        markers = []
        if name in global_disabled:
            markers.append("disabled")
        per_ide = sorted(tool for tool, names in disabled_data.items() if tool != DISABLED_ALL and name in names)
        if per_ide:
            markers.append("disabled in " + ",".join(per_ide))
        suffix = f" [{'; '.join(markers)}]" if markers else ""
        print(f"  - {name}: {transport}{suffix}")

    print("\nTargets:")
    targets = config.get("targets", {})
    for tool, tcfg in sorted(targets.items()):
        target_path = resolve_target_path(tool, tcfg["path"])
        exists = "\u2713" if target_path.exists() else "\u2717"
        strategy = DISABLE_DESCRIPTION.get(tool, "?")
        print(f"  {exists} {tool}: {target_path}  [disable: {strategy}]")

    if disabled_data:
        print(f"\nDisabled ({disabled_file_path(source_dir)}):")
        for key in sorted(disabled_data):
            label = "all IDEs" if key == DISABLED_ALL else key
            print(f"  [{label}]")
            for name in sorted(disabled_data[key]):
                print(f"    - {name}")


def cmd_disable(args: Disable) -> None:
    """Disable one or more MCP servers (per-IDE, persistent)."""
    config = load_config()
    master = load_master(config["source"])
    servers = master.get("mcpServers", {})
    targets = config.get("targets", {})

    if args.ide is not None and args.ide not in targets:
        raise McpShareError(f"Unknown IDE: {args.ide}. Configured: {', '.join(sorted(targets)) or '(none)'}")

    if args.ide is not None and args.ide == "vscode":
        logger.warning(
            "\033[33m%s has no JSON disable mechanism; this entry is recorded but will be ignored on sync.\033[0m",
            args.ide,
        )

    for name in args.servers:
        if name not in servers:
            logger.warning("Server '%s' not currently in master; recording anyway", name)

    disabled_data = load_disabled(config["source"])
    key = args.ide if args.ide else DISABLED_ALL
    entry = list(disabled_data.get(key, []))
    for name in args.servers:
        if name not in entry:
            entry.append(name)
        scope = f"in {args.ide}" if args.ide else "in all IDEs"
        print(f"Disabled: {name} ({scope})")
    disabled_data[key] = entry
    save_disabled(config["source"], disabled_data)


def cmd_enable(args: Enable) -> None:
    """Re-enable one or more previously disabled MCP servers."""
    config = load_config()
    targets = config.get("targets", {})
    if args.ide is not None and args.ide not in targets:
        raise McpShareError(f"Unknown IDE: {args.ide}. Configured: {', '.join(sorted(targets)) or '(none)'}")

    disabled_data = load_disabled(config["source"])
    # When --ide is not given, clear from "*" and every per-IDE entry.
    keys = [args.ide] if args.ide else list(disabled_data)
    for name in args.servers:
        removed_from: list[str] = []
        for key in keys:
            entry = disabled_data.get(key, [])
            if name in entry:
                disabled_data[key] = [s for s in entry if s != name]
                removed_from.append("all IDEs" if key == DISABLED_ALL else key)
        if removed_from:
            print(f"Enabled: {name} (cleared from {', '.join(removed_from)})")
        else:
            print(f"Enabled: {name} (was not disabled)")

    save_disabled(config["source"], disabled_data)


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    cmd = tyro.cli(Command, prog="mcpshare", description="Synchronize MCP configurations between coding agents.")

    try:
        if isinstance(cmd, Init):
            cmd_init(cmd)
        elif isinstance(cmd, Pull):
            cmd_pull(cmd)
        elif isinstance(cmd, Sync):
            cmd_sync(cmd)
        elif isinstance(cmd, Status):
            cmd_status(cmd)
        elif isinstance(cmd, Disable):
            cmd_disable(cmd)
        elif isinstance(cmd, Enable):
            cmd_enable(cmd)
        elif isinstance(cmd, Version):
            print(f"mcpshare {__version__}")
    except McpShareError as exc:
        logger.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
