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
Google Gemini CLI, Google Antigravity, and OpenCode.
"""

import importlib.metadata
import json
import logging
import os
import re
import shutil
import sys
import tomllib
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

# Config key listing which tools ``mcpshare pull`` collects servers from, and
# the default list used by ``mcpshare init`` (and when older configs omit it).
INIT_SOURCE_KEY = "init-source"
DEFAULT_INIT_SOURCES = ("vscode", "copilot")

# Filename of the native settings file (relative to the target directory)
# used by Copilot CLI for its ``disabledMcpServers`` array.
NATIVE_SETTINGS_FILENAME = "settings.json"
NATIVE_DISABLED_KEY = "disabledMcpServers"

# Files mcpshare wrote for Claude Code before it targeted ``.claude.json``.
# Claude never read them; :func:`_cleanup_legacy_claude` removes them.
CLAUDE_LEGACY_FILENAME = "mcp_servers.json"

# Human-readable description of each tool's native disable mechanism,
# shown by ``mcpshare status``. The actual mechanism lives inside each
# writer; this dict is purely cosmetic.
DISABLE_DESCRIPTION = {
    "claude": "omitted from .claude.json",
    "copilot": "settings.json/disabledMcpServers",
    "codex": "enabled = false in TOML",
    "gemini": "settings.json/mcp.excluded",
    "antigravity": '"disabled": true in mcp_config.json',
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


# Claude Code's default config dir.  Its user-scope MCP servers live in
# ``~/.claude.json``, outside this dir (see :func:`resolve_target_path`).
CLAUDE_CONFIG_DIR = Path.home() / ".claude"

# Default config locations for each tool (user-level)
DEFAULT_CONFIG_PATHS = {
    "claude": Path.home() / ".claude.json",
    "vscode": _vscode_config_dir() / "mcp.json",
    "copilot": Path.home() / ".copilot" / "mcp-config.json",
    "codex": Path.home() / ".codex" / "config.toml",
    "gemini": Path.home() / ".gemini" / "settings.json",
    "antigravity": Path.home() / ".gemini" / "config" / "mcp_config.json",
    "opencode": Path.home() / ".config" / "opencode" / "opencode.json",
}

# Output filenames when writing to target directories
TARGET_FILENAMES = {
    "claude": ".claude.json",
    "vscode": "mcp.json",
    "copilot": "mcp-config.json",
    "codex": "config.toml",
    "gemini": "settings.json",
    "antigravity": "mcp_config.json",
    "opencode": "opencode.json",
}


def _atomic_write_text(path: Path, content: str) -> None:
    """Write *content* to *path* atomically.

    Content goes to a sibling temp file that is then renamed into place, so a
    reader never sees a half-written file.  A symlinked *path* is written
    through to its target, and an existing file's permission bits carry over
    to the replacement before any bytes hit disk.
    """
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            if path.exists():
                shutil.copymode(path, tmp_path)
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Write *data* as formatted JSON to *path* atomically."""
    _atomic_write_text(path, json.dumps(data, indent=2) + "\n")


def default_config() -> dict[str, Any]:
    """Return a default configuration."""
    source = str(CONFIG_DIR / "master")
    return {
        "source": source,
        INIT_SOURCE_KEY: list(DEFAULT_INIT_SOURCES),
        "mode": "merge",
        "targets": {
            "claude": {"path": str(CLAUDE_CONFIG_DIR)},
            "vscode": {"path": str(DEFAULT_CONFIG_PATHS["vscode"].parent)},
            "copilot": {"path": str(DEFAULT_CONFIG_PATHS["copilot"].parent)},
            "codex": {"path": str(DEFAULT_CONFIG_PATHS["codex"].parent)},
            "gemini": {"path": str(DEFAULT_CONFIG_PATHS["gemini"].parent)},
            "antigravity": {"path": str(DEFAULT_CONFIG_PATHS["antigravity"].parent)},
            "opencode": {
                "path": str(DEFAULT_CONFIG_PATHS["opencode"].parent),
            },
        },
    }


def _resolve_init_sources(config: dict[str, Any]) -> list[str]:
    """Return the ordered, deduplicated list of tools ``mcpshare pull`` reads from.

    Reads ``init-source`` from *config*.  When the key is absent (as in
    configs created by older versions), falls back to
    :data:`DEFAULT_INIT_SOURCES` filtered to the configured ``targets``.

    Raises:
        ConfigParseError: If ``init-source`` is present and not a list of strings.
    """
    targets = config.get("targets", {})
    if INIT_SOURCE_KEY not in config:
        return [tool for tool in DEFAULT_INIT_SOURCES if tool in targets]
    raw = config[INIT_SOURCE_KEY]
    if raw is None:
        return []
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise ConfigParseError(f"Expected list of tool names under '{INIT_SOURCE_KEY}' in {CONFIG_FILE}")
    return list(dict.fromkeys(raw))


def load_config() -> dict[str, Any]:
    """Load configuration from the config file.

    Raises:
        ConfigNotFoundError: If the config file does not exist.
        ConfigParseError: If the config file contains invalid YAML.
    """
    if not CONFIG_FILE.exists():
        raise ConfigNotFoundError(f"Config not found: {CONFIG_FILE}. Run 'mcpshare init' to create one.")
    try:
        with open(CONFIG_FILE) as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        raise ConfigParseError(f"Invalid YAML in {CONFIG_FILE}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigParseError(f"Expected mapping at top of {CONFIG_FILE}, got {type(data).__name__}")
    return data


def save_config(config: dict[str, Any]) -> None:
    """Save configuration to the config file atomically."""
    _atomic_write_text(CONFIG_FILE, yaml.dump(config, default_flow_style=False, sort_keys=False))


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
    """Persist the per-IDE disabled-servers list next to the master atomically."""
    path = disabled_file_path(source_dir)
    # Drop empty entries so the file stays tidy.
    clean = {k: sorted(v) for k, v in data.items() if v}
    if not clean:
        if path.exists():
            path.unlink()
        return
    _atomic_write_text(path, yaml.dump(clean, default_flow_style=False, sort_keys=True))


def disabled_for_tool(data: dict[str, list[str]], tool: str) -> set[str]:
    """Return the set of servers disabled for *tool* (global ``*`` ∪ tool-specific)."""
    return set(data.get(DISABLED_ALL, [])) | set(data.get(tool, []))


# ---------------------------------------------------------------------------
# JSON file helpers
# ---------------------------------------------------------------------------


def _load_json_object(path: Path) -> dict[str, Any]:
    """Load *path* as a JSON object, treating a missing or empty file as ``{}``.

    Some tools create their config file as an empty placeholder before any
    setting is written to it (Antigravity does this with ``mcp_config.json``).
    An empty file holds nothing worth preserving, so it is read as an empty
    object instead of being rejected as invalid JSON.

    Raises:
        ConfigParseError: If the file contains invalid JSON, or its top-level
            value is not an object (nothing could be merged into it safely).
    """
    if not path.exists():
        return {}
    text = path.read_text()
    if not text.strip():
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigParseError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigParseError(f"Expected JSON object at top of {path}, got {type(data).__name__}")
    return data


# ---------------------------------------------------------------------------
# Master config (canonical format)
# ---------------------------------------------------------------------------


def load_master(source_dir: str) -> dict[str, Any]:
    """Load the master MCP config from the source directory.

    A missing or empty master file yields an empty server map.

    Raises:
        ConfigParseError: If the master file contains invalid JSON.
    """
    data = _load_json_object(Path(source_dir) / MASTER_FILENAME)
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
    """Save the master MCP config to the source directory atomically.

    Server entries are written as-is. Disable state is tracked exclusively in
    ``mcp.disabled.yaml`` (see :func:`load_disabled`); the master never
    carries a per-entry ``disabled`` flag.
    """
    _atomic_write_json(Path(source_dir) / MASTER_FILENAME, master)


def read_vscode(path: Path) -> dict[str, Any]:
    """Read MCP servers from VSCode mcp.json format.

    VSCode uses ``{"servers": {"name": {"type": "stdio", ...}}}``.
    Server names are sanitized to kebab-case.

    Raises:
        ConfigParseError: If the file contains invalid JSON.
    """
    data = _load_json_object(path)
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
    """Read user-scope MCP servers from Claude Code ``~/.claude.json``.

    Claude uses ``{"mcpServers": {"name": {"type": "stdio", "command": ...}}}``.

    Raises:
        ConfigParseError: If the file contains invalid JSON.
    """
    data = _load_json_object(path)
    servers = data.get("mcpServers", {})
    return {name: {k: v for k, v in cfg.items() if k not in _READER_DROPPED_KEYS} for name, cfg in servers.items()}


def read_copilot(path: Path) -> dict[str, Any]:
    """Read MCP servers from Copilot CLI mcp-config.json.

    Copilot CLI uses ``{"mcpServers": {"name": {"type": "stdio", ...}}}``.

    Raises:
        ConfigParseError: If the file contains invalid JSON.
    """
    data = _load_json_object(path)
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
    data = tomllib.loads(text)
    servers = data.get("mcp_servers", {})
    return _flatten_nested_servers(servers)


def read_codex(path: Path) -> dict[str, Any]:
    """Read MCP servers from Codex config.toml.

    Codex uses TOML with ``[mcp_servers.<name>]`` sections.  Any
    ``enabled`` field is dropped — disable state is tracked exclusively in
    ``mcp.disabled.yaml`` and not propagated through readers.  Codex
    ``tools`` maps are converted back to lists and ``http_headers`` back to
    ``headers`` for the canonical format.

    Raises:
        ConfigParseError: If the file contains invalid TOML.
    """
    if not path.exists():
        return {}
    text = path.read_text()
    try:
        servers = _parse_toml_mcp_servers(text)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigParseError(f"Invalid TOML in {path}: {exc}") from exc
    for cfg in servers.values():
        cfg.pop("enabled", None)
        if "http_headers" in cfg:
            cfg["headers"] = cfg.pop("http_headers")
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
    data = _load_json_object(path)
    servers = data.get("mcpServers", {})
    return {name: {k: v for k, v in cfg.items() if k not in _READER_DROPPED_KEYS} for name, cfg in servers.items()}


def read_antigravity(path: Path) -> dict[str, Any]:
    """Read MCP servers from Antigravity mcp_config.json.

    Antigravity uses ``{"mcpServers": {"name": {"command": ...}}}`` with
    ``serverUrl`` for remote servers, which becomes the canonical ``url``.

    Raises:
        ConfigParseError: If the file contains invalid JSON.
    """
    data = _load_json_object(path)
    servers = data.get("mcpServers", {})
    return {
        name: {("url" if k == "serverUrl" else k): v for k, v in cfg.items() if k not in _READER_DROPPED_KEYS}
        for name, cfg in servers.items()
    }


def read_opencode(path: Path) -> dict[str, Any]:
    """Read MCP servers from OpenCode opencode.json.

    OpenCode uses ``{"mcp": {"name": {"type": "local", "command": [...], "environment": {}}}}``
    for stdio servers and ``{"type": "remote", "url": ..., "headers": {...}}``
    for remote servers.  Any ``enabled`` flag is ignored.

    Raises:
        ConfigParseError: If the file contains invalid JSON.
    """
    data = _load_json_object(path)
    mcp = data.get("mcp", {})
    result: dict[str, Any] = {}
    for name, cfg in mcp.items():
        entry: dict[str, Any] = {}
        if "url" in cfg:
            entry["url"] = cfg["url"]
            headers = cfg.get("headers")
            if headers is not None:
                entry["headers"] = headers
        else:
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
    "claude": read_claude,
    "vscode": read_vscode,
    "copilot": read_copilot,
    "codex": read_codex,
    "gemini": read_gemini,
    "antigravity": read_antigravity,
    "opencode": read_opencode,
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


def _prefer_vscode_vars(a: Any, b: Any) -> Any:
    """Given two values equal after ``${env:VAR}`` expansion, keep ``${env:VAR}`` placeholders."""
    if isinstance(a, str) and isinstance(b, str):
        return a if _VSCODE_ENV_RE.search(a) else b
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [_prefer_vscode_vars(x, y) for x, y in zip(a, b, strict=True)]
    if isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys():
        return {k: _prefer_vscode_vars(a[k], b[k]) for k in a}
    return a


def write_vscode(path: Path, servers: dict[str, Any], **_: Any) -> tuple[int, list[str]]:
    """Write MCP servers in VSCode format.

    Produces ``{"servers": {"name": {"type": "stdio"|"http", ...}}}``.
    The type is determined by the server configuration: ``"http"`` for
    URL-based servers and ``"stdio"`` for command-based servers.  Other
    top-level keys (such as VSCode's ``inputs`` array) are preserved.

    Returns a tuple of (server count, list of skipped server descriptions).

    Raises:
        ConfigParseError: If an existing file at *path* is not valid JSON.
    """
    existing = _load_json_object(path)
    vscode_servers = {}
    for name, cfg in servers.items():
        entry = {"type": _server_type(cfg)}
        entry.update(cfg)
        vscode_servers[name] = entry
    existing["servers"] = vscode_servers
    _atomic_write_json(path, existing)
    return len(vscode_servers), []


# Canonical keys outside Claude Code's server schema: ``tools`` is a Codex
# allow-list, and disable state lives only in ``mcp.disabled.yaml``.
_CLAUDE_DROPPED_KEYS = frozenset({"tools", "disabled"})


def write_claude(
    path: Path, servers: dict[str, Any], *, disabled_names: Iterable[str] = (), **_: Any
) -> tuple[int, list[str]]:
    """Write MCP servers into Claude Code ``~/.claude.json``.

    Replaces only the top-level ``mcpServers`` object (Claude's user scope);
    every other key in the file is Claude's own state and is preserved.
    Each entry gets ``"type": "stdio"|"http"``, since Claude rejects a
    remote server that has a ``url`` but no ``type``.  Claude has no global
    disable list (its ``disabledMcpServers`` is per project), so names in
    *disabled_names* are left out of ``mcpServers`` entirely.

    Returns a tuple of (server count, list of skipped server descriptions).

    Raises:
        ConfigParseError: If an existing file at *path* is not valid JSON.
    """
    disabled = set(disabled_names)
    existing = _load_json_object(path)
    claude_servers = {}
    for name, cfg in servers.items():
        if name in disabled:
            continue
        entry = {"type": _server_type(cfg)}
        entry.update({k: v for k, v in cfg.items() if k not in _CLAUDE_DROPPED_KEYS})
        claude_servers[name] = entry
    existing["mcpServers"] = claude_servers
    _atomic_write_json(path, existing)
    return len(claude_servers), []


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

    Raises:
        ConfigParseError: If an existing file at *path* is not valid JSON.
    """
    existing = _load_json_object(path)
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
    _atomic_write_json(path, existing)
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

    Canonical ``headers`` entries are emitted as ``http_headers``, which is
    the key Codex expects for HTTP transport headers.

    Returns a tuple of (server count, list of skipped server descriptions).

    Raises:
        ConfigParseError: If an existing file at *path* is not valid TOML.
    """
    disabled = set(disabled_names)
    existing_lines: list[str] = []
    if path.exists():
        text = path.read_text()
        try:
            tomllib.loads(text)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigParseError(f"Invalid TOML in {path}: {exc}") from exc
        in_mcp = False
        for line in text.splitlines():
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
            codex_key = "http_headers" if key == "headers" else key
            parts.append(f"{codex_key} = {_format_toml_value(val)}")
    parts.append("")

    _atomic_write_text(path, "\n".join(parts))
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

    Raises:
        ConfigParseError: If an existing file at *path* is not valid JSON.
    """
    existing = _load_json_object(path)
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

    _atomic_write_json(path, existing)
    return len(servers), []


# Canonical keys outside Antigravity's server schema: it has no ``type``,
# ``tools`` is a Codex allow-list (Antigravity only has a deny-list), and
# disable state lives only in ``mcp.disabled.yaml``.
_ANTIGRAVITY_DROPPED_KEYS = frozenset({"type", "tools", "disabled"})


def write_antigravity(
    path: Path, servers: dict[str, Any], *, disabled_names: Iterable[str] = (), **_: Any
) -> tuple[int, list[str]]:
    """Write MCP servers in Antigravity mcp_config.json format.

    Produces ``{"mcpServers": {"name": {"command": ...}}}``, the file shared
    by the Antigravity app, IDE and CLI.  Canonical ``url`` entries are
    emitted as ``serverUrl``, the only remote-server key Antigravity
    accepts.  Names in *disabled_names* receive ``"disabled": true``
    (Antigravity's native per-entry disable flag).  Other top-level keys
    are preserved.  Antigravity creates ``mcp_config.json`` as an empty
    file until the user adds a server, which is treated as an empty object.

    Returns a tuple of (server count, list of skipped server descriptions).

    Raises:
        ConfigParseError: If an existing file at *path* is not valid JSON.
    """
    disabled = set(disabled_names)
    existing = _load_json_object(path)
    ag_servers = {}
    for name, cfg in servers.items():
        entry = {("serverUrl" if k == "url" else k): v for k, v in cfg.items() if k not in _ANTIGRAVITY_DROPPED_KEYS}
        if name in disabled:
            entry["disabled"] = True
        ag_servers[name] = entry
    existing["mcpServers"] = ag_servers
    _atomic_write_json(path, existing)
    return len(ag_servers), []


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

    Raises:
        ConfigParseError: If an existing file at *path* is not valid JSON.
    """
    disabled = set(disabled_names)
    existing = _load_json_object(path)
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
    _atomic_write_json(path, existing)
    return len(oc_servers), []


WRITERS = {
    "claude": write_claude,
    "vscode": write_vscode,
    "copilot": write_copilot,
    "codex": write_codex,
    "gemini": write_gemini,
    "antigravity": write_antigravity,
    "opencode": write_opencode,
}


def _write_native_disabled_list(settings_path: Path, disabled_names: set[str]) -> set[str]:
    """Set the top-level ``disabledMcpServers`` array in *settings_path*.

    Preserves every other key in the file.  When *disabled_names* is empty
    the key is removed entirely.  Returns the set of names that were
    previously listed (so the caller can warn about anything that mcpshare
    is about to silently re-enable).

    The write is atomic (see :func:`_atomic_write_json`).

    Raises:
        ConfigParseError: If an existing *settings_path* is not valid JSON.
    """
    data = _load_json_object(settings_path)
    previous_raw = data.get(NATIVE_DISABLED_KEY, [])
    previous = {str(item) for item in previous_raw} if isinstance(previous_raw, list) else set()
    sorted_names = sorted(disabled_names)
    if sorted_names:
        data[NATIVE_DISABLED_KEY] = sorted_names
    else:
        data.pop(NATIVE_DISABLED_KEY, None)
    _atomic_write_json(settings_path, data)
    return previous


def _emit_native_disabled(
    tool: str,
    settings_path: Path,
    disabled_to_emit: set[str],
    known_servers: set[str],
) -> None:
    """Write the native disabled list for *tool* and log drift warnings.

    Helper for :func:`write_copilot`, whose IDE tracks disabled servers in
    ``settings.json`` → ``disabledMcpServers``.
    Warns when mcpshare is about to silently re-enable a server that the
    user had disabled natively in the IDE (mcpshare-owns: it wins, but the
    user gets a clear pointer to the command that would preserve the
    previous state).
    """
    try:
        previous = _write_native_disabled_list(settings_path, disabled_to_emit)
    except ConfigParseError as exc:
        logger.error(
            "Cannot update %s (%s). Leaving file untouched.",
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
    """Return the full file path for a tool's config inside *target_dir*.

    Claude Code is the exception and mirrors Claude's own lookup: its
    default config dir ``~/.claude`` maps to ``~/.claude.json`` in the home
    directory, while any other dir (as with ``CLAUDE_CONFIG_DIR``) holds
    its own ``.claude.json``.
    """
    if tool == "claude":
        claude_dir = Path(target_dir).expanduser()
        if claude_dir.resolve() == (Path.home() / ".claude").resolve():
            return Path.home() / TARGET_FILENAMES[tool]
        return claude_dir / TARGET_FILENAMES[tool]
    return Path(target_dir) / TARGET_FILENAMES[tool]


def _cleanup_legacy_claude(target_dir: str) -> None:
    """Remove what older mcpshare versions wrote into the Claude config dir.

    Those versions wrote ``mcp_servers.json`` plus a ``disabledMcpServers``
    array in ``settings.json``.  Claude Code reads neither, so both only
    mislead.  Every other ``settings.json`` key is preserved.
    """
    claude_dir = Path(target_dir).expanduser()
    legacy_path = claude_dir / CLAUDE_LEGACY_FILENAME
    if legacy_path.exists():
        legacy_path.unlink()
        logger.info("Removed legacy %s (never read by Claude Code)", legacy_path)

    settings_path = claude_dir / NATIVE_SETTINGS_FILENAME
    try:
        settings = _load_json_object(settings_path)
    except ConfigParseError as exc:
        logger.error(
            "Cannot clean up %s (%s). Leaving file untouched.",
            settings_path,
            exc,
        )
        return
    if NATIVE_DISABLED_KEY in settings:
        del settings[NATIVE_DISABLED_KEY]
        _atomic_write_json(settings_path, settings)
        logger.info("Removed legacy %s from %s", NATIVE_DISABLED_KEY, settings_path)


# Canonical server keys that a given tool's config format cannot represent
# (dropped by its writer).  When ``collect`` compares two sources, a missing
# key that the source's format cannot store is not treated as a conflict and
# is preserved from the side that has it.
_UNSUPPORTED_CANONICAL_KEYS: dict[str, frozenset[str]] = {
    "claude": frozenset({"tools"}),
    "antigravity": frozenset({"tools"}),
    "opencode": frozenset({"tools", "cwd"}),
}


def _configs_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Compare two server configs for equality after ``${env:VAR}`` expansion."""
    return _resolve_vscode_vars(a) == _resolve_vscode_vars(b)


def _merge_equivalent_configs(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Merge two equivalent configs, keeping ``${env:VAR}`` placeholders and any extra keys."""
    merged: dict[str, Any] = {}
    for key in dict.fromkeys((*a, *b)):
        if key in a and key in b:
            merged[key] = _prefer_vscode_vars(a[key], b[key])
        elif key in a:
            merged[key] = a[key]
        else:
            merged[key] = b[key]
    return merged


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
    """Read MCP servers from the configured ``init-source`` tools and merge into master.

    Which tools are read (and in what priority order) is controlled by the
    ``init-source`` list in ``config.yaml`` (defaulting to
    :data:`DEFAULT_INIT_SOURCES` when omitted).

    Disable state lives exclusively in ``mcp.disabled.yaml`` (see
    :func:`load_disabled`); readers strip any legacy per-entry ``disabled``
    field so it never enters the master.  ``${env:VAR}`` placeholders are
    preserved when another source holds the expanded environment value, and
    keys that a tool's schema cannot store (e.g. Codex ``tools`` in Claude)
    do not trigger false conflicts.

    When two sources define the same server name with genuinely different
    settings, the user is prompted to pick one (when *interactive* is True
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
    unrepresentable: dict[str, frozenset[str]] = {name: frozenset() for name in servers}

    can_prompt = interactive and sys.stdin.isatty()

    targets = config.get("targets", {})
    default_targets = default_config()["targets"]
    for tool in _resolve_init_sources(config):
        if tool not in READERS:
            logger.warning(
                "Skipping unknown %s tool: %s (known: %s)",
                INIT_SOURCE_KEY,
                tool,
                ", ".join(sorted(READERS)),
            )
            continue
        target_dir = targets[tool]["path"] if tool in targets else default_targets[tool]["path"]
        target_path = resolve_target_path(tool, target_dir)
        tool_servers = READERS[tool](target_path)
        if not tool_servers:
            continue
        logger.info("Collected %d server(s) from %s", len(tool_servers), tool)
        unrep_new = _UNSUPPORTED_CANONICAL_KEYS.get(tool, frozenset())
        for name, new_cfg in tool_servers.items():
            if name not in servers:
                servers[name] = new_cfg
                sources[name] = tool
                unrepresentable[name] = unrep_new
                continue
            cur_cfg = servers[name]
            unrep_cur = unrepresentable[name]
            cur_proj = {k: v for k, v in cur_cfg.items() if k in new_cfg or k not in unrep_new}
            new_proj = {k: v for k, v in new_cfg.items() if k in cur_cfg or k not in unrep_cur}
            if _configs_equal(cur_proj, new_proj):
                servers[name] = _merge_equivalent_configs(cur_cfg, new_cfg)
                unrepresentable[name] = unrep_cur & unrep_new
            else:
                current_source = sources.get(name, "?")
                if can_prompt:
                    use_new = _prompt_conflict(name, current_source, cur_cfg, tool, new_cfg)
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
                    unrepresentable[name] = unrep_new

    # Warn about server IDs that don't follow naming conventions.
    for name in sorted(servers):
        for warning in _validate_server_id(name):
            logger.warning("\033[33m  ⚠ %s\033[0m", warning)

    master["mcpServers"] = servers
    return master


def distribute(config: dict[str, Any], master: dict[str, Any]) -> list[str]:
    """Write the master MCP config to all configured targets atomically.

    Every master server is written to every configured tool.  Disable
    state comes exclusively from ``mcp.disabled.yaml`` (see
    :func:`load_disabled`) and is forwarded to each writer via the
    ``disabled_names`` kwarg.  Each writer is responsible for applying it
    using whatever native mechanism its IDE supports:

    * **Claude Code** — servers written to ``mcpServers`` in
      ``~/.claude.json``; disabled servers are omitted, since Claude has
      no global disable list.
    * **Copilot CLI** — server written to ``mcp-config.json`` as normal;
      disabled names recorded in ``settings.json`` → ``disabledMcpServers``
      (mcpshare-owns: the array is overwritten to exactly match mcpshare's
      view).
    * **Codex** — server written with ``enabled = false`` inside its
      ``[mcp_servers.<name>]`` TOML block.
    * **Gemini CLI** — server written to ``mcpServers``; disabled names
      added to ``mcp.excluded`` in the same ``settings.json``.
    * **Antigravity** — server written to ``mcpServers`` in
      ``mcp_config.json`` with ``"disabled": true``.
    * **OpenCode** — server written to ``mcp`` with ``"enabled": false``.
    * **VSCode** — ``disabled_names`` is ignored; VSCode has no JSON-level
      disable mechanism, so every server is written enabled.

    Every writer replaces only the MCP servers section in place (via an
    atomic temp-file rename), so removed servers never linger while
    unrelated keys and sections (such as VSCode's ``inputs`` array) are
    preserved.  A target file that cannot be parsed is left untouched
    rather than clobbered: the error is logged, that target is skipped,
    and the remaining targets are still written.

    Returns the names of the tools that could not be updated.
    """
    servers = master.get("mcpServers", {})
    disabled_data = load_disabled(config["source"])
    known_servers = set(servers)
    failed: list[str] = []

    for tool, tcfg in config.get("targets", {}).items():
        if tool not in WRITERS:
            logger.warning("Skipping unknown tool: %s", tool)
            continue
        disabled_set = disabled_for_tool(disabled_data, tool) & known_servers

        target_path = resolve_target_path(tool, tcfg["path"])
        # VSCode handles ${env:VAR} natively; resolve for all other targets.
        resolved = servers if tool == "vscode" else _resolve_vscode_vars(servers)
        try:
            count, skipped = WRITERS[tool](target_path, resolved, disabled_names=disabled_set)
        except ConfigParseError as exc:
            logger.error("\033[31mCannot update %s (%s). Leaving file untouched.\033[0m", tool, exc)
            failed.append(tool)
            continue
        logger.info("Wrote %d server(s) to %s (%s)", count, tool, target_path)
        for name in sorted(disabled_set):
            logger.info("\033[33m  ↳ disabled: %s\033[0m", name)
        for reason in skipped:
            logger.info("\033[31m  ↳ skipped %s\033[0m", reason)
        if tool == "claude":
            _cleanup_legacy_claude(tcfg["path"])
    return failed


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
    """Distribute master MCP config to all targets.

    Raises:
        McpShareError: If any target could not be updated (so the process
            exits non-zero); the other targets are still written.
    """
    config = load_config()
    master = load_master(config["source"])
    logger.info("Distributing to targets...")
    failed = distribute(config, master)
    if failed:
        raise McpShareError(f"Sync incomplete: could not update {', '.join(failed)} (see errors above).")
    logger.info("Sync complete.")


def cmd_status(args: Status) -> None:
    """Show current configuration status."""
    config = load_config()
    source_dir = config["source"]
    master = load_master(source_dir)
    servers = master.get("mcpServers", {})
    disabled_data = load_disabled(source_dir)

    init_sources = _resolve_init_sources(config)
    # Status output is user-facing, so print() is intentional here.
    print(f"Config:       {CONFIG_FILE}")
    print(f"Source:       {source_dir}")
    print(f"Init sources: {', '.join(init_sources) if init_sources else '(none)'}")
    print(f"Mode:         {config.get('mode', 'merge')}")
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
