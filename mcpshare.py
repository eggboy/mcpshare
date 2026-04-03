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
    """Save the master MCP config to the source directory."""
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
        entry = {k: v for k, v in cfg.items() if k != "type"}
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
    return data.get("mcpServers", {})


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
        entry = {k: v for k, v in cfg.items() if k != "type"}
        result[name] = entry
    return result


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

    Codex uses TOML with ``[mcp_servers.<name>]`` sections.
    """
    if not path.exists():
        return {}
    text = path.read_text()
    return _parse_toml_mcp_servers(text)


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
    return data.get("mcpServers", {})


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
    # "copilot": read_copilot,
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


def write_vscode(path: Path, servers: dict[str, Any]) -> tuple[int, list[str]]:
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
        entry.update({k: v for k, v in cfg.items() if k != "disabled"})
        vscode_servers[name] = entry
    existing["servers"] = vscode_servers
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
        f.write("\n")
    return len(vscode_servers), []


def write_claude(path: Path, servers: dict[str, Any]) -> tuple[int, list[str]]:
    """Write MCP servers in Claude Code format.

    Produces ``{"mcpServers": {"name": {"command": ...}}}``.

    Returns a tuple of (server count, list of skipped server descriptions).
    """
    existing = {}
    if path.exists():
        with open(path) as f:
            existing = json.load(f)
    existing["mcpServers"] = servers
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
        f.write("\n")
    return len(servers), []


def write_copilot(path: Path, servers: dict[str, Any]) -> tuple[int, list[str]]:
    """Write MCP servers in Copilot CLI format.

    Produces ``{"mcpServers": {"name": {"type": "stdio"|"http", ...}}}``.
    The type is determined by the server configuration: ``"http"`` for
    URL-based servers and ``"stdio"`` for command-based servers.

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
        pairs = ", ".join(f"{k} = {_format_toml_value(v)}" for k, v in value.items())
        return f"{{ {pairs} }}"
    return str(value)


def _toml_key(name: str) -> str:
    """Return a TOML-safe key, quoting if it contains dots or spaces."""
    if "." in name or " " in name:
        return f'"{name}"'
    return name


def write_codex(path: Path, servers: dict[str, Any]) -> tuple[int, list[str]]:
    """Write MCP servers in Codex TOML format.

    Produces ``[mcp_servers.<name>]`` sections.  Server names containing
    dots or spaces are quoted to prevent TOML from treating them as nested
    tables.  Non-MCP content in an existing file is preserved.

    Returns a tuple of (server count, list of skipped server descriptions).
    """
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

    # Remove trailing blank lines
    while existing_lines and not existing_lines[-1].strip():
        existing_lines.pop()

    parts = existing_lines
    for name, cfg in servers.items():
        parts.append("")
        parts.append(f"[mcp_servers.{_toml_key(name)}]")
        for key, val in cfg.items():
            if key == "disabled":
                continue
            parts.append(f"{key} = {_format_toml_value(val)}")
    parts.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts))
    return len(servers), []


def write_gemini(path: Path, servers: dict[str, Any]) -> tuple[int, list[str]]:
    """Write MCP servers in Gemini CLI format.

    Produces ``{"mcpServers": {"name": {"command": ...}}}``.
    Preserves existing non-MCP settings.

    Returns a tuple of (server count, list of skipped server descriptions).
    """
    existing = {}
    if path.exists():
        with open(path) as f:
            existing = json.load(f)
    existing["mcpServers"] = {name: {k: v for k, v in cfg.items() if k != "disabled"} for name, cfg in servers.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
        f.write("\n")
    return len(servers), []


def write_opencode(path: Path, servers: dict[str, Any]) -> tuple[int, list[str]]:
    """Write MCP servers in OpenCode format.

    For stdio servers, converts ``command``/``args`` to a single ``command``
    list, ``env`` to ``environment``, and adds ``"type": "local"``.
    For HTTP/SSE servers (those with ``url``), uses ``"type": "remote"``
    with a ``url`` field.  Preserves existing non-MCP settings.

    Returns a tuple of (server count, list of skipped server descriptions).
    """
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


def resolve_target_path(tool: str, target_dir: str) -> Path:
    """Return the full file path for a tool's config inside *target_dir*."""
    return Path(target_dir) / TARGET_FILENAMES[tool]


def collect(config: dict[str, Any]) -> dict[str, Any]:
    """Read MCP servers from all configured targets and merge into master.

    Disabled flags from the existing master are preserved even when a target
    provides the same server without the flag.

    Returns the merged master dict.
    """
    source_dir = config["source"]
    master = load_master(source_dir)
    servers = dict(master.get("mcpServers", {}))

    # Remember which servers were disabled before merging.
    disabled_servers = {name for name, cfg in servers.items() if cfg.get("disabled")}

    targets = config.get("targets", {})
    for tool, tcfg in targets.items():
        if tool not in READERS:
            logger.warning("Skipping unknown tool: %s", tool)
            continue
        target_path = resolve_target_path(tool, tcfg["path"])
        tool_servers = READERS[tool](target_path)
        if tool_servers:
            logger.info("Collected %d server(s) from %s", len(tool_servers), tool)
            servers.update(tool_servers)

    # Re-apply disabled flags that were present in the master.
    for name in disabled_servers:
        if name in servers:
            servers[name]["disabled"] = True

    # Warn about server IDs that don't follow naming conventions.
    for name in sorted(servers):
        for warning in _validate_server_id(name):
            logger.warning("\033[33m  ⚠ %s\033[0m", warning)

    master["mcpServers"] = servers
    return master


def distribute(config: dict[str, Any], master: dict[str, Any]) -> None:
    """Write the master MCP config to all configured targets."""
    servers = master.get("mcpServers", {})
    targets = config.get("targets", {})
    for tool, tcfg in targets.items():
        if tool not in WRITERS:
            logger.warning("Skipping unknown tool: %s", tool)
            continue
        target_path = resolve_target_path(tool, tcfg["path"])
        # VSCode handles ${env:VAR} natively; resolve for all other targets.
        tool_servers = servers if tool == "vscode" else _resolve_vscode_vars(servers)
        count, skipped = WRITERS[tool](target_path, tool_servers)
        logger.info("Wrote %d server(s) to %s (%s)", count, tool, target_path)
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


@dataclass
class Sync:
    """Distribute master MCP config to all targets."""


@dataclass
class Status:
    """Show current config status."""


@dataclass
class Disable:
    """Disable one or more MCP servers in the master config."""

    servers: tuple[str, ...]
    """Names of the servers to disable."""


@dataclass
class Enable:
    """Enable one or more previously disabled MCP servers."""

    servers: tuple[str, ...]
    """Names of the servers to enable."""


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
    master = collect(config)
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

    # Status output is user-facing, so print() is intentional here.
    print(f"Config: {CONFIG_FILE}")
    print(f"Source: {source_dir}")
    print(f"Mode:   {config.get('mode', 'merge')}")
    print(f"Master servers ({len(servers)}):")
    for name in sorted(servers):
        cfg = servers[name]
        transport = cfg["url"] if "url" in cfg else cfg.get("command", "?")
        disabled = " [disabled]" if cfg.get("disabled") else ""
        print(f"  - {name}: {transport}{disabled}")

    print("\nTargets:")
    targets = config.get("targets", {})
    for tool, tcfg in sorted(targets.items()):
        target_path = resolve_target_path(tool, tcfg["path"])
        exists = "\u2713" if target_path.exists() else "\u2717"
        print(f"  {exists} {tool}: {target_path}")


def cmd_disable(args: Disable) -> None:
    """Disable one or more MCP servers in the master config."""
    config = load_config()
    master = load_master(config["source"])
    servers = master.get("mcpServers", {})
    for name in args.servers:
        if name not in servers:
            raise McpShareError(f"Server not found: {name}")
    for name in args.servers:
        servers[name]["disabled"] = True
        print(f"Disabled: {name}")
    save_master(config["source"], master)


def cmd_enable(args: Enable) -> None:
    """Enable one or more previously disabled MCP servers."""
    config = load_config()
    master = load_master(config["source"])
    servers = master.get("mcpServers", {})
    for name in args.servers:
        if name not in servers:
            raise McpShareError(f"Server not found: {name}")
    for name in args.servers:
        servers[name].pop("disabled", None)
        print(f"Enabled: {name}")
    save_master(config["source"], master)


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
