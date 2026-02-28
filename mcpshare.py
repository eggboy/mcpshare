#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pyyaml>=6.0",
# ]
# ///
"""mcpshare - Synchronize MCP configurations between coding agents.

Supports VSCode, Claude Code, GitHub Copilot CLI, OpenAI Codex,
Google Gemini CLI, and OpenCode.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

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
    "claude": Path.home() / ".claude" / "settings.json",
    "vscode": _vscode_config_dir() / "mcp.json",
    "copilot": Path.home() / ".copilot" / "mcp-config.json",
    "codex": Path.home() / ".codex" / "config.toml",
    "gemini": Path.home() / ".gemini" / "settings.json",
    "opencode": Path.home() / ".config" / "opencode" / "opencode.json",
}

# Output filenames when writing to target directories
TARGET_FILENAMES = {
    "claude": ".mcp.json",
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
    Server names with spaces are sanitized by replacing spaces with underscores.

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
        safe_name = name.replace(" ", "_")
        entry = {k: v for k, v in cfg.items() if k != "type"}
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
    """Parse MCP servers from a TOML config string (Codex format).

    Uses ``tomllib`` (Python 3.11+) when available, otherwise falls back
    to a minimal parser for the ``[mcp_servers.<name>]`` sections.
    """
    try:
        import tomllib

        data = tomllib.loads(text)
        servers = data.get("mcp_servers", {})
        return _flatten_nested_servers(servers)
    except ImportError:
        pass

    # Minimal fallback parser
    servers: dict[str, Any] = {}
    current_name: str | None = None
    current: dict = {}
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[mcp_servers.") and stripped.endswith("]"):
            if current_name is not None:
                servers[current_name] = current
            current_name = stripped[len("[mcp_servers.") : -1].strip('" ')
            current = {}
        elif current_name is not None and "=" in stripped:
            key, _, value = stripped.partition("=")
            key = key.strip()
            value = value.strip()
            if value.startswith('"') and value.endswith('"'):
                current[key] = value[1:-1]
            elif value.startswith("[") and value.endswith("]"):
                items = value[1:-1].split(",")
                current[key] = [i.strip().strip('"') for i in items if i.strip()]
            elif value.startswith("{") and value.endswith("}"):
                pairs = value[1:-1].split(",")
                d = {}
                for p in pairs:
                    if "=" in p:
                        k, _, v = p.partition("=")
                        d[k.strip().strip('"')] = v.strip().strip('"')
                current[key] = d
            elif value in ("true", "false"):
                current[key] = value == "true"
            else:
                try:
                    current[key] = int(value)
                except ValueError:
                    current[key] = value
    if current_name is not None:
        servers[current_name] = current
    return servers


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
    return "http" if "url" in cfg else "stdio"


def write_vscode(path: Path, servers: dict[str, Any]) -> None:
    """Write MCP servers in VSCode format.

    Produces ``{"servers": {"name": {"type": "stdio"|"http", ...}}}``.
    The type is determined by the server configuration: ``"http"`` for
    URL-based servers and ``"stdio"`` for command-based servers.
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


def write_claude(path: Path, servers: dict[str, Any]) -> None:
    """Write MCP servers in Claude Code format.

    Produces ``{"mcpServers": {"name": {"command": ...}}}``.
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


def write_copilot(path: Path, servers: dict[str, Any]) -> None:
    """Write MCP servers in Copilot CLI format.

    Produces ``{"mcpServers": {"name": {"type": "stdio"|"http", ...}}}``.
    The type is determined by the server configuration: ``"http"`` for
    URL-based servers and ``"stdio"`` for command-based servers.
    """
    existing = {}
    if path.exists():
        with open(path) as f:
            existing = json.load(f)
    copilot_servers = {}
    for name, cfg in servers.items():
        entry = {"type": _server_type(cfg)}
        entry.update(cfg)
        copilot_servers[name] = entry
    existing["mcpServers"] = copilot_servers
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
        f.write("\n")


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


def write_codex(path: Path, servers: dict[str, Any]) -> None:
    """Write MCP servers in Codex TOML format.

    Produces ``[mcp_servers.<name>]`` sections.  Server names containing
    dots or spaces are quoted to prevent TOML from treating them as nested
    tables.  Non-MCP content in an existing file is preserved.
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
            parts.append(f"{key} = {_format_toml_value(val)}")
    parts.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts))


def write_gemini(path: Path, servers: dict[str, Any]) -> None:
    """Write MCP servers in Gemini CLI format.

    Produces ``{"mcpServers": {"name": {"command": ...}}}``.
    Preserves existing non-MCP settings.
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


def write_opencode(path: Path, servers: dict[str, Any]) -> None:
    """Write MCP servers in OpenCode format.

    For stdio servers, converts ``command``/``args`` to a single ``command``
    list, ``env`` to ``environment``, and adds ``"type": "local"``.
    For HTTP/SSE servers (those with ``url``), uses ``"type": "remote"``
    with a ``url`` field.  Preserves existing non-MCP settings.
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

    Returns the merged master dict.
    """
    source_dir = config["source"]
    master = load_master(source_dir)
    servers = dict(master.get("mcpServers", {}))

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
        WRITERS[tool](target_path, servers)
        logger.info("Wrote %d server(s) to %s (%s)", len(servers), tool, target_path)


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


def cmd_init(args: argparse.Namespace) -> None:
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


def cmd_sync(args: argparse.Namespace) -> None:
    """Synchronize MCP settings between tools."""
    config = load_config()
    mode = config.get("mode", "merge")

    if mode == "merge":
        logger.info("Collecting MCP servers from targets...")
        master = collect(config)
    else:
        logger.info("Loading master config (overwrite mode)...")
        master = load_master(config["source"])

    save_master(config["source"], master)
    total = len(master.get("mcpServers", {}))
    logger.info("Master updated with %d server(s).", total)

    logger.info("Distributing to targets...")
    distribute(config, master)
    logger.info("Sync complete.")


def cmd_status(args: argparse.Namespace) -> None:
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
        print(f"  - {name}: {transport}")

    print("\nTargets:")
    targets = config.get("targets", {})
    for tool, tcfg in sorted(targets.items()):
        target_path = resolve_target_path(tool, tcfg["path"])
        exists = "\u2713" if target_path.exists() else "\u2717"
        print(f"  {exists} {tool}: {target_path}")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="mcpshare",
        description="Synchronize MCP configurations between coding agents.",
    )
    sub = parser.add_subparsers(dest="command")

    init_p = sub.add_parser("init", help="Create default config")
    init_p.add_argument("--force", action="store_true", help="Overwrite existing config")
    init_p.add_argument("--source", metavar="DIR", help="Source folder for master MCP config")

    sub.add_parser("sync", help="Synchronize MCP settings between tools")
    sub.add_parser("status", help="Show current config status")

    return parser


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    parser = build_parser()
    args = parser.parse_args()

    commands = {
        "init": cmd_init,
        "sync": cmd_sync,
        "status": cmd_status,
    }

    if args.command in commands:
        try:
            commands[args.command](args)
        except McpShareError as exc:
            logger.error("%s", exc)
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
