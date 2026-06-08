"""Shared fixtures for mcpshare tests."""

import pytest

import mcpshare


@pytest.fixture
def tmp_home(tmp_path, monkeypatch):
    """Provide an isolated home directory for tests.

    Patches ``CONFIG_DIR`` and ``CONFIG_FILE`` so any test that runs the
    CLI commands writes its config under ``tmp_path``.  The disabled file
    now lives next to the master ``mcp.json`` (see
    :func:`mcpshare.disabled_file_path`), so each test passes the source
    dir explicitly rather than relying on a patched constant.
    """
    monkeypatch.setattr(mcpshare, "CONFIG_DIR", tmp_path / ".config" / "mcpshare")
    monkeypatch.setattr(mcpshare, "CONFIG_FILE", tmp_path / ".config" / "mcpshare" / "config.yaml")
    return tmp_path


@pytest.fixture
def sample_servers():
    """A minimal set of MCP servers in master/canonical format."""
    return {
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
            "env": {},
        },
        "github": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "env": {"GITHUB_TOKEN": "ghp_test"},
        },
    }
