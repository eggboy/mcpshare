"""Tests for mcpshare."""

import argparse
import json
import textwrap
from pathlib import Path

import pytest

import mcpshare

# ---------------------------------------------------------------------------
# Config management tests
# ---------------------------------------------------------------------------


class TestVscodeConfigDir:
    def test_vscode_config_dir_darwin(self, monkeypatch):
        monkeypatch.setattr(mcpshare.sys, "platform", "darwin")
        result = mcpshare._vscode_config_dir()
        assert result == Path.home() / "Library" / "Application Support" / "Code" / "User"

    def test_vscode_config_dir_linux(self, monkeypatch):
        monkeypatch.setattr(mcpshare.sys, "platform", "linux")
        result = mcpshare._vscode_config_dir()
        assert result == Path.home() / ".config" / "Code" / "User"

    def test_vscode_config_dir_win32(self, monkeypatch):
        monkeypatch.setattr(mcpshare.sys, "platform", "win32")
        monkeypatch.setenv("APPDATA", "/fake/appdata")
        result = mcpshare._vscode_config_dir()
        assert result == Path("/fake/appdata") / "Code" / "User"


class TestConfig:
    def test_default_config_structure(self):
        cfg = mcpshare.default_config()
        assert "source" in cfg
        assert "mode" in cfg
        assert "targets" in cfg
        assert cfg["mode"] == "merge"
        for tool in ("claude", "vscode", "copilot", "codex", "gemini", "opencode"):
            assert tool in cfg["targets"]
            assert "path" in cfg["targets"][tool]

    def test_save_and_load_config(self, tmp_home):
        cfg = {"source": "/tmp/master", "mode": "merge", "targets": {}}
        mcpshare.save_config(cfg)
        loaded = mcpshare.load_config()
        assert loaded == cfg

    def test_load_config_raises_when_missing(self, tmp_home):
        """load_config raises ConfigNotFoundError when no config exists."""
        with pytest.raises(mcpshare.ConfigNotFoundError, match="Config not found"):
            mcpshare.load_config()


# ---------------------------------------------------------------------------
# Master config tests
# ---------------------------------------------------------------------------


class TestMaster:
    def test_load_nonexistent_master(self, tmp_path):
        master = mcpshare.load_master(str(tmp_path / "nonexistent"))
        assert master == {"mcpServers": {}}

    def test_save_and_load_master(self, tmp_path, sample_servers):
        source = str(tmp_path / "master")
        master = {"mcpServers": sample_servers}
        mcpshare.save_master(source, master)
        loaded = mcpshare.load_master(source)
        assert loaded["mcpServers"] == sample_servers

    def test_load_master_raises_on_invalid_json(self, tmp_path):
        """load_master raises ConfigParseError on malformed JSON."""
        source = tmp_path / "bad_master"
        source.mkdir()
        (source / "mcp.json").write_text("{not valid json")
        with pytest.raises(mcpshare.ConfigParseError, match="Invalid JSON"):
            mcpshare.load_master(str(source))


# ---------------------------------------------------------------------------
# Reader tests
# ---------------------------------------------------------------------------


class TestReaders:
    def test_read_vscode(self, tmp_path):
        path = tmp_path / "mcp.json"
        data = {
            "servers": {
                "fs": {
                    "type": "stdio",
                    "command": "npx",
                    "args": ["-y", "server-fs"],
                    "env": {},
                }
            }
        }
        path.write_text(json.dumps(data))
        result = mcpshare.read_vscode(path)
        assert "fs" in result
        assert "type" not in result["fs"]
        assert result["fs"]["command"] == "npx"

    def test_read_vscode_replaces_spaces_in_names(self, tmp_path):
        path = tmp_path / "mcp.json"
        data = {
            "servers": {
                "my server": {"type": "stdio", "command": "npx"},
                "another tool": {"type": "stdio", "command": "node"},
            }
        }
        path.write_text(json.dumps(data))
        result = mcpshare.read_vscode(path)
        assert "my_server" in result
        assert "another_tool" in result
        assert "my server" not in result
        assert "another tool" not in result

    def test_read_claude(self, tmp_path):
        path = tmp_path / "settings.json"
        data = {"mcpServers": {"github": {"command": "npx", "args": ["server-gh"], "env": {}}}}
        path.write_text(json.dumps(data))
        result = mcpshare.read_claude(path)
        assert "github" in result
        assert result["github"]["command"] == "npx"

    def test_read_copilot(self, tmp_path):
        path = tmp_path / "mcp-config.json"
        data = {
            "mcpServers": {
                "db": {
                    "type": "stdio",
                    "command": "node",
                    "args": ["server.js"],
                    "env": {},
                }
            }
        }
        path.write_text(json.dumps(data))
        result = mcpshare.read_copilot(path)
        assert "db" in result
        assert "type" not in result["db"]

    def test_read_codex_toml(self, tmp_path):
        path = tmp_path / "config.toml"
        toml_text = textwrap.dedent("""\
            [mcp_servers.search]
            command = "npx"
            args = ["-y", "server-search"]
            env = { API_KEY = "abc" }
        """)
        path.write_text(toml_text)
        result = mcpshare.read_codex(path)
        assert "search" in result
        assert result["search"]["command"] == "npx"
        assert result["search"]["args"] == ["-y", "server-search"]
        assert result["search"]["env"] == {"API_KEY": "abc"}

    def test_read_codex_dotted_server_name(self, tmp_path):
        """Server names with dots (e.g. microsoft.docs.mcp) must be preserved."""
        path = tmp_path / "config.toml"
        toml_text = textwrap.dedent("""\
            [mcp_servers."microsoft.docs.mcp"]
            url = "https://learn.microsoft.com/api/mcp"

            [mcp_servers.simple]
            command = "npx"
            args = ["server"]
        """)
        path.write_text(toml_text)
        result = mcpshare.read_codex(path)
        assert "microsoft.docs.mcp" in result
        assert result["microsoft.docs.mcp"]["url"] == "https://learn.microsoft.com/api/mcp"
        assert "simple" in result

    def test_write_codex_quotes_dotted_names(self, tmp_path):
        """Server names with dots must be quoted in TOML output."""
        path = tmp_path / "config.toml"
        servers = {
            "microsoft.docs.mcp": {"url": "https://learn.microsoft.com/api/mcp"},
            "simple": {"command": "npx", "args": ["server"]},
        }
        mcpshare.write_codex(path, servers)
        content = path.read_text()
        assert '[mcp_servers."microsoft.docs.mcp"]' in content
        assert "[mcp_servers.simple]" in content

    def test_read_gemini(self, tmp_path):
        path = tmp_path / "settings.json"
        data = {
            "theme": "Dark",
            "mcpServers": {"git": {"command": "uvx", "args": ["mcp-server-git"]}},
        }
        path.write_text(json.dumps(data))
        result = mcpshare.read_gemini(path)
        assert "git" in result
        assert result["git"]["command"] == "uvx"

    def test_read_opencode(self, tmp_path):
        path = tmp_path / "opencode.json"
        data = {
            "mcp": {
                "shadcn": {
                    "type": "local",
                    "command": ["npx", "-y", "shadcn@latest", "mcp"],
                    "environment": {"KEY": "val"},
                }
            }
        }
        path.write_text(json.dumps(data))
        result = mcpshare.read_opencode(path)
        assert "shadcn" in result
        assert result["shadcn"]["command"] == "npx"
        assert result["shadcn"]["args"] == ["-y", "shadcn@latest", "mcp"]
        assert result["shadcn"]["env"] == {"KEY": "val"}

    def test_read_nonexistent_returns_empty(self, tmp_path):
        for reader in mcpshare.READERS.values():
            assert reader(tmp_path / "nope") == {}

    @pytest.mark.parametrize("reader_name", ["vscode", "claude", "copilot", "gemini", "opencode"])
    def test_read_raises_on_invalid_json(self, tmp_path, reader_name):
        """JSON-based readers raise ConfigParseError on malformed input."""
        path = tmp_path / "bad.json"
        path.write_text("{not valid json")
        with pytest.raises(mcpshare.ConfigParseError, match="Invalid JSON"):
            mcpshare.READERS[reader_name](path)


# ---------------------------------------------------------------------------
# Writer tests
# ---------------------------------------------------------------------------


class TestWriters:
    def test_write_vscode(self, tmp_path, sample_servers):
        path = tmp_path / "mcp.json"
        mcpshare.write_vscode(path, sample_servers)
        data = json.loads(path.read_text())
        assert "servers" in data
        assert data["servers"]["filesystem"]["type"] == "stdio"
        assert data["servers"]["filesystem"]["command"] == "npx"

    def test_write_vscode_http_type_for_url_servers(self, tmp_path):
        path = tmp_path / "mcp.json"
        servers = {"remote": {"url": "https://example.com/mcp"}}
        mcpshare.write_vscode(path, servers)
        data = json.loads(path.read_text())
        assert data["servers"]["remote"]["type"] == "http"

    def test_write_claude(self, tmp_path, sample_servers):
        path = tmp_path / ".mcp.json"
        mcpshare.write_claude(path, sample_servers)
        data = json.loads(path.read_text())
        assert "mcpServers" in data
        assert "filesystem" in data["mcpServers"]

    def test_write_copilot(self, tmp_path, sample_servers):
        path = tmp_path / "mcp-config.json"
        mcpshare.write_copilot(path, sample_servers)
        data = json.loads(path.read_text())
        assert "mcpServers" in data
        assert data["mcpServers"]["github"]["type"] == "stdio"

    def test_write_copilot_http_type_for_url_servers(self, tmp_path):
        path = tmp_path / "mcp-config.json"
        servers = {"remote": {"url": "https://example.com/mcp"}}
        mcpshare.write_copilot(path, servers)
        data = json.loads(path.read_text())
        assert data["mcpServers"]["remote"]["type"] == "http"

    def test_write_codex_toml(self, tmp_path, sample_servers):
        path = tmp_path / "config.toml"
        mcpshare.write_codex(path, sample_servers)
        content = path.read_text()
        assert "[mcp_servers.filesystem]" in content
        assert "[mcp_servers.github]" in content
        assert 'command = "npx"' in content

    def test_write_gemini(self, tmp_path, sample_servers):
        path = tmp_path / "settings.json"
        # Pre-existing settings should be preserved
        path.write_text(json.dumps({"theme": "Dracula"}))
        mcpshare.write_gemini(path, sample_servers)
        data = json.loads(path.read_text())
        assert data["theme"] == "Dracula"
        assert "mcpServers" in data

    def test_write_opencode(self, tmp_path, sample_servers):
        path = tmp_path / "opencode.json"
        mcpshare.write_opencode(path, sample_servers)
        data = json.loads(path.read_text())
        assert "mcp" in data
        fs = data["mcp"]["filesystem"]
        assert fs["type"] == "local"
        assert fs["command"] == ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"]

    def test_write_preserves_existing_settings(self, tmp_path, sample_servers):
        """Ensure writers don't clobber non-MCP settings."""
        path = tmp_path / "opencode.json"
        path.write_text(json.dumps({"model": "gpt-4", "theme": "dark"}))
        mcpshare.write_opencode(path, sample_servers)
        data = json.loads(path.read_text())
        assert data["model"] == "gpt-4"
        assert data["theme"] == "dark"
        assert "mcp" in data


# ---------------------------------------------------------------------------
# Round-trip tests (read → write → read)
# ---------------------------------------------------------------------------


class TestRoundTrip:
    @pytest.mark.parametrize(
        "tool,writer,reader,filename",
        [
            ("vscode", mcpshare.write_vscode, mcpshare.read_vscode, "mcp.json"),
            ("claude", mcpshare.write_claude, mcpshare.read_claude, ".mcp.json"),
            ("copilot", mcpshare.write_copilot, mcpshare.read_copilot, "mcp-config.json"),
            ("gemini", mcpshare.write_gemini, mcpshare.read_gemini, "settings.json"),
            ("opencode", mcpshare.write_opencode, mcpshare.read_opencode, "opencode.json"),
        ],
    )
    def test_round_trip(self, tmp_path, sample_servers, tool, writer, reader, filename):
        path = tmp_path / filename
        writer(path, sample_servers)
        result = reader(path)
        assert result == sample_servers


# ---------------------------------------------------------------------------
# Collect / Distribute integration tests
# ---------------------------------------------------------------------------


class TestCollectDistribute:
    def _make_config(self, tmp_path):
        source = str(tmp_path / "master")
        targets = {}
        for tool in ("claude", "vscode", "copilot", "codex", "gemini", "opencode"):
            tdir = tmp_path / tool
            tdir.mkdir()
            targets[tool] = {"path": str(tdir)}
        return {"source": source, "mode": "merge", "targets": targets}

    def test_collect_merges_from_targets(self, tmp_path, sample_servers):
        config = self._make_config(tmp_path)
        # Write servers into Claude target
        claude_path = mcpshare.resolve_target_path("claude", config["targets"]["claude"]["path"])
        mcpshare.write_claude(claude_path, {"filesystem": sample_servers["filesystem"]})
        # Write another server into VSCode target
        vscode_path = mcpshare.resolve_target_path("vscode", config["targets"]["vscode"]["path"])
        mcpshare.write_vscode(vscode_path, {"github": sample_servers["github"]})

        master = mcpshare.collect(config)
        assert "filesystem" in master["mcpServers"]
        assert "github" in master["mcpServers"]

    def test_distribute_writes_all_targets(self, tmp_path, sample_servers):
        config = self._make_config(tmp_path)
        master = {"mcpServers": sample_servers}

        mcpshare.distribute(config, master)

        for tool in config["targets"]:
            target_path = mcpshare.resolve_target_path(tool, config["targets"][tool]["path"])
            assert target_path.exists(), f"{tool} config file should exist"

    def test_full_sync_flow(self, tmp_path, sample_servers):
        config = self._make_config(tmp_path)
        # Seed one target
        claude_path = mcpshare.resolve_target_path("claude", config["targets"]["claude"]["path"])
        mcpshare.write_claude(claude_path, sample_servers)

        # Collect
        master = mcpshare.collect(config)
        assert len(master["mcpServers"]) == len(sample_servers)

        # Distribute
        mcpshare.distribute(config, master)

        # Verify all targets have the servers
        for tool in config["targets"]:
            target_path = mcpshare.resolve_target_path(tool, config["targets"][tool]["path"])
            reader = mcpshare.READERS[tool]
            result = reader(target_path)
            assert len(result) == len(sample_servers), f"{tool} should have all servers"


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestCLI:
    def test_init_creates_config(self, tmp_home, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "")
        args = argparse.Namespace(force=False, source=None)
        mcpshare.cmd_init(args)
        assert mcpshare.CONFIG_FILE.exists()

    def test_init_force_overwrites(self, tmp_home, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "")
        args = argparse.Namespace(force=False, source=None)
        mcpshare.cmd_init(args)
        args = argparse.Namespace(force=True, source=None)
        mcpshare.cmd_init(args)
        assert mcpshare.CONFIG_FILE.exists()

    def test_init_with_source_arg(self, tmp_home, tmp_path):
        custom_source = str(tmp_path / "my_master")
        args = argparse.Namespace(force=False, source=custom_source)
        mcpshare.cmd_init(args)
        assert mcpshare.CONFIG_FILE.exists()
        cfg = mcpshare.load_config()
        assert cfg["source"] == custom_source
        assert Path(custom_source).exists()

    def test_init_prompt_uses_entered_value(self, tmp_home, tmp_path, monkeypatch):
        custom_source = str(tmp_path / "prompted_master")
        monkeypatch.setattr("builtins.input", lambda _: custom_source)
        args = argparse.Namespace(force=False, source=None)
        mcpshare.cmd_init(args)
        cfg = mcpshare.load_config()
        assert cfg["source"] == custom_source

    def test_init_prompt_empty_uses_default(self, tmp_home, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "")
        args = argparse.Namespace(force=False, source=None)
        mcpshare.cmd_init(args)
        cfg = mcpshare.load_config()
        assert cfg["source"] == mcpshare.default_config()["source"]

    def test_parser_subcommands(self):
        parser = mcpshare.build_parser()
        args = parser.parse_args(["init", "--force"])
        assert args.command == "init"
        assert args.force is True

        args = parser.parse_args(["init", "--source", "/my/dir"])
        assert args.source == "/my/dir"

        args = parser.parse_args(["sync"])
        assert args.command == "sync"

        args = parser.parse_args(["status"])
        assert args.command == "status"


# ---------------------------------------------------------------------------
# TOML formatting tests
# ---------------------------------------------------------------------------


class TestTomlFormatting:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("hello", '"hello"'),
            (True, "true"),
            (False, "false"),
            (42, "42"),
            (["a", "b"], '["a", "b"]'),
        ],
    )
    def test_format_toml_value(self, value, expected):
        assert mcpshare._format_toml_value(value) == expected

    def test_format_dict(self):
        result = mcpshare._format_toml_value({"k": "v"})
        assert result == '{ k = "v" }'

    def test_codex_preserves_existing_content(self, tmp_path):
        path = tmp_path / "config.toml"
        path.write_text('[settings]\nmodel = "gpt-4"\n')
        mcpshare.write_codex(path, {"fs": {"command": "npx", "args": []}})
        content = path.read_text()
        assert 'model = "gpt-4"' in content
        assert "[mcp_servers.fs]" in content
