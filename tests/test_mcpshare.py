"""Tests for mcpshare."""

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
        assert "my-server" in result
        assert "another-tool" in result
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
        path = tmp_path / "mcp_servers.json"
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
            ("claude", mcpshare.write_claude, mcpshare.read_claude, "mcp_servers.json"),
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
        args = mcpshare.Init(force=False, source=None)
        mcpshare.cmd_init(args)
        assert mcpshare.CONFIG_FILE.exists()

    def test_init_force_overwrites(self, tmp_home, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "")
        args = mcpshare.Init(force=False, source=None)
        mcpshare.cmd_init(args)
        args = mcpshare.Init(force=True, source=None)
        mcpshare.cmd_init(args)
        assert mcpshare.CONFIG_FILE.exists()

    def test_init_with_source_arg(self, tmp_home, tmp_path):
        custom_source = str(tmp_path / "my_master")
        args = mcpshare.Init(force=False, source=custom_source)
        mcpshare.cmd_init(args)
        assert mcpshare.CONFIG_FILE.exists()
        cfg = mcpshare.load_config()
        assert cfg["source"] == custom_source
        assert Path(custom_source).exists()

    def test_init_prompt_uses_entered_value(self, tmp_home, tmp_path, monkeypatch):
        custom_source = str(tmp_path / "prompted_master")
        monkeypatch.setattr("builtins.input", lambda _: custom_source)
        args = mcpshare.Init(force=False, source=None)
        mcpshare.cmd_init(args)
        cfg = mcpshare.load_config()
        assert cfg["source"] == custom_source

    def test_init_prompt_empty_uses_default(self, tmp_home, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "")
        args = mcpshare.Init(force=False, source=None)
        mcpshare.cmd_init(args)
        cfg = mcpshare.load_config()
        assert cfg["source"] == mcpshare.default_config()["source"]


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


# ---------------------------------------------------------------------------
# Disable / Enable tests
# ---------------------------------------------------------------------------


class TestDisableEnable:
    def _setup_master(self, tmp_path, sample_servers):
        source = str(tmp_path / "master")
        master = {"mcpServers": sample_servers}
        mcpshare.save_master(source, master)
        config = {"source": source, "mode": "merge", "targets": {}}
        mcpshare.save_config(config)
        return source

    def test_disable_server(self, tmp_home, tmp_path, sample_servers):
        source = self._setup_master(tmp_path, sample_servers)
        args = mcpshare.Disable(servers=("filesystem",))
        mcpshare.cmd_disable(args)
        master = mcpshare.load_master(source)
        assert master["mcpServers"]["filesystem"]["disabled"] is True

    def test_enable_server(self, tmp_home, tmp_path, sample_servers):
        source = self._setup_master(tmp_path, sample_servers)
        # First disable, then enable
        mcpshare.cmd_disable(mcpshare.Disable(servers=("filesystem",)))
        mcpshare.cmd_enable(mcpshare.Enable(servers=("filesystem",)))
        master = mcpshare.load_master(source)
        assert "disabled" not in master["mcpServers"]["filesystem"]

    def test_disable_nonexistent_raises(self, tmp_home, tmp_path, sample_servers):
        self._setup_master(tmp_path, sample_servers)
        with pytest.raises(mcpshare.McpShareError, match="Server not found"):
            mcpshare.cmd_disable(mcpshare.Disable(servers=("nonexistent",)))

    def test_enable_nonexistent_raises(self, tmp_home, tmp_path, sample_servers):
        self._setup_master(tmp_path, sample_servers)
        with pytest.raises(mcpshare.McpShareError, match="Server not found"):
            mcpshare.cmd_enable(mcpshare.Enable(servers=("nonexistent",)))

    def test_disable_idempotent(self, tmp_home, tmp_path, sample_servers):
        source = self._setup_master(tmp_path, sample_servers)
        mcpshare.cmd_disable(mcpshare.Disable(servers=("filesystem",)))
        mcpshare.cmd_disable(mcpshare.Disable(servers=("filesystem",)))
        master = mcpshare.load_master(source)
        assert master["mcpServers"]["filesystem"]["disabled"] is True

    def test_enable_idempotent(self, tmp_home, tmp_path, sample_servers):
        source = self._setup_master(tmp_path, sample_servers)
        # Enable a server that was never disabled
        mcpshare.cmd_enable(mcpshare.Enable(servers=("filesystem",)))
        master = mcpshare.load_master(source)
        assert "disabled" not in master["mcpServers"]["filesystem"]


# ---------------------------------------------------------------------------
# Disabled field handling in writers
# ---------------------------------------------------------------------------


class TestDisabledFieldInWriters:
    @pytest.fixture
    def servers_with_disabled(self, sample_servers):
        s = dict(sample_servers)
        s["filesystem"] = dict(s["filesystem"], disabled=True)
        return s

    def test_write_copilot_preserves_disabled(self, tmp_path, servers_with_disabled):
        path = tmp_path / "mcp-config.json"
        mcpshare.write_copilot(path, servers_with_disabled)
        data = json.loads(path.read_text())
        assert data["mcpServers"]["filesystem"]["disabled"] is True

    def test_write_claude_preserves_disabled(self, tmp_path, servers_with_disabled):
        path = tmp_path / "mcp_servers.json"
        mcpshare.write_claude(path, servers_with_disabled)
        data = json.loads(path.read_text())
        assert data["mcpServers"]["filesystem"]["disabled"] is True

    def test_write_vscode_strips_disabled(self, tmp_path, servers_with_disabled):
        path = tmp_path / "mcp.json"
        mcpshare.write_vscode(path, servers_with_disabled)
        data = json.loads(path.read_text())
        assert "disabled" not in data["servers"]["filesystem"]

    def test_write_codex_strips_disabled(self, tmp_path, servers_with_disabled):
        path = tmp_path / "config.toml"
        mcpshare.write_codex(path, servers_with_disabled)
        content = path.read_text()
        assert "disabled" not in content
        assert "enabled = false" in content

    def test_write_codex_converts_tools_list_to_map(self, tmp_path):
        """Codex expects tools as a TOML map, not an array."""
        path = tmp_path / "config.toml"
        servers = {"azure-mcp": {"command": "npx", "args": ["-y", "@azure/mcp@latest"], "tools": ["*"]}}
        mcpshare.write_codex(path, servers)
        content = path.read_text()
        assert 'tools = { "*" = { enabled = true } }' in content
        assert 'tools = ["*"]' not in content

    def test_read_codex_translates_enabled_false(self, tmp_path):
        """read_codex converts Codex 'enabled = false' to canonical 'disabled = true'."""
        path = tmp_path / "config.toml"
        path.write_text('[mcp_servers.test-mcp]\ncommand = "echo"\nargs = []\nenabled = false\n')
        result = mcpshare.read_codex(path)
        assert result["test-mcp"]["disabled"] is True
        assert "enabled" not in result["test-mcp"]

    def test_read_codex_tools_map_to_list(self, tmp_path):
        """read_codex converts Codex tools map back to canonical list."""
        path = tmp_path / "config.toml"
        path.write_text('[mcp_servers.test-mcp]\ncommand = "echo"\nargs = []\n\n[mcp_servers.test-mcp.tools]\n"*" = {}\n')
        result = mcpshare.read_codex(path)
        assert result["test-mcp"]["tools"] == ["*"]

    def test_write_gemini_strips_disabled(self, tmp_path, servers_with_disabled):
        path = tmp_path / "settings.json"
        mcpshare.write_gemini(path, servers_with_disabled)
        data = json.loads(path.read_text())
        assert "disabled" not in data["mcpServers"]["filesystem"]

    def test_write_opencode_strips_disabled(self, tmp_path, servers_with_disabled):
        path = tmp_path / "opencode.json"
        mcpshare.write_opencode(path, servers_with_disabled)
        data = json.loads(path.read_text())
        assert "disabled" not in data["mcp"]["filesystem"]


# ---------------------------------------------------------------------------
# Collect preserves disabled flag
# ---------------------------------------------------------------------------


class TestCollectPreservesDisabled:
    def test_collect_preserves_disabled_flag(self, tmp_path, sample_servers):
        source = str(tmp_path / "master")
        disabled_servers = dict(sample_servers)
        disabled_servers["filesystem"] = dict(disabled_servers["filesystem"], disabled=True)
        mcpshare.save_master(source, {"mcpServers": disabled_servers})

        # Set up a vscode target that has the same server without disabled
        vscode_dir = tmp_path / "vscode"
        vscode_dir.mkdir()
        vscode_path = vscode_dir / "mcp.json"
        mcpshare.write_vscode(vscode_path, {"filesystem": sample_servers["filesystem"]})

        config = {
            "source": source,
            "mode": "merge",
            "targets": {"vscode": {"path": str(vscode_dir)}},
        }
        master = mcpshare.collect(config)
        assert master["mcpServers"]["filesystem"]["disabled"] is True


# ---------------------------------------------------------------------------
# Server ID sanitization and validation tests
# ---------------------------------------------------------------------------


class TestSanitizeServerId:
    @pytest.mark.parametrize(
        "input_name,expected",
        [
            ("my server", "my-server"),
            ("My_Server", "my-server"),
            ("UPPER_CASE", "upper-case"),
            ("already-kebab", "already-kebab"),
            ("  leading spaces  ", "leading-spaces"),
            ("double__underscore", "double-underscore"),
            ("mixed - _ stuff", "mixed-stuff"),
        ],
    )
    def test_sanitize_server_id(self, input_name, expected):
        assert mcpshare._sanitize_server_id(input_name) == expected


class TestValidateServerId:
    def test_valid_kebab_with_mcp(self):
        assert mcpshare._validate_server_id("epic-stuff-mcp") == []

    def test_valid_mcp_prefix(self):
        assert mcpshare._validate_server_id("mcp-filesystem") == []

    def test_missing_mcp(self):
        warnings = mcpshare._validate_server_id("filesystem")
        assert len(warnings) == 1
        assert "does not contain 'mcp'" in warnings[0]

    def test_not_kebab_case(self):
        warnings = mcpshare._validate_server_id("My_Server")
        assert any("not kebab-case" in w for w in warnings)

    def test_dotted_name_skips_kebab_check(self):
        """Dotted names (e.g. TOML nested keys) skip the kebab-case check."""
        warnings = mcpshare._validate_server_id("microsoft.docs.mcp")
        assert not any("not kebab-case" in w for w in warnings)
