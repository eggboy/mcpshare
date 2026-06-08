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
    def _setup_master(self, tmp_path, sample_servers, *, with_targets=False):
        source = str(tmp_path / "master")
        master = {"mcpServers": sample_servers}
        mcpshare.save_master(source, master)
        targets = {}
        if with_targets:
            for tool in ("vscode", "claude"):
                tdir = tmp_path / tool
                tdir.mkdir(exist_ok=True)
                targets[tool] = {"path": str(tdir)}
        config = {"source": source, "mode": "merge", "targets": targets}
        mcpshare.save_config(config)
        return source

    def test_disable_server_writes_disabled_yaml(self, tmp_home, tmp_path, sample_servers):
        source = self._setup_master(tmp_path, sample_servers)
        mcpshare.cmd_disable(mcpshare.Disable(servers=("filesystem",)))
        data = mcpshare.load_disabled(source)
        assert data == {"*": ["filesystem"]}

    def test_disable_per_ide(self, tmp_home, tmp_path, sample_servers):
        source = self._setup_master(tmp_path, sample_servers, with_targets=True)
        mcpshare.cmd_disable(mcpshare.Disable(servers=("filesystem",), ide="vscode"))
        data = mcpshare.load_disabled(source)
        assert data == {"vscode": ["filesystem"]}

    def test_disable_unknown_ide_raises(self, tmp_home, tmp_path, sample_servers):
        self._setup_master(tmp_path, sample_servers, with_targets=True)
        with pytest.raises(mcpshare.McpShareError, match="Unknown IDE"):
            mcpshare.cmd_disable(mcpshare.Disable(servers=("filesystem",), ide="emacs"))

    def test_disable_vscode_warns(self, tmp_home, tmp_path, sample_servers, caplog):
        """VSCode has no JSON disable mechanism — disable should still record but warn."""
        source = self._setup_master(tmp_path, sample_servers, with_targets=True)
        with caplog.at_level("WARNING"):
            mcpshare.cmd_disable(mcpshare.Disable(servers=("filesystem",), ide="vscode"))
        assert "no JSON disable mechanism" in caplog.text
        assert mcpshare.load_disabled(source) == {"vscode": ["filesystem"]}

    def test_enable_clears_global(self, tmp_home, tmp_path, sample_servers):
        source = self._setup_master(tmp_path, sample_servers)
        mcpshare.cmd_disable(mcpshare.Disable(servers=("filesystem",)))
        mcpshare.cmd_enable(mcpshare.Enable(servers=("filesystem",)))
        assert mcpshare.load_disabled(source) == {}

    def test_enable_clears_only_specified_ide(self, tmp_home, tmp_path, sample_servers):
        source = self._setup_master(tmp_path, sample_servers, with_targets=True)
        mcpshare.cmd_disable(mcpshare.Disable(servers=("filesystem",), ide="vscode"))
        mcpshare.cmd_disable(mcpshare.Disable(servers=("filesystem",), ide="claude"))
        mcpshare.cmd_enable(mcpshare.Enable(servers=("filesystem",), ide="vscode"))
        data = mcpshare.load_disabled(source)
        assert data == {"claude": ["filesystem"]}

    def test_enable_without_ide_clears_everywhere(self, tmp_home, tmp_path, sample_servers):
        source = self._setup_master(tmp_path, sample_servers, with_targets=True)
        mcpshare.cmd_disable(mcpshare.Disable(servers=("filesystem",)))
        mcpshare.cmd_disable(mcpshare.Disable(servers=("filesystem",), ide="vscode"))
        mcpshare.cmd_enable(mcpshare.Enable(servers=("filesystem",)))
        assert mcpshare.load_disabled(source) == {}

    def test_disable_nonexistent_warns_but_records(self, tmp_home, tmp_path, sample_servers, caplog):
        source = self._setup_master(tmp_path, sample_servers)
        with caplog.at_level("WARNING"):
            mcpshare.cmd_disable(mcpshare.Disable(servers=("nonexistent",)))
        assert "not currently in master" in caplog.text
        assert mcpshare.load_disabled(source) == {"*": ["nonexistent"]}

    def test_enable_nonexistent_is_silent_success(self, tmp_home, tmp_path, sample_servers):
        source = self._setup_master(tmp_path, sample_servers)
        # Should not raise even though never disabled.
        mcpshare.cmd_enable(mcpshare.Enable(servers=("nonexistent",)))
        assert mcpshare.load_disabled(source) == {}

    def test_disable_idempotent(self, tmp_home, tmp_path, sample_servers):
        source = self._setup_master(tmp_path, sample_servers)
        mcpshare.cmd_disable(mcpshare.Disable(servers=("filesystem",)))
        mcpshare.cmd_disable(mcpshare.Disable(servers=("filesystem",)))
        assert mcpshare.load_disabled(source) == {"*": ["filesystem"]}

    def test_enable_idempotent(self, tmp_home, tmp_path, sample_servers):
        source = self._setup_master(tmp_path, sample_servers)
        mcpshare.cmd_enable(mcpshare.Enable(servers=("filesystem",)))
        assert mcpshare.load_disabled(source) == {}


# ---------------------------------------------------------------------------
# Disabled field handling in writers
# ---------------------------------------------------------------------------


class TestDisabledKeyRemoval:
    """Per-entry 'disabled' state is owned exclusively by mcp.disabled.yaml.

    Defense-in-depth: every reader drops any on-disk 'disabled' key so
    legacy or hand-edited files can't smuggle it into the master.
    """

    @pytest.fixture
    def on_disk_with_disabled(self, sample_servers):
        s = dict(sample_servers)
        s["filesystem"] = dict(s["filesystem"], disabled=True)
        return s

    def test_read_vscode_drops_disabled(self, tmp_path, on_disk_with_disabled):
        path = tmp_path / "mcp.json"
        path.write_text(json.dumps({"servers": {n: {"type": "stdio", **c} for n, c in on_disk_with_disabled.items()}}))
        result = mcpshare.read_vscode(path)
        assert "disabled" not in result["filesystem"]

    def test_read_claude_drops_disabled(self, tmp_path, on_disk_with_disabled):
        path = tmp_path / "mcp_servers.json"
        path.write_text(json.dumps({"mcpServers": on_disk_with_disabled}))
        result = mcpshare.read_claude(path)
        assert "disabled" not in result["filesystem"]

    def test_read_copilot_drops_disabled(self, tmp_path, on_disk_with_disabled):
        path = tmp_path / "mcp-config.json"
        path.write_text(
            json.dumps({"mcpServers": {n: {"type": "stdio", **c} for n, c in on_disk_with_disabled.items()}})
        )
        result = mcpshare.read_copilot(path)
        assert "disabled" not in result["filesystem"]

    def test_read_gemini_drops_disabled(self, tmp_path, on_disk_with_disabled):
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"mcpServers": on_disk_with_disabled}))
        result = mcpshare.read_gemini(path)
        assert "disabled" not in result["filesystem"]

    def test_read_codex_drops_enabled_field(self, tmp_path):
        """read_codex strips 'enabled = false' without translating it.

        Disable state lives in mcp.disabled.yaml; Codex's native flag must
        not bleed into the canonical master.
        """
        path = tmp_path / "config.toml"
        path.write_text('[mcp_servers.test-mcp]\ncommand = "echo"\nargs = []\nenabled = false\n')
        result = mcpshare.read_codex(path)
        assert "enabled" not in result["test-mcp"]
        assert "disabled" not in result["test-mcp"]

    def test_write_codex_converts_tools_list_to_map(self, tmp_path):
        """Codex expects tools as a TOML map, not an array."""
        path = tmp_path / "config.toml"
        servers = {"azure-mcp": {"command": "npx", "args": ["-y", "@azure/mcp@latest"], "tools": ["*"]}}
        mcpshare.write_codex(path, servers)
        content = path.read_text()
        assert 'tools = { "*" = { enabled = true } }' in content
        assert 'tools = ["*"]' not in content

    def test_write_codex_emits_enabled_false_for_disabled_names(self, tmp_path, sample_servers):
        """write_codex marks names in disabled_names with 'enabled = false'."""
        path = tmp_path / "config.toml"
        mcpshare.write_codex(path, sample_servers, disabled_names={"filesystem"})
        content = path.read_text()
        assert "[mcp_servers.filesystem]" in content
        assert "enabled = false" in content
        # github is not disabled
        github_block = content.split("[mcp_servers.github]")[1].split("[mcp_servers")[0]
        assert "enabled = false" not in github_block

    def test_read_codex_tools_map_to_list(self, tmp_path):
        """read_codex converts Codex tools map back to canonical list."""
        path = tmp_path / "config.toml"
        path.write_text(
            '[mcp_servers.test-mcp]\ncommand = "echo"\nargs = []\n\n[mcp_servers.test-mcp.tools]\n"*" = {}\n'
        )
        result = mcpshare.read_codex(path)
        assert result["test-mcp"]["tools"] == ["*"]


# ---------------------------------------------------------------------------
# Collect preserves disabled flag
# ---------------------------------------------------------------------------


class TestCollectStripsDisabled:
    def test_collect_strips_disabled_from_master(self, tmp_path, sample_servers):
        """Master's per-entry 'disabled' flag is stripped on collect.

        mcp.disabled.yaml is the sole source of truth; the master must never
        carry a 'disabled' key after a pull.
        """
        source = str(tmp_path / "master")
        seeded = dict(sample_servers)
        seeded["filesystem"] = dict(seeded["filesystem"], disabled=True)
        # Write directly: save_master would strip it for us, so bypass.
        Path(source).mkdir(parents=True, exist_ok=True)
        (Path(source) / mcpshare.MASTER_FILENAME).write_text(json.dumps({"mcpServers": seeded}))

        vscode_dir = tmp_path / "vscode"
        vscode_dir.mkdir()
        mcpshare.write_vscode(vscode_dir / "mcp.json", {"filesystem": sample_servers["filesystem"]})

        config = {
            "source": source,
            "mode": "merge",
            "targets": {"vscode": {"path": str(vscode_dir)}},
        }
        master = mcpshare.collect(config)
        assert "disabled" not in master["mcpServers"]["filesystem"]


# ---------------------------------------------------------------------------
# Conflict resolution during collect
# ---------------------------------------------------------------------------


class TestCollectConflicts:
    def _config_with_two_vscode_like_targets(self, tmp_path):
        """Two targets that both use the canonical vscode JSON layout.

        We reuse `vscode` and `copilot` because they are the only two readers
        currently registered in `READERS`.
        """
        source = str(tmp_path / "master")
        targets = {}
        for tool in ("vscode", "copilot"):
            tdir = tmp_path / tool
            tdir.mkdir()
            targets[tool] = {"path": str(tdir)}
        return {"source": source, "mode": "merge", "targets": targets}

    def _seed(self, tmp_path, vscode_cfg, copilot_cfg):
        config = self._config_with_two_vscode_like_targets(tmp_path)
        vscode_path = mcpshare.resolve_target_path("vscode", config["targets"]["vscode"]["path"])
        copilot_path = mcpshare.resolve_target_path("copilot", config["targets"]["copilot"]["path"])
        mcpshare.write_vscode(vscode_path, {"shared-mcp": vscode_cfg})
        mcpshare.write_copilot(copilot_path, {"shared-mcp": copilot_cfg})
        return config

    def test_no_prompt_when_configs_match(self, tmp_path, monkeypatch):
        cfg = {"command": "npx", "args": ["-y", "foo"]}
        config = self._seed(tmp_path, cfg, cfg)

        def fail_prompt(*_a, **_kw):
            raise AssertionError("should not prompt when configs are identical")

        monkeypatch.setattr(mcpshare, "_prompt_conflict", fail_prompt)
        monkeypatch.setattr(mcpshare.sys.stdin, "isatty", lambda: True)

        master = mcpshare.collect(config, interactive=True)
        assert master["mcpServers"]["shared-mcp"] == cfg

    def test_prompt_keep_current(self, tmp_path, monkeypatch):
        vscode_cfg = {"command": "npx", "args": ["-y", "vscode-version"]}
        copilot_cfg = {"command": "npx", "args": ["-y", "copilot-version"]}
        config = self._seed(tmp_path, vscode_cfg, copilot_cfg)

        calls = []

        def fake_prompt(name, cur_src, cur_cfg, new_src, new_cfg):
            calls.append((name, cur_src, new_src))
            return False  # keep current (vscode)

        monkeypatch.setattr(mcpshare, "_prompt_conflict", fake_prompt)
        monkeypatch.setattr(mcpshare.sys.stdin, "isatty", lambda: True)

        master = mcpshare.collect(config, interactive=True)
        assert calls == [("shared-mcp", "vscode", "copilot")]
        assert master["mcpServers"]["shared-mcp"] == vscode_cfg

    def test_prompt_use_new(self, tmp_path, monkeypatch):
        vscode_cfg = {"command": "npx", "args": ["-y", "vscode-version"]}
        copilot_cfg = {"command": "npx", "args": ["-y", "copilot-version"]}
        config = self._seed(tmp_path, vscode_cfg, copilot_cfg)

        monkeypatch.setattr(mcpshare, "_prompt_conflict", lambda *a, **kw: True)
        monkeypatch.setattr(mcpshare.sys.stdin, "isatty", lambda: True)

        master = mcpshare.collect(config, interactive=True)
        assert master["mcpServers"]["shared-mcp"] == copilot_cfg

    def test_non_interactive_silently_takes_later(self, tmp_path, monkeypatch):
        vscode_cfg = {"command": "npx", "args": ["-y", "vscode-version"]}
        copilot_cfg = {"command": "npx", "args": ["-y", "copilot-version"]}
        config = self._seed(tmp_path, vscode_cfg, copilot_cfg)

        def fail_prompt(*_a, **_kw):
            raise AssertionError("should not prompt in non-interactive mode")

        monkeypatch.setattr(mcpshare, "_prompt_conflict", fail_prompt)
        monkeypatch.setattr(mcpshare.sys.stdin, "isatty", lambda: True)

        master = mcpshare.collect(config, interactive=False)
        # Later target (copilot) wins, matching legacy last-write semantics.
        assert master["mcpServers"]["shared-mcp"] == copilot_cfg

    def test_no_tty_skips_prompt(self, tmp_path, monkeypatch):
        vscode_cfg = {"command": "npx", "args": ["-y", "vscode-version"]}
        copilot_cfg = {"command": "npx", "args": ["-y", "copilot-version"]}
        config = self._seed(tmp_path, vscode_cfg, copilot_cfg)

        def fail_prompt(*_a, **_kw):
            raise AssertionError("should not prompt without a TTY")

        monkeypatch.setattr(mcpshare, "_prompt_conflict", fail_prompt)
        monkeypatch.setattr(mcpshare.sys.stdin, "isatty", lambda: False)

        master = mcpshare.collect(config, interactive=True)
        assert master["mcpServers"]["shared-mcp"] == copilot_cfg

    def test_disabled_flag_only_is_not_a_conflict(self, tmp_path, monkeypatch):
        cfg = {"command": "npx", "args": ["-y", "foo"]}
        source = str(tmp_path / "master")
        # Master has the server marked disabled (legacy on-disk state).
        Path(source).mkdir(parents=True, exist_ok=True)
        (Path(source) / mcpshare.MASTER_FILENAME).write_text(
            json.dumps({"mcpServers": {"foo-mcp": dict(cfg, disabled=True)}})
        )
        vscode_dir = tmp_path / "vscode"
        vscode_dir.mkdir()
        mcpshare.write_vscode(vscode_dir / "mcp.json", {"foo-mcp": cfg})
        config = {
            "source": source,
            "mode": "merge",
            "targets": {"vscode": {"path": str(vscode_dir)}},
        }

        def fail_prompt(*_a, **_kw):
            raise AssertionError("disabled-only diff should not be treated as a conflict")

        monkeypatch.setattr(mcpshare, "_prompt_conflict", fail_prompt)
        monkeypatch.setattr(mcpshare.sys.stdin, "isatty", lambda: True)

        master = mcpshare.collect(config, interactive=True)
        # The flag is stripped during collect; mcp.disabled.yaml owns disable state.
        assert "disabled" not in master["mcpServers"]["foo-mcp"]


# ---------------------------------------------------------------------------
# Distribute: file deletion and per-IDE disabled list filtering
# ---------------------------------------------------------------------------


class TestDistributeBehavior:
    def _make_config(self, tmp_path, tools=("vscode", "copilot")):
        targets = {}
        for tool in tools:
            tdir = tmp_path / tool
            tdir.mkdir(exist_ok=True)
            targets[tool] = {"path": str(tdir)}
        source = tmp_path / "master"
        source.mkdir(exist_ok=True)
        return {"source": str(source), "mode": "merge", "targets": targets}

    def test_dedicated_file_is_recreated_not_merged(self, tmp_home, tmp_path, sample_servers):
        """An old VSCode mcp.json with stale top-level keys is removed before write."""
        config = self._make_config(tmp_path, tools=("vscode",))
        vscode_path = mcpshare.resolve_target_path("vscode", config["targets"]["vscode"]["path"])
        # Seed with an unrelated stale top-level key.
        vscode_path.write_text(json.dumps({"servers": {"stale-mcp": {"command": "old"}}, "stale_top_key": 1}))
        master = {"mcpServers": {"filesystem": sample_servers["filesystem"]}}

        mcpshare.distribute(config, master)

        data = json.loads(vscode_path.read_text())
        assert "stale_top_key" not in data, "dedicated MCP file must be wiped before write"
        assert set(data["servers"]) == {"filesystem"}

    # ---- Gemini native disable: mcp.excluded in settings.json ----

    def test_gemini_writes_disabled_to_mcp_excluded(self, tmp_home, tmp_path, sample_servers):
        """Gemini's native disable: keep the server in mcpServers, list it in mcp.excluded."""
        config = self._make_config(tmp_path, tools=("gemini",))
        mcpshare.save_config(config)
        mcpshare.save_disabled(config["source"], {"*": ["filesystem"]})
        mcpshare.distribute(config, {"mcpServers": sample_servers})

        gemini_data = json.loads(
            mcpshare.resolve_target_path("gemini", config["targets"]["gemini"]["path"]).read_text()
        )
        # Server is written normally — it stays visible to Gemini's UI.
        assert "filesystem" in gemini_data["mcpServers"]
        # Disable is applied via the native blocklist.
        assert gemini_data["mcp"]["excluded"] == ["filesystem"]

    def test_gemini_preserves_other_mcp_keys(self, tmp_home, tmp_path, sample_servers):
        """Updating mcp.excluded must not clobber other mcp.* keys (e.g. allowed)."""
        config = self._make_config(tmp_path, tools=("gemini",))
        mcpshare.save_config(config)
        gemini_path = mcpshare.resolve_target_path("gemini", config["targets"]["gemini"]["path"])
        gemini_path.parent.mkdir(parents=True, exist_ok=True)
        gemini_path.write_text(json.dumps({"theme": "dark", "mcp": {"allowed": ["my-trusted"]}}))
        mcpshare.save_disabled(config["source"], {"*": ["filesystem"]})

        mcpshare.distribute(config, {"mcpServers": sample_servers})

        data = json.loads(gemini_path.read_text())
        assert data["theme"] == "dark"
        assert data["mcp"]["allowed"] == ["my-trusted"]
        assert data["mcp"]["excluded"] == ["filesystem"]

    def test_gemini_drops_mcp_section_when_nothing_disabled(self, tmp_home, tmp_path, sample_servers):
        """No disabled servers and no other mcp.* keys → the mcp section is removed."""
        config = self._make_config(tmp_path, tools=("gemini",))
        mcpshare.save_config(config)

        mcpshare.distribute(config, {"mcpServers": sample_servers})

        data = json.loads(mcpshare.resolve_target_path("gemini", config["targets"]["gemini"]["path"]).read_text())
        assert "filesystem" in data["mcpServers"]
        assert "mcp" not in data

    def test_gemini_per_ide_disable_isolated(self, tmp_home, tmp_path, sample_servers):
        """A gemini-only disable doesn't affect opencode's payload."""
        config = self._make_config(tmp_path, tools=("gemini", "opencode"))
        mcpshare.save_config(config)
        mcpshare.save_disabled(config["source"], {"gemini": ["filesystem"]})
        mcpshare.distribute(config, {"mcpServers": sample_servers})

        gemini_data = json.loads(
            mcpshare.resolve_target_path("gemini", config["targets"]["gemini"]["path"]).read_text()
        )
        opencode_data = json.loads(
            mcpshare.resolve_target_path("opencode", config["targets"]["opencode"]["path"]).read_text()
        )
        assert gemini_data["mcp"]["excluded"] == ["filesystem"]
        assert opencode_data["mcp"]["filesystem"].get("enabled") is not False

    # ---- OpenCode native disable: per-entry "enabled": false ----

    def test_opencode_writes_disabled_with_enabled_false(self, tmp_home, tmp_path, sample_servers):
        """OpenCode's native disable: per-entry "enabled": false."""
        config = self._make_config(tmp_path, tools=("opencode",))
        mcpshare.save_config(config)
        mcpshare.save_disabled(config["source"], {"*": ["filesystem"]})
        mcpshare.distribute(config, {"mcpServers": sample_servers})

        opencode_data = json.loads(
            mcpshare.resolve_target_path("opencode", config["targets"]["opencode"]["path"]).read_text()
        )
        # Both servers are written.
        assert "filesystem" in opencode_data["mcp"]
        assert "github" in opencode_data["mcp"]
        # Only the disabled one carries enabled = false.
        assert opencode_data["mcp"]["filesystem"]["enabled"] is False
        assert "enabled" not in opencode_data["mcp"]["github"]

    # ---- "none" strategy (vscode: write everything as enabled) ----

    def test_vscode_writes_disabled_server_as_enabled(self, tmp_home, tmp_path, sample_servers):
        """VSCode has no JSON disable; disabled servers are written as-enabled."""
        config = self._make_config(tmp_path, tools=("vscode",))
        mcpshare.save_config(config)
        mcpshare.save_disabled(config["source"], {"*": ["filesystem"]})

        mcpshare.distribute(config, {"mcpServers": sample_servers})

        vscode_data = json.loads(
            mcpshare.resolve_target_path("vscode", config["targets"]["vscode"]["path"]).read_text()
        )
        assert "filesystem" in vscode_data["servers"], "VSCode receives every server"
        assert "disabled" not in vscode_data["servers"]["filesystem"]

    def test_vscode_ignores_per_ide_disable(self, tmp_home, tmp_path, sample_servers):
        """A vscode-specific disable entry is ignored on sync (VSCode = 'none')."""
        config = self._make_config(tmp_path, tools=("vscode",))
        mcpshare.save_config(config)
        mcpshare.save_disabled(config["source"], {"vscode": ["filesystem"]})

        mcpshare.distribute(config, {"mcpServers": sample_servers})

        vscode_data = json.loads(
            mcpshare.resolve_target_path("vscode", config["targets"]["vscode"]["path"]).read_text()
        )
        assert "filesystem" in vscode_data["servers"]

    # ---- in_file_flag strategy (codex) ----

    def test_codex_writes_disabled_with_enabled_false(self, tmp_home, tmp_path, sample_servers):
        config = self._make_config(tmp_path, tools=("codex",))
        mcpshare.save_config(config)
        mcpshare.save_disabled(config["source"], {"*": ["filesystem"]})

        mcpshare.distribute(config, {"mcpServers": sample_servers})

        codex_path = mcpshare.resolve_target_path("codex", config["targets"]["codex"]["path"])
        content = codex_path.read_text()
        assert "[mcp_servers.filesystem]" in content
        assert "[mcp_servers.github]" in content
        # Only the filesystem block carries 'enabled = false' (Codex's native disable).
        fs_block = content.split("[mcp_servers.filesystem]")[1].split("[mcp_servers")[0]
        gh_block = content.split("[mcp_servers.github]")[1].split("[mcp_servers")[0]
        assert "enabled = false" in fs_block
        assert "enabled = false" not in gh_block

    # ---- settings_json strategy (claude, copilot) ----

    def test_claude_writes_disabled_to_settings_json(self, tmp_home, tmp_path, sample_servers):
        """Disabled servers appear in mcp_servers.json AND in settings.json."""
        config = self._make_config(tmp_path, tools=("claude",))
        mcpshare.save_config(config)
        mcpshare.save_disabled(config["source"], {"*": ["filesystem"]})

        mcpshare.distribute(config, {"mcpServers": sample_servers})

        claude_dir = Path(config["targets"]["claude"]["path"])
        servers_data = json.loads((claude_dir / "mcp_servers.json").read_text())
        settings_data = json.loads((claude_dir / "settings.json").read_text())

        # Server is written as normal.
        assert "filesystem" in servers_data["mcpServers"]
        assert "disabled" not in servers_data["mcpServers"]["filesystem"]
        # Native disable list reflects mcpshare's view.
        assert settings_data["disabledMcpServers"] == ["filesystem"]

    def test_copilot_writes_disabled_to_settings_json(self, tmp_home, tmp_path, sample_servers):
        config = self._make_config(tmp_path, tools=("copilot",))
        mcpshare.save_config(config)
        mcpshare.save_disabled(config["source"], {"*": ["filesystem"]})

        mcpshare.distribute(config, {"mcpServers": sample_servers})

        copilot_dir = Path(config["targets"]["copilot"]["path"])
        servers_data = json.loads((copilot_dir / "mcp-config.json").read_text())
        settings_data = json.loads((copilot_dir / "settings.json").read_text())

        assert "filesystem" in servers_data["mcpServers"]
        assert "disabled" not in servers_data["mcpServers"]["filesystem"]
        assert settings_data["disabledMcpServers"] == ["filesystem"]

    def test_settings_json_preserves_unrelated_keys(self, tmp_home, tmp_path, sample_servers):
        """Updating disabledMcpServers must not clobber other settings.json keys."""
        config = self._make_config(tmp_path, tools=("claude",))
        mcpshare.save_config(config)
        claude_dir = Path(config["targets"]["claude"]["path"])
        # Seed settings.json with unrelated keys the user owns.
        (claude_dir / "settings.json").write_text(json.dumps({"theme": "dark", "permissions": {"allow": ["Read(*)"]}}))
        mcpshare.save_disabled(config["source"], {"*": ["filesystem"]})

        mcpshare.distribute(config, {"mcpServers": sample_servers})

        settings_data = json.loads((claude_dir / "settings.json").read_text())
        assert settings_data["theme"] == "dark"
        assert settings_data["permissions"] == {"allow": ["Read(*)"]}
        assert settings_data["disabledMcpServers"] == ["filesystem"]

    def test_settings_json_overwrites_native_disabled_list(self, tmp_home, tmp_path, sample_servers):
        """mcpshare-owns: the disabledMcpServers array is replaced, not merged."""
        config = self._make_config(tmp_path, tools=("claude",))
        mcpshare.save_config(config)
        claude_dir = Path(config["targets"]["claude"]["path"])
        # User had a different server disabled natively.
        (claude_dir / "settings.json").write_text(json.dumps({"disabledMcpServers": ["github", "some-orphan"]}))
        # mcpshare disables only filesystem; github stays enabled.
        mcpshare.save_disabled(config["source"], {"*": ["filesystem"]})

        mcpshare.distribute(config, {"mcpServers": sample_servers})

        settings_data = json.loads((claude_dir / "settings.json").read_text())
        assert settings_data["disabledMcpServers"] == ["filesystem"]

    def test_settings_json_re_enable_warning(self, tmp_home, tmp_path, sample_servers, caplog):
        """When mcpshare re-enables a natively-disabled server, log a warning."""
        config = self._make_config(tmp_path, tools=("claude",))
        mcpshare.save_config(config)
        claude_dir = Path(config["targets"]["claude"]["path"])
        # User had 'github' disabled natively in Claude.
        (claude_dir / "settings.json").write_text(json.dumps({"disabledMcpServers": ["github"]}))
        # mcpshare doesn't know about that disable.
        with caplog.at_level("WARNING"):
            mcpshare.distribute(config, {"mcpServers": sample_servers})
        assert "re-enabling github" in caplog.text

    def test_settings_json_removed_when_no_disabled(self, tmp_home, tmp_path, sample_servers):
        """If no servers are disabled, the disabledMcpServers key is removed."""
        config = self._make_config(tmp_path, tools=("claude",))
        mcpshare.save_config(config)
        claude_dir = Path(config["targets"]["claude"]["path"])
        (claude_dir / "settings.json").write_text(json.dumps({"theme": "dark"}))

        mcpshare.distribute(config, {"mcpServers": sample_servers})

        settings_data = json.loads((claude_dir / "settings.json").read_text())
        assert "disabledMcpServers" not in settings_data
        assert settings_data["theme"] == "dark"

    # ---- master.disabled is ignored: only mcp.disabled.yaml matters ----

    def test_master_disabled_flag_is_ignored_by_gemini(self, tmp_home, tmp_path, sample_servers):
        """A stray 'disabled: true' on master has no effect on Gemini's output."""
        config = self._make_config(tmp_path, tools=("gemini",))
        mcpshare.save_config(config)
        seeded = dict(sample_servers)
        seeded["filesystem"] = dict(seeded["filesystem"], disabled=True)

        mcpshare.distribute(config, {"mcpServers": seeded})

        gemini_data = json.loads(
            mcpshare.resolve_target_path("gemini", config["targets"]["gemini"]["path"]).read_text()
        )
        # mcp.disabled.yaml is empty, so filesystem must be written enabled.
        assert "filesystem" in gemini_data["mcpServers"]
        assert "mcp" not in gemini_data, "no disabled entries → no mcp.excluded"

    def test_master_disabled_flag_is_ignored_settings_json_strategy(self, tmp_home, tmp_path, sample_servers):
        """A stray 'disabled: true' on master never reaches settings.json."""
        config = self._make_config(tmp_path, tools=("claude",))
        mcpshare.save_config(config)
        seeded = dict(sample_servers)
        seeded["filesystem"] = dict(seeded["filesystem"], disabled=True)

        mcpshare.distribute(config, {"mcpServers": seeded})

        claude_dir = Path(config["targets"]["claude"]["path"])
        settings_path = claude_dir / "settings.json"
        assert not settings_path.exists() or "disabledMcpServers" not in json.loads(settings_path.read_text())

    def test_disabled_persists_across_sync_after_master_regen(self, tmp_home, tmp_path, sample_servers):
        """Core scenario: master is wiped and regenerated; mcp.disabled.yaml still hides it."""
        config = self._make_config(tmp_path, tools=("gemini",))
        mcpshare.save_config(config)
        mcpshare.save_disabled(config["source"], {"*": ["filesystem"]})

        master_path = Path(config["source"]) / mcpshare.MASTER_FILENAME
        master_path.write_text(json.dumps({"mcpServers": sample_servers}))

        master = mcpshare.load_master(config["source"])
        mcpshare.distribute(config, master)

        gemini_data = json.loads(
            mcpshare.resolve_target_path("gemini", config["targets"]["gemini"]["path"]).read_text()
        )
        # Server is written, but recorded in mcp.excluded (Gemini's native disable).
        assert "filesystem" in gemini_data["mcpServers"]
        assert gemini_data["mcp"]["excluded"] == ["filesystem"]


# ---------------------------------------------------------------------------
# Distribute helpers (unit-level)
# ---------------------------------------------------------------------------


class TestWriteNativeDisabledList:
    """Unit tests for the read-modify-write of settings.json."""

    def test_creates_new_file_with_disabled_list(self, tmp_path):
        path = tmp_path / "settings.json"
        previous = mcpshare._write_native_disabled_list(path, {"foo-mcp"})
        assert previous == set()
        assert json.loads(path.read_text()) == {"disabledMcpServers": ["foo-mcp"]}

    def test_preserves_unrelated_keys(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"theme": "dark", "permissions": ["Read"]}))

        mcpshare._write_native_disabled_list(path, {"foo-mcp", "bar-mcp"})

        data = json.loads(path.read_text())
        assert data["theme"] == "dark"
        assert data["permissions"] == ["Read"]
        assert data["disabledMcpServers"] == ["bar-mcp", "foo-mcp"]

    def test_empty_set_removes_key(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"theme": "dark", "disabledMcpServers": ["old-mcp"]}))

        mcpshare._write_native_disabled_list(path, set())

        data = json.loads(path.read_text())
        assert "disabledMcpServers" not in data
        assert data["theme"] == "dark"

    def test_returns_previous_names(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"disabledMcpServers": ["a-mcp", "b-mcp"]}))

        previous = mcpshare._write_native_disabled_list(path, {"c-mcp"})

        assert previous == {"a-mcp", "b-mcp"}

    def test_handles_non_list_previous_value(self, tmp_path):
        """Pre-existing settings.json with a wrong-type value shouldn't crash."""
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"disabledMcpServers": "garbage"}))

        previous = mcpshare._write_native_disabled_list(path, {"a-mcp"})

        assert previous == set()
        assert json.loads(path.read_text())["disabledMcpServers"] == ["a-mcp"]


# ---------------------------------------------------------------------------
# Disabled persistence helpers
# ---------------------------------------------------------------------------


class TestDisabledHelpers:
    def test_load_disabled_missing_returns_empty(self, tmp_home, tmp_path):
        source = str(tmp_path / "master")
        assert mcpshare.load_disabled(source) == {}

    def test_save_and_load_roundtrip(self, tmp_home, tmp_path):
        source = str(tmp_path / "master")
        mcpshare.save_disabled(source, {"*": ["a", "b"], "vscode": ["c"]})
        assert mcpshare.load_disabled(source) == {"*": ["a", "b"], "vscode": ["c"]}

    def test_save_empty_removes_file(self, tmp_home, tmp_path):
        source = str(tmp_path / "master")
        mcpshare.save_disabled(source, {"*": ["a"]})
        assert mcpshare.disabled_file_path(source).exists()
        mcpshare.save_disabled(source, {"*": []})
        assert not mcpshare.disabled_file_path(source).exists()

    def test_disabled_file_lives_next_to_master(self, tmp_home, tmp_path):
        source = str(tmp_path / "some" / "nested" / "master")
        mcpshare.save_disabled(source, {"*": ["a"]})
        # File should be created at <source>/mcp.disabled.yaml, not in CONFIG_DIR.
        assert (Path(source) / mcpshare.DISABLED_FILENAME).exists()

    def test_disabled_for_tool_unions_global_and_specific(self):
        data = {"*": ["a"], "vscode": ["b"], "claude": ["c"]}
        assert mcpshare.disabled_for_tool(data, "vscode") == {"a", "b"}
        assert mcpshare.disabled_for_tool(data, "claude") == {"a", "c"}
        assert mcpshare.disabled_for_tool(data, "gemini") == {"a"}

    def test_load_disabled_invalid_yaml_raises(self, tmp_home, tmp_path):
        source = tmp_path / "master"
        source.mkdir()
        (source / mcpshare.DISABLED_FILENAME).write_text("not: [valid")
        with pytest.raises(mcpshare.ConfigParseError):
            mcpshare.load_disabled(str(source))

    def test_legacy_disabled_migrates_on_first_load(self, tmp_home, tmp_path):
        """An old ~/.config/mcpshare/disabled.yaml is moved next to the master."""
        source = tmp_path / "master"
        source.mkdir()
        legacy_path = mcpshare._legacy_disabled_path()
        legacy_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_path.write_text("'*':\n  - legacy-server\n")

        data = mcpshare.load_disabled(str(source))

        assert data == {"*": ["legacy-server"]}
        assert not legacy_path.exists(), "legacy file should be removed after migration"
        assert (source / mcpshare.DISABLED_FILENAME).exists()

    def test_legacy_disabled_not_migrated_if_new_exists(self, tmp_home, tmp_path):
        """If the new location already exists, the legacy file is left alone."""
        source = tmp_path / "master"
        source.mkdir()
        (source / mcpshare.DISABLED_FILENAME).write_text("'*':\n  - new-server\n")
        legacy_path = mcpshare._legacy_disabled_path()
        legacy_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_path.write_text("'*':\n  - legacy-server\n")

        data = mcpshare.load_disabled(str(source))

        assert data == {"*": ["new-server"]}
        assert legacy_path.exists(), "legacy file is preserved when new file already exists"


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
