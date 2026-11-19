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
        assert cfg["init-source"] == ["vscode", "copilot"]
        assert "mode" in cfg
        assert "targets" in cfg
        assert cfg["mode"] == "merge"
        for tool in ("claude", "vscode", "copilot", "codex", "gemini", "antigravity", "opencode"):
            assert tool in cfg["targets"]
            assert "path" in cfg["targets"][tool]

    def test_default_claude_target_is_config_dir(self):
        cfg = mcpshare.default_config()
        assert cfg["targets"]["claude"]["path"] == str(Path.home() / ".claude")

    def test_save_and_load_config(self, tmp_home):
        cfg = {"source": "/tmp/master", "init-source": ["vscode", "claude"], "mode": "merge", "targets": {}}
        mcpshare.save_config(cfg)
        loaded = mcpshare.load_config()
        assert loaded == cfg
        assert not mcpshare.CONFIG_FILE.with_name(mcpshare.CONFIG_FILE.name + ".tmp").exists()

    def test_load_config_raises_when_missing(self, tmp_home):
        """load_config raises ConfigNotFoundError when no config exists."""
        with pytest.raises(mcpshare.ConfigNotFoundError, match="Config not found"):
            mcpshare.load_config()

    def test_load_config_raises_on_invalid_yaml(self, tmp_home):
        mcpshare.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        mcpshare.CONFIG_FILE.write_text(":::invalid: [yaml")
        with pytest.raises(mcpshare.ConfigParseError, match="Invalid YAML"):
            mcpshare.load_config()

    @pytest.mark.parametrize(
        "config,expected",
        [
            pytest.param(
                {"targets": {"vscode": {}, "copilot": {}, "claude": {}}},
                ["vscode", "copilot"],
                id="omitted-falls-back-to-defaults",
            ),
            pytest.param(
                {"targets": {"claude": {}}},
                [],
                id="omitted-filters-to-configured-targets",
            ),
            pytest.param(
                {"init-source": ["vscode", "claude", "vscode"], "targets": {}},
                ["vscode", "claude"],
                id="explicit-preserves-order-and-deduplicates",
            ),
            pytest.param({"init-source": [], "targets": {"vscode": {}}}, [], id="explicit-empty-list"),
            pytest.param({"init-source": None, "targets": {"vscode": {}}}, [], id="explicit-null"),
        ],
    )
    def test_resolve_init_sources(self, config, expected):
        assert mcpshare._resolve_init_sources(config) == expected

    @pytest.mark.parametrize("bad_value", ["vscode", {"vscode": True}, [123]])
    def test_resolve_init_sources_rejects_non_string_list(self, bad_value):
        with pytest.raises(mcpshare.ConfigParseError, match="init-source"):
            mcpshare._resolve_init_sources({"init-source": bad_value, "targets": {}})


# ---------------------------------------------------------------------------
# Master config tests
# ---------------------------------------------------------------------------


class TestResolveTargetPath:
    def test_claude_default_dir_maps_to_home_claude_json(self, tmp_path, monkeypatch):
        """~/.claude → ~/.claude.json, where Claude Code keeps user-scope servers."""
        monkeypatch.setenv("HOME", str(tmp_path))
        result = mcpshare.resolve_target_path("claude", str(tmp_path / ".claude"))
        assert result == tmp_path / ".claude.json"

    def test_claude_default_dir_with_tilde(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        assert mcpshare.resolve_target_path("claude", "~/.claude") == tmp_path / ".claude.json"

    def test_claude_custom_dir_holds_its_own_claude_json(self, tmp_path, monkeypatch):
        """A non-default dir behaves like CLAUDE_CONFIG_DIR: <dir>/.claude.json."""
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        custom = tmp_path / "claude-config"
        assert mcpshare.resolve_target_path("claude", str(custom)) == custom / ".claude.json"

    def test_other_tools_join_filename(self, tmp_path):
        assert mcpshare.resolve_target_path("copilot", str(tmp_path)) == tmp_path / "mcp-config.json"


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

    def test_load_master_empty_file_returns_empty_servers(self, tmp_path):
        """An empty (e.g. freshly touched) master file is not an error."""
        source = tmp_path / "empty_master"
        source.mkdir()
        (source / "mcp.json").write_text("")
        assert mcpshare.load_master(str(source)) == {"mcpServers": {}}


# ---------------------------------------------------------------------------
# JSON file helper tests
# ---------------------------------------------------------------------------


class TestLoadJsonObject:
    def test_missing_file_returns_empty_object(self, tmp_path):
        assert mcpshare._load_json_object(tmp_path / "nope.json") == {}

    @pytest.mark.parametrize("content", ["", "\n", "   \n\t  "])
    def test_empty_or_whitespace_file_returns_empty_object(self, tmp_path, content):
        """Antigravity creates mcp_config.json as a 0-byte placeholder; that must parse as {}."""
        path = tmp_path / "mcp_config.json"
        path.write_text(content)
        assert mcpshare._load_json_object(path) == {}

    def test_returns_parsed_object(self, tmp_path):
        path = tmp_path / "f.json"
        path.write_text(json.dumps({"a": {"b": 1}}))
        assert mcpshare._load_json_object(path) == {"a": {"b": 1}}

    def test_invalid_json_raises_with_path(self, tmp_path):
        path = tmp_path / "f.json"
        path.write_text("{not valid json")
        with pytest.raises(mcpshare.ConfigParseError, match=f"Invalid JSON in {path}"):
            mcpshare._load_json_object(path)

    @pytest.mark.parametrize("content", ["[]", "null", '"text"', "42"])
    def test_non_object_top_level_raises(self, tmp_path, content):
        """A writer can't merge mcpServers into a list/null/scalar, so refuse rather than clobber."""
        path = tmp_path / "f.json"
        path.write_text(content)
        with pytest.raises(mcpshare.ConfigParseError, match="Expected JSON object"):
            mcpshare._load_json_object(path)


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
        path = tmp_path / ".claude.json"
        data = {
            "numStartups": 3,
            "mcpServers": {"github": {"type": "stdio", "command": "npx", "args": ["server-gh"], "env": {}}},
        }
        path.write_text(json.dumps(data))
        result = mcpshare.read_claude(path)
        assert result == {"github": {"command": "npx", "args": ["server-gh"], "env": {}}}

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

    def test_write_codex_uses_http_headers(self, tmp_path):
        """Codex calls HTTP transport headers ``http_headers``."""
        path = tmp_path / "config.toml"
        servers = {
            "remote": {
                "url": "https://example.com/mcp",
                "headers": {"Authorization": "Basic redacted"},
            }
        }

        mcpshare.write_codex(path, servers)

        parsed = mcpshare._parse_toml_mcp_servers(path.read_text())
        assert parsed["remote"]["http_headers"] == {"Authorization": "Basic redacted"}
        assert "headers" not in parsed["remote"]

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

    def test_read_antigravity(self, tmp_path):
        path = tmp_path / "mcp_config.json"
        data = {
            "mcpServers": {
                "git": {"command": "uvx", "args": ["mcp-server-git"], "env": {"A": "1"}, "cwd": "/tmp"},
                "remote": {"serverUrl": "https://example.com/mcp", "headers": {"X-Key": "k"}, "disabled": True},
            }
        }
        path.write_text(json.dumps(data))
        result = mcpshare.read_antigravity(path)
        assert result == {
            "git": {"command": "uvx", "args": ["mcp-server-git"], "env": {"A": "1"}, "cwd": "/tmp"},
            "remote": {"url": "https://example.com/mcp", "headers": {"X-Key": "k"}},
        }

    def test_read_antigravity_missing_file(self, tmp_path):
        assert mcpshare.read_antigravity(tmp_path / "nope") == {}

    def test_read_antigravity_raises_on_invalid_json(self, tmp_path):
        path = tmp_path / "mcp_config.json"
        path.write_text("{not valid json")
        with pytest.raises(mcpshare.ConfigParseError, match="Invalid JSON"):
            mcpshare.read_antigravity(path)

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

    @pytest.mark.parametrize("reader_name", ["vscode", "claude", "copilot", "gemini", "antigravity", "opencode"])
    def test_read_raises_on_invalid_json(self, tmp_path, reader_name):
        """JSON-based readers raise ConfigParseError on malformed input."""
        path = tmp_path / "bad.json"
        path.write_text("{not valid json")
        with pytest.raises(mcpshare.ConfigParseError, match="Invalid JSON"):
            mcpshare.READERS[reader_name](path)

    def test_read_codex_raises_on_invalid_toml(self, tmp_path):
        path = tmp_path / "config.toml"
        path.write_text("[mcp_servers.broken\ncommand = ")
        with pytest.raises(mcpshare.ConfigParseError, match="Invalid TOML"):
            mcpshare.read_codex(path)

    @pytest.mark.parametrize(
        "reader",
        [
            mcpshare.read_vscode,
            mcpshare.read_claude,
            mcpshare.read_copilot,
            mcpshare.read_gemini,
            mcpshare.read_antigravity,
            mcpshare.read_opencode,
        ],
    )
    def test_read_empty_file_returns_empty(self, tmp_path, reader):
        """An empty placeholder file holds no servers; it is not a parse error."""
        path = tmp_path / "empty.json"
        path.write_text("")
        assert reader(path) == {}


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
        path = tmp_path / ".claude.json"
        mcpshare.write_claude(path, sample_servers)
        data = json.loads(path.read_text())
        assert "mcpServers" in data
        assert data["mcpServers"]["filesystem"]["type"] == "stdio"
        assert data["mcpServers"]["filesystem"]["command"] == "npx"

    def test_write_claude_http_type_for_url_servers(self, tmp_path):
        """Claude rejects a remote server without an explicit type."""
        path = tmp_path / ".claude.json"
        servers = {"remote": {"url": "https://example.com/mcp", "headers": {"X-Key": "k"}}}
        mcpshare.write_claude(path, servers)
        data = json.loads(path.read_text())
        assert data["mcpServers"]["remote"] == {
            "type": "http",
            "url": "https://example.com/mcp",
            "headers": {"X-Key": "k"},
        }

    def test_write_claude_preserves_claude_state(self, tmp_path, sample_servers):
        """Only mcpServers is replaced; Claude's other state in .claude.json survives."""
        path = tmp_path / ".claude.json"
        path.write_text(
            json.dumps(
                {
                    "numStartups": 42,
                    "projects": {"/some/repo": {"allowedTools": [], "mcpServers": {}}},
                    "mcpServers": {"stale-mcp": {"type": "stdio", "command": "old"}},
                }
            )
        )
        path.chmod(0o600)
        mcpshare.write_claude(path, sample_servers)
        data = json.loads(path.read_text())
        assert data["numStartups"] == 42
        assert data["projects"] == {"/some/repo": {"allowedTools": [], "mcpServers": {}}}
        assert set(data["mcpServers"]) == {"filesystem", "github"}
        assert path.stat().st_mode & 0o777 == 0o600
        assert not path.with_name(".claude.json.tmp").exists()

    def test_write_claude_keeps_symlink(self, tmp_path, sample_servers):
        """A symlinked .claude.json (e.g. from a dotfiles repo) stays a symlink."""
        real = tmp_path / "dotfiles" / "claude.json"
        real.parent.mkdir()
        real.write_text(json.dumps({"numStartups": 1}))
        link = tmp_path / ".claude.json"
        link.symlink_to(real)
        mcpshare.write_claude(link, sample_servers)
        assert link.is_symlink()
        data = json.loads(real.read_text())
        assert data["numStartups"] == 1
        assert set(data["mcpServers"]) == {"filesystem", "github"}

    def test_write_claude_omits_disabled_servers(self, tmp_path, sample_servers):
        """Claude has no global disable list, so disabled servers are left out."""
        path = tmp_path / ".claude.json"
        count, skipped = mcpshare.write_claude(path, sample_servers, disabled_names={"filesystem"})
        data = json.loads(path.read_text())
        assert set(data["mcpServers"]) == {"github"}
        assert count == 1
        assert skipped == []

    def test_write_claude_drops_non_claude_keys(self, tmp_path):
        """Codex 'tools' allow-lists and stray 'disabled' flags are not written."""
        path = tmp_path / ".claude.json"
        servers = {"azure-mcp": {"command": "npx", "args": ["-y", "x"], "tools": ["a", "b"], "disabled": True}}
        mcpshare.write_claude(path, servers)
        data = json.loads(path.read_text())
        assert data["mcpServers"]["azure-mcp"] == {"type": "stdio", "command": "npx", "args": ["-y", "x"]}

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

    def test_write_antigravity(self, tmp_path, sample_servers):
        path = tmp_path / "mcp_config.json"
        mcpshare.write_antigravity(path, sample_servers)
        data = json.loads(path.read_text())
        # stdio entries pass through unchanged; Antigravity has no 'type' field.
        assert data["mcpServers"] == sample_servers

    def test_write_antigravity_server_url_for_url_servers(self, tmp_path):
        """Antigravity only accepts 'serverUrl' for remote servers, not 'url'."""
        path = tmp_path / "mcp_config.json"
        servers = {"remote": {"url": "https://example.com/mcp", "headers": {"X-Key": "k"}}}
        mcpshare.write_antigravity(path, servers)
        data = json.loads(path.read_text())
        assert data["mcpServers"]["remote"] == {"serverUrl": "https://example.com/mcp", "headers": {"X-Key": "k"}}

    def test_write_antigravity_preserves_other_keys(self, tmp_path, sample_servers):
        path = tmp_path / "mcp_config.json"
        path.write_text(json.dumps({"futureKey": {"a": 1}, "mcpServers": {"stale-mcp": {"command": "old"}}}))
        mcpshare.write_antigravity(path, sample_servers)
        data = json.loads(path.read_text())
        assert data["futureKey"] == {"a": 1}
        assert set(data["mcpServers"]) == {"filesystem", "github"}

    def test_write_antigravity_drops_non_antigravity_keys(self, tmp_path):
        """'type', Codex 'tools' allow-lists and stray 'disabled' flags are not written."""
        path = tmp_path / "mcp_config.json"
        servers = {
            "azure-mcp": {"type": "stdio", "command": "npx", "args": ["-y", "x"], "tools": ["a"], "disabled": True}
        }
        mcpshare.write_antigravity(path, servers)
        data = json.loads(path.read_text())
        assert data["mcpServers"]["azure-mcp"] == {"command": "npx", "args": ["-y", "x"]}

    def test_write_antigravity_disabled_flag(self, tmp_path, sample_servers):
        path = tmp_path / "mcp_config.json"
        count, skipped = mcpshare.write_antigravity(path, sample_servers, disabled_names={"filesystem"})
        data = json.loads(path.read_text())
        assert data["mcpServers"]["filesystem"]["disabled"] is True
        assert "disabled" not in data["mcpServers"]["github"]
        assert (count, skipped) == (2, [])

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

    # Every JSON writer does a read-modify-write of its target file, so each
    # must cope with the file existing but holding nothing (or garbage).
    JSON_WRITERS = [
        pytest.param(mcpshare.write_vscode, "mcp.json", "servers", id="vscode"),
        pytest.param(mcpshare.write_claude, ".claude.json", "mcpServers", id="claude"),
        pytest.param(mcpshare.write_copilot, "mcp-config.json", "mcpServers", id="copilot"),
        pytest.param(mcpshare.write_gemini, "settings.json", "mcpServers", id="gemini"),
        pytest.param(mcpshare.write_antigravity, "mcp_config.json", "mcpServers", id="antigravity"),
        pytest.param(mcpshare.write_opencode, "opencode.json", "mcp", id="opencode"),
    ]

    @pytest.mark.parametrize("writer,filename,section", JSON_WRITERS)
    @pytest.mark.parametrize("content", ["", "  \n"], ids=["empty", "whitespace"])
    def test_write_treats_empty_existing_file_as_empty_object(
        self, tmp_path, sample_servers, writer, filename, section, content
    ):
        """Regression: Antigravity ships a 0-byte mcp_config.json, which used to crash json.load."""
        path = tmp_path / filename
        path.write_text(content)
        count, skipped = writer(path, sample_servers)
        data = json.loads(path.read_text())
        assert set(data[section]) == {"filesystem", "github"}
        assert (count, skipped) == (2, [])

    @pytest.mark.parametrize("writer,filename,section", JSON_WRITERS)
    def test_write_raises_on_invalid_json_and_leaves_file_untouched(
        self, tmp_path, sample_servers, writer, filename, section
    ):
        """A file we can't parse may hold settings we can't see; never overwrite it."""
        path = tmp_path / filename
        path.write_text("{not valid json")
        with pytest.raises(mcpshare.ConfigParseError, match="Invalid JSON"):
            writer(path, sample_servers)
        assert path.read_text() == "{not valid json"
        assert not path.with_name(filename + ".tmp").exists()

    @pytest.mark.parametrize("writer,filename,section", JSON_WRITERS)
    def test_write_raises_on_non_object_top_level(self, tmp_path, sample_servers, writer, filename, section):
        path = tmp_path / filename
        path.write_text("[1, 2, 3]")
        with pytest.raises(mcpshare.ConfigParseError, match="Expected JSON object"):
            writer(path, sample_servers)
        assert path.read_text() == "[1, 2, 3]"

    def test_write_codex_raises_on_invalid_toml_and_leaves_file_untouched(self, tmp_path, sample_servers):
        path = tmp_path / "config.toml"
        path.write_text("[settings\nbroken = ")
        with pytest.raises(mcpshare.ConfigParseError, match="Invalid TOML"):
            mcpshare.write_codex(path, sample_servers)
        assert path.read_text() == "[settings\nbroken = "
        assert not path.with_name("config.toml.tmp").exists()

    ALL_WRITERS = [
        pytest.param(mcpshare.write_vscode, "mcp.json", "{}\n", id="vscode"),
        pytest.param(mcpshare.write_claude, ".claude.json", "{}\n", id="claude"),
        pytest.param(mcpshare.write_copilot, "mcp-config.json", "{}\n", id="copilot"),
        pytest.param(mcpshare.write_codex, "config.toml", '[settings]\nmodel = "o3"\n', id="codex"),
        pytest.param(mcpshare.write_gemini, "settings.json", "{}\n", id="gemini"),
        pytest.param(mcpshare.write_antigravity, "mcp_config.json", "{}\n", id="antigravity"),
        pytest.param(mcpshare.write_opencode, "opencode.json", "{}\n", id="opencode"),
    ]

    @pytest.mark.parametrize("writer,filename,seed", ALL_WRITERS)
    def test_all_writers_are_atomic_preserve_symlinks_and_permissions(
        self, tmp_path, sample_servers, writer, filename, seed
    ):
        """Every writer writes through symlinks, preserves file mode, and leaves no .tmp file."""
        real_dir = tmp_path / "dotfiles"
        real_dir.mkdir()
        real = real_dir / filename
        real.write_text(seed)
        real.chmod(0o600)
        link_dir = tmp_path / "target"
        link_dir.mkdir()
        link = link_dir / filename
        link.symlink_to(real)

        writer(link, sample_servers)

        assert link.is_symlink()
        assert real.stat().st_mode & 0o777 == 0o600
        assert not real_dir.joinpath(filename + ".tmp").exists()
        assert not link_dir.joinpath(filename + ".tmp").exists()


# ---------------------------------------------------------------------------
# Round-trip tests (read → write → read)
# ---------------------------------------------------------------------------


class TestRoundTrip:
    @pytest.mark.parametrize(
        "tool,writer,reader,filename",
        [
            ("vscode", mcpshare.write_vscode, mcpshare.read_vscode, "mcp.json"),
            ("claude", mcpshare.write_claude, mcpshare.read_claude, ".claude.json"),
            ("copilot", mcpshare.write_copilot, mcpshare.read_copilot, "mcp-config.json"),
            ("codex", mcpshare.write_codex, mcpshare.read_codex, "config.toml"),
            ("gemini", mcpshare.write_gemini, mcpshare.read_gemini, "settings.json"),
            ("antigravity", mcpshare.write_antigravity, mcpshare.read_antigravity, "mcp_config.json"),
            ("opencode", mcpshare.write_opencode, mcpshare.read_opencode, "opencode.json"),
        ],
    )
    def test_round_trip(self, tmp_path, sample_servers, tool, writer, reader, filename):
        path = tmp_path / filename
        writer(path, sample_servers)
        result = reader(path)
        assert result == sample_servers

    @pytest.mark.parametrize(
        "writer,reader,filename",
        [
            pytest.param(mcpshare.write_vscode, mcpshare.read_vscode, "mcp.json", id="vscode"),
            pytest.param(mcpshare.write_claude, mcpshare.read_claude, ".claude.json", id="claude"),
            pytest.param(mcpshare.write_copilot, mcpshare.read_copilot, "mcp-config.json", id="copilot"),
            pytest.param(mcpshare.write_codex, mcpshare.read_codex, "config.toml", id="codex"),
            pytest.param(mcpshare.write_gemini, mcpshare.read_gemini, "settings.json", id="gemini"),
            pytest.param(mcpshare.write_antigravity, mcpshare.read_antigravity, "mcp_config.json", id="antigravity"),
            pytest.param(mcpshare.write_opencode, mcpshare.read_opencode, "opencode.json", id="opencode"),
        ],
    )
    def test_round_trip_remote_url_server(self, tmp_path, writer, reader, filename):
        """Remote URL + headers round-trips across all 7 tools (including serverUrl and http_headers)."""
        path = tmp_path / filename
        servers = {"remote": {"url": "https://example.com/mcp", "headers": {"X-Key": "k"}}}
        writer(path, servers)
        assert reader(path) == servers


# ---------------------------------------------------------------------------
# Collect / Distribute integration tests
# ---------------------------------------------------------------------------


class TestCollectDistribute:
    def _make_config(self, tmp_path):
        source = str(tmp_path / "master")
        targets = {}
        for tool in ("claude", "vscode", "copilot", "codex", "gemini", "antigravity", "opencode"):
            tdir = tmp_path / tool
            tdir.mkdir()
            targets[tool] = {"path": str(tdir)}
        return {"source": source, "init-source": list(targets), "mode": "merge", "targets": targets}

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

    def test_collect_only_reads_configured_init_sources(self, tmp_path, sample_servers):
        """Tools absent from init-source are never pulled, even when configured in targets."""
        config = self._make_config(tmp_path)
        config["init-source"] = ["vscode", "claude"]

        mcpshare.write_vscode(
            mcpshare.resolve_target_path("vscode", config["targets"]["vscode"]["path"]),
            {"filesystem": sample_servers["filesystem"]},
        )
        mcpshare.write_claude(
            mcpshare.resolve_target_path("claude", config["targets"]["claude"]["path"]),
            {"github": sample_servers["github"]},
        )
        mcpshare.write_gemini(
            mcpshare.resolve_target_path("gemini", config["targets"]["gemini"]["path"]),
            {"ignored-mcp": {"command": "uvx", "args": ["ignored"]}},
        )

        master = mcpshare.collect(config, interactive=False)
        assert set(master["mcpServers"]) == {"filesystem", "github"}

    def test_collect_warns_and_skips_unknown_init_source(self, tmp_path, sample_servers, caplog):
        """A typo like 'cluade' in init-source logs a clear warning and continues."""
        config = self._make_config(tmp_path)
        config["init-source"] = ["vscode", "cluade"]
        mcpshare.write_vscode(
            mcpshare.resolve_target_path("vscode", config["targets"]["vscode"]["path"]),
            {"filesystem": sample_servers["filesystem"]},
        )

        with caplog.at_level("WARNING"):
            master = mcpshare.collect(config, interactive=False)

        assert set(master["mcpServers"]) == {"filesystem"}
        assert "cluade" in caplog.text

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

    def _setup_sync(self, tmp_path, sample_servers, tools):
        source = str(tmp_path / "master")
        mcpshare.save_master(source, {"mcpServers": sample_servers})
        targets = {}
        for tool in tools:
            tdir = tmp_path / tool
            tdir.mkdir(exist_ok=True)
            targets[tool] = {"path": str(tdir)}
        mcpshare.save_config({"source": source, "mode": "merge", "targets": targets})
        return targets

    def test_sync_completes_with_empty_antigravity_placeholder(self, tmp_home, tmp_path, sample_servers):
        """The reported crash: `mcpshare sync` against Antigravity's 0-byte mcp_config.json."""
        targets = self._setup_sync(tmp_path, sample_servers, ("antigravity",))
        ag_path = mcpshare.resolve_target_path("antigravity", targets["antigravity"]["path"])
        ag_path.touch()

        mcpshare.cmd_sync(mcpshare.Sync())

        assert set(json.loads(ag_path.read_text())["mcpServers"]) == {"filesystem", "github"}

    def test_sync_fails_when_a_target_cannot_be_updated(self, tmp_home, tmp_path, sample_servers):
        """A skipped target turns into a non-zero exit via McpShareError, after the others are written."""
        targets = self._setup_sync(tmp_path, sample_servers, ("antigravity", "opencode"))
        ag_path = mcpshare.resolve_target_path("antigravity", targets["antigravity"]["path"])
        oc_path = mcpshare.resolve_target_path("opencode", targets["opencode"]["path"])
        ag_path.write_text("{not valid json")

        with pytest.raises(mcpshare.McpShareError, match="Sync incomplete: could not update antigravity"):
            mcpshare.cmd_sync(mcpshare.Sync())

        assert ag_path.read_text() == "{not valid json"
        assert set(json.loads(oc_path.read_text())["mcp"]) == {"filesystem", "github"}


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

    def test_toml_key_escapes_special_characters(self):
        payload = 'evil"]\ncommand = "/bin/sh"\nargs = ["-c", "id > /tmp/pwned"]\n#'
        quoted = mcpshare._toml_key(payload)
        assert "\n" not in quoted
        assert '\n' not in quoted
        assert '\\"' in quoted
        assert '\\n' in quoted

    def test_write_codex_prevents_toml_injection(self, tmp_path):
        path = tmp_path / "config.toml"
        payload = 'evil"]\ncommand = "/bin/sh"\nargs = ["-c", "id > /tmp/pwned"]\n#'
        servers = {payload: {"command": "safe-cmd", "args": ["safe-arg"]}}
        mcpshare.write_codex(path, servers)
        content = path.read_text()

        # The serialized TOML must be parseable and must have exactly one server
        # with the malicious payload as its key, rather than an injected server.
        parsed = mcpshare.read_codex(path)
        assert set(parsed.keys()) == {payload}
        assert parsed[payload]["command"] == "safe-cmd"
        assert parsed[payload]["args"] == ["safe-arg"]


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
        path = tmp_path / ".claude.json"
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

    @pytest.mark.parametrize("init_order", [["vscode", "copilot"], ["copilot", "vscode"]])
    def test_vscode_env_placeholder_does_not_conflict_or_leak_secret_after_sync(
        self, tmp_path, monkeypatch, init_order
    ):
        """Regression: sync expands ${env:VAR} into Copilot; pull must not conflict or store the secret."""
        monkeypatch.setenv("GH_MCP_TOKEN", "s3cr3t-value")
        source = str(tmp_path / "master")
        targets = {}
        for tool in ("vscode", "copilot"):
            tdir = tmp_path / tool
            tdir.mkdir()
            targets[tool] = {"path": str(tdir)}
        config = {"source": source, "init-source": init_order, "mode": "merge", "targets": targets}

        seeded = {"gh-mcp": {"command": "npx", "args": ["-y", "gh"], "env": {"TOKEN": "${env:GH_MCP_TOKEN}"}}}
        mcpshare.save_master(source, {"mcpServers": seeded})
        mcpshare.distribute(config, {"mcpServers": seeded})

        def fail_prompt(*_a, **_kw):
            raise AssertionError("${env:VAR} vs its resolved value must not prompt as a conflict")

        monkeypatch.setattr(mcpshare, "_prompt_conflict", fail_prompt)
        monkeypatch.setattr(mcpshare.sys.stdin, "isatty", lambda: True)

        # Both interactive and non-interactive pulls preserve ${env:GH_MCP_TOKEN}.
        assert mcpshare.collect(config, interactive=True)["mcpServers"] == seeded
        assert mcpshare.collect(config, interactive=False)["mcpServers"] == seeded

    def test_vscode_env_placeholder_upgrades_literal_on_initial_pull(self, tmp_path, monkeypatch):
        """When master is empty and copilot precedes vscode, the ${env:VAR} placeholder still wins."""
        monkeypatch.setenv("GH_MCP_TOKEN", "s3cr3t-value")
        config = self._seed(
            tmp_path,
            vscode_cfg={"command": "npx", "env": {"TOKEN": "${env:GH_MCP_TOKEN}"}},
            copilot_cfg={"command": "npx", "env": {"TOKEN": "s3cr3t-value"}},
        )
        config["init-source"] = ["copilot", "vscode"]

        master = mcpshare.collect(config, interactive=False)
        assert master["mcpServers"]["shared-mcp"]["env"]["TOKEN"] == "${env:GH_MCP_TOKEN}"

    @pytest.mark.parametrize("init_order", [["vscode", "claude"], ["claude", "vscode"]])
    def test_schema_unsupported_keys_do_not_conflict_or_get_stripped_on_pull(self, tmp_path, monkeypatch, init_order):
        """Claude drops Codex 'tools' on write; pulling from Claude must neither conflict nor erase 'tools'."""
        source = str(tmp_path / "master")
        targets = {}
        for tool in ("vscode", "claude", "opencode"):
            tdir = tmp_path / tool
            tdir.mkdir()
            targets[tool] = {"path": str(tdir)}
        config = {"source": source, "init-source": [*init_order, "opencode"], "mode": "merge", "targets": targets}

        seeded = {
            "azure-mcp": {"command": "npx", "args": ["-y", "@azure/mcp"], "cwd": "/tmp", "tools": ["*"]},
        }
        mcpshare.save_master(source, {"mcpServers": seeded})
        mcpshare.distribute(config, {"mcpServers": seeded})

        def fail_prompt(*_a, **_kw):
            raise AssertionError("schema-dropped keys must not trigger a conflict")

        monkeypatch.setattr(mcpshare, "_prompt_conflict", fail_prompt)
        monkeypatch.setattr(mcpshare.sys.stdin, "isatty", lambda: True)

        assert mcpshare.collect(config, interactive=False)["mcpServers"] == seeded


# ---------------------------------------------------------------------------
# Distribute: in-place MCP section replacement and per-IDE disabled filtering
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

    def test_vscode_replaces_stale_servers_while_preserving_inputs(self, tmp_home, tmp_path, sample_servers):
        """Removed servers disappear from VSCode mcp.json, while VSCode's top-level 'inputs' survives."""
        config = self._make_config(tmp_path, tools=("vscode",))
        vscode_path = mcpshare.resolve_target_path("vscode", config["targets"]["vscode"]["path"])
        inputs = [{"id": "api-key", "type": "promptString", "description": "API Key", "password": True}]
        vscode_path.write_text(
            json.dumps(
                {
                    "inputs": inputs,
                    "servers": {"stale-mcp": {"type": "stdio", "command": "old"}},
                }
            )
        )
        master = {"mcpServers": {"filesystem": sample_servers["filesystem"]}}

        mcpshare.distribute(config, master)

        data = json.loads(vscode_path.read_text())
        assert data["inputs"] == inputs
        assert set(data["servers"]) == {"filesystem"}

    def test_vscode_invalid_json_is_left_untouched_not_deleted(self, tmp_home, tmp_path, sample_servers, caplog):
        """A syntax typo in VSCode mcp.json must not cause sync to delete the user's file."""
        config = self._make_config(tmp_path, tools=("vscode",))
        vscode_path = mcpshare.resolve_target_path("vscode", config["targets"]["vscode"]["path"])
        vscode_path.write_text("{not valid json")

        with caplog.at_level("ERROR"):
            failed = mcpshare.distribute(config, {"mcpServers": sample_servers})

        assert failed == ["vscode"]
        assert vscode_path.read_text() == "{not valid json"

    # ---- unparseable / empty target files ----

    def test_antigravity_empty_placeholder_file_is_synced(self, tmp_home, tmp_path, sample_servers):
        """Regression: Antigravity creates a 0-byte mcp_config.json; sync must fill it, not crash."""
        config = self._make_config(tmp_path, tools=("antigravity",))
        ag_path = mcpshare.resolve_target_path("antigravity", config["targets"]["antigravity"]["path"])
        ag_path.touch()

        failed = mcpshare.distribute(config, {"mcpServers": sample_servers})

        assert failed == []
        assert set(json.loads(ag_path.read_text())["mcpServers"]) == {"filesystem", "github"}

    def test_invalid_target_file_is_skipped_and_others_still_written(self, tmp_home, tmp_path, sample_servers, caplog):
        """One unparseable shared-purpose file must not abort the sync or get clobbered."""
        config = self._make_config(tmp_path, tools=("antigravity", "gemini"))
        ag_path = mcpshare.resolve_target_path("antigravity", config["targets"]["antigravity"]["path"])
        gemini_path = mcpshare.resolve_target_path("gemini", config["targets"]["gemini"]["path"])
        ag_path.write_text("{not valid json")

        with caplog.at_level("ERROR"):
            failed = mcpshare.distribute(config, {"mcpServers": sample_servers})

        assert failed == ["antigravity"]
        assert ag_path.read_text() == "{not valid json"
        assert "Cannot update antigravity" in caplog.text
        assert str(ag_path) in caplog.text
        assert set(json.loads(gemini_path.read_text())["mcpServers"]) == {"filesystem", "github"}

    def test_copilot_invalid_settings_json_is_left_untouched(self, tmp_home, tmp_path, sample_servers, caplog):
        """A malformed Copilot settings.json blocks only the disabled list, not mcp-config.json."""
        config = self._make_config(tmp_path, tools=("copilot",))
        copilot_dir = Path(config["targets"]["copilot"]["path"])
        (copilot_dir / "settings.json").write_text("{not valid json")
        mcpshare.save_disabled(config["source"], {"*": ["filesystem"]})

        with caplog.at_level("ERROR"):
            mcpshare.distribute(config, {"mcpServers": sample_servers})

        assert (copilot_dir / "settings.json").read_text() == "{not valid json"
        assert "Cannot update" in caplog.text
        servers_data = json.loads((copilot_dir / "mcp-config.json").read_text())
        assert set(servers_data["mcpServers"]) == {"filesystem", "github"}

    def test_claude_cleanup_tolerates_empty_settings_json(self, tmp_home, tmp_path, sample_servers, caplog):
        """An empty ~/.claude/settings.json has nothing to clean up and is not an error."""
        config = self._make_config(tmp_path, tools=("claude",))
        claude_dir = Path(config["targets"]["claude"]["path"])
        (claude_dir / "settings.json").write_text("")

        with caplog.at_level("ERROR"):
            failed = mcpshare.distribute(config, {"mcpServers": sample_servers})

        assert failed == []
        assert "Cannot clean up" not in caplog.text
        assert (claude_dir / "settings.json").read_text() == ""

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

    # ---- in_file_flag strategy (antigravity) ----

    def test_antigravity_writes_disabled_true(self, tmp_home, tmp_path, sample_servers):
        config = self._make_config(tmp_path, tools=("antigravity",))
        mcpshare.save_config(config)
        mcpshare.save_disabled(config["source"], {"*": ["filesystem"]})

        mcpshare.distribute(config, {"mcpServers": sample_servers})

        ag_data = json.loads(
            mcpshare.resolve_target_path("antigravity", config["targets"]["antigravity"]["path"]).read_text()
        )
        # Server stays listed so it can be toggled in Antigravity's UI.
        assert ag_data["mcpServers"]["filesystem"]["disabled"] is True
        assert "disabled" not in ag_data["mcpServers"]["github"]

    def test_antigravity_per_ide_disable(self, tmp_home, tmp_path, sample_servers):
        """An antigravity-only disable entry affects Antigravity alone."""
        config = self._make_config(tmp_path, tools=("antigravity", "opencode"))
        mcpshare.save_config(config)
        mcpshare.save_disabled(config["source"], {"antigravity": ["filesystem"]})

        mcpshare.distribute(config, {"mcpServers": sample_servers})

        ag_data = json.loads(
            mcpshare.resolve_target_path("antigravity", config["targets"]["antigravity"]["path"]).read_text()
        )
        opencode_data = json.loads(
            mcpshare.resolve_target_path("opencode", config["targets"]["opencode"]["path"]).read_text()
        )
        assert ag_data["mcpServers"]["filesystem"]["disabled"] is True
        assert "enabled" not in opencode_data["mcp"]["filesystem"]

    def test_master_disabled_flag_is_ignored_by_antigravity(self, tmp_home, tmp_path, sample_servers):
        """A stray 'disabled: true' on master never reaches mcp_config.json."""
        config = self._make_config(tmp_path, tools=("antigravity",))
        mcpshare.save_config(config)
        seeded = dict(sample_servers)
        seeded["filesystem"] = dict(seeded["filesystem"], disabled=True)

        mcpshare.distribute(config, {"mcpServers": seeded})

        ag_data = json.loads(
            mcpshare.resolve_target_path("antigravity", config["targets"]["antigravity"]["path"]).read_text()
        )
        assert "disabled" not in ag_data["mcpServers"]["filesystem"]

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

    # ---- omit strategy (claude: ~/.claude.json has no global disable) ----

    def test_claude_omits_disabled_from_claude_json(self, tmp_home, tmp_path, sample_servers):
        """Disabled servers are left out of .claude.json; settings.json is not touched."""
        config = self._make_config(tmp_path, tools=("claude",))
        mcpshare.save_config(config)
        mcpshare.save_disabled(config["source"], {"*": ["filesystem"]})

        mcpshare.distribute(config, {"mcpServers": sample_servers})

        claude_dir = Path(config["targets"]["claude"]["path"])
        claude_data = json.loads((claude_dir / ".claude.json").read_text())
        assert set(claude_data["mcpServers"]) == {"github"}
        assert not (claude_dir / "settings.json").exists()

    def test_claude_per_ide_disable(self, tmp_home, tmp_path, sample_servers):
        """A claude-only disable entry omits the server from Claude alone."""
        config = self._make_config(tmp_path, tools=("claude", "copilot"))
        mcpshare.save_config(config)
        mcpshare.save_disabled(config["source"], {"claude": ["filesystem"]})

        mcpshare.distribute(config, {"mcpServers": sample_servers})

        claude_data = json.loads(
            mcpshare.resolve_target_path("claude", config["targets"]["claude"]["path"]).read_text()
        )
        copilot_data = json.loads(
            mcpshare.resolve_target_path("copilot", config["targets"]["copilot"]["path"]).read_text()
        )
        assert "filesystem" not in claude_data["mcpServers"]
        assert "filesystem" in copilot_data["mcpServers"]

    def test_claude_json_is_merged_not_recreated(self, tmp_home, tmp_path, sample_servers):
        """distribute must never delete .claude.json: it holds all of Claude's state."""
        config = self._make_config(tmp_path, tools=("claude",))
        claude_path = mcpshare.resolve_target_path("claude", config["targets"]["claude"]["path"])
        claude_path.write_text(
            json.dumps(
                {
                    "numStartups": 7,
                    "projects": {"/repo": {"disabledMcpServers": ["x-mcp"]}},
                    "mcpServers": {"stale-mcp": {"type": "stdio", "command": "old"}},
                }
            )
        )

        mcpshare.distribute(config, {"mcpServers": sample_servers})

        data = json.loads(claude_path.read_text())
        assert data["numStartups"] == 7
        assert data["projects"] == {"/repo": {"disabledMcpServers": ["x-mcp"]}}
        assert set(data["mcpServers"]) == {"filesystem", "github"}

    def test_claude_legacy_files_cleaned_up(self, tmp_home, tmp_path, sample_servers):
        """Old mcp_servers.json and settings.json/disabledMcpServers are removed."""
        config = self._make_config(tmp_path, tools=("claude",))
        claude_dir = Path(config["targets"]["claude"]["path"])
        (claude_dir / "mcp_servers.json").write_text(json.dumps({"mcpServers": {}}))
        (claude_dir / "settings.json").write_text(
            json.dumps({"theme": "dark", "disabledMcpServers": ["filesystem"], "permissions": {"allow": []}})
        )

        mcpshare.distribute(config, {"mcpServers": sample_servers})

        assert not (claude_dir / "mcp_servers.json").exists()
        settings_data = json.loads((claude_dir / "settings.json").read_text())
        assert settings_data == {"theme": "dark", "permissions": {"allow": []}}

    def test_claude_cleanup_skips_invalid_settings_json(self, tmp_home, tmp_path, sample_servers, caplog):
        """A malformed settings.json is left untouched and the sync still succeeds."""
        config = self._make_config(tmp_path, tools=("claude",))
        claude_dir = Path(config["targets"]["claude"]["path"])
        (claude_dir / "settings.json").write_text("{not valid json")

        with caplog.at_level("ERROR"):
            mcpshare.distribute(config, {"mcpServers": sample_servers})

        assert (claude_dir / "settings.json").read_text() == "{not valid json"
        assert "Cannot clean up" in caplog.text
        assert (claude_dir / ".claude.json").exists()

    # ---- settings_json strategy (copilot) ----

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
        config = self._make_config(tmp_path, tools=("copilot",))
        mcpshare.save_config(config)
        copilot_dir = Path(config["targets"]["copilot"]["path"])
        # Seed settings.json with unrelated keys the user owns.
        (copilot_dir / "settings.json").write_text(json.dumps({"theme": "dark", "permissions": {"allow": ["Read(*)"]}}))
        mcpshare.save_disabled(config["source"], {"*": ["filesystem"]})

        mcpshare.distribute(config, {"mcpServers": sample_servers})

        settings_data = json.loads((copilot_dir / "settings.json").read_text())
        assert settings_data["theme"] == "dark"
        assert settings_data["permissions"] == {"allow": ["Read(*)"]}
        assert settings_data["disabledMcpServers"] == ["filesystem"]

    def test_settings_json_overwrites_native_disabled_list(self, tmp_home, tmp_path, sample_servers):
        """mcpshare-owns: the disabledMcpServers array is replaced, not merged."""
        config = self._make_config(tmp_path, tools=("copilot",))
        mcpshare.save_config(config)
        copilot_dir = Path(config["targets"]["copilot"]["path"])
        # User had a different server disabled natively.
        (copilot_dir / "settings.json").write_text(json.dumps({"disabledMcpServers": ["github", "some-orphan"]}))
        # mcpshare disables only filesystem; github stays enabled.
        mcpshare.save_disabled(config["source"], {"*": ["filesystem"]})

        mcpshare.distribute(config, {"mcpServers": sample_servers})

        settings_data = json.loads((copilot_dir / "settings.json").read_text())
        assert settings_data["disabledMcpServers"] == ["filesystem"]

    def test_settings_json_re_enable_warning(self, tmp_home, tmp_path, sample_servers, caplog):
        """When mcpshare re-enables a natively-disabled server, log a warning."""
        config = self._make_config(tmp_path, tools=("copilot",))
        mcpshare.save_config(config)
        copilot_dir = Path(config["targets"]["copilot"]["path"])
        # User had 'github' disabled natively in Copilot.
        (copilot_dir / "settings.json").write_text(json.dumps({"disabledMcpServers": ["github"]}))
        # mcpshare doesn't know about that disable.
        with caplog.at_level("WARNING"):
            mcpshare.distribute(config, {"mcpServers": sample_servers})
        assert "re-enabling github" in caplog.text

    def test_settings_json_removed_when_no_disabled(self, tmp_home, tmp_path, sample_servers):
        """If no servers are disabled, the disabledMcpServers key is removed."""
        config = self._make_config(tmp_path, tools=("copilot",))
        mcpshare.save_config(config)
        copilot_dir = Path(config["targets"]["copilot"]["path"])
        (copilot_dir / "settings.json").write_text(json.dumps({"theme": "dark"}))

        mcpshare.distribute(config, {"mcpServers": sample_servers})

        settings_data = json.loads((copilot_dir / "settings.json").read_text())
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
        config = self._make_config(tmp_path, tools=("copilot",))
        mcpshare.save_config(config)
        seeded = dict(sample_servers)
        seeded["filesystem"] = dict(seeded["filesystem"], disabled=True)

        mcpshare.distribute(config, {"mcpServers": seeded})

        copilot_dir = Path(config["targets"]["copilot"]["path"])
        settings_path = copilot_dir / "settings.json"
        assert not settings_path.exists() or "disabledMcpServers" not in json.loads(settings_path.read_text())

    def test_master_disabled_flag_is_ignored_by_claude(self, tmp_home, tmp_path, sample_servers):
        """A stray 'disabled: true' on master neither omits the server nor reaches .claude.json."""
        config = self._make_config(tmp_path, tools=("claude",))
        mcpshare.save_config(config)
        seeded = dict(sample_servers)
        seeded["filesystem"] = dict(seeded["filesystem"], disabled=True)

        mcpshare.distribute(config, {"mcpServers": seeded})

        claude_data = json.loads(
            mcpshare.resolve_target_path("claude", config["targets"]["claude"]["path"]).read_text()
        )
        assert "filesystem" in claude_data["mcpServers"]
        assert "disabled" not in claude_data["mcpServers"]["filesystem"]

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

    def test_empty_settings_file_is_treated_as_empty_object(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text("")

        previous = mcpshare._write_native_disabled_list(path, {"a-mcp"})

        assert previous == set()
        assert json.loads(path.read_text()) == {"disabledMcpServers": ["a-mcp"]}

    def test_invalid_settings_file_raises_config_parse_error(self, tmp_path):
        """Callers (see _emit_native_disabled) rely on ConfigParseError, not a raw JSONDecodeError."""
        path = tmp_path / "settings.json"
        path.write_text("{not valid json")
        with pytest.raises(mcpshare.ConfigParseError, match="Invalid JSON"):
            mcpshare._write_native_disabled_list(path, {"a-mcp"})
        assert path.read_text() == "{not valid json"


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
